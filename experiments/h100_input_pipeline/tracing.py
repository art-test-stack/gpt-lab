"""Short, separate instrumentation and interval-based GPU idle accounting."""
from contextlib import contextmanager
import json

import torch

from recipe import union_length


@contextmanager
def ranges(trainer, loader):
    """Only profile runs pay Python annotation overhead; no extra synchronization."""
    forward, backward, optimizer = trainer.compute_loss, torch.Tensor.backward, trainer.optimizer_step
    copy_tensor = torch.Tensor.copy_

    def compute(payload):
        with torch.profiler.record_function("compute/forward"):
            return forward(payload)

    def back(tensor, *args, **kwargs):
        with torch.profiler.record_function("compute/backward"):
            return backward(tensor, *args, **kwargs)

    def step(log):
        with torch.profiler.record_function("compute/optimizer"):
            return optimizer(log)

    def transfer(tensor, source, *args, **kwargs):
        if tensor is loader.gpu and source is loader.cpu:
            with torch.profiler.record_function("input/transfer"):
                return copy_tensor(tensor, source, *args, **kwargs)
        return copy_tensor(tensor, source, *args, **kwargs)

    trainer.compute_loss, trainer.optimizer_step = compute, step
    torch.Tensor.backward, torch.Tensor.copy_ = back, transfer
    try:
        yield
    finally:
        trainer.compute_loss, trainer.optimizer_step = forward, optimizer
        torch.Tensor.backward, torch.Tensor.copy_ = backward, copy_tensor


class FetchRange:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return self

    def __next__(self):
        with torch.profiler.record_function("input/fetch"):
            return next(self.loader)


def summarize(path, *, expected_microsteps, expected_updates=2):
    events = json.loads(path.read_text())["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X" and event.get("dur", 0) > 0]
    gpu = [(e["ts"], e["ts"]+e["dur"]) for e in spans
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    fetch = [(e["ts"], e["ts"]+e["dur"]) for e in spans if e["name"] == "input/fetch"]
    transfer = [(e["ts"], e["ts"]+e["dur"]) for e in spans if e["name"] == "input/transfer"]
    updates = [e for e in spans if e["name"].startswith("diagnostic/update/")]
    if not gpu or not updates or not fetch or not transfer:
        raise RuntimeError("Missing CUDA activity, update, fetch or transfer ranges; inspect raw trace")
    if len(updates) != expected_updates:
        raise RuntimeError(f"Expected {expected_updates} complete profiled updates")
    result = []
    for update in updates:
        start, stop = update["ts"], update["ts"]+update["dur"]
        counts = {name:sum(e["name"] == name and start <= e["ts"] and e["ts"]+e["dur"] <= stop
                           for e in spans)
                  for name in ("input/fetch","input/transfer","compute/forward","compute/backward","compute/optimizer")}
        expected = {name:(1 if name == "compute/optimizer" else expected_microsteps) for name in counts}
        if counts != expected:
            raise RuntimeError(f"Incomplete update ranges: {counts}; expected {expected}")
        intervals = sorted((max(a,start), min(b,stop)) for a,b in gpu if b>start and a<stop)
        cursor, gaps = start, []
        for a,b in intervals:
            if a>cursor:
                gaps.append((cursor,a))
            cursor = max(cursor,b)
        if cursor<stop:
            gaps.append((cursor,stop))
        def overlap(a,b,items):
            return union_length([(max(a,x),min(b,y)) for x,y in items if x<b and y>a])
        result.append(dict(update=update["name"], range_counts=counts, wall_ms=(stop-start)/1000,
                           gpu_idle_ms=sum(b-a for a,b in gaps)/1000,
                           idle_overlapping_fetch_ms=sum(overlap(a,b,fetch) for a,b in gaps)/1000,
                           idle_overlapping_transfer_ms=sum(overlap(a,b,transfer) for a,b in gaps)/1000,
                           largest_gaps=[dict(start_us=a, duration_ms=(b-a)/1000,
                                              fetch_overlap_ms=overlap(a,b,fetch)/1000,
                                              transfer_overlap_ms=overlap(a,b,transfer)/1000)
                                         for a,b in sorted(gaps,key=lambda pair:pair[1]-pair[0],reverse=True)[:20]]))
    return dict(updates=result, interpretation="Idle is the complement of the union of CUDA kernels, memcpy and memset across streams. CPU range overlap is temporal association, not proof of causation. Fetch includes transfer, so overlap columns are not additive. Profile overhead is excluded from throughput measurements.")
