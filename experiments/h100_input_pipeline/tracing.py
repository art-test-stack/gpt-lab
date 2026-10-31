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


def summarize(path, *, expected_microsteps, expected_updates=2, expected_steps=None):
    events = json.loads(path.read_text())["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X" and event.get("dur", 0) > 0]
    gpu = [(e["ts"], e["ts"]+e["dur"]) for e in spans
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    # Kineto can reuse a numeric pid for CPU and GPU lanes: use thread metadata too.
    cpu_pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_labels"
                and "CPU" in e.get("args",{}).get("labels","").split(",")}
    cpu_threads = {(e["pid"],e["tid"]) for e in events if e.get("ph") == "M"
                   and e.get("name") == "thread_name" and e["pid"] in cpu_pids
                   and e.get("args",{}).get("name","").startswith("thread ")}
    if not cpu_threads:
        raise RuntimeError("Missing CPU process/thread metadata; cannot identify host ranges")
    host = [e for e in spans if e.get("cat") == "user_annotation"
            and (e.get("pid"),e.get("tid")) in cpu_threads]
    updates = sorted((e for e in host if e["name"].startswith("diagnostic/update/")),key=lambda e:e["ts"])
    fetch = [(e["ts"],e["ts"]+e["dur"]) for e in host if e["name"] == "input/fetch"]
    transfer = [(e["ts"],e["ts"]+e["dur"]) for e in host if e["name"] == "input/transfer"]
    if not gpu or not updates or not fetch or not transfer:
        raise RuntimeError("Missing CUDA activity, update, fetch or transfer ranges; inspect raw trace")
    if len(updates) != expected_updates:
        raise RuntimeError(f"Expected {expected_updates} complete profiled updates")
    steps = [int(e["name"].removeprefix("diagnostic/update/")) for e in updates]
    if expected_steps is not None and steps != list(expected_steps):
        raise RuntimeError(f"Expected update steps {list(expected_steps)}; got {steps}")
    result = []
    for update in updates:
        start, stop = update["ts"], update["ts"]+update["dur"]
        window = [e for e in host if e["pid"] == update["pid"]
                  and start <= e["ts"] and e["ts"]+e["dur"] <= stop]
        counts = {name:sum(e["name"] == name and start <= e["ts"] and e["ts"]+e["dur"] <= stop
                           for e in window)
                  for name in ("input/fetch","input/transfer","compute/forward","compute/backward","compute/optimizer")}
        expected = {name:(1 if name == "compute/optimizer" else expected_microsteps) for name in counts}
        if counts != expected:
            raise RuntimeError(f"Incomplete update ranges: {counts}; expected {expected}")
        fetch = [(e["ts"],e["ts"]+e["dur"]) for e in window if e["name"] == "input/fetch"]
        transfer = [(e["ts"],e["ts"]+e["dur"]) for e in window if e["name"] == "input/transfer"]
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
        idle = sum(b-a for a,b in gaps)
        idle_fetch = sum(overlap(a,b,fetch) for a,b in gaps)
        idle_transfer = sum(overlap(a,b,transfer) for a,b in gaps)
        both = [(max(a,x),min(b,y)) for a,b in fetch for x,y in transfer if x<b and y>a]
        idle_both = sum(overlap(a,b,both) for a,b in gaps)
        result.append(dict(update=update["name"], host_pid=update["pid"],host_tid=update["tid"],range_counts=counts,
                           host_range_union_ms={name:union_length([(e["ts"],e["ts"]+e["dur"]) for e in window if e["name"] == name])/1000 for name in counts},
                           wall_ms=(stop-start)/1000,
                           gpu_busy_union_ms=(stop-start-idle)/1000,gpu_idle_ms=idle/1000,
                           idle_overlapping_fetch_ms=idle_fetch/1000,
                           idle_overlapping_transfer_ms=idle_transfer/1000,
                           exclusive_idle_association_ms=dict(fetch_without_transfer=(idle_fetch-idle_both)/1000,
                               transfer_without_fetch=(idle_transfer-idle_both)/1000,fetch_and_transfer=idle_both/1000,
                               neither=(idle-idle_fetch-idle_transfer+idle_both)/1000),
                           largest_gaps=[dict(start_us=a, duration_ms=(b-a)/1000,
                                              fetch_overlap_ms=overlap(a,b,fetch)/1000,
                                              transfer_overlap_ms=overlap(a,b,transfer)/1000)
                                         for a,b in sorted(gaps,key=lambda pair:pair[1]-pair[0],reverse=True)[:20]]))
    return dict(updates=result,host_range_selection="CPU user_annotation events on CPU-labelled process/thread metadata; numeric CPU/GPU pids may collide",
                interpretation="Idle is the complement of the union of CUDA kernels, memcpy and memset across streams. CPU range overlap is temporal association, not proof of causation. Fetch includes transfer, so marginal overlap columns are not additive. The exclusive idle association buckets partition idle time without double-counting; they describe temporal association, not causal attribution. GPU annotations remain in the raw trace but are neither host ranges nor CUDA busy intervals. Profile overhead is excluded from throughput measurements.")
