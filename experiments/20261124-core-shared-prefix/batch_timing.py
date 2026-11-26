from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import torch


def batches(plans, prepare, prefetch):
    if not prefetch:
        for plan in plans:
            yield prepare(plan)
        return
    iterator = iter(plans)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare, next(iterator))
        for plan in iterator:
            ready = future.result()
            future = pool.submit(prepare, plan)
            yield ready
        yield future.result()


def time_sweep(plans, prepare, score, device, prefetch):
    if device.type == "mps":
        torch.mps.synchronize()
    start = perf_counter()
    for batch in batches(plans, prepare, prefetch):
        score(batch)
    if device.type == "mps":
        torch.mps.synchronize()
    return perf_counter() - start


def logits_result(logits, nll, winner):
    return {
        "logits": [tensor.detach().cpu() for tensor in logits],
        "nll": None if nll is None else torch.as_tensor(nll).detach().cpu(),
        "winner": winner,
    }


def observe_rows(core, model, rows, starts, ends, *, task_type):
    """Record real logits and the reductions from the supplied CORE implementation."""
    captured = []
    hook = model.register_forward_hook(lambda _, args, output: captured.append(output.logits))
    try:
        losses, _ = core.forward_model(model, rows)
    finally:
        hook.remove()
    logits = [captured[0][i, start - 1:end - 1] for i, (start, end) in enumerate(zip(starts, ends))]
    nll, winner = None, None
    if task_type != "language_modeling":
        nll = [losses[i, start - 1:end - 1].mean().item() for i, (start, end) in enumerate(zip(starts, ends))]
        winner = nll.index(min(nll))
    return logits_result(logits, nll, winner)
