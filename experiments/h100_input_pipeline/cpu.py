"""CPU preflight using the actual Parquet -> tokenizer -> production loader path."""
import argparse
import json
import os
from pathlib import Path
import time

from recipe import timing, BATCH_SIZES, resolve_cpu_batch


def production_factory(cache, batch_size=16, seq_len=2048):
    import torch
    from gpt_lab.data.loader import build_dataloader
    from gpt_lab.tokenizer.tokenizer import Tokenizer
    from gpt_lab.tokenizer.auto import load_tokenizer
    from gpt_lab.utils.special_tokens import SpecialTokens
    from gpt_lab.utils.distributed import ComputeContext
    loaded = load_tokenizer("gpt2", special_tokens=SpecialTokens(), tokenizer_dir=Path(cache)/"tokenizers")
    tokenizer = Tokenizer.from_config(loaded.config)
    return lambda: build_dataloader(name="climbmix-base", batch_size=batch_size,
                                    seq_len=seq_len, tokenizer=tokenizer,
                                    cachedir=Path(cache)/"data", comp_info=ComputeContext(torch.device("cpu")))


def preflight(run_name, warmup, measured):
    import hashlib
    import platform
    import torch
    from cache import digest_batch
    cache = Path("/cache/gpt_lab")
    data = cache / "data/climbmix-base"
    shards = sorted(data.glob("shard_*.parquet"))
    if len(shards) < 5 or [p.name for p in shards] != [f"shard_{i:05d}.parquet" for i in range(len(shards))]:
        raise RuntimeError("Need at least five existing contiguous production shards; no downloads")
    meta = json.loads((data/"meta.json").read_text())
    boundary = meta.get("max_shards")
    if type(boundary) is not int or not 4 <= boundary < len(shards):
        raise RuntimeError("Invalid existing train/validation boundary")
    if any(p.stat().st_size == 0 for p in shards):
        raise RuntimeError("Empty cached shard")
    if (Path("/runs")/run_name).exists():
        raise FileExistsError("Choose a fresh run name")
    results = {}
    for batch_size in BATCH_SIZES:
        resolved = resolve_cpu_batch(batch_size)
        print("Resolved CPU preflight batch: " + json.dumps(resolved),flush=True)
        setup_start = time.perf_counter()
        factory = production_factory(cache,batch_size=resolved["device_batch_size"],seq_len=resolved["seq_len"])
        tokenizer_setup = time.perf_counter()-setup_start
        count = resolved["n_acc_steps"]*(warmup+measured)
        # Separate startup from sustained CPU-only preparation. No H2D evidence here.
        result = measure(factory,count)
        loader = factory()
        digest = hashlib.sha256()
        for _ in range(count+1):
            digest.update(digest_batch(next(loader)).encode())
        result.update(tokenizer_setup_seconds=tokenizer_setup,data_order_sha256=digest.hexdigest(),
                      resolved_batch=resolved,source="production Modal cache",
                      host=platform.platform(),torch=torch.__version__,
                      torch_threads=torch.get_num_threads(),cpu_count=os.cpu_count(),
                      note="CPU-only preparation; excludes H2D and concurrent training contention")
        results[str(batch_size)] = result
    result = dict(batches=results,shards=[dict(name=p.name,bytes=p.stat().st_size) for p in shards],metadata=meta)
    return result


def measure(factory, count):
    start = time.perf_counter()
    loader = factory()
    initialized = time.perf_counter()
    raw = next(loader)
    first = time.perf_counter()
    samples = []
    for _ in range(count):
        before = time.perf_counter()
        raw = next(loader)
        samples.append(time.perf_counter()-before)
    return dict(initialization_seconds=initialized-start, first_batch_seconds=first-initialized,
                sustained=timing(samples, raw[0].numel()), batches=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=1200)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("batches must be positive")
    print(json.dumps(measure(production_factory(args.cache), args.batches), indent=2))
