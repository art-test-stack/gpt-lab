"""Measure the four PR #60 loaders against a single cached shard."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from tempfile import TemporaryDirectory


SCRATCH = Path(".tmp/codex/rust-dataloader/four-backend")
SOURCE = Path(".tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet")
PATHS = {
    "A": ("Scalar Python", "python", 1),
    "B": ("Batched Python", "python", 64),
    "C": ("Batched Python + Rust packing", "rust-packing", 64),
    "D": ("End-to-end Rust", "rust", 64),
}
ORDERS = ("ABCD", "BDAC", "CADB", "DCBA")
BATCH, CONTEXT, THREADS = 16, 2048, 4


def worker(mode, key, source):
    import pyarrow as pa
    import torch

    from gpt_lab.data.loader import build_dataloader
    from gpt_lab.tokenizer import Tokenizer
    from gpt_lab.utils.distributed import ComputeContext

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizer = Tokenizer.from_pretrained("gpt2", source="tiktoken")
    bos = tokenizer.get_bos_token_id()
    context = ComputeContext(torch.device("cpu"))
    _, backend, encode_batch = PATHS[key]
    shard_id = int(source.stem.split("_")[-1])
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=SCRATCH, prefix="shard-cache-") as temporary:
        dataset = Path(temporary) / "climbmix"
        dataset.mkdir()
        (dataset / source.name).symlink_to(source)
        start = time.perf_counter()
        loader = build_dataloader(
            dataset.name, BATCH, CONTEXT, tokenizer, split="val", cachedir=dataset.parent,
            max_shards=shard_id, comp_info=context, tokenizer_threads=THREADS,
            tokenizer_batch_size=encode_batch, dataloader_backend=backend,
        )()
        if mode == "time":
            batches = sum(1 for _ in loader)
            seconds = time.perf_counter() - start
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() != "Darwin":
                rss_bytes *= 1024
            return dict(batches=batches, seconds=seconds, peak_rss_bytes=rss_bytes)

        batches = content_tokens = 0
        digest = hashlib.sha256()
        for x, y, state in loader:
            assert x.shape == y.shape == (BATCH, CONTEXT)
            digest.update(x.numpy().tobytes())
            digest.update(y.numpy().tobytes())
            digest.update(state.model_dump_json().encode())
            content_tokens += int((x[:, 0] != bos).sum()) + int((y != bos).sum())
            batches += 1
        return dict(batches=batches, content_tokens=content_tokens, sha256=digest.hexdigest())


def run(source):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

    def fresh(mode, key):
        invocation = [sys.executable, str(Path(__file__).resolve()), "--shard", str(source), "--worker", mode, key]
        result = subprocess.run(invocation, env=env, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout + result.stderr, file=sys.stderr)
            result.check_returncode()
        return json.loads(result.stdout.splitlines()[-1])

    meta = source.with_name("meta.json")
    metadata_before = meta.read_bytes() if meta.is_file() else None
    verified = {}
    for key in PATHS:
        result = fresh("verify", key)
        print("Parity", key, result, flush=True)
        verified[key] = result
    assert all(result == verified["A"] for result in verified.values()), "Full-shard digest mismatch"
    reference = verified["A"]
    assert reference["batches"] > 0, "Selected shard did not emit a complete batch"

    results = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "shard": str(source), "batch_size": BATCH,
        "context": CONTEXT, "tokenizer_threads": THREADS, "orders": ORDERS,
        "verification": verified, "runs": [],
    }
    output = SCRATCH / "benchmarks.json"
    for repeat, order in enumerate(ORDERS, 1):
        for position, key in enumerate(order, 1):
            values = fresh("time", key)
            assert values["batches"] == reference["batches"]
            elapsed = values["seconds"]
            record = dict(
                **values, repeat=repeat, position=position, backend=PATHS[key][0],
                source_tokens_per_second=reference["content_tokens"] / elapsed,
                milliseconds_per_batch=1_000 * elapsed / values["batches"],
            )
            results["runs"].append(record)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"Round {repeat} position {position} {key}", json.dumps(record), flush=True)
    assert (meta.read_bytes() if meta.is_file() else None) == metadata_before

    for name, _, _ in PATHS.values():
        samples = [row for row in results["runs"] if row["backend"] == name]
        rates = [row["source_tokens_per_second"] for row in samples]
        latency = statistics.median(row["milliseconds_per_batch"] for row in samples)
        rss = max(row["peak_rss_bytes"] for row in samples) / 2**20
        print(f"| {name} | {statistics.median(rates):,.0f} | {min(rates):,.0f}–{max(rates):,.0f} | {rss:.1f} | {latency:.3f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, default=SOURCE)
    parser.add_argument("--worker", nargs=2)
    args = parser.parse_args()
    source = args.shard.absolute()
    if not source.is_file():
        parser.error(f"Shard is missing: {source}")
    if args.worker:
        print(json.dumps(worker(*args.worker, source)))
    else:
        run(source)
