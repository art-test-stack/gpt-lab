"""Compare the installed Python and Rust training loaders on a cached text shard."""

import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
BATCH, THREADS, CHUNK = 16, 4, 64
CONTEXTS = (512, 1024, 2048, 4096, 8192)
STRATEGIES = ("rolling-16", "block-best-fit")
ORDERS = (("python", "rust"), ("rust", "python")) * 2


def worker(dataset, context, strategy, backend=None, batches=None):
    import numpy as np
    import pyarrow as pa
    import torch

    from gpt_lab.data import build_dataloader
    from gpt_lab.tokenizer import Tokenizer
    from gpt_lab.utils.distributed import ComputeContext

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizer = Tokenizer.from_pretrained("gpt2", source="tiktoken")
    compute = ComputeContext(torch.device("cpu"))

    def build(backend):
        return build_dataloader(
            dataset.name, BATCH, context, tokenizer, split="train", cachedir=dataset.parent,
            max_shards=1, comp_info=compute, dataloader_backend=backend, packing_strategy=strategy,
            tokenizer_threads=THREADS, tokenizer_batch_size=CHUNK,
        )

    if backend is not None:
        start = perf_counter()
        emitted = sum(1 for _ in islice(build(backend), batches))
        elapsed = perf_counter() - start
        if emitted != batches:
            raise ValueError(f"{backend}: expected {batches} batches, emitted {emitted}")
        return dict(seconds=elapsed, batches=emitted)

    python, rust = build("python"), build("rust")
    digest = hashlib.sha256()
    tokens = 0
    bos = tokenizer.get_bos_token_id()
    for batch in range(1, sys.maxsize):
        x, y, state = next(python)
        other_x, other_y, other_state = next(rust)
        serialized = state.model_dump_json()
        for field, equal in (
            ("x", torch.equal(x, other_x)),
            ("y", torch.equal(y, other_y)),
            ("DataLoaderState", serialized == other_state.model_dump_json()),
        ):
            if not equal:
                raise ValueError(f"{strategy}, context {context}, rank {compute.rank}, batch {batch}: {field} mismatch")
        if state.epoch != 1:
            if batch == 1:
                raise ValueError(f"Shard does not fill one epoch-1 batch at context {context}")
            return dict(batches=batch - 1, emitted_source_tokens=tokens, sha256=digest.hexdigest())
        x, y = x.numpy(), y.numpy()
        tokens += int(np.count_nonzero(x[:, 0] != bos) + np.count_nonzero(y != bos))
        digest.update(x.tobytes())
        digest.update(y.tobytes())
        digest.update(serialized.encode())


def run(shard, contexts, output):
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1")
    environment.pop("PYTHONPATH", None)

    def fresh(dataset, context, strategy, *timing):
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker",
                   str(dataset), str(context), strategy, *map(str, timing)]
        result = subprocess.run(command, env=environment, cwd=dataset.parent, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout + result.stderr, file=sys.stderr)
            result.check_returncode()
        return json.loads(result.stdout.splitlines()[-1])

    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True).stdout.strip() or cpu
    with shard.open("rb") as file:
        shard_hash = hashlib.file_digest(file, "sha256").hexdigest()
    results = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        platform=platform.platform(), cpu=cpu, python=platform.python_version(),
        shard=shard.name, shard_sha256=shard_hash, batch_size=BATCH, rank=0, world_size=1,
        tokenizer="gpt2", tokenizer_threads=THREADS, tokenizer_batch_size=CHUNK,
        orders=ORDERS, results=[],
    )
    metadata = shard.with_name("meta.json")
    original = metadata.read_bytes() if metadata.is_file() else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="gpt-lab-benchmark-") as temporary:
        dataset = Path(temporary) / "climbmix"
        dataset.mkdir()
        # A single-shard train manifest uses slot zero. Both backends read the original bytes.
        (dataset / "shard_00000.parquet").symlink_to(shard)
        for context in contexts:
            for strategy in STRATEGIES:
                verified = fresh(dataset, context, strategy)
                samples = dict(context=context, strategy=strategy, **verified, runs=[])
                results["results"].append(samples)
                print(context, strategy, "parity", verified, flush=True)
                for repeat, order in enumerate(ORDERS, 1):
                    for backend in order:
                        observed = fresh(dataset, context, strategy, backend, verified["batches"])
                        rate = verified["emitted_source_tokens"] / observed["seconds"]
                        samples["runs"].append(dict(repeat=repeat, backend=backend, **observed, tokens_per_second=rate))
                        print(f"  {repeat} {backend}: {rate:,.0f} source tokens/s", flush=True)
                output.write_text(json.dumps(results, indent=2) + "\n")
    if (metadata.read_bytes() if metadata.is_file() else None) != original:
        raise ValueError(f"Source metadata changed: {metadata}")

    for result in results["results"]:
        python, rust = ([run["tokens_per_second"] for run in result["runs"] if run["backend"] == backend]
                        for backend in ("python", "rust"))

        def summary(rates):
            return f"{median(rates)/1e6:.3f} ({min(rates)/1e6:.3f}–{max(rates)/1e6:.3f})"

        print(f"| {result['context']} | {result['strategy']} | {result['batches']:,} | "
              f"{result['emitted_source_tokens']:,} | {summary(python)} | {summary(rust)} | "
              f"{median(rust)/median(python):.2f}× |", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, help="Path to a cached ClimbMix Parquet shard")
    parser.add_argument("--output", type=Path, help="Path for per-run JSON")
    parser.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        directory, context, strategy, *timing = args.worker
        backend, batches = (timing[0], int(timing[1])) if timing else (None, None)
        print(json.dumps(worker(Path(directory), int(context), strategy, backend, batches)))
        return
    if args.shard is None or not args.shard.is_file() or args.output is None:
        parser.error("Pass an existing --shard and a path for --output")
    if min(args.contexts) < 1:
        parser.error("Contexts must be positive")
    run(args.shard.resolve(), args.contexts, args.output)


if __name__ == "__main__":
    main()
