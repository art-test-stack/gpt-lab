"""Full finite-corpus CPU benchmark. Run from the repository root."""

import argparse
import hashlib
import importlib.util
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


BASELINE = "61f066250420afbc121524037b996ee4a3524c0a"
SCRATCH = Path(".tmp/codex/rust-dataloader")
BATCH_SIZE = 16
CONTEXT = 2048


def baseline_loader():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:src/gpt_lab/data/loader.py"])
    path = SCRATCH / "baseline_loader.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("baseline_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(args):
    if args.corpus != "climbmix":
        return _measure(args, None)
    source = Path(args.shard)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dataset-", dir=SCRATCH) as temporary:
        dataset = Path(temporary) / "dataset"
        dataset.mkdir()
        shard = dataset / source.name
        shard.symlink_to(source.resolve(strict=True))
        return _measure(args, shard)


def _measure(args, shard):
    import numpy as np
    import pyarrow as pa
    import torch

    from gpt_lab.data import loader as current
    from gpt_lab.tokenizer import Tokenizer
    from gpt_lab.utils.distributed import ComputeContext
    from gpt_lab.utils.schemas import DataLoaderState
    from experiments.dataloaders import make_synthetic_bin_idx, SyntheticPretokenizedDataset

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    module = baseline_loader() if args.backend == "baseline" else current
    kwargs = {}
    if args.backend != "baseline":
        kwargs["dataloader_backend"] = "rust-packing" if args.backend == "rust" else "python"
    if shard is not None:
        tokenizer = Tokenizer.from_pretrained("gpt2", source="tiktoken")
        bos = tokenizer.get_bos_token_id()
        start = time.perf_counter()
        factory = module.build_dataloader(
            name=shard.parent.name, cachedir=shard.parent.parent,
            batch_size=BATCH_SIZE, seq_len=CONTEXT, tokenizer=tokenizer,
            split="val", max_shards=int(shard.stem.split("_")[-1]),
            comp_info=ComputeContext(device=torch.device("cpu")),
            tokenizer_threads=args.threads,
            **({"tokenizer_batch_size": 64} if args.backend != "baseline" else {}),
            **kwargs,
        )
        iterator = factory()
    else:
        tokens, offsets = make_synthetic_bin_idx()
        fixture = SyntheticPretokenizedDataset(tokens, offsets)
        bos = 0
        documents = [np.concatenate(([bos], fixture.get_doc(i).numpy())) for i in range(len(fixture))]
        cursor = DataLoaderState()

        class Documents:
            bos_token_id = bos
            split = "val"
            start_state = None

            def __iter__(self):
                for document in documents:
                    yield document, cursor

        start = time.perf_counter()
        iterator = module.DistDataLoader(Documents(), BATCH_SIZE, CONTEXT, device="cpu", **kwargs)

    elapsed = time.perf_counter() - start
    batches = content = 0
    digest = hashlib.sha256()
    while True:
        start = time.perf_counter()
        try:
            x, y, state = next(iterator)
        except StopIteration:
            elapsed += time.perf_counter() - start
            break
        elapsed += time.perf_counter() - start
        # Together these contain every packed position once. Inspection is untimed.
        content += x.shape[0] * (x.shape[1] + 1) - int((x[:, 0] == bos).sum()) - int((y == bos).sum())
        digest.update(x[:, :1].numpy().tobytes())
        digest.update(y.numpy().tobytes())
        digest.update(state.model_dump_json().encode())
        batches += 1
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    return dict(corpus=args.corpus, backend=args.backend, threads=args.threads,
                batches=batches, content_tokens=content, seconds=elapsed,
                tokens_per_second=content / elapsed, peak_rss_bytes=rss_bytes,
                sha256=digest.hexdigest(), baseline=BASELINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default=".tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet")
    parser.add_argument("--threads", type=int, default=4, help="Tokenizer worker budget")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", default=str(SCRATCH / "benchmark.json"))
    parser.add_argument("--corpus", choices=["climbmix", "pretokenized"])
    parser.add_argument("--backend", choices=["baseline", "python", "rust"])
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    if args.backend:
        if not args.corpus:
            parser.error("--backend requires --corpus")
        print(json.dumps(measure(args)))
        return
    runs = []
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    corpora = [args.corpus] if args.corpus else ["climbmix", "pretokenized"]
    for corpus in corpora:
        backends = ["baseline", "python", "rust"] if corpus == "climbmix" else ["python", "rust"]
        for repeat in range(args.repeats):
            for backend in backends[::1 if repeat % 2 == 0 else -1]:
                command = [sys.executable, __file__, "--corpus", corpus, "--backend", backend,
                           "--shard", args.shard, "--threads", str(args.threads)]
                completed = subprocess.run(command, env=env, text=True, capture_output=True)
                if completed.returncode:
                    sys.stderr.write(completed.stdout + completed.stderr)
                    completed.check_returncode()
                result = json.loads(completed.stdout.splitlines()[-1])
                result["repeat"] = repeat
                runs.append(result)
                print(json.dumps(result), flush=True)
        assert len({(r["batches"], r["content_tokens"], r["sha256"]) for r in runs if r["corpus"] == corpus}) == 1, "Backend output mismatch"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(runs, indent=2) + "\n")
    print("| Corpus | Backend | Median content tokens/s | Range | Peak RSS MiB |")
    print("|---|---|---:|---:|---:|")
    for corpus in corpora:
        for backend in ("baseline", "python", "rust"):
            group = [r for r in runs if r["corpus"] == corpus and r["backend"] == backend]
            if group:
                rates = [r["tokens_per_second"] for r in group]
                peak = max(r["peak_rss_bytes"] for r in group) / 2**20
                print(f"| {corpus} | {backend} | {statistics.median(rates):,.0f} | {min(rates):,.0f}–{max(rates):,.0f} | {peak:.1f} |")


if __name__ == "__main__":
    main()
