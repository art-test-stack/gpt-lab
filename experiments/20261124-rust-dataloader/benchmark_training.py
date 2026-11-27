"""Compare full Rust training packers on one cached text shard.

The benchmark archives committed Python and Rust sources, stops their training
source after one epoch, and adds split counters only to separate untimed builds.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tarfile
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[2]
RUST_PATHS = (Path("packages/rust-dataloader"), Path("experiments/20261124-rust-dataloader/rust"))
BASELINE = "f008399"
CONTEXTS = (512, 1024, 2048, 4096, 8192)
ARMS = {"A": ("rolling-16", "baseline"), "B": ("greedy-block", "baseline"),
        "C": ("exact-pair-block", "candidate")}
ORDERS = ("ABC", "BCA", "CAB", "CBA", "ACB", "BAC")
BATCH, THREADS = 16, 4

EPOCH_END = """                if !self.training {
                    self.exhausted = true;
                    return Ok(None);
                }
                if !self.had_documents {
                    return Err("No training documents for this rank".into());
                }
                self.cursor.shard_idx = 0;
                self.cursor.epoch += 1;
                self.had_documents = false;
                self.resuming = false;"""
FINITE_END = """                if self.training && !self.had_documents {
                    return Err("No training documents for this rank".into());
                }
                self.exhausted = true;
                return Ok(None);"""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Benchmark overlay expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def instrument(source):
    source = replace_once(source, "    exhausted: bool,",
                          "    exhausted: bool,\n    counts: (usize, usize, usize),")
    source = replace_once(source, "                exhausted: false,",
                          "                exhausted: false,\n                counts: (0, 0, 0),")
    anchor = "                let take = (tokens.len() - self.offset).min(self.capacity - position);"
    source = replace_once(source, anchor, anchor + """
                if self.offset == 0 {
                    self.counts.0 += 1;
                    if tokens.len() <= self.capacity {
                        self.counts.1 += 1;
                        if take < tokens.len() {
                            self.counts.2 += 1;
                        }
                    }
                }""")
    anchor = "    fn fill(&mut self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<State> {"
    return replace_once(source, anchor, """    fn packing_counts(&self) -> (usize, usize, usize) {
        self.counts
    }

""" + anchor)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def rust_path(revision):
    manifests = git("ls-tree", "--name-only", revision, "--",
                    *(str(path / "Cargo.toml") for path in RUST_PATHS)).decode().splitlines()
    if len(manifests) != 1:
        raise ValueError(f"Expected one Rust dataloader crate at {revision}, found {manifests}")
    return Path(manifests[0]).parent


def prepare(scratch, candidate, offline):
    revisions = {"baseline": git("rev-parse", BASELINE).decode().strip(),
                 "candidate": git("rev-parse", candidate).decode().strip()}
    env = os.environ.copy()
    env.update(PYO3_PYTHON=sys.executable, PYO3_BUILD_EXTENSION_MODULE="1",
               CARGO_PROFILE_RELEASE_STRIP="none", RUSTFLAGS="-C strip=none")
    if platform.system() == "Darwin":
        env.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
    for name, revision in revisions.items():
        rust = rust_path(revision)
        snapshot = scratch / name
        snapshot.mkdir()
        archive = git("archive", revision, "--", "src/gpt_lab", str(rust))
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(snapshot, filter="data")
        for variant in ("timing", "counts"):
            crate = snapshot / variant
            shutil.copytree(snapshot / rust, crate)
            source = crate / "src/parquet_source.rs"
            source.write_text(replace_once(source.read_text(), EPOCH_END, FINITE_END))
            if variant == "counts":
                source = crate / "src/text_loader.rs"
                source.write_text(instrument(source.read_text()))
            build = ["cargo", "build", "--release", "--locked", "--manifest-path",
                     str(crate / "Cargo.toml"), "--target-dir", str(scratch / "target")]
            if offline:
                build.append("--offline")
            subprocess.run(build, env=env, check=True)
            extension = crate / "extension"
            extension.mkdir()
            library = "libgpt_lab_rust_packer.dylib" if platform.system() == "Darwin" else "libgpt_lab_rust_packer.so"
            shutil.copyfile(scratch / "target/release" / library,
                            extension / ("gpt_lab_rust_packer" + sysconfig.get_config_var("EXT_SUFFIX")))
    return revisions


def setup():
    import pyarrow as pa
    import torch
    from gpt_lab.tokenizer import Tokenizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    return Tokenizer.from_pretrained("gpt2", source="tiktoken")


def histogram(source, contexts):
    import pyarrow.parquet as pq

    tokenizer = setup()
    lengths = Counter()
    with pq.ParquetFile(source) as parquet:
        for batch in parquet.iter_batches(batch_size=64, columns=["text"]):
            lengths.update(map(len, tokenizer.encode_batch(
                batch["text"].to_pylist(), prepend_bos=True, num_threads=THREADS,
            )))
        assert sum(lengths.values()) == parquet.metadata.num_rows
    return {"documents": sum(lengths.values()),
            "row_fit": {str(context): sum(count for length, count in lengths.items()
                                         if length <= context + 1) for context in contexts}}


def worker(mode, arm, source, batch_size, context, scratch):
    import torch
    from gpt_lab.data.loader import build_dataloader
    from gpt_lab.utils.distributed import ComputeContext

    tokenizer = setup()
    bos = tokenizer.get_bos_token_id()
    compute = ComputeContext(torch.device("cpu"))
    with TemporaryDirectory(prefix="shard-", dir=scratch) as cache:
        dataset = Path(cache) / "climbmix"
        dataset.mkdir()
        # This single-shard training manifest owns slot zero and links the original bytes.
        (dataset / "shard_00000.parquet").symlink_to(source)
        start = time.perf_counter()
        loader = build_dataloader(
            dataset.name, batch_size, context, tokenizer, split="train",
            cachedir=dataset.parent, max_shards=1, comp_info=compute,
            dataloader_backend="rust", tokenizer_threads=THREADS, tokenizer_batch_size=64,
            packing_strategy="rolling-16" if arm == "A" else "block-best-fit",
        )
        if mode == "time":
            batches = sum(1 for _ in loader)
            return {"seconds": time.perf_counter() - start, "batches": batches}
        digest = hashlib.sha256()
        batches = content_tokens = 0
        emitted_counts = (0, 0, 0)
        for x, y, state in loader:
            assert x.shape == y.shape == (batch_size, context)
            assert state.epoch == 1 and state.packing_signature is not None
            digest.update(x.numpy().tobytes())
            digest.update(y.numpy().tobytes())
            digest.update(state.model_dump_json().encode())
            content_tokens += int((x[:, 0] != bos).sum()) + int((y != bos).sum())
            batches += 1
            if mode == "counts":
                emitted_counts = loader._rust_text_loader.packing_counts()
        result = dict(batches=batches, emitted_source_tokens=content_tokens,
                      tensor_state_sha256=digest.hexdigest())
        if mode == "counts":
            native = loader._rust_text_loader
            counts = native.packing_counts()
            assert next(loader, None) is None and native.packing_counts() == counts
            result.update(documents=counts[0], row_fit_documents=counts[1],
                          split_row_fit_documents=counts[2],
                          final_tail_documents=counts[0] - emitted_counts[0],
                          final_tail_row_fit_documents=counts[1] - emitted_counts[1],
                          final_tail_split_row_fit_documents=counts[2] - emitted_counts[2])
        return result


def fresh(mode, arm, source, context, scratch, batch_size=BATCH):
    snapshot = scratch / ARMS[arm][1]
    variant = "counts" if mode == "counts" else "timing"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(snapshot / variant / "extension"), str(snapshot / "src")))
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    invocation = [
        sys.executable, str(Path(__file__).resolve()), "--scratch", str(scratch),
        "--shard", str(source), "--context", str(context), "--batch-size", str(batch_size),
        "--worker", mode, arm,
    ]
    result = subprocess.run(invocation, env=env, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr)
        result.check_returncode()
    return json.loads(result.stdout.splitlines()[-1])


def self_check(scratch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cases = [
        ([" a" * 5] * 3 + [" b" * 10], 2, 7, (2, 25, 4, 3, 2)),
        ([" c" * 7], 1, 7, (1, 7, 1, 1, 0)),
        ([" c" * 7, " d"], 1, 7, (1, 7, 2, 2, 0)),
    ]
    with TemporaryDirectory(prefix="fixture-", dir=scratch) as directory:
        source = Path(directory) / "fixture.parquet"
        for texts, batch_size, context, expected in cases:
            pq.write_table(pa.table({"text": texts}), source)
            for arm in ARMS:
                counts = fresh("counts", arm, source, context, scratch, batch_size)
                actual = tuple(counts[field] for field in (
                    "batches", "emitted_source_tokens", "documents",
                    "row_fit_documents", "split_row_fit_documents"))
                assert actual == expected, (arm, actual, expected)
                verified = fresh("verify", arm, source, context, scratch, batch_size)
                assert all(counts[key] == value for key, value in verified.items())
                if len(texts) == 2:
                    assert counts["final_tail_documents"] == 1
                    assert counts["final_tail_row_fit_documents"] == 1
                    assert counts["final_tail_split_row_fit_documents"] == 0
        pq.write_table(pa.table({"text": [" a" * 8, " b" * 5, " c" * 3]}), source)
        for arm in ARMS:
            counts = fresh("counts", arm, source, 9, scratch, 1)
            assert counts["documents"] == counts["row_fit_documents"] == 3
            assert counts["split_row_fit_documents"] == (0 if arm == "C" else 1)
    print("GPT-2 split, exact-fit, final-tail, and [9, 6, 4] checks passed.", flush=True)


def run(source, contexts, scratch, output, revisions):
    metadata = source.with_name("meta.json")
    metadata_before = metadata.read_bytes() if metadata.exists() else None
    lengths = fresh("histogram", "A", source, contexts[0], scratch)
    with source.open("rb") as shard:
        source_sha256 = hashlib.file_digest(shard, "sha256").hexdigest()
    report = {
        "revisions": revisions, "platform": platform.platform(), "source": source.name,
        "source_sha256": source_sha256,
        "batch_size": BATCH, "tokenizer": "gpt2", "tokenizer_threads": THREADS,
        "tokenizer_batch_size": 64, "split": "train", "epochs": 1, "orders": ORDERS,
        "histogram": lengths, "contexts": {},
        "counting": (
            "Throughput counts non-BOS tokens across the unique positions in each emitted batch. "
            "Row-fit includes BOS and uses length <= context+1. Split counts include the discarded final tail."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        metrics = {}
        for arm in ARMS:
            counts = fresh("counts", arm, source, context, scratch)
            verified = fresh("verify", arm, source, context, scratch)
            assert all(counts[key] == value for key, value in verified.items())
            assert counts["documents"] == lengths["documents"]
            assert counts["row_fit_documents"] == lengths["row_fit"][str(context)]
            assert 0 <= counts["split_row_fit_documents"] <= counts["row_fit_documents"]
            metrics[arm] = counts
        row = {"verification": metrics, "runs": []}
        report["contexts"][str(context)] = row
        print("Counts", context, metrics, flush=True)
        for repeat, order in enumerate(ORDERS, 1):
            for position, arm in enumerate(order, 1):
                timed = fresh("time", arm, source, context, scratch)
                assert timed["batches"] == metrics[arm]["batches"]
                numerator = metrics[arm]["emitted_source_tokens"]
                sample = dict(
                    **timed, arm=arm, backend=ARMS[arm][0], revision=revisions[ARMS[arm][1]],
                    repeat=repeat, position=position, emitted_source_tokens=numerator,
                    source_tokens_per_second=numerator / timed["seconds"],
                )
                row["runs"].append(sample)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print("Run", context, sample, flush=True)
        cells = []
        for arm in ARMS:
            rates = [run["source_tokens_per_second"] for run in row["runs"] if run["arm"] == arm]
            cells.append(f"{statistics.median(rates):,.0f} ({min(rates):,.0f}–{max(rates):,.0f})")
        print(f"| {context:,} | {' | '.join(cells)} |", flush=True)
    metadata_after = metadata.read_bytes() if metadata.exists() else None
    assert metadata_after == metadata_before
    report["source_metadata_unchanged"] = True
    report["source_metadata_sha256"] = hashlib.sha256(metadata_before).hexdigest() if metadata_before is not None else None
    output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, help="Cached ClimbMix Parquet shard (required unless running --self-check)")
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--context", type=int, nargs="+", default=CONTEXTS)
    parser.add_argument("--scratch", type=Path, help="Parent for temporary builds (default: system temporary directory)")
    parser.add_argument("--output", type=Path, help="Results JSON path (default: comparison.json in --scratch or the current directory)")
    parser.add_argument("--offline", action="store_true", help="Build using cached Cargo dependencies")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--batch-size", type=int, default=BATCH, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.context) < 1 or args.batch_size < 1:
        parser.error("Context and batch size must be positive")
    if not args.worker and any(context not in CONTEXTS for context in args.context):
        parser.error(f"Contexts must be drawn from {CONTEXTS}")
    source = args.shard.resolve() if args.shard is not None else None
    if (args.worker or not args.self_check) and (source is None or not source.is_file()):
        parser.error("A cached Parquet file is required; pass --shard")
    scratch = args.scratch
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    if args.worker:
        mode, arm = args.worker
        result = histogram(source, CONTEXTS) if mode == "histogram" else worker(
            mode, arm, source, args.batch_size, args.context[0], scratch,
        )
        print(json.dumps(result))
        return
    with TemporaryDirectory(prefix="build-", dir=scratch) as temporary:
        work = Path(temporary).resolve()
        revisions = prepare(work, args.candidate, args.offline)
        self_check(work)
        if not args.self_check:
            output = (args.output or (scratch or Path.cwd()) / "comparison.json").resolve()
            run(source, args.context, work, output, revisions)


if __name__ == "__main__":
    main()
