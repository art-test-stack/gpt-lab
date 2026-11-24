# Rust dataloader prototype

Baseline is `61f066250420afbc121524037b996ee4a3524c0a`, fetched from `origin/master` before creating `experiment/rust-dataloader`.

`build_dataloader(..., tokenizer_batch_size=64)` batches encoding for adapters that declare `supports_exact_batch_encoding = True` on their concrete class. Their `encode_batch` must return deterministic, unpadded sequences in source order that match scalar encoding, including BOS. Checkpoint restore encodes buffered documents individually. The tiktoken adapter used for GPT-2 opts in and forwards `num_threads=tokenizer_threads`. Other adapters and pretokenized input use the scalar path. Packing lookahead stays at 16 documents when BOS is available and one otherwise.

`build_dataloader(..., packing_backend="rust")` or `DistDataLoader(..., packing_backend="rust")` selects the optional extension. Rust selects documents and fills the existing CPU batch buffer from contiguous `int64` documents. Python owns the shard order, refill, and checkpoint. Cursors advance per admitted document, so unused encodings are recreated after resume. The default is scalar tokenization and Python packing, with the same checkpoint signature and validation exhaustion behavior as the baseline.

If a refill error escapes during training, discard the iterator. Rebuild with `build_dataloader(..., resume_state=last_committed_state)` using the last committed `DataLoaderState`. The failed batch may already have consumed part of a document.

## Reproduce on macOS arm64

From the repository root, use the project’s Python 3.12 CPU environment. Build the extension separately using Rust and Cargo. The linker argument below is required on the macOS 27.2.1 test host.

```sh
scratch="$PWD/.tmp/codex/rust-dataloader"
mkdir -p "$scratch"
export PYO3_PYTHON="$PWD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
env -u PYTHONPATH "$PYO3_PYTHON" -m pytest -q -p no:cacheprovider \
  --basetemp "$scratch/pytest-default" tests/test_sharder.py tests/test_bpb.py

MACOSX_DEPLOYMENT_TARGET=14.0 cargo rustc --release --locked \
  --manifest-path experiments/20261124-rust-dataloader/rust/Cargo.toml \
  --target-dir "$scratch/target" -- -C link-arg=-Wl,-fixup_chains
extension_suffix=$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ln -sf "$scratch/target/release/libgpt_lab_rust_packer.dylib" \
  "$scratch/gpt_lab_rust_packer$extension_suffix"
export PYTHONPATH="$scratch:$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYO3_PYTHON" -m pytest -q -p no:cacheprovider --basetemp "$scratch/pytest-parity" \
  experiments/20261124-rust-dataloader/test_parity.py \
  experiments/20261124-rust-dataloader/test_benchmark.py

TIKTOKEN_CACHE_DIR=.tmp/codex/training-data-inspection/tiktoken \
  "$PYO3_PYTHON" experiments/20261124-rust-dataloader/benchmark.py \
  --shard .tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet \
  --threads 4 --repeats 4
```

The locked build also works with `--offline` when its crates are cached. The crate pins the locally available PyO3 0.22.2, a yanked release. The standard Python install does not use the crate.

The ClimbMix input is the cached `shard_01061.parquet`, with SHA-256 `9025ed81d751886e7810238f2c71a4d68ee907860f3c506d1cd3e74bda135fd8`, 86,016 documents and 84 row groups. The harness reads it in full using finite validation iteration and loads GPT-2 from the existing tiktoken cache. `--shard` accepts a local shard. The harness links it into a temporary dataset directory under its original shard ID so it can write benchmark metadata there. The pretokenized run uses 10,000 documents from `experiments/dataloaders.py` and prepares contiguous token arrays before timing.

## Parity

Training compares tensors and full checkpoint state after each of 350 batches on each of two logical CPU ranks, including row-group, shard, and epoch transitions. Both Python-to-Rust and Rust-to-Python resume use serialized checkpoints with buffered documents and a retained suffix, then compare against uninterrupted Python. Validation covers scalar tokenizers, batched tokenizers, and pretokenized input without BOS. Separate cases cover a padding adapter, an incomplete validation batch, and checkpoint recovery after a refill error interrupts a batch with a partially consumed document. The benchmark tests verify source metadata with three shards and both present and absent metadata, then read from the original cache using its normal train and validation splits.

## Measurement

Apple M2 CPU, 16 GiB RAM, macOS 27.2.1, Python 3.12.12, Torch 2.11.0, Arrow 22.0.0, and tiktoken 0.12.0. Batch size 16, context length 2,048, and one rank. The tokenizer has a four-thread budget and Torch and Arrow each use one thread. Each measurement uses a fresh process. Four rounds alternate forward and reverse backend order.

Timing includes loader initialization and every batch request through exhaustion. Token counting and digest calculation are outside the timed calls. Each packed position contributes at most once, excluding document and continuation BOS. Peak RSS is the highest process peak across the four repetitions. The harness writes individual observations to `.tmp/codex/rust-dataloader/benchmark.json`.

| Input | Backend | Median source tokens/s | Run-to-run range | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| ClimbMix | Baseline Python | 3,596,586 | 3,270,609–3,635,999 | 335.5 |
| ClimbMix | Batched Python | 5,857,447 | 5,675,450–6,123,779 | 396.4 |
| ClimbMix | Batched Python + Rust | 6,822,029 | 6,650,974–6,966,206 | 402.5 |
| Pretokenized | Python packing | 77,632,018 | 75,540,803–78,344,376 | 296.5 |
| Pretokenized | Rust packing | 278,686,929 | 239,118,316–312,335,754 | 297.2 |

Each ClimbMix run emitted 54,014,008 source tokens across 1,651 full batches. Each pretokenized run emitted 5,232,954 source tokens across 160 full batches. The tensor and state digests matched across backends and repetitions for each input.

Batched Python raised median ClimbMix throughput by 62.9%. Rust added 16.5% over batched Python at about 6 MiB higher peak RSS. It reached 3.59× Python throughput on pretokenized input. Use batched Python as the first optimization and keep Rust opt-in for runs where loader time is a bottleneck. These measurements cover local CPU loader throughput.
