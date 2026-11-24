# Rust dataloader prototype

Baseline is `61f066250420afbc121524037b996ee4a3524c0a`, fetched from `origin/master` before creating `experiment/rust-dataloader`.

`build_dataloader(..., tokenizer_batch_size=64)` batches encoding for adapters that declare `supports_exact_batch_encoding = True` on their concrete class. Their `encode_batch` must return deterministic, unpadded sequences in source order that match scalar encoding, including BOS. Checkpoint restore encodes buffered documents individually. The tiktoken adapter used for GPT-2 opts in and forwards `num_threads=tokenizer_threads`. Other adapters and pretokenized input use the scalar path. Packing lookahead stays at 16 documents when BOS is available and one otherwise.

`build_dataloader(..., packing_backend="rust")` or `DistDataLoader(..., packing_backend="rust")` selects the optional extension. Rust selects documents and fills the existing CPU batch buffer from contiguous `int64` documents. For a document that spans batches, it copies only the tokens the current batch consumes. Python owns the shard order, refill, and checkpoint. Cursors advance per admitted document, so unused encodings are recreated after resume. The default is scalar tokenization and Python packing, with the same checkpoint signature and validation exhaustion behavior as the baseline.

`build_dataloader(..., dataloader_backend="rust")` selects the end-to-end CPU path for cached Parquet text with the repository’s tiktoken tokenizer. Python supplies the rank-owned shard order, actual tokenizer parameters, batch shape, thread budget, and checkpoint. Rust reads and tokenizes the text, packs the next batch, and copies it into the existing CPU tensor in one extension call. PyTorch transfers that tensor to the training device. CoreBPE is initialized once per iterator from [tiktoken 0.12.0](https://github.com/openai/tiktoken/tree/97e49cbadd500b5cc9dbb51a486f0b42e6701bee). It encodes literal special-token strings as ordinary text and prepends the configured BOS.

`gpt-lab train auto` and `gpt-lab train resume` accept `--dataloader-backend {python,rust-packing,rust}` for both training and validation. `python` is the default. The two Rust selections default to tokenization chunks of 64 documents and accept the same training checkpoints as Python. Pass `--tokenizer-batch-size 64` to compare `python` and `rust-packing` with the same tokenization chunk size for training and validation. The end-to-end path requires local shards and non-null UTF-8 text. On Unix, an `EIO` while reading a row group causes Rust to reopen the shard before retrying the same group. It makes up to five attempts with 1, 2, 4, and 8 second delays.

If a refill error escapes during training, discard the iterator. Rebuild with `build_dataloader(..., resume_state=last_committed_state)` using the last committed `DataLoaderState`. The failed batch may already have consumed part of a document.

## Reproduce on macOS arm64

From the repository root, use the project’s Python 3.12 CPU environment. The optional extension uses PyO3 0.30.1 and was built with Rust 1.91. On the macOS 27.2.1 test host, disabling Rust’s post-link stripping prevents [misaligned dynamic libraries](https://github.com/rust-lang/rust/issues/157750).

```sh
scratch="$PWD/.tmp/codex/rust-dataloader"
mkdir -p "$scratch"
export PYO3_PYTHON="$PWD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
env -u PYTHONPATH "$PYO3_PYTHON" -m pytest -q -p no:cacheprovider \
  --basetemp "$scratch/pytest-default" tests/test_sharder.py tests/test_bpb.py

PYO3_BUILD_EXTENSION_MODULE=1 CARGO_HOME="$scratch/cargo" CARGO_NET_GIT_FETCH_WITH_CLI=true \
  MACOSX_DEPLOYMENT_TARGET=14.0 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS="-C strip=none" \
  cargo build --release --locked \
  --manifest-path experiments/20261124-rust-dataloader/rust/Cargo.toml \
  --target-dir "$scratch/target-native"
env -u PYO3_BUILD_EXTENSION_MODULE CARGO_HOME="$scratch/cargo" \
  MACOSX_DEPLOYMENT_TARGET=14.0 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS="-C strip=none" \
  cargo test --release --locked --lib \
  --manifest-path experiments/20261124-rust-dataloader/rust/Cargo.toml \
  --target-dir "$scratch/target-native"
extension_suffix=$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ln -sf "$scratch/target-native/release/libgpt_lab_rust_packer.dylib" \
  "$scratch/gpt_lab_rust_packer$extension_suffix"
export PYTHONPATH="$scratch:$PWD${PYTHONPATH:+:$PYTHONPATH}"

export TIKTOKEN_CACHE_DIR=.tmp/codex/training-data-inspection/tiktoken
"$PYO3_PYTHON" -m pytest -q -p no:cacheprovider --basetemp "$scratch/pytest-parity" \
  experiments/20261124-rust-dataloader/test_parity.py \
  experiments/20261124-rust-dataloader/test_benchmark.py \
  experiments/20261124-rust-dataloader/test_native_text.py

# Run the four-backend comparison with untimed parity passes.
"$PYO3_PYTHON" experiments/20261124-rust-dataloader/benchmark_four.py \
  --shard .tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet

# Run the packing-only comparison.
"$PYO3_PYTHON" experiments/20261124-rust-dataloader/benchmark.py \
  --shard .tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet \
  --threads 4 --repeats 4
```

The locked build works with `--offline` when its Cargo dependencies are cached. The standard Python install requires no Rust build. The native text parity tests require an existing GPT-2 cache under `TIKTOKEN_CACHE_DIR`.

The ClimbMix input is the cached `shard_01061.parquet`, with SHA-256 `9025ed81d751886e7810238f2c71a4d68ee907860f3c506d1cd3e74bda135fd8`, 86,016 documents and 84 row groups. The harness reads it in full using finite validation iteration and loads GPT-2 from the existing tiktoken cache. `--shard` accepts a local shard. The harness links it into a temporary dataset directory under its original shard ID so it can write benchmark metadata there. The pretokenized run uses 10,000 documents from `experiments/dataloaders.py` and prepares contiguous token arrays before timing.

`benchmark_four.py` records untimed verification and all 16 timed observations in `.tmp/codex/rust-dataloader/four-backend/benchmarks.json`. In the original `benchmark.py`, `--backend rust` selects Rust packing.

## Parity

Training compares tensors and full checkpoint state after each of 350 batches on each of two logical CPU ranks, including row-group, shard, and epoch transitions. Both Python-to-Rust and Rust-to-Python resume use serialized checkpoints with buffered documents and a retained suffix, then compare against uninterrupted Python. Validation covers scalar tokenizers, batched tokenizers, and pretokenized input without BOS. Separate cases cover a padding adapter, an incomplete validation batch, and checkpoint recovery after a refill error interrupts a batch with a partially consumed document. The benchmark tests verify source metadata with three shards and both present and absent metadata, then read from the original cache using its normal train and validation splits.

The native text tests use GPT-2 to compare all three backends across both CPU ranks and resume in both directions. The fixture includes empty documents, Unicode, literal special-token strings, and rank-strided validation row groups.

Rust unit tests inject typed row-group read failures to check reopening, cursor stability, the five-attempt limit, and immediate propagation of other errors.

## Four-backend CPU benchmark

At `7eb26da`, end-to-end Rust delivered **105.1% higher median throughput** than Rust packing alone on the same Apple M2 host. Median time per batch fell from 5.571 to 2.718 ms, and peak RSS fell by 48.7 MiB. Rust packing alone was 5.0% faster than batched Python at the median, with overlapping run ranges.

| Backend | Median source tokens/s | Run-to-run range | Peak RSS (MiB) | Median ms/batch |
| --- | ---: | ---: | ---: | ---: |
| Scalar Python | 3,554,772 | 3,551,132–3,667,770 | 321.5 | 9.203 |
| Batched Python | 5,595,782 | 5,220,132–5,677,075 | 383.6 | 5.847 |
| Batched Python + Rust packing | 5,875,133 | 5,329,612–6,587,603 | 364.1 | 5.571 |
| End-to-end Rust | 12,047,747 | 9,953,271–12,535,069 | 315.4 | 2.718 |

All runs used the Apple M2 CPU with 16 GiB RAM and macOS 27.2.1, cached ClimbMix shard `01061`, GPT-2, batch 16, context 2,048, a four-thread tokenizer budget, and 64-document chunks for batched paths. Torch and Arrow each used one thread. The four rounds used fresh CPU processes with a warm filesystem cache. Taking A through D in table order, the orders were `ABCD`, `BDAC`, `CADB`, and `DCBA`. Timing covered loader construction through exhaustion. Time per batch is each run’s elapsed time divided by its emitted batch count. Peak RSS is the highest process peak across the four runs.

Separate untimed passes matched the full `x`, `y`, and `DataLoaderState` digest for 1,651 batches and 54,014,008 source tokens per backend. All eight real GPT-2 fixture cases passed with scalar and batched Python references on two logical CPU ranks. The cases covered row-group, shard, and epoch transitions, long-document continuation, finite validation exhaustion, and resume in both directions from lookahead plus a pending suffix. The fixture included empty text, Unicode, and literal `<|bos|>` and `<|endoftext|>` strings.

In a separate in-memory check on the same host, one pretokenized document without BOS spanned 64 batches at the same shape. Median Rust packing iterator time fell from 45.1 ms to 2.45 ms after limiting the copy to the current batch. Each median uses five runs after a warmup.

## Original packing-only benchmark

These measurements predate the PyO3 upgrade and the end-to-end backend. The earlier Rust packing result was 6.82M source tokens/s on Apple M2.

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
