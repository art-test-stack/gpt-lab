# Rust dataloader prototype

Baseline is `61f066250420afbc121524037b996ee4a3524c0a`, fetched from `origin/master` before creating `experiment/rust-dataloader`.

`build_dataloader(..., tokenizer_batch_size=64)` batches encoding for adapters that declare `supports_exact_batch_encoding = True` on their concrete class. Their `encode_batch` must return deterministic, unpadded sequences in source order that match scalar encoding, including BOS. Checkpoint restore encodes buffered documents individually. The tiktoken adapter used for GPT-2 opts in and forwards `num_threads=tokenizer_threads`. Other adapters and pretokenized input use the scalar path. The default packing lookahead is 16 documents when BOS is available and one otherwise.

`build_dataloader(..., packing_backend="rust")` or `DistDataLoader(..., packing_backend="rust")` selects the optional extension. Rust selects documents and fills the existing CPU batch buffer from contiguous `int64` documents. For a document that spans batches, it copies only the tokens the current batch consumes. Python owns the shard order, refill, and checkpoint. Cursors advance per admitted document, so unused encodings are recreated after resume. The default is scalar tokenization and Python packing, with the same checkpoint signature and validation exhaustion behavior as the baseline.

`build_dataloader(..., dataloader_backend="rust")` selects the end-to-end CPU path for cached Parquet text with the repository’s tiktoken tokenizer. Python supplies the rank-owned shard order, actual tokenizer parameters, batch shape, thread budget, and checkpoint. Rust reads and tokenizes the text, packs the next batch, and copies it into the existing CPU tensor in one extension call. PyTorch transfers that tensor to the training device. CoreBPE is initialized once per iterator from [tiktoken 0.12.0](https://github.com/openai/tiktoken/tree/97e49cbadd500b5cc9dbb51a486f0b42e6701bee). It encodes literal special-token strings as ordinary text and prepends the configured BOS.

`gpt-lab train auto` and `gpt-lab train resume` accept `--dataloader-backend {python,rust-packing,rust}` for both training and validation. `python` is the default. The two Rust selections default to tokenization chunks of 64 documents and accept the same training checkpoints as Python. Pass `--tokenizer-batch-size 64` to compare `python` and `rust-packing` with the same tokenization chunk size for training and validation. The end-to-end path requires local shards and non-null UTF-8 text. On Unix, an `EIO` while reading a row group causes Rust to reopen the shard before retrying the same group. It makes up to five attempts with 1, 2, 4, and 8 second delays.

If a refill error escapes during training, discard the iterator. Rebuild with `build_dataloader(..., resume_state=last_committed_state)` using the last committed `DataLoaderState`. The failed batch may already have consumed part of a document.

`--dataloader-backend rust --packing-strategy block-best-fit` admits 64 consecutive training documents and drains each block before admitting more. Blocks can cross row-group and shard boundaries within an epoch. The final block of each epoch can be shorter. The packer first tries an exact single-document fit, then an exact pair. The fallback selects the largest fitting document or splits the shortest when none fits. Ties follow source order and document suffixes are retained.

Rolling-16 remains the default and keeps its existing checkpoint signature. Exact-pair blocks have a distinct signature from the greedy blocks at `f008399`. Rebuild the optional extension for exact-pair runs and resume greedy-block checkpoints on the greedy implementation. Validation uses rolling-16 in both configurations so loss comparisons can use the same held-out batches.

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

Run both sweep regimes in the same environment using the cached shard.

```sh
runner=experiments/20261124-rust-dataloader/benchmark_four.py
shard=.tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet
for shape in 16:2048 16:512 16:1024 16:4096 16:8192 64:512 32:1024 8:4096 4:8192; do
  "$PYO3_PYTHON" "$runner" --shard "$shard" --batch-size "${shape%:*}" --context "${shape#*:}"
done
```

For each nondefault shape, the runner writes `benchmarks-b<B>-t<T>.json` in the same directory.

## Parity

Training compares tensors and full checkpoint state after each of 350 batches on each of two logical CPU ranks, including row-group, shard, and epoch transitions. Both Python-to-Rust and Rust-to-Python resume use serialized checkpoints with buffered documents and a retained suffix, then compare against uninterrupted Python. Validation covers scalar tokenizers, batched tokenizers, and pretokenized input without BOS. Separate cases cover a padding adapter, an incomplete validation batch, and checkpoint recovery after a refill error interrupts a batch with a partially consumed document. The benchmark tests verify source metadata with three shards and both present and absent metadata, then read from the original cache using its normal train and validation splits.

The native text tests use GPT-2 to compare all three backends across both CPU ranks and resume in both directions. The fixture includes empty documents, Unicode, literal special-token strings, and rank-strided validation row groups.

Exact-pair blocks matched a Python reference for 1,100 batches on each of two logical CPU ranks. Resumes from a partially consumed document and buffered block references matched exact tensors and state across row-group, shard, and epoch transitions. Tests reject greedy-block checkpoints under exact-pair packing and reject checkpoints across rolling and block packing. An older rolling-16 checkpoint from `8d24e93` resumed for 100 identical batches on each rank. The previous rolling and greedy comparison used identical tensors and state for all 47 held-out batches per rank.

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

## M2 context sweep

The rolling-16 sweep used the original ClimbMix shard and the code at `8d24e93` on the same M2. Each shape had four counterbalanced runs per backend and a separate untimed pass per backend. Context 2,048 at batch 16 serves both the batch-16 and fixed-32,768-target regimes. Throughput is in millions of emitted source tokens per second. Parentheses show the minimum and maximum run.

| Batch | Context | Matched batches | Scalar Python | Batched Python | Rust packing | End-to-end Rust |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 512 | 6,605 | 3.664 (3.564–3.671) | 5.732 (5.226–5.822) | 6.935 (6.054–6.997) | 12.241 (9.868–12.739) |
| 64 | 512 | 1,651 | 3.476 (3.416–3.515) | 5.517 (5.241–5.522) | 6.533 (6.195–6.589) | 12.560 (12.485–12.718) |
| 16 | 1,024 | 3,302 | 3.717 (3.566–3.757) | 5.844 (4.619–6.069) | 6.870 (5.819–7.152) | 12.930 (12.156–13.147) |
| 32 | 1,024 | 1,651 | 3.530 (3.492–3.552) | 5.558 (5.466–5.611) | 6.473 (6.323–6.587) | 12.341 (11.970–12.467) |
| 16 | 2,048 | 1,651 | 3.601 (3.501–3.685) | 5.988 (5.650–6.191) | 6.403 (6.321–6.679) | 12.995 (12.579–13.190) |
| 16 | 4,096 | 825 | 3.456 (2.545–3.607) | 5.609 (3.754–5.735) | 6.152 (5.822–6.519) | 10.294 (8.189–12.256) |
| 8 | 4,096 | 1,651 | 3.461 (2.572–3.519) | 5.116 (4.118–5.627) | 6.322 (6.182–6.588) | 11.337 (9.077–12.538) |
| 16 | 8,192 | 412 | 3.569 (3.180–3.623) | 5.646 (5.488–5.745) | 6.456 (6.303–6.552) | 12.057 (11.485–12.546) |
| 4 | 8,192 | 1,651 | 3.203 (3.039–3.460) | 5.279 (4.677–5.324) | 5.259 (3.897–6.105) | 10.340 (7.579–11.049) |

End-to-end Rust had 1.67–2.03× the median throughput of Rust packing across the nine shapes. Within each shape, every backend produced the same full digest of `x`, `y`, and validation `DataLoaderState` through exhaustion.

## Training-mode packing benchmark

`benchmark_training.py` compares full Rust rolling-16 and greedy blocks at `f008399` with exact-pair blocks at the candidate commit. It builds each revision from Git into a temporary directory and runs `split="train"` through one epoch. The timing builds add only the epoch stop. Separate builds count splits and verify their emitted tensors and state against the timing builds.

```sh
.venv/bin/python experiments/20261124-rust-dataloader/benchmark_training.py \
  --candidate fe0f648 --shard .tmp/codex/pr54-validation/recheck/climbmix-base/shard_01061.parquet
```

Use the GPT-2 cache from the setup above. Add `--offline` when the locked Cargo dependencies are cached. The runner saves individual observations to `.tmp/codex/rust-dataloader/training-comparison/comparison.json` by default.

Throughput counts non-BOS tokens at unique positions in emitted full batches, using the first column of `x` and all of `y`. Row-fit eligibility uses the encoded document length including BOS and the packed row width of `context + 1`. A split is counted once at the first cut. The untimed counts cover all source documents, and the JSON records the counts added during the final incomplete batch separately.

On Apple M2, all arms used cached ClimbMix shard `01061`, batch 16, GPT-2, a four-thread tokenizer budget, and 64-document tokenization chunks. Each arm ran in a fresh process in every one of the six run orders. Timing covered loader construction through exhaustion. The [per-run JSON](benchmark_training_m2.json) includes all 90 samples, the shard hash, tensor and state digests, and the emitted source-token count used in every rate.

Median millions of emitted source tokens/s with the observed run range in parentheses.

| Context | Rolling-16 | Greedy blocks | Exact-pair blocks | Exact-pair vs greedy |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 9.968 (9.410–10.605) | 9.305 (6.383–10.327) | 9.503 (7.918–11.276) | +2.1% |
| 1,024 | 11.589 (9.412–12.045) | 11.827 (11.268–12.385) | 11.532 (11.187–11.842) | -2.5% |
| 2,048 | 10.352 (9.090–12.014) | 10.755 (10.247–12.206) | 10.116 (8.985–10.857) | -5.9% |
| 4,096 | 10.604 (4.854–11.453) | 10.901 (4.714–12.010) | 10.753 (10.154–11.978) | -1.4% |
| 8,192 | 6.964 (5.639–9.133) | 7.942 (7.470–8.051) | 7.767 (7.241–8.424) | -2.2% |

Row-fit documents split in one epoch. Each percentage uses the row-fit count for its context.

| Context | Row-fit documents | Rolling-16 splits | Greedy block splits | Exact-pair splits |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 60,233 | 21,800 (36.19%) | 19,982 (33.17%) | 12,417 (20.61%) |
| 1,024 | 72,291 | 19,472 (26.94%) | 18,120 (25.07%) | 9,336 (12.91%) |
| 2,048 | 80,978 | 15,667 (19.35%) | 14,487 (17.89%) | 6,801 (8.40%) |
| 4,096 | 84,608 | 10,310 (12.19%) | 9,384 (11.09%) | 3,858 (4.56%) |
| 8,192 | 85,672 | 5,861 (6.84%) | 5,265 (6.15%) | 1,820 (2.12%) |

Exact-pair selection reduced row-fit splits by 38–65% versus greedy blocks. Median throughput changed by −5.9% to +2.1% and run ranges overlapped at every context. Greedy and exact-pair counts include six and one final-tail splits at context 4,096, and eight and four at context 8,192.

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
