# Rust dataloader experiment

The [project README](../../README.md) covers installation and backend selection. The training benchmark measures Python and end-to-end Rust within each of two packing strategies. Both implementations use the same input bytes, GPT-2 tokenizer, four tokenizer threads, 64-document tokenization chunks, batch size 16 and one CPU rank. Torch and PyArrow each use one thread.

The untimed pass runs the two installed training loaders in lockstep and compares every input, target and serialized `DataLoaderState` through the first batch that admits epoch-two data. A mismatch identifies the strategy, context, rank and batch. The timing runs stop before that boundary, using the same number of complete batches in each matched pair. Each timed process starts the timer just before loader construction and stops after the last measured batch. The untimed pass warms the filesystem cache. Four fresh-process repetitions alternate Python-first and Rust-first. The numerator counts non-BOS tokens in the measured batches, using the first column of `x` and all of `y` so each packed position is counted once.

The benchmark links the input bytes into a temporary, single-shard training dataset at physical slot 0. Run it on the original cached `shard_01061.parquet` (SHA-256 `9025ed81d751886e7810238f2c71a4d68ee907860f3c506d1cd3e74bda135fd8`, 86,016 documents, 84 row groups). The runner checks that the original cache metadata stays byte-identical if present and stays absent otherwise.

## Reproduce

From the repository root, with Cargo and the path to your cached ClimbMix shard:

```sh
: "${CLIMBMIX_SHARD:?Set CLIMBMIX_SHARD to the path of shard_01061.parquet}"
mkdir -p .tmp/codex
scratch=$(mktemp -d .tmp/codex/paired-dataloader.XXXXXX)
scratch=$(cd "$scratch" && pwd)
uv sync --locked --extra cpu --group dev --no-editable
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$scratch/tiktoken}"
.venv/bin/python -c 'from gpt_lab.tokenizer import Tokenizer; Tokenizer.from_pretrained("gpt2", source="tiktoken")'
env -u PYTHONPATH .venv/bin/python -I experiments/20261124-rust-dataloader/benchmark.py \
  --shard "$CLIMBMIX_SHARD" --output "$scratch/paired.json"

env -u PYTHONPATH GPT_LAB_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q \
  tests/test_dataloader_parity.py tests/test_native_dataloader.py tests/cli/test_cli.py tests/cli/test_workflows.py
cargo test --locked --manifest-path packages/rust-dataloader/Cargo.toml --lib
```

## Historical measurements

The [report pinned at `b33bcc8`](https://github.com/art-test-stack/gpt-lab/blob/b33bcc8c8e1c4228186cbd94783caf95ca6f280a/experiments/20261124-rust-dataloader/README.md) preserves the original packing-only run, the four-backend finite-validation runs and the Rust-only training split study. It links the matching runners and the training study's [90 raw observations](https://github.com/art-test-stack/gpt-lab/blob/b33bcc8c8e1c4228186cbd94783caf95ca6f280a/experiments/20261124-rust-dataloader/benchmark_training_m2.json). The historical training runner archives `f008399` (rolling and greedy blocks) and `fe0f648` (exact pairs), and instruments separate builds for final-tail split accounting.
