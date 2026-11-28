# Rust dataloader experiment

The [project README](../../README.md) covers installation and backend selection. The training benchmark measures Python and end-to-end Rust within each of two packing strategies. Both implementations use the same input bytes, GPT-2 tokenizer, four tokenizer threads, 64-document tokenization chunks, batch size 16 and one CPU rank. Torch and PyArrow each use one thread.

The untimed pass runs the two installed training loaders in lockstep and compares every input, target and serialized `DataLoaderState` through the first batch that admits epoch-two data. A mismatch identifies the strategy, context, rank and batch. The timing runs stop before that boundary, using the same number of complete batches in each matched pair. Each timed process starts the timer just before loader construction and stops after the last measured batch. The untimed pass warms the filesystem cache. Four fresh-process repetitions alternate Python-first and Rust-first. The numerator counts non-BOS tokens in the measured batches, using the first column of `x` and all of `y` so each packed position is counted once.

The benchmark links the input bytes into a temporary, single-shard training dataset at physical slot 0. Run it on the original cached `shard_01061.parquet` (SHA-256 `9025ed81d751886e7810238f2c71a4d68ee907860f3c506d1cd3e74bda135fd8`, 86,016 documents, 84 row groups). The runner checks that the original cache metadata stays byte-identical if present and stays absent otherwise.

## Paired training results

End-to-end Rust reached **1.96–2.31×** the median throughput of batched Python. All ten comparisons matched across 25,589 complete training batches and the ten epoch-crossing batches. The run used an Apple M2 on macOS 27.2.1, Python 3.12.12, PyTorch 2.11.0, PyArrow 22.0.0 and tiktoken 0.12.0. The [raw results](benchmark_paired_m2.json) record all 80 runs, the tested commit `4de4d9a` and the parity digests. Context 512 overlapped the final local install and test checks. The table includes every repetition.

Throughput is in millions of emitted non-BOS source tokens per second. Parentheses show the minimum and maximum run.

| Context | Packing | Batches per run | Source tokens per run | Batched Python | Full Rust | Speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | rolling-16 | 6,605 | 54,022,291 | 2.604 (2.449–2.980) | 6.019 (4.678–6.967) | 2.31× |
| 512 | block-best-fit | 6,604 | 54,023,422 | 4.408 (3.758–5.043) | 9.684 (7.622–11.768) | 2.20× |
| 1,024 | rolling-16 | 3,302 | 54,014,048 | 4.270 (3.950–5.136) | 8.821 (6.333–9.910) | 2.07× |
| 1,024 | block-best-fit | 3,302 | 54,024,176 | 4.473 (3.527–4.947) | 9.975 (7.279–10.708) | 2.23× |
| 2,048 | rolling-16 | 1,651 | 54,014,008 | 3.969 (3.189–4.746) | 8.823 (6.292–10.787) | 2.22× |
| 2,048 | block-best-fit | 1,651 | 54,022,890 | 4.837 (4.763–4.900) | 9.477 (8.152–10.578) | 1.96× |
| 4,096 | rolling-16 | 825 | 53,981,212 | 5.113 (4.686–5.166) | 10.327 (9.769–10.934) | 2.02× |
| 4,096 | block-best-fit | 825 | 53,987,722 | 5.351 (5.118–5.446) | 10.577 (10.128–11.406) | 1.98× |
| 8,192 | rolling-16 | 412 | 53,915,669 | 5.609 (5.354–5.649) | 11.623 (11.019–11.808) | 2.07× |
| 8,192 | block-best-fit | 412 | 53,919,817 | 5.509 (5.144–5.526) | 11.247 (9.600–11.827) | 2.04× |

## Correctness checks

The installed-wheel suite passed 158 focused tests outside the checkout with `PYTHONPATH` unset. The GPT-2 fixture compares 1,100 batches per rank and strategy, including row-group, shard and epoch transitions, literal special-token text and long-document suffixes. Both resume directions restore serialized checkpoints with lookahead and a pending suffix in epochs 1 and 2. Rolling-16 also resumes the [frozen master checkpoints](../../tests/fixtures/rolling16_master.json) from `61f0662`. Validation parity checks finite exhaustion, and the workflow test checks that only training receives block-best-fit. Both packers pass the `[9, 6, 4]` example.

All five Rust unit tests passed. The transient EIO test fails the active row-group reader and checks the next emitted text and cursor. A separate test checks the five-attempt limit. A clean Python-only installation also emitted batches with Cargo and rustc off `PATH`. [Python-only and native CI](https://github.com/art-test-stack/gpt-lab/actions/runs/46084934177)

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
