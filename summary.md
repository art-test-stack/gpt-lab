# Dataloader packing and benchmark overhaul

## Summary

This branch makes token packing explicit, measurable, and resumable. It fixes the default stream loader so tokens and adjacent transitions are preserved across rows and batches, adds a lossless BOS-aligned strategy, retains the legacy destructive best-fit strategy for comparison, and rebuilds the dataloader benchmark around policy-matched correctness and performance measurements.

## Packing strategies

| Strategy | Behavior | Trade-off |
| --- | --- | --- |
| `stream` | Treats documents as one continuous `B*T+1` token stream and carries the final token into the next batch. | Default and lossless, but rows do not necessarily start with BOS. |
| `bos_aligned` | Starts every row with BOS, retains document continuations, and inserts a synthetic BOS when a document spans rows. | Lossless for complete batches, with synthetic-BOS overhead. |
| `bos_bestfit_crop` | Uses the legacy nanochat-style best-fit packer to keep rows full and BOS-aligned. | May permanently discard document suffixes and retains approximate resume behavior. |

Neither `stream` nor `bos_aligned` emits padded or incomplete batches. A finite source stops when it cannot fill the next batch.

BOS alignment is a layout policy, not document isolation: without segment-aware attention masks, tokens can still attend across document boundaries within a row.

## Key changes

- Reworked `DistDataLoader` to preserve FIFO token order, retain long-document tails, and maintain transitions across row and batch boundaries.
- Added explicit `stream`, `bos_aligned`, and `bos_bestfit_crop` selection through the public API, configuration, and training CLI.
- Added exact checkpoint state for `stream` and `bos_aligned`, including pending tokens, carry tokens, continuation state, the active strategy, and packing statistics.
- Corrected shard resume offsets so checkpoints continue from the next unread document instead of replaying the previous chunk.
- Added `PackingStats` counters for source-token flow, destructive crops, skipped transitions, synthetic BOS tokens, intentional BOS boundaries, and buffered state.
- Updated the README with each strategy's guarantees and limitations.

## Benchmark changes

The dataloader benchmark now compares implementations only when they implement the same packing policy. It:

- runs correctness checks before timing;
- separates flat-stream and destructive BOS-best-fit comparisons;
- supports on-the-fly and pretokenized inputs;
- measures loader and device-transfer time without model execution;
- rotates implementation order across trials and excludes warmup work;
- reports throughput, latency, source-token utilization, transition coverage, supervision utilization, crop rate, BOS alignment, and buffering;
- writes reproducible metadata plus JSON, CSV, HTML, and optional plot artifacts.

No benchmark result numbers are committed; results depend on the selected corpus, tokenizer, and device.

## Test coverage

New tests cover token and transition preservation, long-document tails, destructive-crop accounting, BOS-aligned packing, deterministic validation, exact checkpoint resume, incomplete-batch rejection, benchmark accounting invariants, and warmup exclusion.

The full non-slow test suite passes: **59 tests passed**.
