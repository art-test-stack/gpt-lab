# Small CPU packing examples

Run from the checkout with its CPU environment:

```sh
PYTHONPATH=src uv run --locked --extra cpu --group dev python experiments/20261101-packing-policies/cpu_examples.py
```

Measured on 2026-11-01 with PyTorch 2.9.1, arm64 CPU, one PyTorch thread and world size 1. No model runs: token tensors are int64; model precision is not applicable. Documents of lengths 3, 9, 23 and 65 (including real BOS) repeat in order. Batch size is 2, sequence length 16, best-fit lookahead 8 documents. Each policy warms up for 10 batches, then measures 200 batches / 6,400 target positions. Timing includes `next(loader)`, packing-state snapshots and CPU copies, with correctness assertions outside the timed interval.

| Policy | Elapsed | Target positions/s | Source tokens read during measurement | Discarded suffix tokens | Synthetic BOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| stream | 6.762 ms | 946,512 | 6,400 | 0 | 0 |
| bos_aligned | 9.513 ms | 672,758 | 6,400 | 0 | 400 |
| bos_bestfit_crop | 16.488 ms | 388,167 | 20,000 | 13,200 | 0 |

These are single short synthetic runs, not performance conclusions. The policies produce different training contexts and consume different source volumes. The script excludes tokenization, Parquet I/O, model execution, distributed work and GPU transfer. It cannot select a training default or support GPU performance claims; stream remains the default. The draft's policy-matched comparison entry point is retained at `scripts/benchmark/dataloaders.py`, including the shared crop packer.

Focused checks cover flat-stream shifts/carry, real and synthetic row starts, preserved internal BOS, short/long documents, crop accounting, deterministic validation, policy-matched benchmark accounting, configuration persistence and conflicting resume overrides. JSON-reloaded continuation checks compare inputs, targets and source state; tiny local Parquet fixtures exercise real source cursors. Agent B's PR #20 cursor commits also cover row-group offsets and partial chunks, shard/epoch boundaries, changed chunk sizes, legacy/unknown cursor versions, multiple rank assignments, retries and file cleanup.

Exact packing continuation requires packing-state version 1 and unchanged source/tokenizer, shape, settings and rank topology. Old checkpoints lacking the buffered state cannot provide it. No paid compute or GPU checks were run.
