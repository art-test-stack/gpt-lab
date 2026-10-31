# Preserved communication results

Recommendation: **baseline**, selected by default. Opt-in modes `buckets`,
`pipeline`, `backward` remain selectable through `--communication`; no optimizer
equations, parameter owners or storage/communication dtypes are changed by moving files.

Original starting SHA: `192bbe2921e4774d24324b2d07852311b7bc4dcf`.
Muon tested SHA / preserved `exp/comm-buckets` tip:
`8956f6e2cb8aae856c467c40ff2e3f37824dff75`.
NorMuon tested SHA / preserved `exp/comm-optimizer-sweeps` tip:
`2795b42e96d444b285ffff5dedd3955e68550804`.
The latter includes every commit of the former. Both branches stay intact.

All runs: 4 A100-SXM4-80GB, Torch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5,
SwiGLU depth 12/width 768, BF16 storage policy, FA2, batch 16/GPU, sequence 2048,
global tokens 524288/update, horizon 2317/LR warmup 300. Full confirmation:
150 updates, discard first 25; optimizer profiles compared only within optimizer.

| Full confirmation | Global tokens/s | Median / p95 ms | Peak allocated GiB/GPU |
|---|---:|---:|---:|
| Muon 64 MiB baseline | 618857 | 847.73 / 849.68 | 25.39 |
| Muon 64 MiB buckets | 615255 | 852.54 / 854.57 | 28.32 |
| Muon 128 MiB baseline | 617488 | 849.45 / 852.48 | 25.39 |
| Muon 128 MiB pipeline | 615619 | 852.04 / 855.29 | 27.53 |
| NorMuon 128 MiB baseline | 613868 | 854.18 / 858.06 | 25.39 |
| NorMuon 128 MiB pipeline | 611848 | 857.00 / 860.55 | 27.53 |

Muon confirmations regressed 0.582% / 0.303%; NorMuon pipeline regressed
0.323% in its 30-sample screen and 0.329% in confirmation. Muon 128 MiB screen
was only +0.018%; neither meets the >=1% gain required in both stages.
NorMuon screen results (all retained): baseline 616358, buckets 614317,
pipeline 614365, backward 614295 tokens/s. All alternative screen modes used
27.53 GiB versus baseline 25.39 GiB. Reserved memory was 30.34 GiB/GPU.

## Padding pathology and corrected layout

At 64 MiB, twelve 18 MiB whole matrices became singleton owner-packed
72 MiB inputs, each with 54 MiB padding: **648 MiB extra padding/rank** plus
576 bytes of ordinary owner padding. Optimizer collectives rose from 28 to 48;
reduction input grew from 1095.190 to 1743.190 MiB/rank. Transport pool: 2.923 GiB.

At 128 MiB owner-balanced groups fit together: packed FP32 RS layout was
2×72, 1×126, 1×99, 1×63.002197, 1×128 and 1×19.375 MiB;
BF16 7×73.6875 MiB; and a 96-byte FP32 all-reduce.
Only 576 bytes padding remained, no oversized bucket. Total collectives 29
(including one 4-byte safety all-reduce), same reduction payload as baseline,
2.132 GiB transport pool. Removing padding saved ~0.79 GiB but did not establish
a throughput improvement. These are API/layout bytes, not measured link traffic.

## Correctness and profiling limits

Saved four-rank NCCL fixtures passed Muon and NorMuon modes B/C/D against their
own baseline: parameter, owner-local gradient and optimizer-state max errors 0.
NorMuon gate ran `force_eager`, four updates/mode/cap. This is a transport check,
not compiled ANVIL validation. Timing used native production training.
No separate profiler run established NCCL overlap, SM residency or actual link
bandwidth; no repeated-trial confidence intervals or long quality comparison.

External artifacts stay on Modal `gpt-lab-runs`:
`comm-a100-20261031-r2`, `comm-a100-128m-20261031`,
`comm-normuon-128m-20261031`; initial NCCL retries: `comm-nccl-20261031`,
`comm-nccl-20261031-r2`. See preserved historical notes for original commands.
The compact NorMuon summary is [normuon-summary.json](normuon-summary.json).
The original notes retain tested source names and paths verbatim, even though
the current implementations moved. Legacy script paths are compatibility shims.
