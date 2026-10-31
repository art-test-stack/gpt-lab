# Optimizer-specific communication sweeps

This branch, `exp/comm-optimizer-sweeps`, starts at
`8956f6e2cb8aae856c467c40ff2e3f37824dff75` and changes the Modal orchestration,
its checks, and documentation only. The communication implementation, kernels,
optimizer profiles, model, trainer, and data loader are unchanged. No new
optimizer or learning-quality run is included. ANVIL remains separate.

## Preserved Muon source and results

Keep `exp/comm-buckets` and its original checkout at
`8956f6e2cb8aae856c467c40ff2e3f37824dff75`. Its original launcher requires that
exact local HEAD and published branch tip to profile the existing results.
The new branch checks its own attached branch's published tip instead; it does
not move, merge, or republish `exp/comm-buckets`.

Both user-launched Muon sweeps completed at the original SHA:

- 64 MiB: `gpt-lab-runs/comm-a100-20261031-r2`;
  [Modal run](https://modal.com/apps/testardarthur/main/ap-9hY9gytbHf4kKYe9LGeNp0).
- 128 MiB: `gpt-lab-runs/comm-a100-128m-20261031`, completed
  2026-10-31 10:37:04 UTC. Its manifest and per-candidate results were downloaded
  read-only after completion. The agent did not repeat or interrupt the sweep.

Configuration: 4 x NVIDIA A100-SXM4-80GB, PyTorch 2.9.1+cu128, CUDA 12.8,
NCCL 2.27.5, depth 12, width 768, head dimension 128, SwiGLU, BF16 compute with
the existing mixed parameter-storage policy, FA2 loaded and FA3 absent. Explicit
Muon profile `configs/optim.yaml`, microbatch 16/GPU, sequence 2048, four
accumulation microsteps, global batch 524,288, seed 42, scheduler horizon 2317,
LR warmup 300. Captured backward, evaluation, generation and checkpoint writes
were disabled. The 128 MiB baseline and pipeline had identical recorded initial
model/optimizer fingerprints and consumed data-position sequences.

## Measured Muon performance

The original screens used 35 total updates, excluding 25 warmup updates: only
10 timed samples each. Full confirmations used 150 total/125 timed updates.
Throughput below is global steady wall tokens/s; step times include work
between native train steps. Peak allocated memory is after warmup, per GPU;
all four ranks reported the same values.

| Full run | Global tokens/s | Median / p95 step ms | Peak allocated GiB/GPU | Versus its baseline |
|---|---:|---:|---:|---:|
| 64 MiB A baseline | 618,857 | 847.73 / 849.68 | 25.39 | reference |
| 64 MiB B immediate wait | 615,255 | 852.54 / 854.57 | 28.32 | -0.582% |
| 128 MiB A baseline | 617,488 | 849.45 / 852.48 | 25.39 | reference |
| 128 MiB C pipeline | 615,619 | 852.04 / 855.29 | 27.53 | -0.303% |

All runs reserved 30.34 GiB/GPU. At 128 MiB the native synchronized train-step
throughputs were 617,886 baseline and 616,002 pipeline tokens/s. Process startup
through the first completed update was 48.44/48.08 seconds for these warm-cache
full runs; image building and queueing are excluded. Startup from the first cold
screen must not be compared with later processes reusing the compiler cache.

| 128 MiB screen | Global tokens/s | Median / p95 step ms | Versus A |
|---|---:|---:|---:|
| A baseline | 619,499 | 845.42 / 850.42 | reference |
| B immediate wait | 619,213 | 845.88 / 850.51 | -0.046% |
| C pipeline | 619,612 | 846.00 / 848.59 | +0.018% |
| D backward hooks | 618,246 | 848.39 / 851.43 | -0.202% |

Pipeline was the fastest nonbaseline candidate in the short screen, not a
confirmed improvement. Its full confirmation regressed. Retain baseline. The
128 MiB candidate was only 0.059% faster than the earlier 64 MiB candidate,
while the two baseline runs differed by 0.221%; these separate sessions do not
establish a reliable speed improvement from bucket size alone. There were no
repeated trials or confidence intervals.

## Actual 128 MiB bucket layout

The full pipeline JSON contains this gradient layout, identical on each rank.
Lengths below are packed inputs including padding. MiB means 2^20 bytes.

| Count | Collective | Dtype | Packed MiB each | Padding each |
|---:|---|---|---:|---:|
| 2 | reduce-scatter | FP32 | 72 | 0 |
| 1 | reduce-scatter | FP32 | 126 | 0 |
| 1 | reduce-scatter | FP32 | 99 | 0 |
| 1 | reduce-scatter | FP32 | 63.002197 | 576 bytes |
| 1 | all-reduce | FP32 | 96 bytes | 0 |
| 7 | reduce-scatter | BF16 | 73.6875 | 0 |
| 1 | reduce-scatter | FP32 | 128 | 0 |
| 1 | reduce-scatter | FP32 | 19.375 | 0 |

Every reduce-scatter has equal useful owner lengths except the 63.002197 MiB
bucket: three owners have 4,128,912 elements and one has 4,128,768 elements,
producing 576 bytes of padding. No bucket is oversized. Whole-matrix ownership
is preserved; dtype boundaries remain intact.

| Per-update accounting | Baseline A | 64 MiB B/C/D | 128 MiB B/C/D |
|---|---:|---:|---:|
| Gradient collectives | 15: 2 AR + 13 RS | 34: 1 AR + 33 RS | 15: 1 AR + 14 RS |
| Parameter all-gathers | 13 | 13 | 13 |
| Safety control all-reduces | 0 | 1, four bytes | 1, four bytes |
| Total optimizer collectives | 28 | 48 | 29 |
| Reduction input MiB/rank | 1095.190 | 1743.190 | 1095.190 |
| Padding | 576 bytes | 648 MiB + 576 bytes | 576 bytes |
| Gather output MiB/rank | 1095.190 | 1095.190 | 1095.190 |
| Extra persistent transport GiB/rank | 0 | 2.923 | 2.132 |
| Ring-send estimate GiB/rank | 1.604 | 2.079 | 1.604 |

These are API/layout bytes, not measured NVLink/PCIe traffic. The baseline's
zero extra persistent pool does not imply zero temporary communication memory.
Counts exclude trainer metrics and initial setup exchanges. At 64 MiB, twelve
18 MiB whole matrices became separate 72 MiB owner-packed inputs, each with
54 MiB padding. At 128 MiB, owner-balanced matrices fit together. Removing that
648 MiB padding saves about 0.79 GiB of transport storage and peak allocation
per GPU, but does not produce a measured throughput win here.

Actual four-rank NCCL fixture checks passed for all Muon modes, with zero
reported parameter/local-gradient/optimizer-state errors. The 128 MiB sweep's
small NorMuon compatibility check covered pipeline, also with zero errors.
These eager small-fixture checks are not 150-update compiled full-model equality
checks. Optimizer CUDA time, packing cost, exposed wait and useful overlap are
still unmeasured: no separate profiler run has been launched.

## New optimizer-specific launcher

`--optimizer muon|normuon` now selects the optimizer for correctness, every
training command, manifests, validation, compatibility checks and profiling.
Default remains Muon. The native runner already supports both profiles; its
implementation is unchanged. NorMuon explicitly uses `configs/normuon.yaml`.

- Each sweep first gates training on a four-rank NCCL fixture comparing B/C/D
  against the selected optimizer's own A baseline. It aborts on failures/skips.
- Screen all four modes for **55 total updates: 25 warmup + 30 measured**.
- Confirm A and the fastest nonbaseline candidate for **150 total updates**,
  excluding the first 25. The candidate is compared even if its screen is weak.
- As approved, recommend a candidate only if it has **at least 1% higher
  throughput in both screen and confirmation**. Otherwise retain baseline.
  This is a conservative decision threshold, not a confidence interval.
- `candidate_mode` identifies the tested candidate; `recommended_mode` records
  the final decision. Do not interpret a fastest candidate as a baseline win.
- Fresh processes reset seed/model/data for each candidate. Initial optimizer
  state, group fingerprints and data-position comparisons are made **within one
  optimizer**. Muon states/updates are not required to match NorMuon states/updates.
  Old Muon JSON is not passed into NorMuon fingerprint validation.
- Architecture, BF16/FA2, original precision/ownership/communication, microbatch
  16/GPU, sequence 2048, global batch 524,288, horizon 2317 and LR warmup 300 are
  matched. CUDA capture stays off. No evaluation, sampling or checkpoint writes.
- Results persist after each candidate; failure logs persist before aborting.
  Process-group and child-process deadlines and restart/retry settings are
  unchanged. Profile traces are collected separately from timed runs.
- Profile/compatibility inputs must be completed sweeps with the **same SHA and
  selected optimizer**. Profile always compares A with `candidate_mode`, even
  when the recommendation is to retain A, and inherits the sweep's bucket size.

Local checks cover both optimizers' dry runs and command paths, selected NCCL
fixture configuration, same-optimizer fingerprint checks, branch-specific source
verification, same-SHA/optimizer result validation, 55/150-update caps, the 1%
decision rule in both stages, profile bucket inheritance, timeouts and result
persistence. Candidate execution is simulated in these launcher tests; no GPU
performance or NCCL result is claimed for the newly prepared NorMuon sweep.
Validation: 20 focused local launcher/native-CLI checks passed; the Gloo and
GPU transport tests were not selected because the communication implementation
is unchanged. `git diff --check` passed, and source/profile files were checked
byte-for-byte against the preserved Muon SHA.

## User-launched commands

From the new clean `exp/comm-optimizer-sweeps` checkout, with the existing Modal
cache/results volumes and `gpt-lab` secret (`HF_TOKEN`):

```bash
SHA=$(git rev-parse HEAD)

# No Modal job: verify the pushed source and selected optimizer.
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py sweep --optimizer normuon \
  --sha "$SHA" --run-name comm-normuon-128m-20261031 --bucket-mib 128 --dry-run

# User-launched 4 x A100-80GB NCCL gate, screen, then confirmation.
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py sweep --optimizer normuon \
  --sha "$SHA" --run-name comm-normuon-128m-20261031 --bucket-mib 128

# Optional, only after that sweep completes: separate NorMuon profiler run.
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py profile --optimizer normuon \
  --sha "$SHA" --run-name comm-normuon-profile-20261031 \
  --results-run comm-normuon-128m-20261031
```

Do not repeat the Muon sweep. To profile its existing 128 MiB results, use the
**preserved original checkout and SHA**, not the new launcher:

```bash
cd /Users/arthurtestard/worktrees/gpt-lab/comm-backward/.tmp/codex/comm-buckets/worktree
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py profile \
  --sha 8956f6e2cb8aae856c467c40ff2e3f37824dff75 \
  --run-name comm-muon128-profile-20261031 \
  --results-run comm-a100-128m-20261031
```

No additional compute has been launched by the agent. The original Muon source
pin and results are reused as evidence without rerunning them.
