# Compiled versus captured forward/backward

This experiment tests whether explicitly capturing compiled forward/backward
improves training throughput over existing compiled execution. Capture remains
opt-in (`--cuda-graph-mode forward_backward`); the default is `disabled`.
Optimizer updates and communication are outside capture in both modes.

Started from pinned NorMuon commit `192bbe2921e4774d24324b2d07852311b7bc4dcf`.
The results below were measured at `a5d46414cca0351276b65f30f873d40034f9a7f0`
on branch `exp/cuda-graphs`.

## Correctness and boundaries

Each replay consumes one fixed-shape microbatch using persistent inputs and
gradients. Accumulation and in-place gradient clearing preserve gradient storage.
Warmup restores parameters, buffers, gradients and CPU/CUDA RNG state. Scheduling,
finite-loss checks and dataloader work stay outside capture, so LR and other
scheduler values follow the logical optimizer step. Compiler-managed graphs are
disabled inside the explicit capture boundary.

Before timing, a small production transformer compares compiled and captured
losses, accumulated gradients, parameters and full optimizer state over six
updates with changing schedules. The eight-rank CUDA/NCCL gate passed for both
one and two accumulation microsteps. These are small-model checks, not proof of
all configurations. FP16/scaler, FP8 and custom loss overrides are unsupported.

## Matched measurement

Run `cg-8-b32-matched-v2`: eight A100-SXM4-80GB GPUs, BF16, 422,314,440 parameters,
12 layers, width 768, fused attention, sequence 2048, microbatch 32, one accumulation
microstep, global batch 524,288 tokens, seed 42, and `configs/normuon.yaml`.
Both modes ran sequentially on the same allocation with matching initial state
and data position. Horizon 2317 and warmup 300 were preserved; execution stopped
after 150 updates. Evaluation and checkpoint saving were disabled.

| Metric | Compiled | Captured |
| --- | ---: | ---: |
| Steady global tokens/s (updates 11–150) | 1,298,919 | 1,292,484 |
| Mean training step (ms) | 403.32 | 405.36 |
| Process elapsed, including startup (s) | 199.88 | 199.93 |
| Explicit compilation/warmup/capture setup (s) | 0 | 86.23 |
| Peak reserved GPU memory per rank (GiB) | 59.05 | 68.90 |

The baseline's zero explicit capture setup does not mean zero compilation cost;
its compilation occurs in the first update. Process elapsed excludes the shared
correctness gate and allocation/image setup. This single trial found no speedup
(captured throughput was 0.5% lower) and higher reserved memory. Keep compiled
execution as the default.

A separate batch-32 profile (`cg-8-b32-profile-v1`) recorded three updates per
mode. On rank 0, host launch calls fell from 2,541 to 1,092, while time without
recorded GPU activity stayed around 2%. Captured execution added gradient
add/fill kernels. Matrix multiplications accounted for about 55% of summed
kernel time. Profile timings are not throughput benchmarks, and timeline activity
does not measure SM utilization. Physical topology was unavailable from
`nvidia-smi`; CUDA peer access alone does not establish NVLink connectivity.

## Reproduce (user launches paid compute)

From a clean checkout whose HEAD matches its remote branch, run:

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python scripts/modal_cuda_graphs.py --gpus 8 --mode paired --require-sxm --run-name cg-8-b32-matched-new -- --optim-config-path configs/normuon.yaml --board dummy --device-batch-size 32
```

For a separate 14-update profile, add `--profile` before `--` and use a new run
name. `--require-sxm` checks the allocated GPU model; it cannot select SXM hardware.
`--board dummy` disables W&B, while JSON results and profiler files remain saved.
Artifacts live on the `gpt-lab-runs` Modal volume under the run name. Open
`disabled/trace_rank_0.json` and `forward_backward/trace_rank_0.json` in Perfetto.
See the [main README](../../README.md#explicit-cuda-capture-experimental) for setup
and supported launcher modes.

Optimizer/communication capture and ANVIL/bucketed-communication comparisons
were deferred. Existing rank-local loss reporting and the momentum schedule
were flagged separately; these measurements retain the pinned behavior.
