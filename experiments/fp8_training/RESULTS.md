# FP8 results retained without promoting the option

Implementation SHA: `1d1e31f`; explicit attention-kernel SHA:
`3fcd250669bb2f0bb5c25397c058f8cf7bd7d429`.
Runs `gpt-lab-runs/fp8-1h100-03-bf16` and
`gpt-lab-runs/fp8-1h100-03-fp8-forward` used that same SHA, H100 80GB HBM3,
Torch 2.9.1+cu128, FA3, the same SwiGLU/storage/optimizer/data/schedules.
Steps 2–9: BF16 1419.6675 ms / 369303 tok/s, FP8 forward 1403.585 ms /
373535 tok/s. BF16-peak-normalized utilization 36.48% / 36.89%, against
989 dense BF16 TFLOP/s, unchanged useful FLOPs/token. Startup-containing step 0:
76.50 / 54.87 seconds. Max paired loss delta 0.000024.

The ~1.15% apparent gain has no matched repeated-control significance result.
Flags and class names do not prove compiled FP8 kernels. Native compiled
forward dispatch, full-update attribution and a reliable speedup remain pending
until the prepared batch-16 diagnostic completes. No FP8 backward, caches,
head or QKV conversion is enabled. Stored parameter/gradient/optimizer dtypes,
sharing, full-vocabulary loss and checkpoint semantics are preserved.
External logs/traces remain on Modal; no large trace is committed.
