# Optional FP8 training

Question: can native FP8 FFN compute improve the same SwiGLU model?
A = BF16; B = existing opt-in FFN-forward FP8 with BF16 backward. Backward,
head and QKV FP8 extensions are unimplemented. FP8 is never enabled by default.
Setup: H100, FA3, 12×768 SwiGLU, batch 16×2048, 16 accumulation microsteps,
horizon 2317/warmup 300; torchao 0.15.0, PyTorch 2.9.1. A100 retains BF16.
The ten-update H100 pair showed ~1.15% higher throughput, insufficient to
establish reliable speedup or native compiled dispatch. See [RESULTS.md](RESULTS.md).
Reproduction: `python experiments/fp8_training/run_h100_compare.py --help`;
prefer the prepared [complete diagnostic](../h100_fp8_diagnostic/README.md).
Backend precision/storage/checkpoint details and prior commands: [BACKEND.md](BACKEND.md).

The native converter also accepts ReLU-squared FFNs, preserving parameter identities
and checkpoint keys. CPU wiring checks cover both architectures; native ReLU-squared
CUDA dispatch remains a required hardware check. Prepare it with
`python experiments/fp8_training/check_native.py --activation relu_squared --output <fresh-directory>`
on an H100 with the GPU and FP8 extras installed. This command runs GPU work.
