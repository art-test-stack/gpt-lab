# H100 compiled FP8 diagnostic — batch 16

Question: do faster FP8 forward GEMMs survive conversion and whole-update costs?
One H100 worker profiles BF16/FP8, then unprofiled BF16→FP8→BF16 with identical
initial state/data, 20 warmup + 50 measured updates per leg. SwiGLU/FA3,
batch 16×2048, 16 microsteps, horizon 2317/LR warmup 300 are fixed.
Production pin: `3fcd250669bb2f0bb5c25397c058f8cf7bd7d429`; complete harness
first committed at `cd7ca922d5d062a4b7f537c1d2e6e672b8a91514`.
Native compiled FP8 dispatch and reliable speedup remain **pending**. Existing
`h100-fp8-diag-01` artifacts do not establish a completed diagnostic; use a fresh name.
Reproduce (user launch only):
`uv run --locked --extra cpu --with modal==1.6.2 python experiments/h100_fp8_diagnostic/launch.py --run-name h100-fp8-diag-consolidated-01 --detach`.
Setup, checks, trace accounting and external artifact retrieval: [HARNESS.md](HARNESS.md).
