# H100 input-pipeline investigation — no demonstrated improvement

Question: does production input preparation/transfer limit training throughput?
Retained selectable paths: production, CPU-cached inputs and GPU-resident inputs,
with production controls repeated. Historical batch-16 and batch-32 results are
both retained; no new batch-32 experiment is added. BF16/FA3, 12×768 SwiGLU,
sequence 2048, global batch 524288, horizon 2317/LR warmup 300; 25 warmup/50 measured.
Caching changed throughput <0.5%, below the conservative variation threshold.
Historical batch 32 gained ~0.70% while peak memory rose 27.88→50.96 GiB.
Profiler intervals are temporal associations, not additive/causal stage costs.
SHAs, original runs/traces, all variants and setup: [RESULTS.md](RESULTS.md).
Reproduction entry point: `python experiments/h100_input_pipeline/launch.py --help`.
Production source and input-pipeline defaults remain unchanged.
