# CUDA graph capture
Tests explicit compiled forward/backward capture against compiled execution; optimizer and communication stay outside capture.
Pinned measured implementation: `a5d46414cca0351276b65f30f873d40034f9a7f0`; this archive adds no new GPU result.
On 8 A100-SXM4 GPUs, BF16, sequence 2048 and batch 32/GPU, capture was 0.5% slower in one matched trial.
Peak reserved memory rose from 59.05 to 68.90 GiB/GPU; keep capture disabled by default.
Small distributed loss/gradient/parameter/state equivalence passed; this does not validate every configuration.
Detailed setup, limits and reproduction: [preserved report](../../docs/cuda-graphs/README.md); compact evidence: [results.json](results.json).
Launchers/checks retain their paths: `scripts/modal_cuda_graphs.py`, `scripts/benchmark/cuda_graph_equivalence.py`, `tests/test_cuda_graph.py`.
No repeated compute; optimizer capture, FP8 capture and combined communication capture remain unsupported or deferred.
