# Communication buckets and overlap
A: original transport; B: immediate-wait buckets; C: post-backward pipeline; D: backward-ready hooks.
Tested 4 x A100-SXM4-80GB, BF16, SwiGLU, seq 2048, microbatch 16/GPU, global batch 524288; horizon 2317.
Muon 64 MiB B: -0.582%; Muon 128 MiB C: -0.303%; NorMuon 128 MiB C: -0.329% versus each matched baseline.
128 MiB removes excess padding (576 bytes remain), but adds ~2.13 GiB/GPU and one control collective; no demonstrated speedup.
Tiny four-rank NCCL gradients/updates/state matched exactly; eager fixtures do not establish full-model compiled equivalence or useful overlap.
Recommendation: baseline communication and capture off; preserve all modes as explicit options; no learning-quality conclusion.
Entry: `python experiments/communication_buckets/modal_communication.py --help`; checks: `pytest experiments/communication`.
Tracked evidence: `results.json`, `RESULTS.md`, `history-buckets.md`; large evidence: Modal `gpt-lab-runs/<run_id>` recorded in the summary.
