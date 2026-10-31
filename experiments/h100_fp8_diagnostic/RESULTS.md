# Diagnostic status and provenance

The complete batch-16 harness is tracked from
`cd7ca922d5d062a4b7f537c1d2e6e672b8a91514`; production remains pinned to
`3fcd250669bb2f0bb5c25397c058f8cf7bd7d429` rather than following branch HEAD.
No accepted native compiled FP8 dispatch or reliable speedup is established.
Those gates remain pending until a complete user-launched diagnostic passes.
Existing artifacts at Modal `gpt-lab-runs/h100-fp8-diag-01` must not be treated
as proof of completion; the historical short pair is reported separately in
`../fp8_training/RESULTS.md` and does not replace these gates.

Controls: same SwiGLU, FA3, full-vocabulary objective, optimizer and storage
policy; 16×2048 microbatch, 16 microsteps, 524288 tokens/update, horizon 2317,
LR warmup 300. Profiles cover two warmed complete updates. Independent
BF16→FP8→BF16 legs each warm 20 and measure 50 updates with initial state,
RNG, data position and schedule restored. No batch-32 recipe was added.

No paid compute was launched during consolidation. Original harness scripts
and their fixed recipe were retained unchanged; only the explanatory files were
organized. A fresh run should save its traces/reports under Modal
`gpt-lab-runs/h100-fp8-diag-consolidated-01`, not overwrite the old artifact.
