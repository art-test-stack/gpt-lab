# Optimizer comparison: existing implementations
Preserves explicit Muon, NorMuon (A), and NorMuon with ANVIL spectral maps (B); this is not full ANVIL2.
B adds all six guarded BF16 spectral maps from update zero, retaining NorMuon row normalization and one momentum rail.
Twin-rail momentum, delayed blending, terminal averaging and parameter-storage changes are not implemented.
Profiles: `configs/muon.yaml`, `configs/a_normuon.yaml`, `configs/b_spectral.yaml`; select with `--optim-config-path`.
Muon beta2/decay are 0.9/0.28; NorMuon A/B are 0.95/0.1; base matrix LR is 0.02 for all three.
CPU spectral/norm/state/resume and owner-update checks passed; compiled spectral GPU/NCCL checks and learning quality remain unverified.
Recommendation: capture off, baseline communication, explicit profile; GPU validation before matched learning runs is Agent A's follow-up.
Entry: `uv run --locked --extra cpu --group dev pytest experiments/optimizer_comparison`; evidence and coefficients: `results.json`, `REPORT.md`.
