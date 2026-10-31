# FFN architecture ablation

Question: isolate internal residual composition (A/B) and FFN structure (A/C).
A is SwiGLU + inner residual; B removes only that residual; C is squared ReLU
with residual preserved. One-A100 synthetic controls showed no useful B gain;
C was ~16% faster with fewer FLOPs, not established equal model quality.
Setup: depth 12, width 768, context 2048, BF16/FA2, batch 16, four microsteps;
real four-A100 C runs are recorded separately in [RESULTS.md](RESULTS.md).
Sources, original run IDs, counts and all six cases remain tracked here.
Native reproduction entry: `gpt-lab train auto --act-func swiglu` or
`--act-func relu_squared`; use the production launcher with matched settings.
Defaults remain squared ReLU; missing/null legacy activation loads SwiGLU.
Historical residual-only B and optional unrun narrow variants are documented in
[HISTORY.md](HISTORY.md); their original source hashes remain in the manifest.
