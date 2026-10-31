# Handoff to Agent A

Parent started at `2e76669989a4f23fc007e5e8f50bb5b965771c4c` on
`exp/anvil-normuon`. Original uncommitted ANVIL edits were preserved unchanged
in commit `90adc9f`; communication history merged in `3db7530`.
Both original communication branch tips remain preserved in RESULTS.md.
The final pushed branch SHA and final working-tree status are in the closing handoff.

Implemented ANVIL mechanism: spectral-only six ANVIL2 polynomial maps, BF16
spectral input guard `d=1.05*sqrt(trace(G_raw))+1e-6`, first Gram reuse and
original NorMuon row rescale/norm restoration. Opt-in `anvil_spectral: true`
in `configs/anvil/b_spectral.yaml`; `a_normuon.yaml` explicitly false. Existing
NorMuon momentum, beta2, LR/decay, parameter/optimizer-state storage and ownership
are retained. The baseline remains unchanged; six maps apply only when opted in.

Twin fast/slow momentum rails, engagement mapping and C/D combinations are
**not implemented**. The proposed mapping in the preserved `docs/anvil2.md` is
not an active schedule: `u=1194*t/H`, engagement u>=514, slow beta .98,
post-engagement fast beta .85 and fast weight .4385; proposed H=2317 boundaries
~466/998/2220. Do not report these as implemented flags or tested schedules.
The ordinary scheduler horizon stays 2317 with LR warmup 300 in prior diagnostics.

Baseline communication is selected unless `configure_communication` is called
explicitly; B/C/D modes stay opt-in. Backward communication requires no scaler
and disabled CUDA capture. CUDA replay remains opt-in; the merge retained both
paths without enabling their combination.

Remaining correctness: native compiled GPU/NCCL spectral option, same-variant
GPU resume and boundaries if twin rails are implemented later. CPU/eager/Gloo
and small compiled CPU results are documented in the original ANVIL note;
communication NCCL results predate ANVIL and do not validate spectral updates.
No new communication sweep, longer loss run or cross-agent review is authorized
by this consolidation. Agent A owns subsequent optimizer decisions.
