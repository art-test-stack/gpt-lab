# Agent C consolidation handoff to Agent A

## Preserved history

- Starting NorMuon/development source: `192bbe2921e4774d24324b2d07852311b7bc4dcf`.
- Existing spectral-only ANVIL work was preserved in
  `90adc9f7d16424ffca86854792e02fe101f8e1c8` before consolidation.
- Existing capture history through `120b0dd91f0ea08d041f686502862b4584c991e3`
  was already included by merge `2e76669989a4f23fc007e5e8f50bb5b965771c4c`.
  Agent A owns capture-folder preservation and any further capture work.
- Communication histories (`exp/comm-buckets` at `8956f6e2cb8aae856c467c40ff2e3f37824dff75`
  and `exp/comm-optimizer-sweeps` at `2795b42e96d444b285ffff5dedd3955e68550804`)
  were included by merge `3db7530892214264592fc871c8652f3b01f941fb`.
- The capture/communication conflict retained graph replay as an opt-in branch,
  with readiness notifications only in ordinary backward. Backward-hook
  communication rejects capture/scaling as before. Compiled execution, original
  communication and existing Muon remain the defaults.
- Source branches and original result-pinned checkouts are not rewritten.
  No unsuccessful experiment was squashed or discarded. No GPU job was rerun.

## Exact optimizer implementation

| Choice | Profile | Transformer base settings |
|---|---|---|
| Existing Muon | `configs/muon.yaml` in this folder | LR .02, momentum .95, beta2 .9, decay .28, five Polar Express maps, existing adaptive/cautious behavior |
| NorMuon baseline A | `configs/a_normuon.yaml` | LR .02, momentum .95, beta2 .95, decay .1, five repeated baseline maps, ordinary decoupled decay |
| NorMuon with ANVIL spectral maps B | `configs/b_spectral.yaml` | Exactly A plus `anvil_spectral: true`: six historical maps and the required 1.05 input guard |

Select a profile explicitly with
`--optim-config-path experiments/optimizer_comparison/configs/<profile>.yaml`.
`anvil_spectral` is an optional Boolean on `opt: normuon`; missing/false retains
the existing NorMuon path. B keeps NorMuon row-wise normalization, norm restoration,
one EMA momentum rail, parameter ownership, storage precision, auxiliary AdamW
groups, and state layout. There is no optimizer registered as `anvil`.

All six maps execute on every B update, including update zero, even though the
matched `ns_steps: 5` field remains for A. There is no delayed activation or
implemented time-varying ANVIL schedule. A 150-update cap exercises B's full
implemented mechanism; the proposed 998-update twin-rail engagement is **not
implemented** and must not be described as active in this handoff. Twin rails,
their historical Nesterov schedule, terminal averaging and storage changes have
not been added. Historical coefficients/provenance and proposed C/D equations
remain in `REPORT.md` explicitly labeled as unimplemented.

## Checks and reproduction

```bash
uv run --locked --extra cpu --group dev pytest experiments/optimizer_comparison
uv run --locked --extra cpu --group dev pytest experiments/communication
python experiments/communication/modal_communication.py --help
```

The relocated checks preserve scalar spectral behavior, tall/wide BF16 matrix
examples, update norms, owner-local state memory and same-variant disk resume.
The real two-process Gloo checks exercise averaged owner updates for Muon,
NorMuon and the spectral option, plus bucket layout/accumulation/resume for the
communication experiment. Defaults and trainer accumulation/capture rejection
are checked locally. Actual validation results are recorded in the final
handoff response; large local test outputs are not committed.

Tracked compact evidence: this folder's `results.json`, `REPORT.md`, and
`../communication/results.json`. The communication JSON includes source SHAs,
run IDs and Modal volume locations for all three completed sweeps, including
negative results. Logs/traces/checkpoints remain outside Git.

## Remaining ownership for Agent A

1. Preserve capture-specific launchers/checks/evidence under `experiments/cuda_graphs/`.
2. Incorporate master fixes consistently before new learning runs: `4fe8aea`,
   merged by `83fc9b1`, averages detached reported loss across ranks and makes
   the default momentum multiplier 1.0 independently of LR warmup/warmdown.
   These fixes are not part of this preservation-only source; historical
   communication/capture results retain their pinned pre-fix schedules.
3. Validate the actual compiled spectral kernel on CUDA: guarded near-rank-one,
   zero/tall/wide matrices, finite updates/norm behavior, sharded NCCL owner-state
   agreement, and same-variant checkpoint/resume. CPU fusion is not bitwise eager
   BF16 equivalence; no GPU spectral correctness claim is made.
4. Prepare the agreed matched 2317-update learning runs, compiled execution,
   capture off, BF16, SwiGLU, baseline communication, held-out full-vocabulary
   CE/BPB after completed updates, and final checkpoints. Keep optimizer-specific
   hyperparameters explicit. No such run or new optimizer mechanism was added by C.
5. After validation, propagate via `feat/comm-backward` and its master PR with
   merge history preserved. C hands ownership to A upon publication and stops
   editing the shared upstream.
