# Current integration and run selection

`exp/anvil-normuon` is the single active branch for today's training experiments
and Modal evidence archive. Agent A owns subsequent work on this branch and its
one consolidated PR. Do not propagate through `feat/comm-backward` or reopen
the already merged FFN PR. Master is not updated by this consolidation.

History is preserved from C's `e8ae43dbaecfe97cdc35e02f47abfcfb993aab51`,
B's published `fix/swiglu` at `8e0b0ea11f28d6896b9cb20cca9f9278b5f0beba`,
and the Modal archive at `8b476492a048dc223c01b9fedf7e24df90176a39`.
C's history already includes both communication branches and CUDA capture.
B's source already includes the distributed reporting and constant-momentum
fixes. The master merge tip `6763425` is not needed as a merge input; its tree
is identical to B's published source tree.

Integration makes the older A100 capture/communication runners select `fa2`
and `swiglu` explicitly because B's CLI rejects legacy `fused` and now defaults
to ReLU squared. Historical result files and pinned commits are unchanged.
These compatibility changes do not turn those diagnostics into the new H100
production recipe.

[selected-runs.json](selected-runs.json) records the user's current production
selection for Muon, NorMuon and NorMuon with ANVIL spectral maps. This replaces
the old depth-12/SwiGLU/fixed-2317-update plan in C's historical handoff. The
literal model-name prefix remains `ic1-d16`, while actual depth is 20; names
also include `d20` to expose that distinction. Derive the new horizon from
data:parameter ratio 10 and report the scaling parameter definition and rounding.

The JSON is a recipe and handoff, not an executable launcher or completed run.
The selected-run launcher, ReLU-squared conversion and post-update evaluation/
checkpoint prerequisites are now implemented locally. Native hardware validation
is still mandatory before long training. FP8 currently means FFN
forward only, with BF16 backward. Evaluation disables FP8 consistently. Keep
initialization, data and schedules matched. Run the three 8-H100 jobs sequentially
under the available eight-GPU concurrency limit. No paid job was launched here.

Archive negative results and existing opt-in execution paths unchanged. Capture
evidence is indexed in [../cuda_graphs/](../cuda_graphs/README.md); communication,
FFN, FP8 and input-pipeline results remain in their existing experiment folders.
The Modal collector does not collect model checkpoints or launch GPUs.

PRs #31, #39 and #43 are superseded only after their head commits are verified
as ancestors of the consolidated branch. Preserve their source branches. Older
unrelated PRs (value embeddings, HF compatibility and dataloader work) are outside
this consolidation; leave them open.

Local integration validation: 264 focused CPU checks passed, five GPU checks
skipped, and eight Node dashboard checks passed. Six distributed checks could
not run successfully because this host rejects Gloo socket operations; they
were excluded from the successful repeat, not counted as passing. No CUDA/NCCL,
native FP8 dispatch, training quality or authenticated archive publication was
validated by this consolidation.

Remaining work stays on `exp/anvil-normuon` and consolidated PR #44. The selected
recipe now explicitly retains the existing tying-aware BF16-embedding/FP32-other
storage policy; no all-FP32 option was added. The approved unspecified CLI defaults
include warmup 40. See README.md for the gated launcher and resolved 8,728-update
horizon. The earlier validation counts above describe consolidation, not proof
that the newly prepared H100 gate has run.

Selected-run prerequisite validation: 178 focused CPU checks passed (including
conversion wiring, trainer counters, checkpoint compatibility, CLI/workflows,
embedding tying, spectral/profile checks and offline launcher plans). The known
socket-dependent distributed-metrics test was deselected. Entrypoint compilation
and `git diff --check` passed. The eight-H100 gate is prepared but has not run;
CUDA/NCCL, native dispatch, memory fit and learning quality remain unverified.

Direct-launch correction: the former aggregate launcher depended on an ignored,
untracked personal `runs/modal_trainbase.py`, so the remote preflight could not
import it. The requested native entry point is now tracked. All three launches
use that file directly with explicit training flags; the aggregate executable
was removed. No `--full-run` or optimizer-selection flag was added. The implicit
15-update cap was removed in favor of the native unset default. Each selected
run performs the hardware gate inside its allocation before native CLI training.

Direct-launch validation: eight focused checks passed, including all three
literal README commands, cap handling, failed-gate ordering, existing diagnostic
plans, and isolated preflight from tracked files with no personal launcher files.
The native CLI parser was exercised through the project interpreter. Hardware
validation has not been rerun and no paid compute was launched.

CLI-authority correction: removed frozen-recipe comparisons, model/run-name
constraints and prior-run dependencies. The launcher accepts native CLI overrides
and supplies parsed settings to the hardware gate. Reporting no longer asserts a
hardcoded horizon, batch or checkpoint/evaluation policy. The revised 20261031
commands use window 258, softcap 12, automatic batch/accumulation, BPB budget
4,194,300 and save-every -1; native scaling resolves 1,048,576 tokens/update,
two microsteps and 4,364 updates. Initial fingerprints are recorded for later
comparison rather than vetoing runs based on labels or stale settings.

Revised-parameter validation: eight focused checks passed, including arbitrary
labels and valid overrides, native parser/preflight in an isolated checkout,
automatic batch/horizon resolution and failed hardware-gate ordering. No paid
compute was launched; native GPU dispatch and memory fit remain unverified.
