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
Agent A must implement its listed pending prerequisites, generate pinned launch
commands and report validation before long training. FP8 currently means FFN
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
