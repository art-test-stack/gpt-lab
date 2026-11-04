# Historical FLOPs reporting repair

`scripts/repair_flops_reporting.py` repairs reporting copies of recorded results.
It does not import the model, train, change checkpoints, or change training budgets.
The formula is the implementation in [PR #45](https://github.com/art-test-stack/gpt-lab/pull/45),
reviewed at `9e2f0783111e104c98b27ecce06407689414dd15`:

```text
F = 6 * (block_parameters + head_parameters)
    + sum(12 * d_model * effective_seq for each layer)
effective_seq = context_length if left_window < 0 else min(left_window, context_length)
```

The label is **nanochat-compatible rectangular attention**. Count the padded
output head even when tied. Input/value lookup tables are excluded. The original
per-GPU peak FLOPs and GPU count remain unchanged. This estimates model training
work under an accounting convention; it is not executed kernel FLOPs.

## Audit before planning

Discover projects through the authenticated W&B Public API. Export every run
before selecting affected runs. `export` calls `scan_history()` without a keys
filter, verifies the history line count and final step, and saves the complete
history, configuration, summary, raw metadata, and a content digest:

```bash
python scripts/repair_flops_reporting.py export \
  --run ENTITY/PROJECT/SOURCE_ID --output .tmp/codex/flops-repair/original/SOURCE_ID
```

Keep these private backups. They can contain machine identity and local paths.
Do not commit them or upload them to the public Space.

Recover architecture and parameter counts from each run's resolved configuration,
source revision, and archived evidence. Record the padded vocabulary, actual FFN
activation/dimensions, actual embedding sharing, all layer windows, context,
world size, and original peak denominator. A legacy tying flag may have been
unused by its source code. Do not interpret it using current defaults.

Each accounting record supplies `source_run_id`, `source_sha`, `evidence`,
`classification`, `original_flops_per_token`, `block_parameters`, `head_parameters`,
`padded_vocab_size`, `d_model`, `context_length`, `n_layers`, and `left_windows`.
Classifications are `nanochat-compatible`, `causal-pair`, or `unresolved`.
The planner must reproduce the original estimator before changing any value.
Unresolved records are rejected; compatible documents are unchanged.

An audit JSON contains `targets`, each with a relative `file`, a JSON `path`
array selecting a run/object, its `accounting` record, and explicit `metric_keys`.
Select only derived fields. A report containing both configured rectangular MFU
and causal-pair MFU needs a target selecting only the causal-pair series.
The published audit linked below contains the actual records and targets.

## Stage and reverse

```bash
python scripts/repair_flops_reporting.py plan \
  --audit .tmp/codex/flops-repair/audit.json \
  --root .tmp/codex/flops-repair/original \
  --output .tmp/codex/flops-repair/staged \
  --manifest .tmp/codex/flops-repair/manifest.json
```

The manifest retains every original/corrected value, source run ID, formula
version, and before/after document digest. Recorded cumulative FLOPs are scaled
directly: their original zero, offsets, and alignment are preserved. Do not
recompute them from the latest token counter. Targets, schedules, times, tokens,
throughput, costs, losses, BPB, CORE, source SHAs, and checkpoints are unchanged.
Sparse nulls stay null. Original metadata stays in `original_attrs` and
`original_metadata`. A second plan against staged documents has no edits.

```bash
python scripts/repair_flops_reporting.py restore \
  --manifest .tmp/codex/flops-repair/manifest.json \
  --root .tmp/codex/flops-repair/staged \
  --output .tmp/codex/flops-repair/restored
```

Restoration rejects modified staged files and verifies the original digest.

## W&B reporting copies

W&B's [supported logging API](https://docs.coreweave.com/products/wandb/track/log)
cannot overwrite arbitrary past steps. This tool never rewinds or deletes an
original. `replay --input STAGED_RUN_JSON --output LOCAL_LOG_DIRECTORY` creates a
deterministic `nc-*` run in the source project, marked `reporting-only` and linked
to its original. Existing matching copies are verified without replaying them;
conflicting or incomplete copies fail instead of creating duplicates.

Every original `_step`, `_timestamp`, and `_runtime` is explicit in
`source/step`, `source/timestamp`, and `source/runtime`. Reserved source summary
fields are explicit `source_summary/*` scalars. W&B generates upload runtime and
creation metadata; these are reporting bookkeeping, never training time.
The original source SHA remains in configuration; Git/code/stat collection is
disabled for the upload. Readback checks every history value and source summary.

The optional official Workspaces client saves a reporting view excluding replaced
source run IDs. Install it only in a temporary environment, not this project's
dependencies. `wandb-workspaces==0.4.5` works with the existing `wandb==0.25.1`:

```bash
uv run --no-project --python 3.12 --with wandb-workspaces==0.4.5 \
  --with wandb==0.25.1 --cache-dir .tmp/codex/flops-repair/uv-cache \
  python scripts/repair_flops_reporting.py dashboard \
  --audit .tmp/codex/flops-repair/audit.json --project ENTITY/PROJECT \
  --receipt .tmp/codex/flops-repair/dashboard.json
```

Retain the receipt: subsequent invocations load and verify that view without
creating another one. Other saved/user views are not overwritten.

## Actual Space publication

The deployed [Space](https://huggingface.co/spaces/ai-testack/gpt-lab) reads
`results.json` and `experiments.json`, including lazy report artifacts referenced
by the latter. The tracked `spaces/results/results.json` is an empty template.
Audit the deployed revision, not that template. Previous deployments used the
authenticated Hugging Face SDK's `upload_folder`; use that same workflow.

Prepare a publication folder from the audited live files. Retain original raw
evidence at its existing paths. Put corrected JSON reporting companions under
`archive/accounting-v1/reporting/`, update archive artifact links/checksums, and
keep clearly named links to their originals. YAML configs can have JSON reporting
companions. Preserve original logs and traces rather than editing their text.
Keep the existing snapshot run IDs and row alignment; point corrected source
links to their reporting-only W&B runs. Publish the reversible manifests and
per-run table alongside the data. Review the publication folder for credentials.

```bash
python scripts/repair_flops_reporting.py publish-space \
  --folder .tmp/codex/flops-repair/publication \
  --expected-revision AUDITED_SPACE_SHA \
  --receipt .tmp/codex/flops-repair/publication-receipt.json
```

Publication uses a parent-commit guard, never deletes unlisted files, and verifies
every uploaded file by SHA-256. With the same receipt/files/revision, repeating
the command only verifies. Roll back by uploading the retained original snapshots
and interface file; original evidence and original W&B runs still exist.

## 2026-11-03 audit

The seven discovered W&B projects contained 85 source runs: 76 compatible, five
causal-pair, and four unresolved early configurations. The Space contained three
training snapshots and 48 evidence entries. Corrections affect 11 evidence
entries; 18 are compatible and 19 have no recoverable full-model accounting
series (including partial-block diagnostics), so those entries are retained.
Two approximately 52% A100 CUDA-graph histories were added unchanged to the
training snapshot after verifying their original rectangular convention.

[Full per-run before/after table](https://huggingface.co/spaces/ai-testack/gpt-lab/blob/main/archive/accounting-v1/per-run.md)
and its linked JSON audits document every run, recovered formula inputs,
per-series factors, unresolved evidence, and reversible changes.

Validation uses six focused offline tests, comparison with the exact PR #45
production estimator for all 81 resolved W&B records, original-measurement
invariants, exact restoration, repeated-application checks, remote history
readback, Space file checksums, and live chart/table inspection. No training or
paid GPU work is required.
