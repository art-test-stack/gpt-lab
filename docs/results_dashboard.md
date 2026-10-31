# Results dashboard

`spaces/results/` is a static Hugging Face Space for reviewing gpt-lab experiments.
It has no backend, training dependencies, or account connection. The viewer reads
one `results.json` training snapshot and an optional `experiments.json` evidence archive,
and can also open a local training snapshot in the browser.
Import replaces the displayed snapshot; it does not upload or persist the file.
Reloading returns to the hosted snapshot.

## What it shows

- Overlay training loss, validation BPB, throughput, and learning-rate multiplier.
- Plot against recorded step, tokens, training seconds, or W&B elapsed seconds.
- Apply optional EMA smoothing to training loss only; evaluation values stay raw.
- Compare CORE checkpoints, per-task accuracy, and recorded evaluation duration.
- Inspect configurations side by side; highlight differences or show only differences.
- Inspect run status and source links; download the selected runs as JSON.

The default snapshot contains no runs. An explicit synthetic preview demonstrates
the interface with generic example runs. Preview data has no relationship to Muon,
gpt-lab performance, or any measured training run. Synthetic exports retain their
label when imported again.

## Archive Modal evidence

The **Experiment archive** view groups Modal reports, logs, configurations,
figures and downloadable profiler traces by experiment, retaining failed and
inconclusive runs. The collector excludes models/checkpoints before downloading,
records checksums and source commits, and uses no Modal compute. See
[`experiments/modal_results/README.md`](../experiments/modal_results/README.md)
for collection and publication commands. Publication preserves existing training
snapshots, Space metadata and earlier archive runs. Unclassified runs remain
visible and can be assigned with an explicit mapping.

## Export existing W&B runs

Run the exporter locally from the repository root using the existing gpt-lab
environment. Authenticate W&B locally through its normal login or `WANDB_API_KEY`
environment variable. The exporter uses the read-only Public API; it does not
create or modify W&B runs. Never put the API key in Space files.

```bash
uv run --locked --extra cpu python scripts/export_results.py \
  --wandb ENTITY/PROJECT/RUN_A \
  --wandb ENTITY/PROJECT/RUN_B \
  --output .results/results.json \
  --space-dir dist/results-space
```

`scan_history()` exports the recorded scalar history without the sampling used by
`run.history()`. No multi-key filter is supplied: that would omit sparse rows
lacking any requested metric. No new metrics are inferred from model weights.
To refresh runs, repeat the same command. To add runs incrementally, use `--append`;
matching run ids replace their previous snapshot rather than duplicating it.

The exporter includes a bounded set of experiment configuration fields, excludes
machine identity, and removes known credential and local-directory keys
recursively. Review the resulting snapshot before publishing; free-form string
values are not a general secret scanner.

## Import local histories

The exporter itself needs only Python 3.12's standard library for local files:

```bash
python scripts/export_results.py \
  --history /path/to/history.jsonl \
  --config /path/to/config.json \
  --id vanilla-muon-seed42 \
  --label "Vanilla Muon · seed 42" \
  --group optimizer-comparison \
  --status finished \
  --append \
  --output .results/results.json \
  --space-dir dist/results-space
```

Accepted histories: JSON list, JSON object with a `history` list, JSONL, or a CSV
with **one row per logged step** and original metric names as columns. Each row
needs `step`, `_step`, or `Step`. Chart exports using namespaced columns such as
`RUN_A - train/loss` need explicit conversion first; they are not silently guessed.
The local config is a JSON object; common groups are `model`, `optimizer`,
`training`, `evaluation`, `data`, `hardware`, and `precision`. Existing gpt-lab
W&B configuration keys are also supported. This change does not add local scalar
logging to the trainer or try to load pickle/checkpoint files in the viewer.

## Snapshot contract for the upcoming optimizer runs

```json
{
  "schema_version": 1,
  "runs": [{
    "id": "vanilla-muon-seed42",
    "label": "Vanilla Muon · seed 42",
    "group": "optimizer-comparison",
    "status": "running",
    "config": {
      "optimizer": {"name": "vanilla-muon", "learning_rate": 0.001},
      "training": {"seed": 42},
      "evaluation": {"protocol_id": "fixed-core-v1", "examples_per_task": 500}
    },
    "history": []
  }]
}
```

The numbers above illustrate configuration syntax, not a proposed optimizer recipe.
For later training scripts, write scalar rows with the current trainer's keys:

| Key | Meaning |
| --- | --- |
| `step` | Actual logged training step; required |
| `train/raw_loss` | Training loss before display smoothing |
| `train/total_tokens` | Cumulative training tokens |
| `train/total_training_time` | Cumulative training seconds, excluding evaluation |
| `_runtime` | Logged elapsed seconds, optional |
| `train/tokens_per_sec` | Recorded global training throughput |
| `lrm` | Learning-rate multiplier |
| `eval/bpb`, `eval/loss` | Recorded validation results |
| `eval/step_time_ms` | Duration of that validation pass |
| `core/core` | Centered CORE score, displayed as a fraction |
| `core/max_per_task` | Configured sample cap per task; -1 means all examples |
| `core/step_time_ms` | Duration of that CORE pass |
| `core/TASK/accuracy` | Task accuracy as a fraction |

Missing values stay missing, including on the time/token axes. The viewer never
interpolates evaluation counters or fills absent scores with zero. Supply counters
alongside evaluation rows when exact token/time alignment is required. W&B can
combine logs sharing a step; the viewer reflects that recorded history rather
than reconstructing the trainer's before/after-update timing.

Summary cards show extrema across recorded checkpoints, while the run recap shows
latest values. They do not claim statistical significance or equal evaluation
protocols. Per-task values are the last recorded value for each task and can come
from different checkpoints; inspect the full history before interpreting them as
one complete evaluation. Check token budgets, seeds, sample caps, demonstration
selection, and protocol ids before comparing experiments. The first version does
not aggregate seeds or compute confidence intervals.

Recorded evaluation time sums distinct `(step, metric)` durations for BPB and CORE.
Repeated logs of the same pass are counted once. Missing timing logs cannot be
recovered. W&B elapsed time is not GPU allocation time or a measured cloud bill;
the dashboard makes no automatic cost estimates.

## Deploy on Hugging Face

Create a **static** Space, then upload the contents of `dist/results-space/` as its
repository root. The folder includes its own Space README metadata and every
runtime asset, plus the exported snapshot. Keep the source repository README
unchanged. No GPU, server, secrets, or W&B access are required in the Space.

Using the Hugging Face Python SDK in a locally authenticated environment:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("YOUR_USERNAME/gpt-lab-results", repo_type="space", space_sdk="static", private=True, exist_ok=True)
api.upload_folder(repo_id="YOUR_USERNAME/gpt-lab-results", repo_type="space", folder_path="dist/results-space")
```

The `private=True` example creates a private Space; it does not change an existing
Space's visibility. Choose the desired visibility in Hugging Face. Re-upload a
refreshed build to update results. Future automation can publish the same snapshot
contract after the three optimizer runs finish.

## Checks

No PyTorch, model downloads, or GPU are needed for the export/viewer checks:

```bash
python -m unittest discover -s tests/dashboard
node --test tests/dashboard/data.test.mjs
node --check spaces/results/app.js
```

The public viewer reads at most 100 runs, 100,000 history rows per run, and a 25 MiB
imported snapshot. Line paths retain all available points. Hover targets are capped
at approximately 300 per series for responsiveness. Fonts have system fallbacks.
