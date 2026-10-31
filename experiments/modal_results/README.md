# Modal results archive

- Collects reports, configurations, logs, figures and profiler traces from `gpt-lab-runs`, grouped by experiment; failed and inconclusive runs stay visible.
- Model weights, optimizer checkpoints, caches, opaque binaries and symlinks are excluded before reading. Known credential fields/patterns are redacted; original/archive hashes record transformations.
- Reads the volume through the client API: no Modal App, GPU allocation or training. Existing hosted `results.json` and Space metadata are preserved.
- Local checks cover filtering, provenance, malformed/failed runs, redaction, repeat imports and publication boundaries. Actual volume access/publication requires the user's authentication; no GPU performance is claimed.
- Collect/preview: `uv run --no-project --with modal==1.6.2 python experiments/modal_results/collect.py --volume gpt-lab-runs --environment main`; serve `dist/modal-results-space` locally with `python -m http.server --directory dist/modal-results-space`.
- Add `--publish --space ai-testack/gpt-lab` with `--with huggingface_hub==1.12.0` to update the existing static Space. Use `--dry-run` to list eligible paths without downloading, or `--local-root PATH` for already downloaded evidence.
- Unknown runs stay under `other`; `--mapping mapping.json` accepts exact run-directory → experiment-slug overrides. `--run-prefix NAME` restricts collection. Raw evidence is built under `dist/`, never committed to the code repository.

Interface-only update (no Modal access or evidence transfer): `uv run --no-project --with huggingface_hub==1.12.0 python experiments/modal_results/publish_viewer.py --space ai-testack/gpt-lab`. The landing page groups experiments, selects runs, and reads small summary reports on demand; training charts remain under Overview.
