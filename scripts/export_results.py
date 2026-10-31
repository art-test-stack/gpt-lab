"""Export read-only run snapshots for spaces/results without importing PyTorch.

Use --wandb ENTITY/PROJECT/RUN (repeatable), or --history JSON/JSONL/CSV with
--config JSON for a local run. W&B access happens only in this local exporter.
The static dashboard itself never receives API credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "spaces" / "results"
CONFIG_FIELDS = {
    "model_name", "run_name", "random_seed", "device", "ds_name", "depth",
    "aspect_ratio", "d_head", "n_kv_heads", "max_seq_len", "vocab_size",
    "tokenizer_model", "pat_str", "tie_embeddings", "window_pattern", "window_size",
    "softcap", "attn_softcap", "attn_impl", "device_batch_size", "total_batch_size",
    "n_acc_steps", "num_steps", "target_flops", "target_param_data_ratio",
    "warmup_steps", "warmdown_ratio", "final_lr_frac", "optim_config",
    "meta_config", "training_config", "model_card", "gpu_info", "git_info",
    "eval_bpb_every", "n_bpb_tokens", "eval_core_every", "n_core_tokens",
    # Also accept a simple configuration structure for local JSON histories.
    "model", "optimizer", "training", "evaluation", "data", "hardware", "precision",
}
PRIVATE_FIELDS = re.compile(r"(^|_)(password|secret|token|api_key|credential|authorization)$", re.I)
PATH_FIELDS = {"dirname", "model_dir", "board_dir", "hostname", "cache_dir", "source_dir"}
SCALARS = {"step", "_runtime", "lrm", "muon_momentum", "weight_decay"}


def public_config(value, key=""):
    """Export experiment settings, omitting known credential and local-path fields."""
    if isinstance(value, dict):
        return {
            str(k): public_config(v, str(k))
            for k, v in value.items()
            if not PRIVATE_FIELDS.search(str(k)) and str(k) not in PATH_FIELDS
        }
    if isinstance(value, (tuple, list)):
        return [public_config(item, key) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def export_config(config):
    selected = {key: value for key, value in config.items() if key in CONFIG_FIELDS}
    if isinstance(selected.get("git_info"), dict):
        selected["git_info"] = {k: v for k, v in selected["git_info"].items() if k in {"commit", "branch", "dirty"}}
    return public_config(selected)


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def normalize_history(rows):
    """Keep sparse metric rows, zero values, and the original recorded step.

    No forward filling, interpolation, or requirement that every metric is present
    in the same row. This is important for separately logged evaluation records.
    """
    history = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"History row {index} is not an object")
        step = numeric(row.get("step", row.get("_step", row.get("Step"))))
        if step is None:
            raise ValueError(f"History row {index} needs a finite step, _step, or Step")
        record = {"step": int(step) if step.is_integer() else step}
        for key, value in row.items():
            if key in SCALARS or key.startswith(("train/", "eval/", "core/", "optim/")):
                number = numeric(value)
                if number is not None:
                    record[key] = number
        history.append(record)
    if len(history) > 100_000:
        raise ValueError("A snapshot supports up to 100,000 history rows per run; export a shorter interval explicitly")
    return sorted(history, key=lambda row: row["step"])


def read_history(path):
    with path.open(encoding="utf-8-sig") as handle:
        if path.suffix.lower() == ".csv":
            return list(csv.DictReader(handle))
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("history")
    if not isinstance(data, list):
        raise ValueError("History JSON must be a list of metric rows or an object with a history list")
    return data


def export_wandb(path, api):
    if len(path.strip("/").split("/")) != 3:
        raise ValueError("Use a W&B run path in ENTITY/PROJECT/RUN_ID format")
    run = api.run(path)
    config = dict(run.config)
    return {
        "id": path, "label": run.name or run.id, "group": run.group or "",
        "status": run.state, "source_url": run.url,
        "config": export_config(config),
        # scan_history without keys preserves asynchronous, sparse metric rows.
        # A keys=[...] filter would discard rows lacking any requested key.
        "history": normalize_history(run.scan_history()),
    }


def write_snapshot(path, runs, append=False):
    if append and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("schema_version") != 1 or not isinstance(existing.get("runs"), list):
            raise ValueError("Cannot append to an incompatible results file")
        merged = {run["id"]: run for run in existing["runs"]}
        merged.update({run["id"]: run for run in runs})
        runs = list(merged.values())
    if len(runs) > 100:
        raise ValueError("A snapshot supports at most 100 runs")
    data = {"schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(), "runs": runs}
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > 25 * 1024 * 1024:
        raise ValueError("Snapshot exceeds 25 MiB; export fewer runs or a shorter interval")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wandb", action="append", metavar="ENTITY/PROJECT/RUN")
    source.add_argument("--history", type=Path, help="Local scalar history JSON, JSONL, or CSV")
    parser.add_argument("--config", type=Path, help="Local run configuration JSON")
    parser.add_argument("--id", default="local-run", help="Stable id for a local run")
    parser.add_argument("--label", help="Display name; only supported for a single run")
    parser.add_argument("--group", default="", help="Group for a local run")
    parser.add_argument("--status", default="unknown", help="Recorded local run status")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="Merge by run id, replacing matching snapshots")
    parser.add_argument("--space-dir", type=Path, help="Also assemble a ready-to-upload static Space in this directory")
    args = parser.parse_args(argv)
    try:
        if args.space_dir and (args.space_dir.resolve() == ASSETS.resolve() or args.space_dir.resolve() in ASSETS.resolve().parents):
            raise ValueError("Choose a build directory separate from the dashboard source, such as dist/results-space")
        if args.wandb:
            if args.config:
                raise ValueError("--config applies only to --history")
            if args.label and len(args.wandb) != 1:
                raise ValueError("--label requires a single run")
            try:
                import wandb
            except ImportError as exc:
                raise ValueError("W&B export requires wandb; use the gpt-lab environment or install wandb") from exc
            api = wandb.Api()
            runs = [export_wandb(path, api) for path in args.wandb]
            if args.label:
                runs[0]["label"] = args.label
        else:
            config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            if not isinstance(config, dict):
                raise ValueError("Config JSON must be an object")
            runs = [{"id": args.id, "label": args.label or args.id, "group": args.group,
                     "status": args.status, "config": export_config(config),
                     "history": normalize_history(read_history(args.history))}]
        write_snapshot(args.output, runs, append=args.append)
        if args.space_dir:
            args.space_dir.mkdir(parents=True, exist_ok=True)
            for name in ("README.md", "index.html", "style.css", "app.js", "data.mjs", "evidence.js", "evidence_data.mjs", "logo.png"):
                shutil.copy2(ASSETS / name, args.space_dir / name)
            shutil.copytree(ASSETS / "vendor", args.space_dir / "vendor", dirs_exist_ok=True)
            destination = args.space_dir / "results.json"
            if destination.resolve() != args.output.resolve():
                shutil.copy2(args.output, destination)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(runs)} run(s) to {args.output}")
    if args.space_dir:
        print(f"Static Space prepared at {args.space_dir}")


if __name__ == "__main__":
    main()
