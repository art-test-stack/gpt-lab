"""Repair derived reporting from explicit, independently audited accounting records.

No model defaults, checkpoint loading, training, or remote mutation during planning.
See docs/flops_reporting_repair.md for the audit contract and replay procedure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


VERSION = "nanochat-rectangular-v1"
CONVENTION = "nanochat-compatible rectangular attention"
FORMULA_SOURCE = "https://github.com/art-test-stack/gpt-lab/pull/45"


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def estimate(record, *, causal=False):
    """PR #45's implementation, using recovered counts and per-layer windows."""
    d, t = record["d_model"], record["context_length"]
    windows = record["left_windows"]
    if not all(isinstance(x, int) and x > 0 for x in [d, t, record["block_parameters"], record["head_parameters"]]):
        raise ValueError("Accounting requires positive, recorded dimensions and parameter counts")
    if len(windows) != record["n_layers"] or not windows or any(not isinstance(w, int) or w < -1 for w in windows):
        raise ValueError("Supply every layer's recovered left window (-1 means full attention)")
    if record["head_parameters"] != record["padded_vocab_size"] * d:
        raise ValueError("Count the padded output head even when tied")
    projection = 6 * (record["block_parameters"] + record["head_parameters"])
    attention = 0
    for window in windows:
        if causal:
            span = t if window < 0 else min(window + 1, t)
            pairs = span * (span + 1) // 2 + (t - span) * span
            attention += 12 * d * pairs / t
        else:
            effective_seq = t if window < 0 else min(window, t)
            attention += 12 * d * effective_seq
    return projection + attention


def validate(record):
    if record["classification"] == "unresolved":
        raise ValueError(f"Unresolved run: {record['source_run_id']}")
    if record["classification"] not in {"nanochat-compatible", "causal-pair"}:
        raise ValueError("Unknown estimator classification")
    if not record.get("source_sha") or not record.get("evidence"):
        raise ValueError("Source SHA and recovery evidence are required")
    expected = estimate(record, causal=record["classification"] == "causal-pair")
    if not math.isclose(expected, record["original_flops_per_token"], rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("Recovered estimator does not reproduce original FLOPs/token")
    return estimate(record)


def repair(document, record, metric_keys):
    """Scale recorded counters, preserving their original offset and sparse rows."""
    corrected = validate(record)
    original = record["original_flops_per_token"]
    identity = digest(record)
    marker = document.get("accounting_correction") if isinstance(document, dict) else None
    if marker:
        if marker.get("formula_version") != VERSION or marker.get("audit_record_sha256") != identity:
            raise ValueError("Document already carries a different accounting repair")
        return copy.deepcopy(document), []
    if corrected == original:
        return copy.deepcopy(document), []
    keys = set(metric_keys)
    forbidden = {"target_flops", "gpu_peak_flops", "peak_flops_per_gpu", "n_steps", "n_total_tokens"}
    if keys & forbidden or any(not ("flop" in k.lower() or "mfu" in k.lower() or k == "bf16_peak_normalized_utilization_percent") for k in keys):
        raise ValueError("Only explicitly audited, derived accounting fields may change")
    result = copy.deepcopy(document)
    changes = []

    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"accounting_correction", "original_attrs", "original_metadata"}:
                    continue
                here = path + [key]
                if key in keys:
                    if child is None:
                        continue
                    if key == "flops_convention" and isinstance(child, str):
                        new = CONVENTION
                    elif isinstance(child, bool) or not isinstance(child, (int, float)) or not math.isfinite(child):
                        raise ValueError(f"Accounting field must be finite: {here}")
                    elif key in {"n_flops_per_token", "flops_per_token"}:
                        if not math.isclose(child, original, rel_tol=1e-12):
                            raise ValueError(f"Conflicting FLOPs/token at {here}")
                        new = corrected
                    else:
                        new = child * (corrected / original)
                    if new != child:
                        changes.append({"path": here, "original": child, "corrected": new})
                        value[key] = new
                elif isinstance(child, (dict, list)):
                    walk(child, here)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, path + [i])

    walk(result, [])
    if changes:
        result["accounting_correction"] = {
            "formula_version": VERSION, "convention": CONVENTION,
            "formula_source": FORMULA_SOURCE, "source_run_id": record["source_run_id"],
            "source_sha": record["source_sha"], "audit_record_sha256": identity,
            "original_flops_per_token": original, "corrected_flops_per_token": corrected,
            "factor": corrected / original,
        }
        changes.append({"path": ["accounting_correction"], "absent_original": True,
                        "original": None, "corrected": result["accounting_correction"]})
    return result, changes


def at(document, path):
    for key in path:
        document = document[key]
    return document


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("File escapes the supplied root")
    return path


def plan(audit, root, output):
    """Stage corrected copies; keep complete original values in a reversible manifest."""
    if Path(root).resolve() == Path(output).resolve():
        raise ValueError("Use a separate staging directory; preserve the original export")
    documents, originals, changes = {}, {}, []
    for target in audit["targets"]:
        name, path = target["file"], target.get("path", [])
        if name not in documents:
            documents[name] = read(safe_path(root, name))
            originals[name] = copy.deepcopy(documents[name])
        value = at(documents[name], path)
        fixed, edits = repair(value, target["accounting"], target["metric_keys"])
        if path:
            at(documents[name], path[:-1])[path[-1]] = fixed
        else:
            documents[name] = fixed
        changes.extend({"file": name, **edit, "path": path + edit["path"],
                        "source_run_id": target["accounting"]["source_run_id"]} for edit in edits)
    files = {}
    for name, value in documents.items():
        if digest(value) != digest(originals[name]):
            write(safe_path(output, name), value)
            files[name] = {"original_sha256": digest(originals[name]), "corrected_sha256": digest(value)}
    return {"formula_version": VERSION, "formula_source": FORMULA_SOURCE,
            "convention": CONVENTION, "audit_sha256": digest(audit), "files": files, "changes": changes}


def restore(manifest, root, output):
    """Restore staged JSON documents, rejecting files changed since this plan."""
    for name, hashes in manifest["files"].items():
        value = read(safe_path(root, name))
        if digest(value) != hashes["corrected_sha256"]:
            raise ValueError(f"Corrected file changed: {name}")
        for change in reversed(manifest["changes"]):
            if change["file"] != name:
                continue
            parent = at(value, change["path"][:-1])
            key = change["path"][-1]
            if change.get("absent_original"):
                del parent[key]
            else:
                parent[key] = change["original"]
        if digest(value) != hashes["original_sha256"]:
            raise ValueError(f"Restoration failed: {name}")
        write(safe_path(output, name), value)


def export_run(run, output):
    """Export all unsampled rows and original metadata before planning a mutation."""
    rows = list(run.scan_history(page_size=1000))  # No keys filter: keep sparse rows.
    expected = run._attrs.get("historyLineCount")
    if expected is not None and len(rows) != expected:
        raise ValueError(f"Incomplete history: {run.id}: {len(rows)} != {expected}")
    if rows and max(row["_step"] for row in rows) != run.lastHistoryStep:
        raise ValueError(f"Missing final step: {run.id}")
    document = {"source_run_id": "/".join(run.path), "url": run.url,
                "name": run.name, "group": run.group, "state": run.state,
                "config": dict(run.config), "summary": dict(run.summary._json_dict),
                "original_metadata": run.metadata, "original_attrs": run._attrs, "history": rows}
    write(Path(output) / "run.json", document)
    write(Path(output) / "backup.json", {"sha256": digest(document), "rows": len(rows),
                                         "last_step": run.lastHistoryStep})
    return document


def replay(document, output, api, wandb):
    """Create one deterministic reporting-only run. Never resume a source run."""
    correction = document["accounting_correction"]
    source = correction["source_run_id"]
    entity, project, _ = source.split("/")
    run_id = "nc-" + hashlib.sha256((VERSION + ":" + source).encode()).hexdigest()[:20]
    matches = list(api.runs(f"{entity}/{project}", filters={"name": run_id}))
    signature = digest(document)
    if matches:
        existing = api.run(f"{entity}/{project}/{run_id}")
        if existing.config.get("reporting", {}).get("document_sha256") != signature:
            raise ValueError(f"Existing reporting run conflicts: {existing.url}")
        source_summary = {"source_summary/" + k.lstrip("_"): v
                          for k, v in document["summary"].items() if k.startswith("_")}
        if any(existing.summary.get(k) != v for k, v in source_summary.items()):
            existing.summary.update(source_summary)
            existing.summary.update()
        verify_replay(document, list(existing.scan_history()), dict(existing.summary._json_dict))
        return existing.url
    config = copy.deepcopy(document["config"])
    config["reporting"] = {**correction, "reporting_only": True, "document_sha256": signature,
                           "source_url": document["url"],
                           "source_created_at": document["original_attrs"].get("createdAt"),
                           "source_state": document["state"],
                           "source_ended_at": document["original_attrs"].get("heartbeatAt")}
    settings = wandb.Settings(disable_git=True, disable_code=True, disable_job_creation=True,
                              x_disable_stats=True, console="off")
    with wandb.init(entity=entity, project=project, id=run_id, resume="never",
                    name=f"{document['name']} [reporting-only; {VERSION}]",
                    job_type="reporting-only", group=document.get("group") or "",
                    tags=["reporting-only", VERSION], config=config, settings=settings,
                    dir=str(output), notes=f"Accounting repair of {document['url']}. No training occurred. "
                    "Use source/* time and step fields; run creation time is upload time.") as run:
        run.define_metric("source/step")
        run.define_metric("*", step_metric="source/step", step_sync=False)
        for row in document["history"]:
            logged = dict(row)
            for key in ["_step", "_timestamp", "_runtime"]:
                if key in row:
                    logged["source/" + key.lstrip("_")] = row[key]
            logged.setdefault("source/step", row["_step"])
            run.log(logged, step=int(row["_step"]))
        run.summary.update({k: v for k, v in document["summary"].items() if not k.startswith("_")})
        run.summary.update({"source_summary/" + k.lstrip("_"): v
                            for k, v in document["summary"].items() if k.startswith("_")})
        run.summary["accounting_correction"] = correction
        run.summary["reporting_complete"] = True
    result = api.run(f"{entity}/{project}/{run_id}")
    verify_replay(document, list(result.scan_history()), dict(result.summary._json_dict))
    return result.url


def verify_replay(document, rows, summary):
    if len(rows) != len(document["history"]):
        raise ValueError("Replay row count differs")
    for source, uploaded in zip(document["history"], rows):
        for key, value in source.items():
            check_key = "source/" + key.lstrip("_") if key in {"_step", "_timestamp", "_runtime"} else key
            actual = uploaded.get(check_key)
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(f"Replay differs at {source['_step']}: {key}")
            elif value != actual:
                raise ValueError(f"Replay differs at {source['_step']}: {key}")
    for key, value in document["summary"].items():
        actual = summary.get("source_summary/" + key.lstrip("_")) if key.startswith("_") else summary.get(key)
        if actual != value:
            if not (isinstance(value, (int, float)) and isinstance(actual, (int, float))
                    and math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError(f"Replay summary differs: {key}")


def dashboard(audit, project, receipt):
    """Save a reporting view excluding source runs replaced by corrected runs."""
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    entity, name = project.split("/")
    excluded = sorted(r["source_run_id"].split("/")[-1] for r in audit["runs"]
                      if r["classification"] == "causal-pair" and r["source_run_id"].startswith(project + "/"))
    if not excluded:
        raise ValueError("This project has no audited corrections")
    filters = "ID not in " + repr(excluded)
    if receipt.exists():
        saved = read(receipt)
        view = ws.Workspace.from_url(saved["url"])
        if (saved["excluded_source_ids"] != excluded
                or expr.expr_to_filters(view.runset_settings.filters) != expr.expr_to_filters(filters)):
            raise ValueError("Saved view differs from the audit; inspect it before replacing")
        return view.url
    view = ws.Workspace(
        entity=entity, project=name, name=CONVENTION,
        runset_settings=ws.RunsetSettings(filters=filters, pinned_columns=[
            "summary:train/mfu", "summary:train/flops_per_sec", "config:reporting.source_url"]),
        sections=[ws.Section(name=CONVENTION, panels=[
            wr.LinePlot(title=title, x="Step", y=[key], smoothing_type="none")
            for title, key in [("MFU (%)", "train/mfu"), ("FLOPs/s", "train/flops_per_sec"),
                               ("Cumulative training FLOPs", "train/total_training_flops"),
                               ("Measured tokens/s", "train/tokens_per_sec")]])],
    ).save()
    write(receipt, {"url": view.url, "excluded_source_ids": excluded,
                    "formula_version": VERSION, "audit_sha256": digest(audit)})
    return view.url


def publish_space(folder, expected_revision, receipt):
    """Use the existing HF upload-folder workflow with a revision guard and readback."""
    from huggingface_hub import HfApi, hf_hub_download

    repo = "ai-testack/gpt-lab"
    api = HfApi()
    files = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob("*") if p.is_file()}
    if not {"results.json", "experiments.json"} <= files.keys():
        raise ValueError("Prepare both actual Space snapshots before publishing")
    current = api.space_info(repo).sha
    if receipt.exists():
        saved = read(receipt)
        if saved["revision"] != current or saved["files"] != files:
            raise ValueError("Publication receipt or remote revision changed; inspect before replacing")
        revision = current
    else:
        if current != expected_revision:
            raise ValueError(f"Space changed since audit: {current}")
        commit = api.upload_folder(repo_id=repo, repo_type="space", folder_path=folder,
                                   parent_commit=expected_revision,
                                   commit_message="Correct historical FLOPs/MFU reporting; preserve original evidence")
        revision = commit.oid
    for name, expected in files.items():
        downloaded = hf_hub_download(repo, name, repo_type="space", revision=revision,
                                     local_dir=receipt.parent / "verify-space")
        if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published file differs: {name}")
    if not receipt.exists():
        write(receipt, {"repo": repo, "revision": revision, "previous_revision": expected_revision,
                        "formula_version": VERSION, "files": files})
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p = commands.add_parser("restore")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("export")
    p.add_argument("--run", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("replay")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("dashboard")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p = commands.add_parser("publish-space")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--expected-revision", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        write(args.manifest, plan(read(args.audit), args.root, args.output))
    elif args.command == "restore":
        restore(read(args.manifest), args.root, args.output)
    elif args.command == "dashboard":
        print(dashboard(read(args.audit), args.project, args.receipt))
    elif args.command == "publish-space":
        print(publish_space(args.folder, args.expected_revision, args.receipt))
    else:
        import wandb
        args.output.mkdir(parents=True, exist_ok=True)
        api = wandb.Api()
        if args.command == "export":
            export_run(api.run(args.run), args.output)
        else:
            print(replay(read(args.input), args.output, api, wandb))


if __name__ == "__main__":
    main()
