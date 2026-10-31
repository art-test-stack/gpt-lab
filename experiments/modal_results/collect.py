"""Read Modal evidence without compute, build the static archive, optionally publish."""
from __future__ import annotations

import argparse
import codecs
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "spaces/results"
TITLE = {
    "communication_buckets": "Communication bucketing",
    "cuda_graphs": "CUDA graph capture",
    "fp8_training": "FP8 training",
    "h100_input_pipeline": "H100 input pipeline",
    "ffn_ablation": "FFN architectures",
    "optimizer_comparison": "Optimizer comparison",
    "other": "Other / unclassified",
}
BLOCKED_DIRS = {
    "model", "models", "checkpoint", "checkpoints", "weights", "optimizer_states",
    "cache", "__pycache__", ".git", ".venv", "torchinductor", "inductor", "triton",
}
MODEL_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".onnx", ".pkl", ".pickle", ".npy", ".npz", ".wandb"}
TEXT_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".log", ".txt", ".md", ".xml", ".yaml", ".yml", ".py", ".toml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"}
SECRET_KEY = re.compile(r"(?:^|_)(?:token|token_id|token_secret|api_key|apikey|password|secret|credentials?|authorization|cookie|cookies|access_key_id|secret_access_key)$", re.I)
PATH_KEYS = {"hostname", "dirname", "cache_dir", "model_dir", "board_dir", "source_dir", "home"}
TOKEN = re.compile(r"\b(?:hf_[A-Za-z0-9]{16,}|(?:sk|wandb)_[A-Za-z0-9_-]{16,})\b")
ASSIGNMENT = re.compile(r'''(?i)((?:["']?)(?:[a-z0-9_]*(?:api_key|apikey|access_token|hf_token|wandb_api_key|password|secret|authorization))["']?\s*[:=]\s*)(["'])([^\r\n"']*)(\2)''')
BARE_SECRET = re.compile(r"(?i)((?:HF_TOKEN|WANDB_API_KEY|MODAL_TOKEN_ID|MODAL_TOKEN_SECRET|API_KEY|PASSWORD|AUTHORIZATION)\s*=\s*)[^\s,;]+")
SUMMARY_KEYS = {
    "steady_tokens_per_second", "steady_train_loop_tokens_per_second", "steady_throughput",
    "tokens_per_second", "tokens_per_sec", "throughput", "median_ms", "p95_ms", "mean_ms",
    "median_step_ms", "mean_step_ms", "mean_step_seconds", "steady_step_ms", "median_update_ms",
    "steady_step_seconds", "mean_step_time_ms", "mean_update_ms", "time_to_first_update_seconds",
    "capture_startup_seconds", "train_wall_seconds", "startup_to_first_update_rank_max_s",
    "process_startup_to_first_update_rank_max_s", "full_throughput_ratio_to_baseline",
    "peak_allocated_bytes_after_warmup", "peak_reserved_bytes_after_warmup", "allocated_bytes",
    "peak_allocated_bytes", "peak_reserved_bytes", "persistent_bytes", "padding_bytes",
    "collective_count", "reduction_input_bytes", "gather_output_bytes", "median_seconds",
    "median", "p05", "p95", "speedup", "speedup_percent", "relative_l2_error", "max_absolute_error",
    "loss", "bpb", "mfu", "completed_steps", "steady_updates", "measured_updates",
}
JSON_LIMIT = 8 * 1024 * 1024  # Large traces are streamed, never parsed for summaries.


@dataclass
class Entry:
    path: str
    size: int


def safe_path(path):
    path = str(path).removeprefix("/")
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or any(part in (".", "..") for part in parts) or "\\" in path or "\x00" in path:
        raise ValueError(f"Unsafe evidence path: {path!r}")
    return PurePosixPath(*parts)


def exclusion(path):
    p = safe_path(path)
    if any(part.lower() in BLOCKED_DIRS or part.startswith(".") for part in p.parts):
        return "checkpoint/cache/hidden directory"
    suffixes = [suffix.lower() for suffix in p.suffixes]
    if any(suffix in MODEL_SUFFIXES for suffix in suffixes):
        return "model/optimizer/binary state"
    if p.name.lower().startswith(("credentials", "tokenizer", "vocab", "merges")):
        return "credentials/tokenizer data"
    if p.suffix.lower() in TEXT_SUFFIXES | IMAGE_SUFFIXES:
        return None
    if len(suffixes) >= 2 and suffixes[-2:] in ([".json", ".gz"], [".jsonl", ".gz"]):
        return None
    return "unsupported/opaque format"


class LocalSource:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("--local-root must be an existing directory")

    def entries(self):
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
            for name in sorted(files):
                p = Path(directory) / name
                if not p.is_symlink() and p.is_file():
                    yield Entry(p.relative_to(self.root).as_posix(), p.stat().st_size)

    def chunks(self, path):
        with (self.root / path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                yield chunk


class ModalSource:
    def __init__(self, volume, environment):
        try:
            import modal
        except ImportError as error:
            raise ValueError("Modal collection requires modal; see experiments/modal_results/README.md") from error
        self.volume = modal.Volume.from_name(volume, environment_name=environment, create_if_missing=False)

    def entries(self):
        for entry in self.volume.iterdir("/", recursive=True):
            # FileEntryType.FILE is an IntEnum named FILE; directories/symlinks are never read.
            if getattr(entry.type, "name", "") == "FILE":
                yield Entry(str(safe_path(entry.path)), entry.size)

    def chunks(self, path):
        yield from self.volume.read_file(path)


def redact_text(text):
    text = TOKEN.sub("[REDACTED]", text)
    text = ASSIGNMENT.sub(lambda m: m[1] + m[2] + "[REDACTED]" + m[4], text)
    text = BARE_SECRET.sub(lambda m: m[1] + "[REDACTED]", text)
    text = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{12,}", r"\1[REDACTED]", text)
    return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()
                if not SECRET_KEY.search(str(k)) and str(k).lower() not in PATH_KEYS}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def slug(name):
    readable = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:80] or "run"
    return readable + "-" + hashlib.sha256(name.encode()).hexdigest()[:8]


def family(run, paths, documents, mapping):
    if run in mapping:
        result = mapping[run]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", result):
            raise ValueError(f"Invalid experiment name for {run}")
        return result
    for _, document in documents:
        if isinstance(document, dict):
            explicit = document.get("experiment") or document.get("experiment_name")
            if isinstance(explicit, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", explicit):
                return explicit
    text = (run + " " + " ".join(paths)).lower()
    # Specific pipeline families precede broad FP8/H100 names.
    for patterns, result in [
        (("input-pipeline", "input_pipeline", "dataloader", "loader-cache"), "h100_input_pipeline"),
        (("communication", "comm-", "comm_"), "communication_buckets"),
        (("cuda-graph", "cuda_graph", "cg-", "capture"), "cuda_graphs"),
        (("fp8", "gemm_operands"), "fp8_training"),
        (("ffn", "swiglu", "relu-squared"), "ffn_ablation"),
        (("anvil", "optimizer-comparison", "optimizer_comparison"), "optimizer_comparison"),
    ]:
        if any(pattern in text for pattern in patterns):
            return result
    if any(isinstance(doc, dict) and doc.get("mode") in ("forward_backward", "disabled")
           for _, doc in documents):
        return "cuda_graphs"
    return "other"


def measurements(value, source, location="", depth=0):
    if depth > 8:
        return []
    result = []
    if isinstance(value, dict):
        metrics = {key: child for key, child in value.items()
                   if key in SUMMARY_KEYS and isinstance(child, (int, float))
                   and not isinstance(child, bool) and math.isfinite(child)}
        if metrics:
            result.append({"source": source, "context": location or "summary", "values": metrics})
        for key, child in value.items():
            if isinstance(child, dict):
                result.extend(measurements(child, source, f"{location}/{key}".strip("/"), depth + 1))
            elif isinstance(child, list) and len(child) <= 16 and all(isinstance(v, dict) for v in child):
                for index, item in enumerate(child):
                    result.extend(measurements(item, source, f"{location}/{key}/{index}".strip("/"), depth + 1))
    return result


def artifact_kind(path):
    if "trace" in path.lower() and (path.endswith(".json") or path.endswith(".gz")):
        return "trace"
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "figure"
    if suffix in {".log", ".txt", ".xml"}:
        return "log"
    if suffix in {".py", ".toml", ".yaml", ".yml"}:
        return "configuration/source"
    return "report/data"


def stream_text(source, destination):
    """Bound memory for large JSON traces, retaining overlap for credential patterns."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    with source.open("rb") as handle, destination.open("wb") as output:
        while chunk := handle.read(1024 * 1024):
            pending += decoder.decode(chunk)
            # Split only at a line boundary or safe punctuation, never inside a credential.
            cut = max(pending.rfind("\n", 0, max(0, len(pending) - 2048)),
                      pending.rfind(",", 0, max(0, len(pending) - 2048))) + 1
            if cut:
                output.write(redact_text(pending[:cut]).encode())
                pending = pending[cut:]
            if len(pending) > 16 * 1024 * 1024:
                raise ValueError("Unbroken text record exceeds 16 MiB; cannot safely redact it")
        pending += decoder.decode(b"", final=True)
        output.write(redact_text(pending).encode())


def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def copy_artifact(source, entry, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = None
    with tempfile.TemporaryDirectory() as temporary:
        raw = Path(temporary) / "raw"
        with raw.open("wb") as handle:
            for chunk in source.chunks(entry.path):
                handle.write(chunk)
        source_hash = sha256_file(raw)
        compressed = entry.path.lower().endswith((".json.gz", ".jsonl.gz"))
        decoded = raw
        if compressed:
            decoded = Path(temporary) / "decoded"
            with gzip.open(raw, "rb") as handle, decoded.open("wb") as output:
                shutil.copyfileobj(handle, output, 1024 * 1024)
        normalized = Path(temporary) / "public"
        parse_error = None
        if entry.path.lower().endswith(".json") and decoded.stat().st_size <= JSON_LIMIT:
            try:
                document = sanitize(json.loads(decoded.read_text(encoding="utf-8-sig")))
                normalized.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
            except (json.JSONDecodeError, UnicodeError) as error:
                parse_error = type(error).__name__
                stream_text(decoded, normalized)
        elif PurePosixPath(entry.path).suffix.lower() in IMAGE_SUFFIXES - {".svg"}:
            shutil.copyfile(decoded, normalized)
        else:
            stream_text(decoded, normalized)
        if compressed:
            with destination.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as encoded:
                with normalized.open("rb") as handle:
                    shutil.copyfileobj(handle, encoded, 1024 * 1024)
        else:
            shutil.copyfile(normalized, destination)
        archived_hash = sha256_file(destination)
        return document, {
            "name": entry.path, "kind": artifact_kind(entry.path), "bytes": destination.stat().st_size,
            "source_bytes": raw.stat().st_size, "source_sha256": source_hash,
            "sha256": archived_hash, "transformed": source_hash != archived_hash,
            **({"parse_error": parse_error} if parse_error else {}),
        }


def run_metadata(documents):
    metadata = {"status": "unknown", "commits": [], "started_at": None}
    # Prefer explicit root manifests over nested fixture/profile statuses.
    priority = {"manifest.json": 0, "launch.json": 1, "report.json": 2, "summary.json": 3}
    for path, data in sorted(documents, key=lambda pair: (len(PurePosixPath(pair[0]).parts), priority.get(PurePosixPath(pair[0]).name, 9), pair[0])):
        if not isinstance(data, dict):
            continue
        if len(PurePosixPath(path).parts) == 1 and metadata["status"] == "unknown" and isinstance(data.get("status"), str):
            metadata["status"] = data["status"]
        for key in ("started_at", "started", "timestamp", "exported_at"):
            if not metadata["started_at"] and isinstance(data.get(key), str):
                metadata["started_at"] = data[key]
        for key in ("sha", "git_sha", "production_sha", "source_sha"):
            sha = data.get(key)
            if isinstance(sha, str) and re.fullmatch(r"[a-fA-F0-9]{40}", sha):
                if sha not in metadata["commits"]:
                    metadata["commits"].append(sha)
    return metadata


def inventory(source, prefixes=()):
    kept, skipped = [], Counter()
    for entry in source.entries():
        path = str(safe_path(entry.path))
        if prefixes and not any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes):
            continue
        reason = exclusion(path)
        if reason:
            skipped[reason] += 1
        else:
            kept.append(Entry(path, entry.size))
    return sorted(kept, key=lambda e: e.path), dict(skipped)


def collect(source, output, *, volume="local", environment="local", prefixes=(), mapping=None):
    output = Path(output).resolve()
    if output == ROOT or output == ASSETS or output in ASSETS.parents:
        raise ValueError("Choose a separate build directory, not the repository or dashboard source")
    if isinstance(source, LocalSource) and (output == source.root or output in source.root.parents or source.root in output.parents):
        raise ValueError("The build directory and input evidence directory must not contain each other")
    previous = None
    if (output / "experiments.json").exists():
        previous = json.loads((output / "experiments.json").read_text())
    selected, skipped = inventory(source, prefixes)
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for entry in selected:
        p = PurePosixPath(entry.path)
        run = p.parts[0] if len(p.parts) > 1 else "volume-root"
        groups.setdefault(run, []).append(entry)
    runs, errors, managed = [], [], []
    for run, entries in sorted(groups.items()):
        # Stage outside the archive: classification can inspect bounded report JSON first.
        with tempfile.TemporaryDirectory() as directory:
            stage, artifacts, documents = Path(directory), [], []
            for entry in entries:
                relative = PurePosixPath(entry.path)
                relative = PurePosixPath(*relative.parts[1:]) if len(relative.parts) > 1 else relative
                destination = stage / relative
                try:
                    document, artifact = copy_artifact(source, entry, destination)
                except (OSError, ValueError, UnicodeError, EOFError) as error:
                    errors.append({"path": entry.path, "error": type(error).__name__, "message": redact_text(str(error))})
                    continue
                artifact["relative_path"] = relative.as_posix()
                artifacts.append(artifact)
                if document is not None:
                    documents.append((relative.as_posix(), document))
            experiment = family(run, [e.path for e in entries], documents, mapping or {})
            archive_root = PurePosixPath("archive") / experiment / slug(f"{environment}:{volume}:{run}")
            for artifact in artifacts:
                target = archive_root / "files" / artifact.pop("relative_path")
                relative = PurePosixPath(target).relative_to(archive_root / "files")
                destination = output / target
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / relative, destination)
                artifact["path"] = target.as_posix()
                managed.append(artifact["path"])
            stats = run_metadata(documents)
            values = []
            for name, document in documents:
                if artifact_kind(name) != "trace":
                    values.extend(measurements(document, name))
            runs.append({
                "id": f"{environment}:{volume}:{run}", "name": run,
                "experiment": experiment, **stats, "measurements": values,
                "source": {"type": "modal-volume" if isinstance(source, ModalSource) else "local", "volume": volume, "environment": environment, "directory": run},
                "artifacts": sorted(artifacts, key=lambda a: a["name"]),
            })
    catalog = {
        "schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": {"volume": volume, "environment": environment},
        "experiments": [{"id": key, "title": TITLE.get(key, key.replace("_", " ").title())}
                        for key in sorted({run["experiment"] for run in runs})],
        "runs": runs, "excluded": skipped, "errors": errors,
        "notes": ["Missing measurements stay missing; no loss or speedup is inferred.",
                  "Reported measurements retain their source context and units; protocols may differ.",
                  "Artifacts omit known credential fields/patterns; transformed files retain original and archived checksums.",
                  "Collection is a point-in-time read, not an atomic snapshot of running jobs."],
    }
    if previous:
        catalog = merge_catalogs(previous, catalog)
    (output / "experiments.json").write_text(json.dumps(catalog, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    managed.append("experiments.json")
    for name in ("index.html", "app.js", "data.mjs", "style.css", "evidence.js", "evidence_data.mjs", "logo.png", "README.md"):
        shutil.copy2(ASSETS / name, output / name)
        # Leave the existing Space's README/SDK metadata untouched when publishing.
        if name != "README.md":
            managed.append(name)
    shutil.copytree(ASSETS / "vendor", output / "vendor", dirs_exist_ok=True)
    managed.extend(p.relative_to(output).as_posix() for p in (output / "vendor").rglob("*") if p.is_file())
    # Existing hosted training curves are never replaced by this archive command.
    if not (output / "results.json").exists():
        shutil.copy2(ASSETS / "results.json", output / "results.json")
    managed.extend(a["path"] for run in catalog["runs"] for a in run["artifacts"])
    (output / "archive-files.json").write_text(json.dumps(sorted(set(managed)), indent=2) + "\n", encoding="utf-8")
    return catalog


def merge_catalogs(previous, current):
    if previous.get("schema_version") != 1 or not isinstance(previous.get("runs"), list):
        raise ValueError("Cannot merge an incompatible experiment index")
    runs = {run["id"]: run for run in previous["runs"]}
    runs.update({run["id"]: run for run in current["runs"]})
    experiments = {item["id"]: item for item in previous.get("experiments", [])}
    experiments.update({item["id"]: item for item in current["experiments"]})
    return current | {"runs": sorted(runs.values(), key=lambda r: (r["experiment"], r["name"])),
                      "experiments": sorted(experiments.values(), key=lambda e: e["id"])}


def remote_catalog(repo_id, revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="space", filename="experiments.json", revision=revision)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text())


def publish(output, repo_id, api=None, read_remote=remote_catalog):
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise ValueError("Publishing requires huggingface_hub and local HF authentication") from error
        api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="space")
    if getattr(info, "sdk", None) != "static":
        raise ValueError("The destination must be an existing static Hugging Face Space")
    output = Path(output).resolve()
    previous = read_remote(repo_id, info.sha)
    if previous:
        index_path = output / "experiments.json"
        merged = merge_catalogs(previous, json.loads(index_path.read_text()))
        index_path.write_text(json.dumps(merged, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    paths = json.loads((output / "archive-files.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        for relative in paths:
            p = safe_path(relative)
            if str(p) in ("results.json", "README.md") or exclusion(p):
                # JS/HTML are trusted viewer assets, not volume evidence.
                if str(p) not in {"app.js", "data.mjs", "evidence.js", "evidence_data.mjs", "index.html", "style.css"} and not str(p).startswith("vendor/"):
                    raise ValueError(f"Unexpected publication path: {p}")
            source = output / p
            if source.is_symlink() or output not in source.resolve().parents:
                raise ValueError("Publication files must stay inside the build directory")
            target = stage / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return api.upload_folder(
            repo_id=repo_id, repo_type="space", folder_path=str(stage),
            commit_message="Update Modal experiment evidence archive",
            parent_commit=info.sha,
            # No delete_patterns: retain existing training data and earlier evidence.
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--volume", help="Existing Modal volume, e.g. gpt-lab-runs")
    inputs.add_argument("--local-root", type=Path, help="Already downloaded volume contents")
    parser.add_argument("--environment", default="main")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/modal-results-space")
    parser.add_argument("--run-prefix", action="append", default=[], help="Restrict to a volume directory; repeatable")
    parser.add_argument("--mapping", type=Path, help="JSON mapping of exact run directory names to experiment slugs")
    parser.add_argument("--dry-run", action="store_true", help="List eligible files without downloading or publishing")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--space", default="ai-testack/gpt-lab")
    args = parser.parse_args(argv)
    try:
        source = ModalSource(args.volume, args.environment) if args.volume else LocalSource(args.local_root)
        if args.dry_run:
            entries, skipped = inventory(source, args.run_prefix)
            print(json.dumps({"eligible_files": len(entries), "source_bytes": sum(e.size for e in entries), "excluded": skipped,
                              "files": [e.path for e in entries]}, indent=2))
            return
        mapping = json.loads(args.mapping.read_text()) if args.mapping else {}
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("Mapping must be a JSON object of run directory -> experiment slug")
        catalog = collect(source, args.output, volume=args.volume or "local", environment=args.environment if args.volume else "local",
                          prefixes=args.run_prefix, mapping=mapping)
        print(f"Archived {len(catalog['runs'])} runs, {sum(len(r['artifacts']) for r in catalog['runs'])} evidence files into {args.output}")
        if catalog["errors"]:
            raise ValueError(f"{len(catalog['errors'])} files could not be archived; see experiments.json. Publishing was not attempted")
        if args.publish:
            print(publish(args.output, args.space))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
