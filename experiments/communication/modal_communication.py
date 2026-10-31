"""User-launched, SHA-pinned Modal communication handoff. No work runs on import.

Use --dry-run to verify the published source without importing Modal.
Stages: correctness (2 A100s), sweep (4 A100s), compatibility (2 A100s),
profile (4 A100s, separate from timed runs). Existing Modal volumes:
gpt-lab-cache and gpt-lab-runs; existing secret: gpt-lab (HF_TOKEN).
Select --optimizer explicitly. Screens run 55 updates (25 warmup + 30 measured),
then 150-update confirmation. Retain baseline unless both gains reach 1%.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REPO = Path("/opt/gpt-lab")
CACHE = Path("/cache/gpt_lab")
RUNS = Path("/runs")
PYTHON = str(REPO / ".venv/bin/python")
MODES = ("baseline", "buckets", "pipeline", "backward")
WARMUP_UPDATES = 25
SCREEN_UPDATES = 55
MIN_THROUGHPUT_GAIN = 0.01
REQUIRED = (
    "experiments/communication/modal_communication.py", "experiments/communication/communication.py",
    "src/gpt_lab/optim/communication.py", "src/gpt_lab/optim/factory.py",
    "src/gpt_lab/optim/strategy.py", "src/gpt_lab/optim/strategies/muon.py",
    "src/gpt_lab/train/trainer.py", "experiments/communication/test_communication.py",
    "configs/optim.yaml", "configs/normuon.yaml", "experiments/communication/DESIGN.md",
    "experiments/communication/README.md", "experiments/communication/RESULTS.md",
    "experiments/communication/results.json",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def verify_source(sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("--sha must be the complete pushed commit SHA")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("Launcher must belong to this worktree's Git root")
    if git("rev-parse", "HEAD") != sha or git("status", "--porcelain"):
        raise ValueError("Use a clean, committed checkout at --sha")
    git("ls-files", "--error-unmatch", *REQUIRED)
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    remote = git("ls-remote", "--exit-code", "origin", f"refs/heads/{branch}")
    if remote.split()[0] != sha:
        raise ValueError(f"--sha must equal the published origin/{branch} tip")
    repository = git("remote", "get-url", "origin")
    return re.sub(r"^(?:git@github.com:|ssh://git@github.com/)", "https://github.com/", repository)


def preflight(stage, run_name, results_run, sha, optimizer):
    if (RUNS / run_name).exists():
        raise FileExistsError(f"Run {run_name!r} already exists; use a fresh name")
    if stage in ("sweep", "profile"):
        shards = sorted((CACHE / "data/climbmix-base").glob("shard_*.parquet"))
        if len(shards) < 5 or any(p.stat().st_size == 0 for p in shards):
            raise ValueError("Need the existing nonempty climbmix-base train/validation shards; no downloads are performed")
        if [p.name for p in shards] != [f"shard_{i:05d}.parquet" for i in range(len(shards))]:
            raise ValueError("Cached data shards must be contiguous from shard_00000.parquet")
        if not any(p.is_file() for p in (CACHE / "tokenizers/gpt2").rglob("*")):
            raise ValueError("Cached gpt2 tokenizer is missing")
    if stage in ("compatibility", "profile"):
        previous = json.loads((RUNS / results_run / "manifest.json").read_text())
        if (previous["status"] != "completed" or previous["stage"] != "sweep"
                or previous["sha"] != sha or previous["selected_optimizer"] != optimizer
                or previous.get("candidate_mode") not in MODES[1:]):
            raise ValueError("--results-run must be a completed sweep with the same SHA and optimizer")
    return True


def validate_matched(reference, candidate):
    """Fresh processes reset state/data; fail if their recorded reset differs."""
    optimizer = reference["selected_optimizer"]
    if (optimizer not in ("muon", "normuon") or candidate["status"] != "completed"
            or candidate["selected_optimizer"] != optimizer):
        raise ValueError("Matched fingerprints require the same selected optimizer")
    for first, second in zip(reference["initial_fingerprints_per_rank"], candidate["initial_fingerprints_per_rank"], strict=True):
        for key in ("rank", "model_sha256", "optimizer_state_sha256", "optimizer_groups",
                    "first_batch_sha256", "first_batch_position"):
            if first[key] != second[key]:
                raise ValueError(f"Reset mismatch for rank {first['rank']}: {key}")
        if first["first_batch_position"] is None:
            raise ValueError("A reproducible initial dataloader position is required")
        a, b = first["last_consumed_data_positions"], second["last_consumed_data_positions"]
        count = min(len(a), len(b))
        if a[:count] != b[:count] or any(position is None for position in b):
            raise ValueError(f"Consumed data positions diverged on rank {first['rank']}")


def run_process(command, directory, env, volume, timeout):
    """Bound the whole process tree, including distributed startup/hangs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(shlex.join(command), flush=True)
    try:
        with (directory / "console.log").open("w") as log:
            with subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError(f"Distributed candidate exceeded {timeout}s; process tree terminated")
                if code:
                    raise RuntimeError(f"Candidate failed (exit {code}); see {directory}/console.log")
    finally:
        volume.commit()


def correctness(directory, optimizer, modes, world, env, volume, timeout):
    basetemp = REPO / ".tmp/codex/modal-correctness/pytest"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    report = directory / "correctness.json"
    run_env = env | {"GPTLAB_COMM_TEST_WORLD_SIZE": str(world),
                     "GPTLAB_COMM_TEST_OPTIMIZER": optimizer,
                     "GPTLAB_COMM_TEST_MODES": ",".join(modes),
                     "GPTLAB_COMM_TEST_SMOKE_ONLY": "1",
                     "GPTLAB_COMM_TEST_REPORT": str(report)}
    junit = directory / "junit.xml"
    command = [PYTHON, "-m", "pytest", "experiments/communication/test_communication.py",
               "-k", "test_nccl_buckets_accumulation_padding_ties_and_resume", "-s", "-q",
               f"--junitxml={junit}", f"--basetemp={basetemp}"]
    run_process(command, directory, run_env, volume, timeout)
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    if sum(int(s.get("tests", "0")) for s in suites) != 1 or any(
        int(s.get(key, "0")) for s in suites for key in ("failures", "errors", "skipped")
    ):
        raise ValueError("NCCL correctness must execute and pass, not skip")
    rows = [json.loads(report.with_name(f"correctness.rank{rank}.json").read_text()) for rank in range(world)]
    if any(not row["ok"] or row["optimizer"] != optimizer or row["backend"] != "nccl" for row in rows):
        raise ValueError("Incorrect or incomplete per-rank NCCL reports")
    volume.commit()
    return rows


def result_summary(result):
    return {
        key: result[key] for key in (
            "communication", "selected_optimizer", "updates_requested", "steady_updates",
            "steady_train_loop_tokens_per_second", "steady_wall_step_time_ms",
            "steady_train_step_time_ms", "startup_to_first_update_rank_max_s",
            "process_startup_to_first_update_rank_max_s",
            "memory_per_rank", "communication_rank_max_summary",
        )
    }


def run_suite(stage, run_name, results_run, sha, bucket_mib, optimizer, volume, candidate_timeout):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != sha:
        raise RuntimeError("Modal image does not match the pinned source")
    world = 2 if stage in ("correctness", "compatibility") else 4
    directory = RUNS / run_name
    directory.mkdir(parents=True, exist_ok=False)
    temporary = REPO / ".tmp/codex/modal"
    temporary.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "GPTLAB_CACHE_DIR": str(CACHE), "GPTLAB_DTYPE": "bfloat16", "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TMPDIR": str(temporary), "MPLCONFIGDIR": str(temporary / "mpl"),
        "TORCHINDUCTOR_CACHE_DIR": str(temporary / "inductor"),
        "HF_HOME": str(CACHE / "huggingface"),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "TORCH_NCCL_BLOCKING_WAIT": "0",
    }
    manifest = dict(sha=sha, stage=stage, status="running", bucket_mib=bucket_mib,
                    world_size=world, selected_optimizer=optimizer, candidates={},
                    screen_updates=SCREEN_UPDATES, warmup_updates=WARMUP_UPDATES,
                    min_throughput_gain=MIN_THROUGHPUT_GAIN,
                    started_at=datetime.now(timezone.utc).isoformat())

    def persist():
        path = directory / "manifest.json"
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_path.replace(path)
        volume.commit()

    def candidate(label, mode, updates, reference=None, profile=False):
        output = directory / label
        command = [str(REPO / ".venv/bin/torchrun"), "--standalone", f"--nproc_per_node={world}",
                   "--max_restarts=0", "experiments/communication/communication.py",
                   "--optimizer", optimizer, "--communication", mode,
                   "--bucket-mib", str(bucket_mib), "--updates", str(updates),
                   "--warmup-updates", str(WARMUP_UPDATES), "--output-dir", str(output)]
        if profile:
            command.append("--profile")
        manifest["current_candidate"] = label
        persist()
        run_process(command, output, env, volume, candidate_timeout)
        result = json.loads((output / "communication-results.json").read_text())
        if (result["status"] != "completed" or result["selected_optimizer"] != optimizer
                or result["world_size"] != world or result["git_sha"] != sha
                or result["completed_steps_by_rank"] != [updates] * world
                or result["scheduler_horizon_by_rank"] != [2317] * world
                or result["steady_updates"] != updates - WARMUP_UPDATES
                or result["bucket_mib"] != bucket_mib):
            raise ValueError("Candidate source, optimizer, or update/horizon invariants failed")
        if reference is not None:
            validate_matched(reference, result)
        throughput = result["steady_train_loop_tokens_per_second"]
        if not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0:
            raise ValueError("Candidate has no valid timed throughput window")
        manifest["candidates"][label] = result_summary(result)
        persist()
        return result

    persist()
    try:
        probe = ("import torch; assert torch.cuda.device_count()==" + str(world) +
                 "; assert all('A100' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))")
        run_process([PYTHON, "-c", probe], directory / "hardware", env, volume, 120)
        if stage == "correctness":
            manifest["correctness"] = correctness(directory / f"{optimizer}-correctness", optimizer, MODES[1:], world, env, volume, 600)
        elif stage == "compatibility":
            previous = json.loads((RUNS / results_run / "manifest.json").read_text())
            manifest["candidate_mode"] = previous["candidate_mode"]
            manifest["correctness"] = correctness(directory / f"{optimizer}-compatibility", optimizer, [previous["candidate_mode"]], world, env, volume, 600)
        elif stage == "sweep":
            # Gate expensive training on actual NCCL parameter/gradient/state agreement.
            manifest["correctness"] = correctness(directory / f"{optimizer}-correctness", optimizer, MODES[1:], world, env, volume, 600)
            persist()
            baseline = candidate("screen-baseline", "baseline", SCREEN_UPDATES)
            screened = {mode: candidate(f"screen-{mode}", mode, SCREEN_UPDATES, baseline) for mode in MODES[1:]}
            winner = max(MODES[1:], key=lambda mode: screened[mode]["steady_train_loop_tokens_per_second"])
            manifest["candidate_mode"] = winner
            manifest["screen_throughput_ratio_to_baseline"] = (
                screened[winner]["steady_train_loop_tokens_per_second"] / baseline["steady_train_loop_tokens_per_second"])
            persist()
            full_baseline = candidate("full-baseline", "baseline", 150, baseline)
            full_winner = candidate(f"full-{winner}", winner, 150, full_baseline)
            manifest["full_throughput_ratio_to_baseline"] = (
                full_winner["steady_train_loop_tokens_per_second"] / full_baseline["steady_train_loop_tokens_per_second"])
            manifest["recommended_mode"] = (
                winner if min(manifest["screen_throughput_ratio_to_baseline"],
                              manifest["full_throughput_ratio_to_baseline"]) >= 1 + MIN_THROUGHPUT_GAIN
                else "baseline")
            manifest["selection_note"] = "Require at least 1% higher throughput in both stages; this is a conservative threshold, not a confidence interval."
        else:
            previous = json.loads((RUNS / results_run / "manifest.json").read_text())
            bucket_mib = previous["bucket_mib"]
            manifest["bucket_mib"] = bucket_mib
            winner = previous["candidate_mode"]
            manifest["candidate_mode"] = winner
            manifest["recommended_mode"] = previous.get("recommended_mode", "baseline")
            prior_baseline = json.loads((RUNS / results_run / "full-baseline/communication-results.json").read_text())
            baseline = candidate("profile-baseline", "baseline", SCREEN_UPDATES, prior_baseline, profile=True)
            candidate(f"profile-{winner}", winner, SCREEN_UPDATES, baseline, profile=True)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("correctness", "sweep", "compatibility", "profile"))
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-run")
    parser.add_argument("--optimizer", choices=("muon", "normuon"), default="muon")
    parser.add_argument("--bucket-mib", type=float, default=64)
    parser.add_argument("--candidate-timeout", type=int, default=1800)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in (args.run_name, args.results_run):
        if name is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
            parser.error("run names must contain only letters, digits, _ and -")
    if args.stage in ("profile", "compatibility") and not args.results_run:
        parser.error("this stage requires --results-run from the completed sweep")
    if not math.isfinite(args.bucket_mib) or args.bucket_mib <= 0 or args.candidate_timeout < 1:
        parser.error("bucket size and candidate timeout must be positive")
    repository = verify_source(args.sha)
    world = 2 if args.stage in ("correctness", "compatibility") else 4
    if args.dry_run:
        print(json.dumps(vars(args) | {"gpu": f"A100-80GB:{world}", "repository": repository,
                                     "optimizer": args.optimizer, "screen_updates": SCREEN_UPDATES,
                                     "warmup_updates": WARMUP_UPDATES, "confirmation_updates": 150,
                                     "min_throughput_gain": MIN_THROUGHPUT_GAIN}, indent=2))
        return
    import modal
    app = modal.App("gpt-lab-communication")
    cache_volume = modal.Volume.from_name("gpt-lab-cache")
    run_volume = modal.Volume.from_name("gpt-lab-runs")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "ca-certificates")
        .pip_install("uv==0.13.6")
        .run_commands(
            f"git init {REPO}", f"git -C {REPO} remote add origin {shlex.quote(repository)}",
            f"git -C {REPO} fetch --depth 1 origin {args.sha}", f"git -C {REPO} checkout --detach {args.sha}",
            f"git -C {REPO} -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive",
            f"uv sync --locked --default-index https://pypi.org/simple --project {REPO} --extra gpu --group dev",
        )
        .env({"GPTLAB_CACHE_DIR": str(CACHE), "PYTHONUNBUFFERED": "1"})
    )
    check = app.function(image=image, cpu=1, memory=1024, timeout=120, serialized=True,
                         volumes={str(CACHE): cache_volume.with_mount_options(read_only=True),
                                  str(RUNS): run_volume.with_mount_options(read_only=True)})(preflight)
    worker = app.function(image=image, gpu=f"A100-80GB:{world}", cpu=16, memory=65536,
                          timeout=10800, retries=0, max_containers=1, serialized=True,
                          volumes={str(CACHE): cache_volume, str(RUNS): run_volume},
                          secrets=[modal.Secret.from_name("gpt-lab", required_keys=["HF_TOKEN"])])(run_suite)
    with modal.enable_output(), app.run(detach=args.detach):
        check.remote(args.stage, args.run_name, args.results_run, args.sha, args.optimizer)
        values = (args.stage, args.run_name, args.results_run, args.sha, args.bucket_mib, args.optimizer, run_volume, args.candidate_timeout)
        if args.detach:
            call = worker.spawn(*values)
            print(f"Submitted {args.run_name}: {call.object_id}; results in gpt-lab-runs/{args.run_name}")
        else:
            worker.remote(*values)


if __name__ == "__main__":
    main()
