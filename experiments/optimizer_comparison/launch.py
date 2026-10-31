"""User-launched, SHA-pinned validation and sequential selected 8-H100 training."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.optimizer_comparison.recipe import COMMON, SELECTION, arguments
from runs.modal_trainbase import resolve_source


def job_commands(root, mode):
    torchrun = ["/opt/gpt-lab/.venv/bin/torchrun", "--standalone", "--nnodes=1",
                "--nproc_per_node=8", "--max_restarts=0"]
    commands = [torchrun + ["experiments/optimizer_comparison/check_h100.py", "--output", str(root/"validation")]]
    if mode == "train":
        for variant in SELECTION["variants"]:
            directory = root/variant["model_name"]
            commands.append(torchrun + ["experiments/optimizer_comparison/train_selected.py",
                "--output", str(directory), "--expected-initial", str(root/SELECTION["variants"][0]["model_name"]),
                "--", *arguments(variant, directory.name, directory)])
    return commands


def run_commands(commands, run):
    # Sequential blocking calls enforce the eight-GPU concurrency limit; fail closed.
    for command in commands:
        run(command)


def preflight(arguments, run_name):
    import sys
    sys.path.insert(0, "/opt/gpt-lab")
    from runs.modal_trainbase import check_inputs
    return check_inputs(arguments, run_name, 8)


def worker(run_name, mode, sha, volume):
    import os
    import sys
    import time
    from pathlib import Path
    import subprocess
    import json
    sys.path.insert(0, "/opt/gpt-lab")
    from experiments.optimizer_comparison.launch import job_commands, run_commands
    root = Path("/runs")/run_name
    root.mkdir(exist_ok=False)
    env = os.environ | dict(GPTLAB_DTYPE="bfloat16", GPTLAB_CACHE_DIR="/cache/gpt_lab",
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
        GPTLAB_SELECTED_SHA=sha)
    record = dict(sha=sha, mode=mode, status="starting", commands=job_commands(root, mode))
    started = time.perf_counter()
    try:
        def run(command):
            with (root/f"process_{record.get('completed_processes', 0)}.log").open("w", buffering=1) as log:
                with subprocess.Popen(command, cwd="/opt/gpt-lab", env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    if process.wait():
                        raise RuntimeError(f"Process failed; inspect {root}")
            if not record.get("completed_processes", 0):
                gate = json.loads((root/"validation/passed.json").read_text())
                if gate.get("status") != "passed" or gate.get("ranks") != 8:
                    raise RuntimeError("Eight-rank hardware gate did not pass")
            record["completed_processes"] = record.get("completed_processes", 0) + 1
            volume.commit()
        run_commands(record["commands"], run)
        record["status"] = "passed" if mode == "validate" else "finished"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter()-started
        (root/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
        volume.commit()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True, help="Full published exp/anvil-normuon SHA")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("validate", "train"), default="validate")
    parser.add_argument("--dry-run", action="store_true", help="Offline plan; no Git/network/Modal")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("sha must contain 40 lowercase hexadecimal characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,70}", args.run_name):
        parser.error("run-name must contain 1–70 letters/digits/_/-")
    commands = job_commands(Path("/runs")/args.run_name, args.mode)
    print(json.dumps(dict(sha=args.sha, branch="exp/anvil-normuon", gpu="H100:8",
                         mode=args.mode, settings=COMMON, resolved_plan=SELECTION["resolved_plan"], commands=commands), indent=2), flush=True)
    if args.dry_run:
        return
    repository, branch, sha = resolve_source(ROOT)
    if branch != "exp/anvil-normuon" or sha != args.sha:
        parser.error("Use the clean published exp/anvil-normuon checkout at --sha")
    import modal
    app = modal.App("gpt-lab-selected-optimizers")
    cache = modal.Volume.from_name("gpt-lab-cache", environment_name="main")
    runs = modal.Volume.from_name("gpt-lab-runs", environment_name="main")
    image = (modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "ca-certificates")
        .pip_install("uv==0.13.6")
        .run_commands("git init /opt/gpt-lab",
            f"git -C /opt/gpt-lab remote add origin {shlex.quote(repository)}",
            f"git -C /opt/gpt-lab fetch --depth 1 origin {sha}",
            f"git -C /opt/gpt-lab checkout --detach {sha}",
            "git -C /opt/gpt-lab -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive",
            "uv sync --locked --default-index https://pypi.org/simple --project /opt/gpt-lab --extra gpu --extra fp8 --group dev"))
    check = app.function(image=image, cpu=1, memory=1024, timeout=120, serialized=True,
        volumes={"/cache/gpt_lab": cache.with_mount_options(read_only=True),
                 "/runs": runs.with_mount_options(read_only=True)})(preflight)
    run = app.function(image=image, gpu="H100:8", cpu=32, memory=262144,
        timeout=86400, max_containers=1, retries=0, serialized=True,
        volumes={"/cache/gpt_lab": cache, "/runs": runs},
        secrets=[modal.Secret.from_name("gpt-lab", environment_name="main", required_keys=["HF_TOKEN", "WANDB_API_KEY"])])(worker)
    with modal.enable_output(), app.run(detach=args.detach, environment_name="main"):
        check.remote(arguments(SELECTION["variants"][0], args.run_name, Path("/runs")/args.run_name), args.run_name)
        if args.detach:
            print(run.spawn(args.run_name, args.mode, sha, runs).object_id)
        else:
            print(run.remote(args.run_name, args.mode, sha, runs))


if __name__ == "__main__":
    main()
