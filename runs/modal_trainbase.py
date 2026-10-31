"""Train the current branch on Modal after verifying local HEAD matches origin.

Setup: uv pip install 'modal==1.6.2'; modal setup
Run this single file from your GPT-lab checkout:
    python runs/modal_trainbase.py --dry-run
    python runs/modal_trainbase.py --detach --run-name bf16-15 --stop-after-step 15
    python runs/modal_trainbase.py --detach --stop-after-step 30
    python runs/modal_trainbase.py --tie-embeddings 'embs,head;value_embs'

Launcher options configure Modal; other options go to GPT-lab train auto.
An optional -- separates training arguments explicitly. CLI values override
the defaults below: 4 A100s, BF16, 2317-step horizon and 300-step warmup. There is no implicit execution cap;
pass --stop-after-step N for a short diagnostic.
The current branch must have the same HEAD locally and on origin, with no
tracked local edits. Untracked files are ignored; this launcher is tracked in the cloned source.
Origin is queried live; no fetch, pull, push, cherry-picks or local code uploads.
Optimizer defaults come from that exact commit. The branch must support
stop_after_step and completed_steps.
The gpt-lab Modal secret needs HF_TOKEN and WANDB_API_KEY. Existing Volumes:
gpt-lab-cache (/cache/gpt_lab) and gpt-lab-runs (/runs). No data downloads.
Checkpoints default off (including the trainer's final save); --checkpoints
restores normal saves. --dry-run contacts Git only, without importing Modal.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


REPO = Path("/opt/gpt-lab")
CACHE = Path("/cache/gpt_lab")
RUNS = Path("/runs")
PYTHON = str(REPO / ".venv/bin/python")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TRAIN = {
    "model_name": "ic1",
    "ds_name": "climbmix-base",
    "tokenizer_model": "gpt2",
    "depth": 12,
    "aspect_ratio": 64,
    "d_head": 128,
    "max_seq_len": 2048,
    "tie_embeddings": "embs;value_embs;head",
    "window_pattern": "SSSL",
    "window_size": 512,
    "softcap": 18.0,
    "attn_impl": "fa2",
    "act_func": "swiglu",
    "device_batch_size": 16,
    "target_param_data_ratio": 8,
    "total_batch_size": 524288,
    "num_steps": 2317,
    "warmup_steps": 300,
    "warmdown_ratio": 0.65,
    "final_lr_frac": 0.05,
    "board": "wandb",
    "log_every": 1,
    "save_every": 0,
    "eval_bpb_every": 0,
    "eval_core_every": 0,
    "sample_every": 0,
}

# Native CLI execution; no automatic experiment validation or fingerprint hooks.
SHORT_RUNNER = '''import json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from gpt_lab.cli import main
from gpt_lab.train.trainer import Trainer
if os.environ["GPTLAB_CHECKPOINTS"] == "0":
    Trainer.save_checkpoint = lambda self: None
original_train = Trainer.train
def train(self):
    try:
        return original_train(self)
    finally:
        if self.config.comp_info.rank == 0:
            result = dict(completed_steps=self.state.completed_steps,
                          n_steps=self.config.n_steps,
                          stop_after_step=self.config.stop_after_step)
            Path(os.environ["GPTLAB_LAUNCH_DIR"], "resolved_run.json").write_text(
                json.dumps(result, indent=2) + "\\n")
Trainer.train = train
if __name__ == "__main__":
    raise SystemExit(main(["train", "auto", *sys.argv[1:]]))
'''


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, stderr=subprocess.PIPE,
    ).strip()


def resolve_source(directory):
    """Require an attached, clean branch whose commit equals its live origin tip."""
    try:
        branch = git(directory, "symbolic-ref", "--quiet", "--short", "HEAD")
    except subprocess.CalledProcessError as error:
        raise ValueError("Cannot determine the current branch; run from a Git checkout with an attached branch.") from error
    local_sha = git(directory, "rev-parse", "HEAD")
    changes = git(directory, "status", "--porcelain", "--untracked-files=no")
    if changes:
        raise ValueError(f"Tracked local edits are not on origin; commit or discard them before launching:\n{changes}")
    repository = git(directory, "remote", "get-url", "origin")
    remote_ref = f"refs/heads/{branch}"
    try:
        output = git(directory, "ls-remote", "--exit-code", "origin", remote_ref)
    except subprocess.CalledProcessError as error:
        if error.returncode == 2:
            raise ValueError(f"Branch {branch!r} does not exist on origin; push it before launching.") from error
        raise
    refs = dict(line.split()[::-1] for line in output.splitlines())
    remote_sha = refs.get(remote_ref)
    if local_sha != remote_sha:
        raise ValueError(
            f"Current branch {branch!r} does not match origin/{branch}.\n"
            f"Local HEAD: {local_sha}\nOrigin HEAD: {remote_sha}\n"
            "Push local commits or update your checkout, then retry."
        )
    # The build image has no SSH credentials; use HTTPS for GitHub origins.
    repository = re.sub(r"^(?:git@github.com:|ssh://git@github.com/)",
                        "https://github.com/", repository)
    return repository, branch, local_sha


def cli_defaults(settings):
    args = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is not None and value is not False:
            args.extend([flag, str(value)])
    return args


def check_inputs(arguments, run_name, world_size, checkpoints=False, sha=None):
    """Validate native CLI arguments and existing cache on CPU before GPU work."""
    code = ("import json, sys; from gpt_lab.cli.parser import build_parser; "
            "print(json.dumps(vars(build_parser().parse_args(sys.argv[1:]))))")
    settings = json.loads(subprocess.check_output(
        [PYTHON, "-c", code, "train", "auto", *arguments], cwd=REPO, text=True,
    ))
    if (RUNS / run_name).exists():
        raise FileExistsError(f"Run {run_name!r} already exists; choose a new name.")
    if settings["num_steps"] < 1 and settings["num_steps"] != -1:
        raise ValueError("num_steps must be positive or -1 for automatic inference.")
    if settings["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be nonnegative.")
    cap = settings["stop_after_step"]
    if cap is not None and cap < 0:
        raise ValueError("stop_after_step must be nonnegative.")
    microbatch_tokens = world_size * settings["device_batch_size"] * settings["max_seq_len"]
    batch = settings["total_batch_size"]
    if microbatch_tokens < 1:
        raise ValueError("device_batch_size and max_seq_len must be positive.")
    if settings["n_acc_steps"] == -1 and batch != -1 and (batch < microbatch_tokens or batch % microbatch_tokens):
        raise ValueError(f"total_batch_size must be -1 or a positive multiple of {microbatch_tokens}.")
    data = CACHE / "data" / settings["ds_name"]
    shards = sorted(data.glob("shard_*.parquet"))
    if len(shards) < 5 or [p.name for p in shards] != [f"shard_{i:05d}.parquet" for i in range(len(shards))]:
        raise ValueError(f"Need >=5 contiguous cached shards starting at 00000 in {data}.")
    if any(p.stat().st_size == 0 for p in shards):
        raise ValueError("Found an empty Parquet shard.")
    metadata_path = data / "meta.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    val_index = metadata.get("max_shards") or len(shards) - 1
    if type(val_index) is not int or not 4 <= val_index < len(shards):
        raise ValueError(f"Invalid validation shard index: {val_index!r}.")
    expected = dict(name=settings["ds_name"], base_url="", column_name="text", max_shards=val_index)
    if metadata and any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Dataset metadata differs from the loader convention: {metadata!r}.")
    tokenizer = settings["tokenizer_model"] or "gpt2"
    if not any(p.is_file() for p in (CACHE / "tokenizers" / tokenizer).rglob("*")):
        raise ValueError(f"Cached tokenizer {tokenizer!r} is missing.")
    return settings


def run_process(command, log_path, environment):
    print(shlex.join(command), flush=True)
    with log_path.open("w", buffering=1) as log:
        with subprocess.Popen(command, cwd=REPO, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def train(arguments, run_name, sha, branch, environment, checkpoints, settings, world_size, run_volume):
    actual = git(REPO, "rev-parse", "HEAD")
    if actual != sha or git(REPO, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Image source differs from the selected commit.")
    directory = RUNS / run_name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "train_short.py").write_text(SHORT_RUNNER)
    (directory / "requested_settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    command = [str(REPO / ".venv/bin/torchrun"), "--standalone", "--nnodes=1",
               f"--nproc_per_node={world_size}", "--max_restarts=0", str(directory / "train_short.py"),
               *arguments]
    env = os.environ | environment | {
        "GPTLAB_CACHE_DIR": str(CACHE), "GPTLAB_LAUNCH_DIR": str(directory),
        "GPTLAB_CHECKPOINTS": str(int(checkpoints)), "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    record = dict(git_sha=sha, git_branch=branch, repository=git(REPO, "remote", "get-url", "origin"),
                  command=command, environment=environment,
                  checkpoint_export=checkpoints, status="starting",
                  started_at=datetime.now(timezone.utc).isoformat())
    try:
        probe = f"import torch; assert torch.cuda.device_count() == {world_size}"
        if environment["GPTLAB_DTYPE"] == "bfloat16":
            probe += "; assert torch.cuda.is_bf16_supported()"
        if settings["attn_impl"] == "fa2":
            probe += "; from gpt_lab.model.flash_attn import require_flash_attn2; require_flash_attn2()"
        subprocess.run([PYTHON, "-c", probe], cwd=REPO, env=env, check=True)
        record["status"] = "running"
        (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        run_volume.commit()
        run_process(command, directory / "train.log", env)
        resolved = json.loads((directory / "resolved_run.json").read_text())
        expected = resolved["n_steps"]
        if resolved["stop_after_step"] is not None:
            expected = min(expected, resolved["stop_after_step"])
        if resolved["completed_steps"] != expected:
            raise RuntimeError("Completed optimizer-step count differs from the execution cap.")
        record["status"] = "finished"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        resolved = directory / "resolved_run.json"
        if resolved.is_file():
            record.update(json.loads(resolved.read_text()))
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        run_volume.commit()
    return record


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd(), help="Local checkout (default: current directory)")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--sha", help="Require this full published commit SHA")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu", default="A100-80GB", help="Modal GPU type")
    parser.add_argument("--gpus", type=int, default=4, help="GPU count on one node")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", type=int, default=65536, help="GPU worker RAM in MiB")
    parser.add_argument("--wandb-entity", default="art-test-stack-ntnu")
    parser.add_argument("--wandb-project", default="trainbase_ic1")
    parser.add_argument("--wandb-base-url", default="https://api.forge.coreweave.com")
    parser.add_argument("--wandb-app-url", default="https://forge.coreweave.com/wandb")
    parser.add_argument("--timeout", type=int, default=3600, help="GPU function timeout in seconds")
    parser.add_argument("--checkpoints", action="store_true", help="Enable the trainer's checkpoint saves")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve source and print command without using Modal")
    args, overrides = parser.parse_known_args(argv[:split])
    overrides.extend(argv[split + 1:])
    try:
        repository, branch, sha = resolve_source(args.repo_dir)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(getattr(error, "stderr", None) or str(error))
    if args.sha is not None and args.sha != sha:
        parser.error(f"Requested SHA {args.sha} differs from the published checkout {sha}")
    run_name = args.run_name or f"{sha[:8]}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_name):
        parser.error("Use a run name of 1-100 letters/digits/_/-.")
    if min(args.timeout, args.gpus, args.cpu, args.memory) < 1:
        parser.error("timeout, gpus, cpu and memory must be positive.")
    arguments = [*cli_defaults(DEFAULT_TRAIN), *overrides,
                 "--run-name", run_name, "--model-dir", str(RUNS / run_name / "models"),
                 "--board-dir", str(RUNS / run_name / "wandb")]
    environment = dict(GPTLAB_DTYPE=args.dtype, WANDB_ENTITY=args.wandb_entity,
                       WANDB_PROJECT=args.wandb_project, WANDB_BASE_URL=args.wandb_base_url,
                       WANDB_APP_URL=args.wandb_app_url, WANDB_MODE="online")
    print(f"Verified: local {branch} == origin/{branch} == {sha}", flush=True)
    print(shlex.join(["gpt-lab", "train", "auto", *arguments]), flush=True)
    if args.dry_run:
        return
    import modal
    app = modal.App("gpt-lab-train")
    cache_volume = modal.Volume.from_name("gpt-lab-cache")
    run_volume = modal.Volume.from_name("gpt-lab-runs")
    fp8_extra = " --extra fp8" if "--fp8-ffn-forward" in arguments else ""
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "ca-certificates")
        .pip_install("uv==0.13.6")
        .run_commands(
            f"git init {REPO}",
            f"git -C {REPO} remote add origin {shlex.quote(repository)}",
            f"git -C {REPO} fetch --depth 1 origin {sha}",
            f"git -C {REPO} checkout --detach {sha}",
            f"git -C {REPO} -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive",
            f"uv sync --locked --default-index https://pypi.org/simple --project {REPO} --extra gpu{fp8_extra} --group dev",
            f"{PYTHON} -c 'import gpt_lab; import gpt_lab.cli.commands.base; print(gpt_lab.__file__)'",
        )
        .env({"GPTLAB_CACHE_DIR": str(CACHE), "PYTHONUNBUFFERED": "1"})
    )
    check = app.function(image=image, cpu=1, memory=1024, timeout=120,
                         volumes={str(CACHE): cache_volume.with_mount_options(read_only=True),
                                  str(RUNS): run_volume.with_mount_options(read_only=True)},
                         serialized=True)(check_inputs)
    worker = app.function(image=image, gpu=f"{args.gpu}:{args.gpus}", cpu=args.cpu, memory=args.memory,
                          timeout=args.timeout, retries=0, max_containers=1,
                          volumes={str(CACHE): cache_volume, str(RUNS): run_volume},
                          secrets=[modal.Secret.from_name("gpt-lab", required_keys=["HF_TOKEN", "WANDB_API_KEY"])],
                          serialized=True)(train)
    with modal.enable_output(), app.run(detach=args.detach):
        settings = check.remote(arguments, run_name, args.gpus, args.checkpoints, sha)
        values = (arguments, run_name, sha, branch, environment, args.checkpoints, settings, args.gpus, run_volume)
        if args.detach:
            call = worker.spawn(*values)
            print(f"Run {run_name} submitted; function call {call.object_id}", flush=True)
        else:
            worker.remote(*values)


if __name__ == "__main__":
    main()
