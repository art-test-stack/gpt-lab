"""Compare compiled training and explicit forward/backward capture on Modal.

Setup: uv pip install 'modal==1.6.2'; modal setup
Run from the CUDA-graph worktree after committing and pushing its changes:
    python scripts/modal_cuda_graphs.py --gpus 1 --mode verify --run-name cg-check-1
    python scripts/modal_cuda_graphs.py --gpus 8 --mode paired --run-name cg-8-matched
    python scripts/modal_cuda_graphs.py --gpus 8 --mode paired --profile --run-name cg-8-profile

Launcher options configure Modal; other options go to GPT-lab train auto.
An optional -- separates training arguments explicitly. Defaults match the
reference runs/modal_trainbase.py: A100-80GB, BF16, 2317-step horizon,
300-step warmup, global batch 524288 tokens. NorMuon is explicitly selected.
Runs stop after 150 updates. Evaluation, sampling and all checkpoint saves
are disabled. Settings may be overridden identically for both modes.
The current branch must have the same HEAD locally and on origin, with no
tracked local edits. Untracked files (including this launcher) are ignored.
Origin is queried live; no fetch, pull, push, cherry-picks or local code uploads.
Optimizer defaults come from that exact commit. The branch must support
stop_after_step and completed_steps.
The gpt-lab Modal secret needs HF_TOKEN and WANDB_API_KEY. Existing Volumes:
gpt-lab-cache (/cache/gpt_lab) and gpt-lab-runs (/runs). No data downloads.
Every GPU job first runs six compiled-versus-captured updates on a tiny
production transformer with the selected optimizer and real NCCL when distributed.
Timing is blocked unless every rank passes. Paired modes run sequentially on
one allocation in fresh seeded processes and verify identical state/data reset.
Artifacts live under /runs/RUN_NAME: equivalence/, topology.txt, gpus.csv,
comparison.json and per-mode directories with launch/summary/rank JSON and logs.
Summary separates startup/time to first update from aggregate
global tokens/s over updates 11..150 (first ten excluded). Every process uses
its own empty compiler cache, so one run cannot warm the next run's compiler.
Profile jobs run 14 updates per mode and save three-update CPU/CUDA timelines
and kernel tables; their results are not throughput measurements.
--dry-run contacts Git only, without importing Modal or allocating compute.
--require-sxm rejects any GPU other than NVIDIA A100-SXM4-80GB before
training. It validates the allocation; it does not control Modal placement.
Adapted from the user's ~/gpt-lab/runs/modal_trainbase.py.
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
import time


REPO = Path("/opt/gpt-lab")
CACHE = Path("/cache/gpt_lab")
RUNS = Path("/runs")
PYTHON = str(REPO / ".venv/bin/python")

DEFAULT_TRAIN = {
    "model_name": "ic1",
    "optim_config_path": "configs/normuon.yaml",
    "random_seed": 42,
    "device": "cuda",
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
    "attn_impl": "fused",
    "device_batch_size": 16,
    "target_param_data_ratio": 8,
    "total_batch_size": 524288,
    "num_steps": 2317,
    "warmup_steps": 300,
    "warmdown_ratio": 0.65,
    "final_lr_frac": 0.05,
    "stop_after_step": 150,
    "board": "wandb",
    "log_every": 1,
    "save_every": 0,
    "eval_bpb_every": 0,
    "eval_core_every": 0,
    "sample_every": 0,
}

# Keep diagnostic instrumentation outside the production trainer.
SHORT_RUNNER = '''import hashlib, json, os, sys, time
from contextlib import nullcontext
from pathlib import Path
compiler_cache = Path(os.environ["GPTLAB_COMPILER_CACHE"]) / os.environ.get("LOCAL_RANK", "0")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache / "inductor")
os.environ["TRITON_CACHE_DIR"] = str(compiler_cache / "triton")
import torch
from gpt_lab.cli import main
from gpt_lab.train.trainer import Trainer
Trainer.save_checkpoint = lambda self: None
original_train = Trainer.train
original_train_step = Trainer.train_step
original_update_metrics = Trainer.update_metrics
original_prepare_batch = Trainer.prepare_batch
original_optimizer_step = Trainer.optimizer_step
profiling = os.environ.get("GPTLAB_PROFILE") == "1"
def digest(value):
    sha = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            sha.update(str((item.dtype, tuple(item.shape))).encode())
            sha.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                sha.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                add(entry)
        else:
            sha.update(repr(item).encode())
    add(value)
    return sha.hexdigest()
def memory():
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
diagnostic = dict(steps=[], capture_startup_seconds=None,
                  setup_before_first_update_seconds=None,
                  time_to_first_update_seconds=None)
def prepare_batch(self, *args, **kwargs):
    with torch.profiler.record_function("prepare_batch") if profiling else nullcontext():
        result = original_prepare_batch(self, *args, **kwargs)
    if "first_batch_sha256" not in diagnostic:
        diagnostic["first_batch_sha256"] = digest(result[0])
        diagnostic["initial_data_position"] = result[1].model_dump() if result[1] is not None else None
        expected_root = os.environ.get("GPTLAB_EXPECTED_INITIAL")
        if expected_root:
            expected = json.loads((Path(expected_root) / f"rank_{self.config.comp_info.rank}.json").read_text())
            for key in ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
                        "first_batch_sha256", "initial_data_position"):
                if diagnostic[key] != expected[key]:
                    raise RuntimeError(f"Initial reset mismatch before capture: {key}")
    return result
def optimizer_step(self, *args, **kwargs):
    with torch.profiler.record_function("optimizer_and_communication") if profiling else nullcontext():
        return original_optimizer_step(self, *args, **kwargs)
def train_step(self, *args, **kwargs):
    now = time.perf_counter()
    if diagnostic["steps"]:
        diagnostic["steps"][-1]["iteration_seconds"] = now - diagnostic["last_update_started"]
    diagnostic["last_update_started"] = now
    if diagnostic["setup_before_first_update_seconds"] is None:
        diagnostic["setup_before_first_update_seconds"] = now - diagnostic["started"]
    if self.state.step == 10:
        diagnostic["startup_memory"] = memory()
        torch.cuda.reset_peak_memory_stats()
    with torch.profiler.record_function("accumulated_update") if profiling else nullcontext():
        return original_train_step(self, *args, **kwargs)
def update_metrics(self, *args, **kwargs):
    metrics = original_update_metrics(self, *args, **kwargs)
    if diagnostic["time_to_first_update_seconds"] is None:
        diagnostic["time_to_first_update_seconds"] = time.perf_counter() - diagnostic["started"]
    diagnostic["steps"].append(dict(
        step=self.state.step, seconds=metrics["train/step_time_ms"] / 1000,
        tokens=self.config.total_batch_size, loss=metrics["train/raw_loss"],
        lrm=metrics["lrm"], muon_momentum=metrics["muon_momentum"],
        weight_decay=metrics["weight_decay"]))
    if profiling:
        self._diagnostic_profiler.step()
    return metrics
def train(self):
    diagnostic["started"] = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    diagnostic["initial_model_sha256"] = digest(dict(
        parameters=self.model.state_dict(), buffers=dict(self.model.named_buffers())))
    diagnostic["initial_optimizer_sha256"] = digest(self.optimizer.state_dict())
    diagnostic["initial_rng_sha256"] = digest((torch.get_rng_state(), torch.cuda.get_rng_state()))
    original_log = self.board.log
    def log(values, *args, **kwargs):
        if "train/capture_startup_seconds" in values:
            diagnostic["capture_startup_seconds"] = values["train/capture_startup_seconds"]
        return original_log(values, *args, **kwargs)
    self.board.log = log
    def trace_ready(profiler):
        root = Path(os.environ["GPTLAB_LAUNCH_DIR"])
        rank = self.config.comp_info.rank
        profiler.export_chrome_trace(str(root / f"trace_rank_{rank}.json"))
        (root / f"kernels_rank_{rank}.txt").write_text(
            profiler.key_averages().table(sort_by="self_device_time_total", row_limit=40))
    profile_context = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=10, warmup=1, active=3, repeat=1),
        on_trace_ready=trace_ready,
    ) if profiling else nullcontext()
    try:
        with profile_context as profiler:
            self._diagnostic_profiler = profiler
            return original_train(self)
    finally:
        self.board.log = original_log
        if diagnostic["steps"] and "iteration_seconds" not in diagnostic["steps"][-1]:
            diagnostic["steps"][-1]["iteration_seconds"] = time.perf_counter() - diagnostic["last_update_started"]
        result = dict(diagnostic, completed_steps=self.state.completed_steps,
                      n_steps=self.config.n_steps, stop_after_step=self.config.stop_after_step,
                      mode=self.config.cuda_graph_mode, rank=self.config.comp_info.rank,
                      world_size=self.config.comp_info.world_size,
                      n_acc_steps=self.config.n_acc_steps,
                      device_batch_size=self.config.device_batch_size,
                      total_batch_size=self.config.total_batch_size,
                      parameter_count=sum(p.numel() for p in self.model.parameters()),
                      dtype=str(self.dtype), device_name=torch.cuda.get_device_name(self.device),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      final_data_position=self.state.train_loader_state.model_dump() if self.state.train_loader_state is not None else None,
                      steady_memory=memory(),
                      train_wall_seconds=time.perf_counter() - diagnostic["started"])
        result.pop("started")
        result.pop("last_update_started", None)
        Path(os.environ["GPTLAB_LAUNCH_DIR"], f"rank_{result['rank']}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\\n")
Trainer.train = train
Trainer.train_step = train_step
Trainer.update_metrics = update_metrics
Trainer.prepare_batch = prepare_batch
Trainer.optimizer_step = optimizer_step
if __name__ == "__main__":
    raise SystemExit(main(["train", "auto", *sys.argv[1:]]))
'''


def summarize(rank_results):
    """Exclude ten updates; use the slowest rank's full loop time for global throughput."""
    rank_zero = rank_results[0]
    for result in rank_results:
        if (result["n_steps"], result["completed_steps"], result["mode"]) != (
            rank_zero["n_steps"], rank_zero["completed_steps"], rank_zero["mode"]
        ):
            raise RuntimeError("Ranks disagree on completed updates, horizon or capture mode")
    steady = [step for step in rank_zero["steps"] if step["step"] >= 10]
    seconds = max(sum(step["iteration_seconds"] for step in result["steps"] if step["step"] >= 10)
                  for result in rank_results)
    tokens = sum(step["tokens"] for step in steady)
    return dict(
        mode=rank_zero["mode"], world_size=len(rank_results),
        n_steps=rank_zero["n_steps"], completed_steps=rank_zero["completed_steps"],
        dtype=rank_zero["dtype"], device_name=rank_zero["device_name"],
        device_names=[r["device_name"] for r in rank_results],
        memory_by_rank=[dict(rank=r["rank"], startup=r.get("startup_memory"),
                             steady=r.get("steady_memory")) for r in rank_results],
        train_startup_inclusive_seconds=max(r.get("train_wall_seconds", 0) for r in rank_results),
        n_acc_steps=rank_zero["n_acc_steps"], device_batch_size=rank_zero["device_batch_size"],
        total_batch_size=rank_zero["total_batch_size"], parameter_count=rank_zero["parameter_count"],
        setup_before_first_update_seconds=max(r["setup_before_first_update_seconds"] for r in rank_results),
        time_to_first_update_seconds=max(r["time_to_first_update_seconds"] for r in rank_results),
        capture_startup_seconds=max((r["capture_startup_seconds"] or 0) for r in rank_results),
        steady_first_update=11, steady_last_update=rank_zero["completed_steps"],
        steady_updates=len(steady), steady_seconds=seconds,
        steady_mean_train_step_ms=1000 * sum(step["seconds"] for step in steady) / len(steady) if steady else None,
        steady_global_tokens_per_second=tokens / seconds if seconds else None,
    )


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


def check_inputs(arguments, run_name, world_size, check_data=True):
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
    if cap is None or cap <= 10 or (settings["num_steps"] > 0 and settings["num_steps"] < cap):
        raise ValueError("Diagnostics need a cap >10 within the original scheduling horizon.")
    microbatch_tokens = world_size * settings["device_batch_size"] * settings["max_seq_len"]
    batch = settings["total_batch_size"]
    if microbatch_tokens < 1:
        raise ValueError("device_batch_size and max_seq_len must be positive.")
    if settings["n_acc_steps"] == -1 and batch != -1 and (batch < microbatch_tokens or batch % microbatch_tokens):
        raise ValueError(f"total_batch_size must be -1 or a positive multiple of {microbatch_tokens}.")
    if settings["n_acc_steps"] != -1 and (
        settings["n_acc_steps"] < 1 or (batch != -1 and batch != microbatch_tokens * settings["n_acc_steps"])
    ):
        raise ValueError("Explicit n_acc_steps must be positive and match total_batch_size.")
    from gpt_lab.utils.schemas import OptimizerConfig
    OptimizerConfig.from_file(settings["optim_config_path"])
    if world_size == 8 and (
        settings["device_batch_size"], settings["max_seq_len"], batch,
        settings["num_steps"], settings["warmup_steps"], cap, settings["n_acc_steps"]
    ) not in ((16, 2048, 524288, 2317, 300, 150, -1), (16, 2048, 524288, 2317, 300, 150, 2)):
        raise ValueError("8-GPU comparison requires microbatch 16, sequence 2048, global batch 524288, horizon 2317, warmup 300, cap 150 and two accumulation microsteps")
    if not check_data:
        return settings
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


def gpu_probe(world_size, dtype, attn_impl, require_sxm=False):
    probe = (f"import torch; assert torch.cuda.device_count() == {world_size}; "
             f"names = [torch.cuda.get_device_name(i) for i in range({world_size})]; "
             "print('Allocated GPUs:', names, flush=True); "
             "assert all('A100' in name for name in names), names")
    if require_sxm:
        probe += "; assert all(name == 'NVIDIA A100-SXM4-80GB' for name in names), f'SXM4 required; allocated: {names}'"
    if dtype == "bfloat16":
        probe += "; assert torch.cuda.is_bf16_supported()"
    if attn_impl == "fused":
        probe += ("; from gpt_lab.model.flash_attn import _flash_attn2, _flash_attn; "
                  "assert _flash_attn2 is not None or _flash_attn is not None, 'Fused attention unavailable'; "
                  "print('Attention backend:', 'FA3' if _flash_attn is not None else 'FA2')")
    return probe


def train_one(arguments, run_name, sha, branch, environment, settings, world_size, run_volume, directory=None):
    actual = git(REPO, "rev-parse", "HEAD")
    if actual != sha or git(REPO, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Image source differs from the selected commit.")
    directory = directory or RUNS / run_name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "train_short.py").write_text(SHORT_RUNNER)
    command = [str(REPO / ".venv/bin/torchrun"), "--standalone", "--nnodes=1",
               f"--nproc_per_node={world_size}", "--max_restarts=0", str(directory / "train_short.py"),
               *arguments]
    env = os.environ | environment | {
        "GPTLAB_CACHE_DIR": str(CACHE), "GPTLAB_LAUNCH_DIR": str(directory),
        "GPTLAB_COMPILER_CACHE": str(REPO / ".tmp/codex/modal-cuda-graphs" / run_name),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    record = dict(git_sha=sha, git_branch=branch, repository=git(REPO, "remote", "get-url", "origin"),
                  command=command, environment=environment,
                  checkpoint_export=False, settings=settings, status="starting",
                  started_at=datetime.now(timezone.utc).isoformat())
    try:
        record["status"] = "running"
        (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        run_volume.commit()
        print(shlex.join(command), flush=True)
        process_start = time.perf_counter()
        with (directory / "train.log").open("w", buffering=1) as log:
            with subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        process_elapsed = time.perf_counter() - process_start
        results = [json.loads((directory / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
        resolved = results[0]
        expected = resolved["n_steps"]
        if resolved["stop_after_step"] is not None:
            expected = min(expected, resolved["stop_after_step"])
        if any(result["completed_steps"] != expected for result in results):
            raise RuntimeError("Completed optimizer-step count differs from the execution cap.")
        if environment.get("GPTLAB_PROFILE") == "1":
            summary = dict(mode=resolved["mode"], world_size=world_size,
                           completed_steps=resolved["completed_steps"], kind="profile",
                           traces=[f"trace_rank_{rank}.json" for rank in range(world_size)])
        else:
            summary = summarize(results)
        summary.update(git_sha=sha, git_branch=branch,
                       process_startup_inclusive_seconds=process_elapsed)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        record["status"] = "finished"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        resolved = directory / "rank_0.json"
        if resolved.is_file():
            result = json.loads(resolved.read_text())
            record.update({key: result[key] for key in ("n_steps", "completed_steps", "stop_after_step", "mode")})
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        run_volume.commit()
    return record


def verify_pair(left, right):
    """Fresh-process reset is accepted only when initial values and data positions match."""
    fields = ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
              "first_batch_sha256", "initial_data_position", "final_data_position",
              "n_steps", "completed_steps", "n_acc_steps", "device_name", "rank",
              "world_size", "dtype", "total_batch_size", "device_batch_size",
              "parameter_count", "torch_version", "cuda_version")
    if len(left) != len(right):
        raise RuntimeError("Paired runs have different rank counts")
    for rank, (compiled, captured) in enumerate(zip(left, right)):
        for field in fields:
            if field not in compiled or field not in captured or compiled[field] != captured[field]:
                raise RuntimeError(f"Paired reset mismatch on rank {rank}: {field}")
        if compiled["initial_data_position"] is None or compiled["final_data_position"] is None:
            raise RuntimeError("Paired data position could not be verified")


def train(arguments, run_name, sha, branch, environment, settings, world_size, run_volume):
    """One allocation: verify NCCL equivalence, then run independent restored processes."""
    if git(REPO, "rev-parse", "HEAD") != sha or git(REPO, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Image source differs from the selected commit")
    root = RUNS / run_name
    root.mkdir(parents=True, exist_ok=False)
    record = dict(git_sha=sha, world_size=world_size, settings=settings,
                  kind="profile" if environment["GPTLAB_PROFILE"] == "1" else "timing",
                  status="starting")
    started = time.perf_counter()
    try:
        probe = gpu_probe(world_size, environment["GPTLAB_DTYPE"], settings["attn_impl"],
                          environment["GPTLAB_REQUIRE_SXM"] == "1")
        subprocess.run([PYTHON, "-c", probe], cwd=REPO, env=os.environ | environment, check=True)
        for filename, command in (
            ("topology.txt", ["nvidia-smi", "topo", "-m"]),
            ("gpus.csv", ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv"]),
        ):
            (root / filename).write_text(subprocess.check_output(command, text=True))
        command = [str(REPO / ".venv/bin/torchrun"), "--standalone", "--nnodes=1",
                   f"--nproc_per_node={world_size}", "--max_restarts=0",
                   str(REPO / "scripts/benchmark/cuda_graph_equivalence.py"),
                   "--optim-config-path", settings["optim_config_path"],
                   "--attn-impl", settings["attn_impl"], "--n-acc-steps",
                   str(settings["n_acc_steps"] if settings["n_acc_steps"] > 0 else
                       settings["total_batch_size"] // (world_size * settings["device_batch_size"] * settings["max_seq_len"])),
                   "--output-dir", str(root / "equivalence")]
        gate_env = os.environ | environment | dict(
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            TORCHINDUCTOR_CACHE_DIR=str(REPO / ".tmp/codex/modal-cuda-graphs" / run_name / "equivalence"))
        gate_start = time.perf_counter()
        with (root / "equivalence.log").open("w") as log:
            subprocess.run(command, cwd=REPO, env=gate_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        checks = [json.loads((root / "equivalence" / f"rank_{rank}.json").read_text())
                  for rank in range(world_size)]
        if not all(check["passed"] and check["world_size"] == world_size for check in checks):
            raise RuntimeError("Equivalence gate did not pass on every rank")
        record["equivalence_seconds"] = time.perf_counter() - gate_start
        record["equivalence_passed"] = True
        mode = environment["GPTLAB_DIAGNOSTIC_MODE"]
        if mode != "verify":
            modes = ("disabled", "forward_backward") if mode == "paired" else (mode,)
            results = []
            for selected in modes:
                directory = root / selected
                cap = 14 if environment["GPTLAB_PROFILE"] == "1" else settings["stop_after_step"]
                selected_args = [*arguments, "--cuda-graph-mode", selected,
                                 "--stop-after-step", str(cap),
                                 "--run-name", f"{run_name}-{selected}",
                                 "--model-dir", str(directory / "models"),
                                 "--board-dir", str(directory / "wandb")]
                selected_env = environment
                if mode == "paired" and selected == "forward_backward":
                    selected_env = environment | {"GPTLAB_EXPECTED_INITIAL": str(root / "disabled")}
                train_one(selected_args, f"{run_name}-{selected}", sha, branch, selected_env,
                          settings | dict(cuda_graph_mode=selected, stop_after_step=cap),
                          world_size, run_volume, directory=directory)
                ranks = [json.loads((directory / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
                results.append(ranks)
            if len(results) == 2:
                verify_pair(*results)
                record["identical_initial_state_and_data_position"] = True
            record["results"] = {selected: json.loads((root / selected / "summary.json").read_text()) for selected in modes}
        record["status"] = "finished"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["allocation_work_seconds"] = time.perf_counter() - started
        (root / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        run_volume.commit()
    return record


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd(), help="Local checkout (default: current directory)")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--mode", required=True, choices=("disabled", "forward_backward", "paired", "verify"), help="Run one mode, both sequentially on one allocation, or only the equivalence gate")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--gpus", type=int, choices=(1, 4, 8), default=1, help="A100-80GB GPU count on one node")
    parser.add_argument("--profile", action="store_true", help="Separate 14-update profiling job; record three updates after warmup, never use for throughput claims")
    parser.add_argument("--require-sxm", action="store_true", help="Reject non-SXM4 A100 allocations before training; does not control placement")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", type=int, default=65536, help="GPU worker RAM in MiB")
    parser.add_argument("--wandb-entity", default="art-test-stack-ntnu")
    parser.add_argument("--wandb-project", default="trainbase_ic1")
    parser.add_argument("--wandb-base-url", default="https://api.forge.coreweave.com")
    parser.add_argument("--wandb-app-url", default="https://forge.coreweave.com/wandb")
    parser.add_argument("--timeout", type=int, default=3600, help="GPU function timeout in seconds")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve source and print command without using Modal")
    args, overrides = parser.parse_known_args(argv[:split])
    overrides.extend(argv[split + 1:])
    try:
        repository, branch, sha = resolve_source(args.repo_dir)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(getattr(error, "stderr", None) or str(error))
    run_name = args.run_name or f"{sha[:8]}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_name):
        parser.error("Use a run name of 1-100 letters/digits/_/-.")
    if min(args.timeout, args.gpus, args.cpu, args.memory) < 1:
        parser.error("timeout, gpus, cpu and memory must be positive.")
    arguments = [*cli_defaults(DEFAULT_TRAIN), *overrides,
                 "--cuda-graph-mode", args.mode if args.mode in ("disabled", "forward_backward") else "disabled", "--device", "cuda",
                 "--save-every", "0", "--eval-bpb-every", "0",
                 "--eval-core-every", "0", "--sample-every", "0", "--log-every", "1",
                 "--run-name", run_name, "--model-dir", str(RUNS / run_name / "models"),
                 "--board-dir", str(RUNS / run_name / "wandb")]
    environment = dict(GPTLAB_DTYPE=args.dtype, WANDB_ENTITY=args.wandb_entity,
                       WANDB_PROJECT=args.wandb_project, WANDB_BASE_URL=args.wandb_base_url,
                       WANDB_APP_URL=args.wandb_app_url, WANDB_MODE="online",
                       GPTLAB_REQUIRE_SXM=str(int(args.require_sxm)),
                       GPTLAB_DIAGNOSTIC_MODE=args.mode, GPTLAB_PROFILE=str(int(args.profile)),
                       TORCHINDUCTOR_COMPILE_THREADS="1",
                       TORCHINDUCTOR_CUDAGRAPHS="0", PYTORCH_ALLOC_CONF="expandable_segments:False")
    print(f"Verified: local {branch} == origin/{branch} == {sha}", flush=True)
    print(shlex.join(["gpt-lab", "train", "auto", *arguments]), flush=True)
    print(f"Hardware: {args.gpus} x A100-80GB; dtype: {args.dtype}; capture mode: {args.mode}", flush=True)
    print("Gate: six compiled-versus-captured optimizer updates must pass on every rank before timing.", flush=True)
    if args.mode == "paired":
        print("Plan: disabled then forward_backward on the same allocation, with fresh state/data reset.", flush=True)
    if args.profile:
        print("Profile only: 14 updates per mode; record three updates; no throughput comparison.", flush=True)
    if args.require_sxm:
        print("Required device: NVIDIA A100-SXM4-80GB (other allocations will fail before training)", flush=True)
    if args.dry_run:
        return
    import modal
    app = modal.App("gpt-lab-cuda-graphs")
    cache_volume = modal.Volume.from_name("gpt-lab-cache")
    run_volume = modal.Volume.from_name("gpt-lab-runs")
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
            f"uv sync --locked --default-index https://pypi.org/simple --project {REPO} --extra gpu --group dev",
            f"{PYTHON} -c 'import gpt_lab; import gpt_lab.cli.commands.base; print(gpt_lab.__file__)'",
        )
        .env({"GPTLAB_CACHE_DIR": str(CACHE), "PYTHONUNBUFFERED": "1"})
    )
    check = app.function(image=image, cpu=1, memory=1024, timeout=120,
                         volumes={str(CACHE): cache_volume.with_mount_options(read_only=True),
                                  str(RUNS): run_volume.with_mount_options(read_only=True)},
                         serialized=True)(check_inputs)
    worker = app.function(image=image, gpu=f"A100-80GB:{args.gpus}", cpu=args.cpu, memory=args.memory,
                          timeout=args.timeout, retries=0, max_containers=1,
                          volumes={str(CACHE): cache_volume, str(RUNS): run_volume},
                          secrets=[modal.Secret.from_name("gpt-lab", required_keys=["HF_TOKEN", "WANDB_API_KEY"])],
                          serialized=True)(train)
    with modal.enable_output(), app.run(detach=args.detach):
        settings = check.remote(arguments, run_name, args.gpus, args.mode != "verify")
        values = (arguments, run_name, sha, branch, environment, settings, args.gpus, run_volume)
        if args.detach:
            call = worker.spawn(*values)
            print(f"Run {run_name} submitted; function call {call.object_id}", flush=True)
        else:
            worker.remote(*values)


if __name__ == "__main__":
    main()
