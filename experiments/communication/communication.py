"""Run matched, user-launched communication diagnostics through ``train auto``.

The runner never starts a job by itself. Example (from the repository root)::

    uv run --extra gpu python experiments/communication/communication.py \
      --communication pipeline --optimizer muon --updates 35 \
      --bucket-mib 16 --output-dir .tmp/codex/comm-buckets/pipeline

Use the same seed and output settings for each communication mode. ``--profile``
enables CUDA-event optimizer timing and a short torch.profiler trace; these
instrumented numbers should be compared separately from ordinary throughput.
Remaining arguments are passed to ``gpt-lab train auto`` for model/data options.
Bucket and pipeline reductions run in optimizer.step after backward; backward mode
launches reductions from final-microstep readiness hooks and rejects graph capture.
Muon is the default optimizer profile; ``--optimizer normuon`` selects its profile.
The scheduler horizon (2317), BF16, fixed A100/FA2 model setup, update cap, and
disabled eval/sample/save settings are enforced for matched comparisons.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED_AT = time.perf_counter()


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--communication", choices=("baseline", "buckets", "pipeline", "backward"), default="baseline")
    parser.add_argument("--optimizer", choices=("muon", "normuon"), default="muon")
    parser.add_argument("--updates", type=positive_int, default=150)
    parser.add_argument("--bucket-mib", type=positive_float, default=64.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-updates", type=int, default=25)
    parser.add_argument("--profile", nargs="?", const="auto", metavar="TRACE.json", help="Collect optimizer CUDA-event timing and a short torch.profiler trace.")
    args, forwarded = parser.parse_known_args(argv)
    if args.warmup_updates < 0:
        parser.error("--warmup-updates must be nonnegative")
    if "--" in forwarded:
        forwarded = forwarded[forwarded.index("--") + 1 :]
    if "--fp8" in forwarded:
        parser.error("--fp8 is incompatible with this BF16 communication comparison")
    return args, forwarded


def default_cli_args(output_dir: Path) -> list[str]:
    """Defaults matching modal_trainbase; forwarded native CLI args may override."""
    return [
        "--model-name", "ic1",
        "--ds-name", "climbmix-base",
        "--tokenizer-model", "gpt2",
        "--depth", "12",
        "--aspect-ratio", "64",
        "--d-head", "128",
        "--max-seq-len", "2048",
        "--tie-embeddings", "embs;value_embs;head",
        "--window-pattern", "SSSL",
        "--window-size", "512",
        "--softcap", "18",
        "--attn-impl", "fused",
        "--device-batch-size", "16",
        "--target-param-data-ratio", "8",
        "--total-batch-size", "524288",
        "--num-steps", "2317",
        "--warmup-steps", "300",
        "--warmdown-ratio", "0.65",
        "--final-lr-frac", "0.05",
        "--random-seed", "42",
        "--board", "dummy",
        "--log-every", "-1",
        "--model-dir", str(output_dir / "models"),
        "--board-dir", str(output_dir / "board"),
    ]


def invariant_cli_args(optimizer: str = "muon", updates: int = 150) -> list[str]:
    """Settings kept identical even when callers override other CLI defaults."""
    return [
        "--depth", "12",
        "--aspect-ratio", "64",
        "--d-head", "128",
        "--max-seq-len", "2048",
        "--device-batch-size", "16",
        "--total-batch-size", "524288",
        "--random-seed", "42",
        "--attn-impl", "fused",
        "--target-param-data-ratio", "8",
        "--model-name", "ic1",
        "--tokenizer-model", "gpt2",
        "--ds-name", "climbmix-base",
        "--softcap", "18",
        "--window-pattern", "SSSL",
        "--window-size", "512",
        "--tie-embeddings", "embs;value_embs;head",
        "--optim-config-path", str(ROOT / "configs" / ("optim.yaml" if optimizer == "muon" else "normuon.yaml")),
        "--num-steps", "2317",
        "--warmup-steps", "300",
        "--warmdown-ratio", "0.65",
        "--final-lr-frac", "0.05",
        "--stop-after-step", str(updates),
        "--board", "dummy",
        "--eval-bpb-every", "0",
        "--eval-core-every", "0",
        "--sample-every", "0",
        "--save-every", "0",
        "--target-time", "-1",
    ]


def nearest_rank_summary(seconds: list[float]) -> dict[str, float | None]:
    ordered = sorted(seconds)
    if not ordered:
        return {"median_ms": None, "p95_ms": None}
    def percentile(q: float) -> float:
        return ordered[math.ceil(q * len(ordered)) - 1] * 1000
    return {"median_ms": percentile(0.5), "p95_ms": percentile(0.95)}


def baseline_communication_stats(optimizer, world_size: int, active_ids=None) -> dict[str, Any]:
    """Describe the pinned optimizer's planned per-rank collectives and payloads."""
    counts = {"all_reduce": 0, "reduce_scatter": 0, "all_gather": 0}
    payload = {"all_reduce_bytes": 0, "reduce_scatter_input_bytes": 0, "all_gather_output_bytes": 0}
    plans = []
    for group in optimizer.param_groups:
        params = group["params"]
        if group["opt"] == "adamw":
            for parameter in params:
                if active_ids is not None and id(parameter) not in active_ids:
                    continue
                size = parameter.numel() * parameter.element_size()
                if parameter.numel() < 1024:
                    counts["all_reduce"] += 1
                    payload["all_reduce_bytes"] += size
                    plans.append({"kind": "all_reduce", "bytes": size})
                else:
                    counts["reduce_scatter"] += 1
                    counts["all_gather"] += 1
                    payload["reduce_scatter_input_bytes"] += size
                    payload["all_gather_output_bytes"] += size
                    plans.extend(({"kind": "reduce_scatter", "bytes": size}, {"kind": "all_gather", "bytes": size}))
        elif params:
            matrix_bytes = params[0].numel() * params[0].element_size()
            chunk = (len(params) + world_size - 1) // world_size
            padded_bytes = chunk * world_size * matrix_bytes
            useful_bytes = len(params) * matrix_bytes
            counts["reduce_scatter"] += 1
            counts["all_gather"] += 1
            payload["reduce_scatter_input_bytes"] += padded_bytes
            payload["all_gather_output_bytes"] += padded_bytes
            plans.extend(({"kind": "reduce_scatter", "bytes": padded_bytes}, {"kind": "all_gather", "bytes": padded_bytes}))
            plans[-2]["padding_bytes"] = padded_bytes - useful_bytes
    return {
        "mode": "baseline",
        "planned_collective_counts": counts,
        "planned_collective_count": sum(counts.values()),
        "planned_payload_bytes": payload,
        "planned_padding_bytes": sum(plan.get("padding_bytes", 0) for plan in plans),
        "control_collective_count": 0,
        "control_payload_bytes": 0,
        "planned_collectives": plans,
        "gradient_collective_count": counts["all_reduce"] + counts["reduce_scatter"],
        "gather_collective_count": counts["all_gather"],
        "collective_count": sum(counts.values()),
        "reduction_input_bytes": payload["all_reduce_bytes"] + payload["reduce_scatter_input_bytes"],
        "gather_output_bytes": payload["all_gather_output_bytes"],
        "padding_bytes": sum(plan.get("padding_bytes", 0) for plan in plans),
        "persistent_bytes": 0,
        "ring_send_bytes_estimate_per_rank": (world_size - 1) / world_size * (
            2 * payload["all_reduce_bytes"] + payload["reduce_scatter_input_bytes"] + payload["all_gather_output_bytes"]),
        "assumption": ("first-update active set remains fixed" if active_ids is not None else
                       "all parameters active; replaced with first-update active metadata before measurement"),
    }


def install_baseline_timing(optimizer, state: dict[str, Any]) -> None:
    """Profile baseline launch and Work.wait calls without adding stream waits."""
    import torch
    from types import MethodType

    class TimedWork:
        def __init__(self, work, label):
            self.work = work
            self.label = label

        def wait(self, *args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            before = time.perf_counter()
            try:
                return self.work.wait(*args, **kwargs)
            finally:
                state["baseline_current"][self.label + "_wait_exposed_host_ms"] += (time.perf_counter() - before) * 1000
                end.record()
                state["baseline_current"][self.label + "_wait_stream_events"].append((start, end))

    def wrap_gradient_works(info):
        if not info:
            return
        if info.get("work") is not None:
            info["work"] = TimedWork(info["work"], "gradient")
        for pinfo in info.get("param_infos", {}).values():
            if pinfo.get("work") is not None:
                pinfo["work"] = TimedWork(pinfo["work"], "gradient")
        for bucket in info.get("buckets", []):
            if bucket.get("work") is not None:
                bucket["work"] = TimedWork(bucket["work"], "gradient")

    backend = optimizer.backend
    for strategy in backend.strategies.values():
        original_reduce = strategy.dist_reduce

        def timed_reduce(group, world_size, original=original_reduce):
            before = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(group, world_size)
            end.record()
            current = state["baseline_current"]
            current["pack_and_launch_host_ms"] += (time.perf_counter() - before) * 1000
            current["pack_and_launch_stream_events"].append((start, end))
            wrap_gradient_works(result)
            return result

        strategy.dist_reduce = timed_reduce

    original_finish = backend._finish_gathers

    def timed_finish(self, gather_list):
        current = state["baseline_current"]
        for info in gather_list:
            if info.get("work") is not None:
                info["work"] = TimedWork(info["work"], "gather")
        before = time.perf_counter()
        try:
            return original_finish(gather_list)
        finally:
            current["gather_wait_and_copy_host_ms"] += (time.perf_counter() - before) * 1000

    backend._finish_gathers = MethodType(timed_finish, backend)


def device_metadata(trainer, seed: int) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from importlib import import_module

    info = trainer.config.comp_info
    result: dict[str, Any] = {
        "rank": int(info.rank),
        "world_size": int(info.world_size),
        "device": str(info.device),
        "device_type": str(info.device_type),
        "dtype": str(trainer.dtype),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": (torch.cuda.nccl.version() if torch.cuda.is_available() else None),
        "distributed_backend": (dist.get_backend() if dist.is_available() and dist.is_initialized() else None),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(info.device)
        result["gpu"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        }
    config = trainer.model.config
    result["model_config"] = {
        key: getattr(config, key, None)
        for key in ("n_layers", "d_model", "n_heads", "n_kv_heads", "d_head", "vocab_size", "max_context", "attn_impl")
    }
    result["model_config"]["ffn_class"] = type(trainer.model.blocks[0].ffn).__name__
    result["trainer_config"] = {
        "scheduler_horizon": int(trainer.config.n_steps),
        "stop_after_step": trainer.config.stop_after_step,
        "n_acc_steps": int(trainer.config.n_acc_steps),
        "device_batch_size": int(trainer.config.device_batch_size),
        "total_batch_size": int(trainer.config.total_batch_size),
        "seed": seed,
    }
    flash_attention_module = import_module("gpt_lab.model.flash_attn")
    result["attention_kernels"] = {
        "fa2_loaded": flash_attention_module._flash_attn2 is not None,
        "fa3_loaded": flash_attention_module._flash_attn is not None,
    }
    return result


def tensor_digest(tensor) -> str:
    import torch
    cpu = tensor.detach().contiguous().cpu().reshape(-1)
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def named_tensors_digest(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(bytes.fromhex(tensor_digest(tensor)))
    return digest.hexdigest()


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "value"):
        return jsonable(value.value)
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return repr(value)


def optimizer_group_snapshot(optimizer, model) -> list[dict[str, Any]]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    snapshots = []
    for group in optimizer.param_groups:
        snapshot = {key: jsonable(value) for key, value in group.items() if key != "params"}
        snapshot["parameter_names"] = [parameter_names.get(id(parameter), "<unnamed>") for parameter in group["params"]]
        snapshot["parameter_count"] = len(group["params"])
        snapshots.append(snapshot)
    return snapshots


def initial_optimizer_digest(optimizer, group_snapshot, model) -> str:
    saved = optimizer.state_dict()
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    named_state = {}
    for live_group, saved_group in zip(optimizer.param_groups, saved["param_groups"]):
        for parameter, saved_id in zip(live_group["params"], saved_group["params"]):
            state = saved["state"].get(saved_id, {})
            name = parameter_names.get(id(parameter), str(saved_id))
            named_state[name] = jsonable_optimizer_state(state)
    body = json.dumps({"groups": group_snapshot, "state": named_state}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def jsonable_optimizer_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": tensor_digest(value), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(key): jsonable_optimizer_state(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [jsonable_optimizer_state(item) for item in value]
    return jsonable(value)


def data_position(state):
    if state is None:
        return None
    if hasattr(state, "model_dump"):
        return jsonable(state.model_dump(mode="json"))
    return jsonable(state)


def batch_digest(payload) -> str:
    import torch
    digest = hashlib.sha256()
    values = payload if isinstance(payload, (tuple, list)) else (payload,)
    for index, value in enumerate(values):
        digest.update(str(index).encode())
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(bytes.fromhex(tensor_digest(value)))
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def install_runner(args: argparse.Namespace, output_dir: Path, seed: int, resolved_cli: dict[str, Any]) -> None:
    """Install lightweight hooks before dispatching the normal CLI workflow."""
    from gpt_lab.train.trainer import Trainer
    import torch
    import torch.distributed as dist

    state: dict[str, Any] = {
        "steps": [], "optimizer_events": [], "baseline_profile_steps": [],
        "step_wall_starts": [], "step_wall_intervals": [],
        "train_started": None, "first_step": None,
    }
    original_init = Trainer.__init__
    original_train = Trainer.train
    original_train_step = Trainer.train_step
    original_optimizer_step = Trainer.optimizer_step
    original_prepare_batch = Trainer.prepare_batch
    Trainer.save_checkpoint = lambda self: None
    from gpt_lab.model.checkpoint import CheckpointManager
    CheckpointManager.save_training_config = lambda self, trainer_config: None

    def init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        info = self.config.comp_info
        if not torch.cuda.is_available() or info.device_type != "cuda":
            raise RuntimeError("communication performance diagnostics require CUDA; CPU/MPS runs cannot measure NCCL behavior")
        if not dist.is_initialized() or info.world_size < 2 or dist.get_backend() != "nccl":
            raise RuntimeError("communication performance diagnostics require a multi-GPU NCCL process group (launch with torchrun)")
        graph_mode = getattr(self.config, "cuda_graph_mode", None)
        graph_capture = graph_mode not in (None, False, "none", "disabled") or getattr(self, "_forward_backward_graph", None) is not None
        if graph_capture:
            raise ValueError("communication comparisons require eager forward/backward execution without CUDA graph capture")
        if self.dtype != torch.bfloat16 or self.scaler is not None:
            raise ValueError("communication comparisons require BF16 compute without gradient scaling")
        gpu = torch.cuda.get_device_properties(info.device)
        if not gpu.name.startswith("NVIDIA A100"):
            raise ValueError(f"fixed communication diagnostics require an A100, got {gpu.name}")
        if (self.model.config.n_layers != 12 or self.model.config.d_head != 128
                or self.model.config.max_context != 2048 or self.model.config.attn_impl != "fused"):
            raise ValueError("resolved model differs from the fixed depth-12/head-128/context-2048 diagnostic architecture")
        from gpt_lab.model.layers import SwigLUFeedForward
        if len(self.model.blocks) != 12 or any(type(block.ffn) is not SwigLUFeedForward for block in self.model.blocks):
            raise ValueError("fixed communication diagnostics require SwigLUFeedForward in all 12 transformer blocks")
        from importlib import import_module
        flash_attention_module = import_module("gpt_lab.model.flash_attn")
        if flash_attention_module._flash_attn2 is None or flash_attention_module._flash_attn is not None:
            raise RuntimeError("A100 diagnostics require loaded FlashAttention 2 and no FlashAttention 3")
        transformer_groups = [group for group in self.optimizer.param_groups if str(group.get("name", "")).startswith("block.")]
        if not transformer_groups or any(group.get("opt") != args.optimizer for group in transformer_groups):
            raise ValueError(f"resolved transformer optimizer groups do not use selected {args.optimizer}")
        configure = getattr(self.optimizer, "configure_communication", None)
        if configure is None:
            raise RuntimeError("optimizer does not provide configure_communication(); use the communication experiment branch")
        configure(
            mode=args.communication,
            bucket_bytes=int(args.bucket_mib * 1024 * 1024),
            timing=args.profile is not None,
        )
        if args.communication == "backward" and not callable(getattr(self.optimizer, "begin_backward_communication", None)):
            raise RuntimeError("backward mode requires optimizer.begin_backward_communication and the Trainer backward call site")
        state["metadata"] = device_metadata(self, seed)
        state["metadata"]["selected_optimizer"] = args.optimizer
        state["metadata"]["optimizer_groups"] = optimizer_group_snapshot(self.optimizer, self.model)
        print(f"[rank {info.rank}] selected optimizer={args.optimizer}; resolved groups="
              + json.dumps(state["metadata"]["optimizer_groups"], sort_keys=True), flush=True)
        state["metadata"]["initial_model_sha256"] = named_tensors_digest(self.model.state_dict().items())
        state["metadata"]["initial_optimizer_state_sha256"] = initial_optimizer_digest(
            self.optimizer, state["metadata"]["optimizer_groups"], self.model,
        )
        state["communication"] = (
            baseline_communication_stats(self.optimizer, info.world_size)
            if args.communication == "baseline"
            else self.optimizer.communication.stats()
        )
        if args.communication == "baseline" and args.profile is not None:
            install_baseline_timing(self.optimizer, state)

    def prepare_batch(self, raw_batch):
        prepared = original_prepare_batch(self, raw_batch)
        if state.get("first_batch_sha256") is None:
            payload, position, _ = prepared
            state["first_batch_sha256"] = batch_digest(payload)
            state["first_batch_position"] = data_position(position)
        return prepared

    def optimizer_step(self, should_log_step):
        if args.communication == "baseline" and not state.get("baseline_active_checked", False):
            active_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None}
            state["communication"] = baseline_communication_stats(self.optimizer, self.config.comp_info.world_size, active_ids)
            state["baseline_active_checked"] = True
        if args.profile is None or self.device.type != "cuda":
            return original_optimizer_step(self, should_log_step)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if args.communication == "baseline":
            state["baseline_current"] = {
                "pack_and_launch_host_ms": 0.0,
                "pack_and_launch_stream_events": [],
                "gradient_wait_exposed_host_ms": 0.0,
                "gradient_wait_stream_events": [],
                "gather_wait_exposed_host_ms": 0.0,
                "gather_wait_stream_events": [],
                "gather_wait_and_copy_host_ms": 0.0,
            }
        start.record()
        result = original_optimizer_step(self, should_log_step)
        end.record()
        state["optimizer_events"].append((start, end))
        if args.communication == "baseline":
            state["baseline_profile_steps"].append(state.pop("baseline_current"))
        return result

    def train_step(self, *step_args, **step_kwargs):
        step_started = time.perf_counter()
        if state["step_wall_starts"]:
            state["step_wall_intervals"].append(step_started - state["step_wall_starts"][-1])
        state["step_wall_starts"].append(step_started)
        result = original_train_step(self, *step_args, **step_kwargs)
        _, metrics = result
        if state["first_step"] is None:
            state["first_step"] = time.perf_counter()
            state["process_startup_to_first_update_s"] = state["first_step"] - PROCESS_STARTED_AT
        state["steps"].append({
            "tokens": int(metrics["tokens"]),
            "step_time_s": float(metrics["step_time"]),
            "last_consumed_data_position": data_position(self.state.train_loader_state),
        })
        if torch.cuda.is_available() and len(state["steps"]) == args.warmup_updates and args.warmup_updates > 0:
            torch.cuda.reset_peak_memory_stats(self.device)
            state["memory_after_warmup"] = {
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
            }
        profiler = state.get("profiler")
        if profiler is not None:
            profiler.step()
        return result

    def train(self):
        state["train_started"] = time.perf_counter()
        profiler = None
        if args.profile is not None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            rank = int(self.config.comp_info.rank)
            trace_path = output_dir / ("communication-trace.json" if args.profile == "auto" else args.profile)
            trace_path = trace_path.with_name(f"{trace_path.stem}.rank{rank}{trace_path.suffix or '.json'}")
            if trace_path.exists():
                raise FileExistsError(f"profile trace already exists: {trace_path}")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=args.warmup_updates, warmup=1, active=3, repeat=1),
                on_trace_ready=lambda active: active.export_chrome_trace(str(trace_path)),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            profiler.__enter__()
            state["profiler"] = profiler
        if args.warmup_updates == 0:
            torch.cuda.reset_peak_memory_stats(self.device)
            state["memory_after_warmup"] = {
                "allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device),
            }
        try:
            return original_train(self)
        finally:
            error = sys.exc_info()[1]
            if error is not None:
                # A peer may still be in a training collective: do not gather summaries.
                rank = int(self.config.comp_info.rank)
                failure = dict(status="failed", rank=rank, error_type=type(error).__name__,
                               completed_steps=int(self.state.completed_steps))
                (output_dir / f"communication-error.rank{rank}.json").write_text(json.dumps(failure, indent=2) + "\n")
                if profiler is not None:
                    profiler.__exit__(*sys.exc_info())
                raise
            if profiler is not None:
                profiler.__exit__(None, None, None)
            end_time = time.perf_counter()
            # original_train_step ends at Trainer's established synchronization
            # boundary, so CUDA event results are ready without another sync.
            optimizer_ms = [start.elapsed_time(end) for start, end in state["optimizer_events"]]
            if state["step_wall_starts"]:
                state["step_wall_intervals"].append(end_time - state["step_wall_starts"][-1])
            local_steps = state["steps"]
            local_wall = end_time - state["train_started"]
            local_startup = ((state["first_step"] - state["train_started"]) if state["first_step"] else local_wall)
            comm = getattr(self.optimizer, "communication", None)
            comm_stats = comm.stats() if comm is not None else state.get("communication")
            if args.communication == "baseline" and args.profile is not None:
                for measurement in state["baseline_profile_steps"]:
                    measurement["pack_and_launch_stream_ms"] = sum(start.elapsed_time(end) for start, end in measurement.pop("pack_and_launch_stream_events"))
                    for label in ("gradient", "gather"):
                        measurement[label + "_wait_stream_ms"] = sum(start.elapsed_time(end) for start, end in measurement.pop(label + "_wait_stream_events"))
                comm_stats["profile_per_update"] = state["baseline_profile_steps"]
            memory = dict(state.get("memory_after_warmup", {}))
            memory.update({
                "peak_allocated_bytes_after_warmup": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes_after_warmup": torch.cuda.max_memory_reserved(self.device),
                "final_allocated_bytes": torch.cuda.memory_allocated(self.device),
                "final_reserved_bytes": torch.cuda.memory_reserved(self.device),
            })
            local = {
                "rank": int(self.config.comp_info.rank),
                "metadata": state["metadata"],
                "first_batch_sha256": state.get("first_batch_sha256"),
                "first_batch_position": state.get("first_batch_position"),
                "steps": local_steps,
                "loop_wall_step_time_s": state["step_wall_intervals"],
                "train_wall_s": local_wall,
                "startup_to_first_update_s": local_startup,
                "process_startup_to_first_update_s": state.get("process_startup_to_first_update_s"),
                "optimizer_event_ms": optimizer_ms,
                "communication": comm_stats,
                "memory": memory,
                "completed_steps": int(self.state.completed_steps),
                "scheduler_horizon": int(self.config.n_steps),
            }
            if dist.is_available() and dist.is_initialized():
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local)
            else:
                gathered = [local]
            if self.config.comp_info.rank == 0:
                count = min(len(item["steps"]) for item in gathered)
                timed = range(min(args.warmup_updates, count), count)
                max_step_times = [max(item["steps"][index]["step_time_s"] for item in gathered) for index in timed]
                token_counts = [max(item["steps"][index]["tokens"] for item in gathered) for index in timed]
                steady_seconds = sum(max_step_times)
                total_tokens = sum(max(item["steps"][index]["tokens"] for item in gathered) for index in range(count))
                train_wall = max(item["train_wall_s"] for item in gathered)
                loop_wall_count = min(min(len(item["loop_wall_step_time_s"]) for item in gathered), count)
                loop_wall_times = [
                    max(item["loop_wall_step_time_s"][index] for item in gathered)
                    for index in range(min(args.warmup_updates, loop_wall_count), loop_wall_count)
                ]
                loop_wall_tokens = [
                    max(item["steps"][index]["tokens"] for item in gathered)
                    for index in range(min(args.warmup_updates, loop_wall_count), loop_wall_count)
                ]
                steady_optimizer = [
                    max(item["optimizer_event_ms"][index] for item in gathered if len(item["optimizer_event_ms"]) > index)
                    for index in range(min(args.warmup_updates, min(len(item["optimizer_event_ms"]) for item in gathered)),
                                       min(len(item["optimizer_event_ms"]) for item in gathered))
                ]
                communication_rows = [item["communication"] for item in gathered]
                numeric_comm_keys = (
                    "gradient_collective_count", "gather_collective_count", "collective_count",
                    "reduction_input_bytes", "padding_bytes", "gather_output_bytes", "persistent_bytes",
                    "planned_collective_count", "planned_padding_bytes", "control_collective_count",
                    "ring_send_bytes_estimate_per_rank",
                    "control_payload_bytes",
                )
                communication_summary = {
                    key: max(row[key] for row in communication_rows if key in row)
                    for key in numeric_comm_keys if any(key in row for row in communication_rows)
                }
                if communication_rows and "buckets" in communication_rows[0]:
                    per_kind = {}
                    for kind in ("all_reduce", "reduce_scatter"):
                        per_kind[kind] = max(
                            sum(bucket["kind"] == kind for bucket in row["buckets"])
                            for row in communication_rows
                        )
                    communication_summary["gradient_collective_types"] = per_kind
                if communication_rows and "planned_collective_counts" in communication_rows[0]:
                    communication_summary["planned_collective_types"] = {
                        kind: max(row["planned_collective_counts"].get(kind, 0) for row in communication_rows)
                        for kind in ("all_reduce", "reduce_scatter", "all_gather")
                    }
                    communication_summary["planned_payload_bytes"] = {
                        key: max(row["planned_payload_bytes"].get(key, 0) for row in communication_rows)
                        for key in ("all_reduce_bytes", "reduce_scatter_input_bytes", "all_gather_output_bytes")
                    }
                per_update_timing = []
                max_updates = max((len(row.get("per_update", [])) for row in communication_rows), default=0)
                for index in range(min(args.warmup_updates, max_updates), max_updates):
                    values = [row["per_update"][index] for row in communication_rows if len(row.get("per_update", [])) > index]
                    per_update_timing.append({
                        "pack_ms": max(value.get("pack_ms", 0.0) for value in values),
                        "unpack_ms": max(value.get("unpack_ms", 0.0) for value in values),
                        "gradient_reduction_wait_ms": max(value.get("wait_ms", 0.0) for value in values),
                        "validation_control_wait_ms": max(value.get("validation_ms", 0.0) for value in values),
                        "gather_wait_and_copy_ms": max(value.get("gather_wait_and_copy_ms", 0.0) for value in values),
                        "host_pack_ms": max(value.get("host_pack_ms", 0.0) for value in values),
                        "host_unpack_ms": max(value.get("host_unpack_ms", 0.0) for value in values),
                        "host_launch_ms": max(value.get("host_launch_ms", 0.0) for value in values),
                    })
                if per_update_timing:
                    communication_summary["per_update_timing_rank_max_ms"] = per_update_timing
                    communication_summary["timing_warmup_updates_excluded"] = args.warmup_updates
                baseline_profile = [row.get("profile_per_update", []) for row in communication_rows]
                baseline_count = min((len(rows) for rows in baseline_profile), default=0)
                baseline_profile_summary = []
                for index in range(min(args.warmup_updates, baseline_count), baseline_count):
                    samples = [rows[index] for rows in baseline_profile]
                    names = (
                        "pack_and_launch_host_ms", "pack_and_launch_stream_ms",
                        "gradient_wait_exposed_host_ms", "gradient_wait_stream_ms",
                        "gather_wait_exposed_host_ms", "gather_wait_stream_ms",
                        "gather_wait_and_copy_host_ms",
                    )
                    baseline_profile_summary.append({name: max(sample.get(name, 0.0) for sample in samples) for name in names})
                if baseline_profile_summary:
                    communication_summary["baseline_profile_per_update_rank_max_ms"] = baseline_profile_summary
                    communication_summary["baseline_profile_wait_labels"] = {
                        "gradient_wait_exposed_host_ms": "time blocked in baseline gradient Work.wait calls",
                        "gather_wait_exposed_host_ms": "time blocked in baseline gather Work.wait calls",
                        "gather_wait_and_copy_host_ms": "total baseline gather drain and parameter-copy wall time",
                    }
                optimizer_means = [
                    (sum(samples) / len(samples) if samples else None)
                    for item in gathered
                    for samples in [item["optimizer_event_ms"][args.warmup_updates:]]
                ]
                optimizer_maxima = [max(item["optimizer_event_ms"][args.warmup_updates:], default=None) for item in gathered]
                git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                result = {
                    "status": "completed" if all(item["completed_steps"] == args.updates for item in gathered) else "incomplete",
                    "communication": args.communication,
                    "optimizer": args.optimizer,
                    "selected_optimizer": args.optimizer,
                    "seed": seed,
                    "updates_requested": args.updates,
                    "bucket_mib": args.bucket_mib,
                    "profiled": args.profile is not None,
                    "warmup_updates_excluded": args.warmup_updates,
                    "steady_updates": len(max_step_times),
                    "steady_tokens": sum(token_counts),
                    "steady_step_time_rank_max_s": max_step_times,
                    "steady_train_step_time_ms": nearest_rank_summary(max_step_times),
                    "steady_tokens_per_second": (sum(token_counts) / steady_seconds if steady_seconds else None),
                    "steady_train_loop_step_time_rank_max_s": loop_wall_times,
                    "steady_wall_step_time_ms": nearest_rank_summary(loop_wall_times),
                    "steady_train_loop_tokens_per_second": (
                        sum(loop_wall_tokens) / sum(loop_wall_times) if sum(loop_wall_times) else None
                    ),
                    "train_wall_rank_max_s": train_wall,
                    "train_wall_tokens_per_second": (total_tokens / train_wall if train_wall else None),
                    "startup_to_first_update_rank_max_s": max(item["startup_to_first_update_s"] for item in gathered),
                    "process_startup_to_first_update_rank_max_s": max(item["process_startup_to_first_update_s"] for item in gathered),
                    "startup_timing_note": "includes train-loop setup, model compilation, prefetch, and work through the first completed update",
                    "step_time_percentile_method": "nearest rank: sorted sample at ceil(p * N), reported in milliseconds",
                    "steady_optimizer_event_ms_rank_max_per_update": steady_optimizer,
                    "steady_optimizer_event_ms_rank_max_mean": max((value for value in optimizer_means if value is not None), default=None),
                    "steady_optimizer_event_ms_rank_max_max": max((value for value in optimizer_maxima if value is not None), default=None),
                    "communication_rank_max_summary": communication_summary,
                    "control_collective_note": "bucket modes add one 4-byte gradient-layout validation all-reduce per optimizer update; it is reported separately from gradient reductions",
                    "communication_per_rank": communication_rows,
                    "memory_per_rank": [item["memory"] for item in gathered],
                    "metadata_per_rank": [item["metadata"] for item in gathered],
                    "optimizer_groups_per_rank": [item["metadata"]["optimizer_groups"] for item in gathered],
                    "initial_fingerprints_per_rank": [
                        {
                            "rank": item["rank"],
                            "model_sha256": item["metadata"]["initial_model_sha256"],
                            "optimizer_state_sha256": item["metadata"]["initial_optimizer_state_sha256"],
                            "optimizer_groups": item["metadata"]["optimizer_groups"],
                            "first_batch_sha256": item["first_batch_sha256"],
                            "first_batch_position": item["first_batch_position"],
                            "last_consumed_data_positions": [step["last_consumed_data_position"] for step in item["steps"]],
                        }
                        for item in gathered
                    ],
                    "resolved_cli": resolved_cli,
                    "optimizer_profile": str(ROOT / "configs" / ("optim.yaml" if args.optimizer == "muon" else "normuon.yaml")),
                    "completed_steps_by_rank": [item["completed_steps"] for item in gathered],
                    "scheduler_horizon_by_rank": [item["scheduler_horizon"] for item in gathered],
                    "world_size": len(gathered),
                    "git_sha": git_sha,
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                }
                (output_dir / "communication-results.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2), flush=True)

    Trainer.__init__ = init
    Trainer.train = train
    Trainer.train_step = train_step
    Trainer.optimizer_step = optimizer_step
    Trainer.prepare_batch = prepare_batch


def main(argv: list[str] | None = None) -> int:
    args, forwarded = parse_args(list(sys.argv[1:] if argv is None else argv))
    seed = 42
    root_tmp = ROOT / ".tmp" / "codex" / "comm-buckets"
    output_dir = (args.output_dir or root_tmp / f"{args.communication}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}").resolve()
    if (output_dir / "communication-results.json").exists():
        raise SystemExit(f"refusing to overwrite existing results: {output_dir / 'communication-results.json'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch._inductor.config.triton.cudagraphs = False
    if not torch.cuda.is_available():
        raise SystemExit("communication performance diagnostics require CUDA/NCCL; no CUDA device is available")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise SystemExit("communication performance diagnostics require at least two GPUs; launch with torchrun")
    os.environ["GPTLAB_DTYPE"] = "bfloat16"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from datetime import timedelta
    from functools import partial
    import torch.distributed as dist
    dist.init_process_group = partial(dist.init_process_group, timeout=timedelta(seconds=300))

    # Apply reference defaults, then user overrides, then matched-run invariants.
    cli_args = [*default_cli_args(output_dir), *forwarded, *invariant_cli_args(args.optimizer, args.updates)]
    from gpt_lab.cli.parser import build_parser
    resolved_cli = vars(build_parser().parse_args(["train", "auto", *cli_args]))
    resolved_cli.pop("_handler", None)
    install_runner(args, output_dir, seed, resolved_cli)
    from gpt_lab.cli import main as cli_main
    return cli_main(["train", "auto", *cli_args])


if __name__ == "__main__":
    raise SystemExit(main())
