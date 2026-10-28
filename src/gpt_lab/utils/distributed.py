from gpt_lab.utils.logging import log0
import torch
import torch.distributed as dist
from torch import nn
import os, warnings
import json
import subprocess
from functools import lru_cache, cached_property
from dataclasses import dataclass, field
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# dtype detection and management
# -----------------------------------------------------------

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

@lru_cache()
def detect_compute_dtype():
    env = os.environ.get("GPTLAB_DTYPE", None)
    if env is not None and env != "bfloat16":
        raise ValueError("Only BF16 compute is supported; GPTLAB_DTYPE must be bfloat16")
    return torch.bfloat16, "BF16 compute policy"

# -----------------------------------------------------------
# distributed utils
# -----------------------------------------------------------

@lru_cache()
def get_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.set_per_process_memory_fraction(0.)
        device_type = "mps"
    elif torch.xpu.is_available():
        device_type = "xpu"
    else:
        device_type = "cpu"
    
    if device_type in ("xpu", "tpu"):
        warnings.warn(f"Device type {device_type!r} is not fully tested. May exhibit unexpected behavior.")
    return device_type

def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    return dist.is_initialized() and dist.is_available()

@dataclass(frozen=True)
class ComputeContext:
    device: torch.device
    mesh: DeviceMesh | None = None
    compute_dtype: torch.dtype = field(default_factory=lambda: detect_compute_dtype()[0])
    compute_dtype_reason: str = field(default_factory=lambda: detect_compute_dtype()[1])

    def __post_init__(self) -> None:
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
            if capability < (8, 0):
                raise ValueError(
                    "BF16 compute requires a BF16-capable CUDA device (SM 80 or newer); "
                    f"{self.device} has capability {capability[0]}.{capability[1]}"
                )

    @property
    def rank(self) -> int:
        return self.mesh.get_rank() if self.mesh is not None else 0

    @property
    def world_size(self) -> int:
        return self.mesh.size() if self.mesh is not None else 1

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def is_distributed(self) -> bool:
        return self.mesh is not None

    @property
    def dp_mesh(self) -> DeviceMesh | None:
        return self.mesh["dp"] if self.mesh is not None else None

    @property
    def dp_rank(self) -> int:
        return self.dp_mesh.get_local_rank() if self.dp_mesh is not None else 0

    @property
    def device_type(self) -> str:
        return self.device.type

    @cached_property
    def device_name(self) -> str | None:
        if self.device_type == "cpu":
            return "cpu"
        if self.device_type == "cuda":
            return torch.cuda.get_device_name(self.device)
        if self.device_type == "mps":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True, text=True, check=True, timeout=10,
                )
                for gpu in json.loads(result.stdout)["SPDisplaysDataType"]:
                    if gpu.get("spdisplays_vendor") == "sppci_vendor_Apple":
                        return f"{gpu['sppci_model']} ({gpu.get('sppci_cores', 'unknown')}-core GPU)"
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AttributeError) as error:
                log0(f"Unable to identify MPS GPU: {error}", level="warning", logger=logger)
        return None

    @cached_property
    def gpu_peak_flops(self) -> float:
        if self.device_type in {"cpu", "mps"}:
            # The MPS table entry is FP32; a BF16 peak has not been calibrated.
            return float("inf")
        return get_peak_flops(self.device_name) if self.device_name is not None else float("inf")


def _get_comp_info(device_type: str) -> ComputeContext:
    assert device_type in ("cuda", "mps", "cpu"), f"Unsupported device type: {device_type!r}."
    assert not is_ddp_requested() or int(os.environ["WORLD_SIZE"]) == 1 or is_ddp_initialized(), (
        "Distributed training requested but process group not initialized. Call init_dist_groups() first."
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(device_type, local_rank) if device_type == "cuda" else torch.device(device_type)
    mesh = None
    if is_ddp_initialized():
        if is_ddp_requested():
            assert dist.get_world_size() == int(os.environ["WORLD_SIZE"]), "WORLD_SIZE does not match the initialized process group"
            assert dist.get_rank() == int(os.environ["RANK"]), "RANK does not match the initialized process group"
        if device_type == "mps":
            raise ValueError("Distributed execution requires CPU/Gloo or CUDA/NCCL")
        mesh = init_device_mesh(device_type, (dist.get_world_size(),), mesh_dim_names=("dp",))
    return ComputeContext(device=device, mesh=mesh)


_context: ComputeContext | None = None


def get_comp_info(device_type: str | None = None) -> ComputeContext:
    global _context
    if _context is None or (device_type is not None and device_type != _context.device_type):
        _context = _get_comp_info(device_type or get_device_type())
    return _context

def init_dist_groups(device_type: str | None = None, random_seed: int = 42) -> ComputeContext:
    assert int(os.environ.get("WORLD_SIZE", 1)) > 0, "WORLD_SIZE must be positive"
    device_type = device_type or get_device_type()
    assert device_type in ("cuda", "mps", "cpu"), f"Unsupported device type: {device_type!r}."
    torch.manual_seed(random_seed)
    if device_type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.manual_seed(random_seed)
        torch.set_float32_matmul_precision("high")
    elif device_type == "mps":
        torch.mps.manual_seed(random_seed)

    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        assert is_ddp_requested(), "RANK, LOCAL_RANK and WORLD_SIZE must be set for distributed execution."
        if device_type == "mps":
            raise ValueError("Distributed execution requires CPU/Gloo or CUDA/NCCL")
        if not is_ddp_initialized():
            dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")
    global _context
    _context = None
    return get_comp_info(device_type)

def broadcast_model(model: nn.Module, comp_info: ComputeContext) -> None:
    if not comp_info.is_distributed:
        return

    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

    # TODO
    # torch.manual_seed(random_seed + rank)
    # torch.cuda.manual_seed(random_seed + rank)

def cleanup_dist_groups():
    if is_ddp_initialized():
        dist.destroy_process_group()
    global _context
    _context = None

# from: https://github.com/karpathy/nanochat/blob/1076f97/nanochat/common.py
# hardcoded BF16 peak flops for CUDA GPUs, FP32 for the Apple M2
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    # TODO: Select peak FLOPs by compute dtype; CUDA entries currently assume BF16 and M2 assumes FP32.
    name = device_name.lower()

    # Apple advertises 3.6 TFLOPS for the 10-core M2 GPU; other variants stay unknown.
    # https://www.apple.com/newsroom/images/live-action/wwdc-2022/Apple-WWDC22-M2-SoC-220606_big.jpg.large.jpg
    if name == "apple m2 (10-core gpu)":
        return 3.6e12

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        (["a10g"], 70e12),
        (["a10"], 125e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    log0(f"Peak flops undefined for: {device_name}, MFU will show as 0%", level="warning", logger=logger)
    return float('inf')


base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}

base_model_pp_plan = {
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}
