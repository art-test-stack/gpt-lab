from gpt_lab.utils.logging import log0
import torch
import torch.distributed as dist
import os, warnings
from functools import lru_cache

import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# dtype detection and management
# -----------------------------------------------------------

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

@lru_cache()
def detect_compute_dtype():
    env = os.environ.get("GPTLAB_DTYPE", None)
    if env is not None:
        return _DTYPE_MAP[env], f"set via GPTLAB_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via GPTLAB_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"

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

@lru_cache()
def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    return dist.is_initialized() and dist.is_available()

@lru_cache()
def get_base_dist_info():
    is_requested = is_ddp_requested()
    if is_requested:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = 1
        rank = 0
        local_rank = 0
    tp_size = 1 # TODO: implement tensor parallelism support

    if is_ddp_initialized():
        # sanity check that env vars match dist info
        assert world_size == dist.get_world_size(), f"World size from environment variable WORLD_SIZE={world_size} does not match dist.get_world_size()={dist.get_world_size()}"
        assert rank == dist.get_rank(), f"Rank from environment variable RANK={rank} does not match dist.get_rank()={dist.get_rank()}"
        assert torch.cuda.current_device() == local_rank

    return is_requested, world_size, rank, local_rank, tp_size

@lru_cache()
def get_dist_info(device_type: str | None = None, base_dist_info: tuple | None = None):
    if device_type is None:
        device_type = get_device_type()

    assert device_type in ("cuda", "mps", "cpu"), f"Unsupported device type: {device_type!r}."

    is_initialized = is_ddp_initialized()
    assert (not is_ddp_requested() or is_initialized), "Distributed training requested but process group not initialized. Make sure to call `init_process_group()` before get_dist_info()."

    if base_dist_info is not None:
        is_requested, world_size, rank, local_rank, tp_size = base_dist_info
    else:
        is_requested, world_size, rank, local_rank, tp_size = get_base_dist_info()

    dist_groups = dict(device_type=device_type)

    dist_groups["WORLD_SIZE"] = world_size
    dist_groups["IS_DDP_REQUESTED"] = is_requested
    dist_groups["IS_DDP_INITIALIZED"] = is_initialized
    dist_groups["DP_SIZE"] = world_size // tp_size
    dist_groups["RANK"] = rank
    dist_groups["LOCAL_RANK"] = local_rank
    dist_groups["DEVICE"] = torch.device(device_type, local_rank) if device_type == "cuda" else torch.device(device_type)
    dist_groups["DEVICE_TYPE"] = device_type
    if device_type == "cuda":
        dist_groups["DEVICE_NAME"] = torch.cuda.get_device_name(dist_groups["DEVICE"])
        dist_groups["gpu_peak_flops"] = get_peak_flops(dist_groups["DEVICE_NAME"])
    else:
        dist_groups["DEVICE_NAME"] = None # does not matter
        dist_groups["gpu_peak_flops"] = float('inf') # TODO: implement peak flops estimation for xpu, tpu devices
    if is_initialized and rank == 0:
        logger.info(
            "Initialized distributed training. Distributed groups:\n%s",
            "\n".join(f"{k!r:<15}: {v}" for k, v in dist_groups.items())
        )
    compute_dtype, compute_dtype_reason = detect_compute_dtype()
    dist_groups["compute_dtype"] = compute_dtype
    dist_groups["compute_dtype_reason"] = compute_dtype_reason

    return dist_groups


def init_dist_groups(device_type: str | None = None, random_seed: int = 42):
    if device_type is None:
        device_type = get_device_type()

    assert device_type in ("cuda", "mps", "cpu"), f"Unsupported device type: {device_type!r}."

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)

    if device_type == "cuda":
        torch.cuda.manual_seed(random_seed)
    elif device_type == "mps":
        torch.mps.manual_seed(random_seed)
    
    # Precision
    if device_type == "cuda":
        # https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision("high") 
    
    # Init dist groups
    is_requested, world_size, rank, local_rank, tp_size = get_base_dist_info()

    if is_requested:
        assert all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")), "Environment variables RANK, LOCAL_RANK, and WORLD_SIZE must be set for ddp training."
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier() # synchronize after init
        # check that dist groups are set up correctly
        assert is_ddp_initialized(), "Failed to initialize distributed process group."
    else:
        device = torch.device(device_type)

    dist_info = get_dist_info(device_type=device_type, base_dist_info=(is_requested, world_size, rank, local_rank, tp_size))

    return dist_info

def cleanup_dist_groups():
    if is_ddp_initialized():
        dist.destroy_process_group()

# from: https://github.com/karpathy/nanochat/blob/1076f97/nanochat/common.py
# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

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