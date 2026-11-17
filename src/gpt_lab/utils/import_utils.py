from packaging import version
from functools import lru_cache
from gpt_lab.utils.logging import log_all
import logging
import traceback
import inspect
import re

logger = logging.getLogger(__name__)


def validate_attn_impl(value: str) -> str:
    if value in ("sdpa", "fa2", "fa3") or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        return value
    raise ValueError("attn_impl must be sdpa, fa2, fa3, or a kernel repository name (owner/name)")


_attention_kernels = {}


def get_attention_kernel(backend: str):
    # Resolve once before torch.compile; forwards only read the loaded module.
    if backend in _attention_kernels:
        return _attention_kernels[backend]
    import kernels

    if backend == "fa3":
        kernel = kernels.get_kernel("kernels-community/flash-attn3", version=2)
    elif backend == "fa2":
        kernel = kernels.get_kernel("kernels-community/flash-attn2", version=4)
    else:
        validate_attn_impl(backend)
        if "/" not in backend:
            raise ValueError(f"Not a Hub attention kernel: {backend}")
        # Selecting a repository explicitly permits loading that repository's code.
        kernel = kernels.get_kernel(backend, revision="main", trust_remote_code=[backend])
        kernel._gpt_lab_attention_options = {}
        for name in ("flash_attn_func", "flash_attn_qkvpacked_func"):
            function = getattr(kernel, name, None)
            if callable(function):
                parameters = inspect.signature(function).parameters
                options = set(parameters)
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                    options.update(("dropout_p", "alibi_slopes"))
                kernel._gpt_lab_attention_options[name] = options
    if not callable(getattr(kernel, "flash_attn_func", None)):
        raise RuntimeError(f"Attention kernel {backend!r} must export flash_attn_func")
    _attention_kernels[backend] = kernel
    return kernel


@lru_cache()
def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

@lru_cache()
def is_torch_cuda_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    return torch.cuda.is_available()

@lru_cache()
def is_torch_mps_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    return torch.backends.mps.is_available()

@lru_cache()
def is_torch_xpu_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    return hasattr(torch, "xpu") and torch.xpu.is_available()

@lru_cache() 
def is_torch_greater_or_equal(vers: str) -> bool:
    if not is_torch_available():
        return False
    import torch
    return version.parse(torch.__version__) >= version.parse(vers)

@lru_cache()
def is_torch_bf16_gpu_available() -> bool:
    if not is_torch_available():
        return False
    import torch
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    if is_torch_xpu_available():
        return torch.xpu.is_bf16_supported()
    if is_torch_mps_available():
        return torch.backends.mps.is_macos_or_newer(14, 0)
    return False


@lru_cache()
def is_flash_attn3_available_from_kernel() -> bool:
    if not is_torch_cuda_available():
        return False
    try: 
        get_attention_kernel("fa3")
        return True
    except Exception:
        log_all(
            "FlashAttention 3 is not available. Falling back to PyTorch SDPA.\n"
            + traceback.format_exc(),
            level="warning", logger=logger,
        )
        return False
