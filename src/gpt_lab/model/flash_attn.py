from gpt_lab.utils.import_utils import get_attention_kernel, is_torch_cuda_available

import torch
import torch.nn.functional as F
import os
import warnings
import traceback
from typing import Optional, Union
from types import SimpleNamespace
from functools import partial
from threading import Lock

import math

_flash_attn = None
_flash_attn2 = None
_flash_attn_load_error = None
_flash_attn2_load_error = None
_flash_attn_initialized = False
_flash_attn_init_lock = Lock()
flash_attention_is_installed = False


def _initialize_flash_attention():
    """Serialize first use and publish availability only after loading finishes."""
    global _flash_attn, _flash_attn2, _flash_attn_load_error, _flash_attn2_load_error
    global _flash_attn_initialized, flash_attention_is_installed
    if _flash_attn_initialized:
        return
    with _flash_attn_init_lock:
        if _flash_attn_initialized:
            return
        if is_torch_cuda_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 9:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                    _flash_attn = get_attention_kernel("fa3")
                elif major == 8:
                    try:
                        _flash_attn2 = get_attention_kernel("fa2")
                    except Exception as error:
                        _flash_attn2_load_error = error
                        warnings.warn(f'FlashAttention 2 unavailable; retaining PyTorch SDPA: {error}',
                                      RuntimeWarning, stacklevel=2)
            except Exception as error:
                _flash_attn_load_error = error
                warnings.warn(
                    "FlashAttention 3 backend initialization failed.\n"
                    + traceback.format_exc(),
                    RuntimeWarning, stacklevel=2,
                )
        flash_attention_is_installed = _flash_attn is not None
        flash_attn.flash_attention_is_installed = flash_attention_is_installed
        _flash_attn_initialized = True


def require_flash_attn3():
    _initialize_flash_attention()
    if _flash_attn is None:
        raise RuntimeError(
            "FlashAttention 3 was requested but failed to load; refusing SDPA fallback."
        ) from _flash_attn_load_error
    return _flash_attn


def require_flash_attn2():
    _initialize_flash_attention()
    global _flash_attn2, _flash_attn2_load_error
    if (_flash_attn2 is None and _flash_attn2_load_error is None
            and is_torch_cuda_available() and torch.cuda.get_device_capability()[0] >= 8):
        try:
            _flash_attn2 = get_attention_kernel("fa2")
        except Exception as error:
            _flash_attn2_load_error = error
    if _flash_attn2 is None:
        raise RuntimeError(
            "FlashAttention 2 was requested but failed to load; refusing SDPA fallback."
        ) from _flash_attn2_load_error
    return _flash_attn2


def require_attention_kernel(backend):
    if backend == "fa3":
        return require_flash_attn3()
    if backend == "fa2":
        return require_flash_attn2()
    try:
        return get_attention_kernel(backend)
    except Exception as error:
        raise RuntimeError(f"Attention kernel {backend!r} was requested but failed to load") from error


def scaled_dot_product_attention(
        query: torch.Tensor, # B, ..., Tq, E
        key: torch.Tensor, # B, ..., Tk, E
        value: torch.Tensor, # B, ..., Tk, E
        attn_mask: torch.Tensor | None = None, 
        window_size: Optional[tuple[int, int]] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False, 
        return_attn_weights: bool = False,
        scale: Optional[float] = None, 
        enable_gqa: bool = False,
        device: Optional[torch.device | str] = None
    ) -> torch.Tensor:
    "Only used if it is needed to look at attention weights."
    if device is None:
        device = query.device
    if isinstance(device, str):
        device = torch.device(device)
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, "Query, Key and Value must be 4D tensors."
    assert query.size(-1) == key.size(-1) == value.size(-1), "Last dimension of Query, Key and Value must be the same"
    assert query.device == key.device == value.device, f"Query, Key and Value must be on the same device. Got Query device: {query.device}, Key device: {key.device}, Value device: {value.device}."
    assert query.device == device, f"Q, K, V devices and specified device must be the same. Got Query device: {query.device}, Key device: {key.device}, Value device: {value.device}, specified device: {device}."

    assert (not is_causal) or (attn_mask is None), "`is_causal` cannot be True when `attn_mask` is provided. This behavior is given as a imitation of PyTorch's scaled_dot_product_attention."

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1) # B x 1 x 1 x S
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # B x 1 x L x S
        else:
            assert attn_mask.size(0) == query.size(0) and attn_mask.size(0) == key.size(0), f"Attention mask batch size must match Query and Key batch size. Got attn_mask size: {attn_mask.size()} (B,...), Query size: {query.size()} (B,...), Key size: {key.size()} (B,...)."
            assert attn_mask.size(-2) == query.size(-2), f"Attention mask size must match the sequence length of Query. Got attn_mask size: {attn_mask.size()} (B,...,L,S), Query size: {query.size()} (B,...,Hq,L,E), Key size: {key.size()} (B,...,H,S,E)."
            assert attn_mask.size(-1) == key.size(-2), f"Attention mask size must match the batch size and sequence lengths of Query and Key. Got attn_mask size: {attn_mask.size()} (B,...,L,S), Query size: {query.size()} (B,...,Hq,L,E), Key size: {key.size()} (B,...,H,S,E)."

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_bias = torch.zeros((1, 1, L, S), device=device)

    if is_causal:
        causal = torch.tril(torch.ones(L, S, device=device))
        attn_bias = attn_bias.masked_fill(causal == 0, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_mask

    if enable_gqa:
        if query.size(-3) != key.size(-3):
            raise ValueError("For GQA, the number of query heads must match the number of key heads.")
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p, training=True)
    return attn_weight @ value, (attn_weight if return_attn_weights else None)

def _sdpa_fallback(q, k, v, 
                   dropout_p=0.0, softmax_scale=None, window_size=(-1, -1), 
                   alibi_slopes=None, deterministic=False, use_gqa=False,
                   causal=False):
    """
    Args:
        q: (B, Tq, H, D)
        k: (B, Tk, Hkv, D)
        v: (B, Tk, Hkv, D)
        ...
    Returns:
        output: (B, Tq, H, D)
    """
    assert alibi_slopes is None, "Alibi slopes are not supported in the fallback implementation."
    
    Tq, Tk = q.size(1), k.size(1)
    if window_size is None:
        window_size = (-1, -1)
    window_left, window_right = window_size
    q = q.transpose(1, 2)  # B, H, Tq, D
    k = k.transpose(1, 2)  # B, H, Tk, D
    v = v.transpose(1, 2)  # B, H, Tk, D

    has_local_window = window_left >= 0 or (not causal and window_right >= 0)
    if causal and Tq == Tk and not has_local_window:
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=True, scale=softmax_scale, enable_gqa=use_gqa
        )
    elif not causal and not has_local_window:
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=False, scale=softmax_scale, enable_gqa=use_gqa
        )
    else:
        # FlashAttention aligns a shorter query sequence to the bottom-right of
        # the key sequence. PyTorch boolean masks use True for positions that
        # are allowed to attend (not positions that should be masked out).
        query_positions = torch.arange(Tq, device=q.device) + (Tk - Tq)
        key_positions = torch.arange(Tk, device=q.device)
        mask = torch.ones((Tq, Tk), device=q.device, dtype=torch.bool)
        if causal:
            mask &= key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if window_left >= 0:
            mask &= key_positions.unsqueeze(0) >= (
                query_positions - window_left
            ).unsqueeze(1)
        if window_right >= 0:
            mask &= key_positions.unsqueeze(0) <= (
                query_positions + window_right
            ).unsqueeze(1)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, 
            is_causal=False, scale=softmax_scale, enable_gqa=use_gqa
        )
    return output.transpose(1, 2)  # B, Tq, H, D


def _cast_qkv(q, k, v):
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else q.dtype
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _sdpa_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
               window_size=(-1, -1), alibi_slopes=None, deterministic=False, use_packed=False):
    return _sdpa_fallback(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                          causal=causal, window_size=window_size, alibi_slopes=alibi_slopes,
                          deterministic=deterministic, use_gqa=q.size(-2) != k.size(-2))


def _sdpa_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                         window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                         num_heads_q=None):
    return _sdpa_func(*qkv.unbind(2), dropout_p=dropout_p, softmax_scale=softmax_scale,
                      causal=causal, window_size=window_size, alibi_slopes=alibi_slopes,
                      deterministic=deterministic)


def _fa2_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
              window_size=(-1, -1), alibi_slopes=None, deterministic=False, use_packed=False,
              *, _function, _packed):
    q, k, v = _cast_qkv(q, k, v)
    kwargs = dict(dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                  window_size=window_size or (-1, -1), alibi_slopes=alibi_slopes,
                  deterministic=deterministic)
    if use_packed and _packed is not None and q.shape == k.shape == v.shape:
        return _packed(torch.stack((q, k, v), dim=2), **kwargs)
    return _function(q, k, v, **kwargs)


def _fa2_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                        num_heads_q=None, *, _function):
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else qkv.dtype
    return _function(qkv.to(dtype), dropout_p=dropout_p, softmax_scale=softmax_scale,
                     causal=causal, window_size=window_size, alibi_slopes=alibi_slopes,
                     deterministic=deterministic)


def _fa3_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
              window_size=(-1, -1), alibi_slopes=None, deterministic=False, use_packed=False,
              *, _function, _packed):
    if dropout_p != 0.0 or alibi_slopes is not None:
        raise ValueError("FA3 does not support attention dropout or ALiBi in this integration")
    q, k, v = _cast_qkv(q, k, v)
    kwargs = dict(softmax_scale=softmax_scale, causal=causal,
                  window_size=window_size or (-1, -1), deterministic=deterministic)
    if use_packed and _packed is not None and q.shape == k.shape == v.shape:
        return _packed(torch.stack((q, k, v), dim=2), **kwargs)
    return _function(q, k, v, **kwargs)


def _fa3_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                        num_heads_q=None, *, _function):
    if dropout_p != 0.0 or alibi_slopes is not None:
        raise ValueError("FA3 does not support attention dropout or ALiBi in this integration")
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else qkv.dtype
    return _function(qkv.to(dtype), num_heads_q=num_heads_q, softmax_scale=softmax_scale,
                     causal=causal, window_size=window_size or (-1, -1), deterministic=deterministic)


def _hub_call(*args, dropout_p, alibi_slopes, _function, _options, _backend, **kwargs):
    if 'dropout_p' in _options:
        kwargs['dropout_p'] = dropout_p
    elif dropout_p != 0.0:
        raise ValueError(f"Attention kernel {_backend!r} does not support dropout_p")
    if 'alibi_slopes' in _options:
        kwargs['alibi_slopes'] = alibi_slopes
    elif alibi_slopes is not None:
        raise ValueError(f"Attention kernel {_backend!r} does not support alibi_slopes")
    return _function(*args, **kwargs)


def _hub_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
              window_size=(-1, -1), alibi_slopes=None, deterministic=False, use_packed=False,
              *, _function, _packed):
    q, k, v = _cast_qkv(q, k, v)
    kwargs = dict(dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                  window_size=window_size or (-1, -1), alibi_slopes=alibi_slopes,
                  deterministic=deterministic)
    if use_packed and _packed is not None and q.shape == k.shape == v.shape:
        return _packed(torch.stack((q, k, v), dim=2), **kwargs)
    return _function(q, k, v, **kwargs)


def _check_cache_devices(q, k_cache, v_cache, k, v):
    if k is not None and v is not None:
        assert q.device == k.device == v.device, "q, k and v are expected to be on the same device"
    assert q.device == k_cache.device == v_cache.device, "q, k_cache and v_cache are expected to be on the same device"


def _native_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None,
                         rotary_sin=None, cache_seqlens=None, cache_batch_idx=None,
                         block_table=None, softmax_scale=None, causal=True,
                         window_size=(-1, -1), rotary_interleaved=True, alibi_slopes=None,
                         *, _function):
    _check_cache_devices(q, k_cache, v_cache, k, v)
    # RoPE stays external; the other unused cache options stay ignored, as before.
    return _function(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                     cache_batch_idx=cache_batch_idx, softmax_scale=softmax_scale,
                     causal=causal, window_size=window_size)


def _sdpa_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None,
                       rotary_sin=None, cache_seqlens=None, cache_batch_idx=None,
                       block_table=None, softmax_scale=None, causal=True,
                       window_size=(-1, -1), rotary_interleaved=True, alibi_slopes=None):
    _check_cache_devices(q, k_cache, v_cache, k, v)
    cur_pos = cache_seqlens[0].item()  # Existing cache path assumes equal positions within a batch.
    end_pos = cur_pos + q.size(1)
    if k is not None and v is not None:
        k_cache[:, cur_pos:end_pos] = k
        v_cache[:, cur_pos:end_pos] = v
    return _sdpa_func(q, k_cache[:, :end_pos], v_cache[:, :end_pos],
                      softmax_scale=softmax_scale, causal=causal, window_size=window_size)


def _missing_kvcache(*args, _backend, **kwargs):
    raise RuntimeError(f"Attention kernel {_backend!r} does not support KV-cache decoding")


def _unsupported_packed(*args, _backend, **kwargs):
    raise ValueError(f"Unsupported packed attention backend: {_backend}")


def _unsupported_attention(*args, _backend, **kwargs):
    raise ValueError(f"Attention implementation {_backend} is not supported; use sdpa, fa2, fa3, or owner/kernel.")


def _automatic_ampere_qkvpacked(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                               window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                               num_heads_q=None, *, _function):
    """Retain the legacy implicit FA2 fast path; named model backends never use this."""
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else qkv.dtype
    if (qkv.is_cuda and causal and window_size is not None and tuple(window_size) == (512, 0)
            and dropout_p == 0.0 and alibi_slopes is None and qkv.size(-1) <= 256
            and qkv.dtype in (torch.float32, torch.bfloat16) and dtype == torch.bfloat16):
        return _fa2_qkvpacked_func(qkv, dropout_p, softmax_scale, causal, window_size,
                                   alibi_slopes, deterministic, _function=_function)
    return _sdpa_qkvpacked_func(qkv, dropout_p, softmax_scale, causal, window_size,
                                alibi_slopes, deterministic, num_heads_q)


def _load_flash_attn_functions(backend):
    """Select all execution functions once, after rank-local device setup."""
    _initialize_flash_attention()
    automatic = backend is None
    if automatic:
        backend = 'fa3' if flash_attention_is_installed else 'sdpa'
    if backend in ('sdpa', 'torch', 'raw'):
        separate = _sdpa_func if backend == 'sdpa' else partial(_unsupported_attention, _backend=backend)
        packed = _sdpa_qkvpacked_func if automatic else partial(_unsupported_packed, _backend=backend)
        if automatic and _flash_attn2 is not None:
            packed = partial(_automatic_ampere_qkvpacked, _function=_flash_attn2.flash_attn_qkvpacked_func)
        return dict(
            flash_attn_func=separate, flash_attn_qkvpacked_func=packed,
            flash_attn_with_kvcache=_sdpa_with_kvcache, flash_attention_is_installed=flash_attention_is_installed,
            packed_fa3=False, supports_attn_weights=True,
        )

    kernel = require_attention_kernel(backend)
    function = kernel.flash_attn_func
    packed = getattr(kernel, 'flash_attn_qkvpacked_func', None)
    packed = packed if callable(packed) else None
    packed_call = partial(_unsupported_packed, _backend=backend)
    if backend == 'fa2':
        separate = partial(_fa2_func, _function=function, _packed=packed)
    elif backend == 'fa3':
        separate = partial(_fa3_func, _function=function, _packed=packed)
        packed_call = partial(_fa3_qkvpacked_func, _function=packed)
    else:
        options = kernel._gpt_lab_attention_options
        function = partial(_hub_call, _function=function, _options=options['flash_attn_func'], _backend=backend)
        if callable(packed):
            packed = partial(_hub_call, _function=packed, _options=options['flash_attn_qkvpacked_func'], _backend=backend)
        separate = partial(_hub_func, _function=function, _packed=packed)
    cache = getattr(kernel, 'flash_attn_with_kvcache', None)
    cache_call = (partial(_native_with_kvcache, _function=cache) if callable(cache)
                  else partial(_missing_kvcache, _backend=backend))
    return dict(
        flash_attn_func=separate, flash_attn_qkvpacked_func=packed_call,
        flash_attn_with_kvcache=cache_call, flash_attention_is_installed=flash_attention_is_installed,
        packed_fa3=backend == 'fa3' and callable(packed), supports_attn_weights=False,
    )


def load_flash_attn(backend):
    """Bind attention functions during model initialization, after device setup.

    Args:
        backend: "sdpa", "fa2", "fa3", or a Hub repository ("owner/name").
            None retains the implicit default: FA3 when available, otherwise SDPA.
            Explicit kernel requests raise if loading fails.
            Legacy "torch" and "raw" configs retain SDPA cached evaluation and
            layer-level attention-weight inspection; ordinary forwards raise.

    Returns:
        Namespace exposing flash_attn_func, flash_attn_qkvpacked_func, and
        flash_attn_with_kvcache with the signatures documented below, excluding
        the backend argument. Also exposes packed_fa3, supports_attn_weights,
        and flash_attention_is_installed (whether FA3 loaded successfully).
        Explicit packed calls support FA3; backend=None retains legacy dispatch.
    """
    return SimpleNamespace(**_load_flash_attn_functions(backend))


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                    backend=None, use_packed=False):
    """Compute attention from separate Q/K/V tensors.

    Args:
        q: (batch_size, seqlen_q, nheads_q, headdim).
        k: (batch_size, seqlen_k, nheads_kv, headdim).
        v: Same shape as k. MQA/GQA uses fewer KV heads than query heads;
            nheads_q must be divisible by nheads_kv.
        dropout_p: Attention dropout probability; set to 0.0 during evaluation.
            FA3 requires 0.0.
        softmax_scale: Scale applied to QK^T; None uses 1 / sqrt(headdim).
        causal: Apply a causal mask aligned to the bottom right when sequence
            lengths differ.
        window_size: (left, right) context limits; -1 means unlimited.
            Query i attends within [i + seqlen_k - seqlen_q - left,
            i + seqlen_k - seqlen_q + right], subject to the causal mask.
        alibi_slopes: Optional fp32 slopes of shape (nheads_q,) or
            (batch_size, nheads_q). Unsupported by FA3 and SDPA here.
        deterministic: Request deterministic backward from native kernels;
            ignored by the SDPA wrapper.
        backend: Selection described in load_flash_attn. Hub kernels must
            support any requested non-default dropout or ALiBi options.
        use_packed: Use a kernel's packed entrypoint when available and Q/K/V
            shapes match; otherwise retain separate inputs. Ignored by SDPA.

    Returns:
        Attention output of shape (batch_size, seqlen_q, nheads_q, headdim).

    Compatibility entrypoint; models bind functions once with load_flash_attn.
    """
    return _load_flash_attn_functions(backend)['flash_attn_func'](
        q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes,
        deterministic, use_packed)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                             num_heads_q=None, backend=None):
    """Compute attention from already packed Q/K/V tensors.

    Args:
        qkv: (batch_size, seqlen, 3, nheads, headdim), or the FA3 layout
            (batch_size, seqlen, num_heads_q + 2 * num_heads_kv, headdim).
            The implicit SDPA/FA2 paths require the five-dimensional layout.
        dropout_p: Attention dropout probability; use 0.0 during evaluation.
            FA3 requires 0.0.
        softmax_scale: Scale applied to QK^T; None uses 1 / sqrt(headdim).
        causal: Whether to apply causal attention.
        window_size: (left, right) context limits around each query position;
            -1 means unlimited.
        alibi_slopes: Optional fp32 slopes of shape (nheads,) or
            (batch_size, nheads); unsupported by FA3 and SDPA here.
        deterministic: Request deterministic native backward; ignored by SDPA.
        num_heads_q: Number of query heads for FA3's four-dimensional layout;
            ignored by the implicit SDPA/FA2 paths.
        backend: "fa3" for an explicit packed call. None retains implicit
            FA3/SDPA selection and the legacy Ampere FA2 fast path. Other
            explicit backends raise ValueError before backend loading.

    Returns:
        Attention output of shape (batch_size, seqlen, nheads_q, headdim).

    Models use the corresponding function bound by load_flash_attn.
    """
    if backend not in (None, "fa3"):
        raise ValueError(f"Unsupported packed attention backend: {backend}")
    return _load_flash_attn_functions(backend)['flash_attn_qkvpacked_func'](
        qkv, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, num_heads_q)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None,
                           rotary_sin=None, cache_seqlens=None, cache_batch_idx=None,
                           block_table=None, softmax_scale=None, causal=True,
                           window_size=(-1, -1), rotary_interleaved=True, alibi_slopes=None,
                           backend=None):
    """Attend to a KV cache, optionally appending new keys and values in place.

    Args:
        q: (batch_size, seqlen_q, nheads_q, headdim).
        k_cache: Preallocated (batch_size_cache, seqlen_cache, nheads_kv, headdim).
        v_cache: Same shape as k_cache. The cache must have capacity for any
            appended tokens. MQA/GQA requires nheads_q divisible by nheads_kv.
        k: Optional new keys (batch_size, seqlen_new, nheads_kv, headdim).
        v: Optional new values with the same shape as k; provide both together.
        rotary_cos, rotary_sin: Ignored; apply RoPE before calling this wrapper.
        cache_seqlens: Existing cache lengths passed to native kernels.
            SDPA requires an indexable tensor, reads cache_seqlens[0], and
            assumes equal positions across the batch and seqlen_new == seqlen_q.
            This wrapper does not advance the supplied lengths.
        cache_batch_idx: Optional native cache row indices, typically an int32
            tensor of shape (batch_size,). Ignored by SDPA, which uses batch
            rows directly.
        block_table: Ignored; paged KV caches are not supported by this wrapper.
        softmax_scale: Scale applied to QK^T; None uses 1 / sqrt(headdim).
        causal: Apply causal attention with bottom-right mask alignment.
        window_size: (left, right) context limits as in flash_attn_func.
        rotary_interleaved: Ignored; RoPE is handled externally.
        alibi_slopes: Ignored by this cache wrapper.
        backend: Selection described in load_flash_attn. Explicit native
            backends must export flash_attn_with_kvcache or the call raises.

    Returns:
        Attention output of shape (batch_size, seqlen_q, nheads_q, headdim).
        Native cache kernels are intended for inference without backward.

    Models use the corresponding function bound by load_flash_attn.
    """
    return _load_flash_attn_functions(backend)['flash_attn_with_kvcache'](
        q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens, cache_batch_idx,
        block_table, softmax_scale, causal, window_size, rotary_interleaved, alibi_slopes)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_qkvpacked_func=flash_attn_qkvpacked_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    flash_attention_is_installed=flash_attention_is_installed,
)
