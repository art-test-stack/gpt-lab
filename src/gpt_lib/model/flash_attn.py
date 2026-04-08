from gpt_lib.utils.import_utils import is_torch_cuda_available, is_flash_attn3_available_from_kernel

import torch
import torch.nn.functional as F
import os
from typing import Optional, Union
from types import SimpleNamespace

import math

_flash_attn = None
if is_torch_cuda_available():
    if is_flash_attn3_available_from_kernel():
        try:
            major, _ = torch.cuda_get_device_capability()
            if major >= 9:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                import kernels
                _flash_attn = kernels.get_kernel('varunneal/flash-attention-3').flash_attn_interface
        except:
            pass

flash_attention_is_installed = _flash_attn is not None

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
                   alibi_slopes=None, deterministic=False, enable_gqa=False):
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
    window = window_size[0]
    q = q.transpose(1, 2)  # B, H, Tq, D
    k = k.transpose(1, 2)  # B, H, Tk, D
    v = v.transpose(1, 2)  # B, H, Tk, D
    
    device = q.device
    if Tq == 1:
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=False, scale=softmax_scale
        )
    elif Tq == Tk:
        if window < 0 or window >= Tq:
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p,
                is_causal=True, scale=softmax_scale, enable_gqa=enable_gqa
            )
        else:
            mask = torch.triu(torch.ones((Tq, Tq), device=device), diagonal=1)
            mask = torch.logical_or(mask, torch.tril(torch.ones((Tq, Tq), device=device), diagonal=-window-1))
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p,
                is_causal=False, scale=softmax_scale, enable_gqa=enable_gqa
            )
    else:
        prefix_len = Tk - Tq
        mask = torch.zeros((Tq, Tk), device=device)
        mask = torch.zeros((Tq, Tk), device=device, dtype=torch.bool)
        mask = mask.masked_fill(torch.tril(torch.ones((Tq, Tk), device=device), diagonal=-prefix_len-1) == 1, True)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, 
            is_causal=False, scale=softmax_scale, enable_gqa=enable_gqa
        )
    return output.transpose(1, 2)  # B, Tq, H, D


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                          window_size=(-1, -1), alibi_slopes=None, deterministic=False):
    if flash_attention_is_installed:
        return _flash_attn.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            # alibi_slopes=alibi_slopes,
            # deterministic=deterministic
        )
    assert alibi_slopes is None, "Alibi slopes are not supported when FlashAttention is not installed."
    B, T, H, D_head_times_3 = qkv.size()
    q, k, v = qkv.split(D_head_times_3 // 3, dim=-1)
    
    return _sdpa_fallback(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        # deterministic=deterministic
    )

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False):
    """
    Args:
        q: (B, Tq, H, D)
        k: (B, Tk, H, D)
        v: (B, Tk, H, D)
        ... 
    Returns:
        output: (B, Tq, H, D)
    """
    if flash_attention_is_installed:
        return _flash_attn.flash_attn_func(
            q, k,v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic
        )
    return _sdpa_fallback(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        # causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic
    )

def flash_attn_with_kvcache(
    q: torch.Tensor, # (B, Tq, H, D)
    k_cache: torch.Tensor, # (Bc, Tc, Hkv, D)
    v_cache: torch.Tensor, # (Bc, Tc, Hkv, D)
    k: Optional[torch.Tensor] = None, # (B, Tk, Hkv, D)
    v: Optional[torch.Tensor] = None, # (B, Tk, Hkv, D)
    rotary_cos=None, # ignored. handled outside
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means 'infinite' context window
    rotary_interleaved=True,
    alibi_slopes=None,
):  
    if (k is not None) and (v is not None):
        assert q.device == k.device == v.device, "q, k and v are expected to be on the same device"
    assert q.device == k_cache.device == v_cache.device, "q, k_cache and v_cache are expected to be on the same device"
    if flash_attention_is_installed:
        return _flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k,
            v=v,
            # rotary_cos=rotary_cos,
            # rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            # block_table=block_table,
            softmax_scale=softmax_scale,
            # causal=causal,
            window_size=window_size,
            # rotary_interleaved=rotary_interleaved,
            # alibi_slopes=alibi_slopes
        )
    Tq = q.size(1)
    
    cur_pos = cache_seqlens[0].item() # TODO: change for batch support -> efficiently get max cur_pos
    end_pos = cur_pos + Tq

    if k is not None and v is not None:
        k_cache[:,cur_pos:end_pos,:,:] = k
        v_cache[:,cur_pos:end_pos,:,:] = v
    
    k = k_cache[:,:end_pos,:,:]
    v = v_cache[:,:end_pos,:,:]
    enable_gqa = (k.size(-2) != q.size(-2)) # GQA if Hq != Hkv
    return _sdpa_fallback(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        window_size=window_size,
        enable_gqa=enable_gqa
    )
    
    

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_qkvpacked_func=flash_attn_qkvpacked_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    flash_attention_is_installed=flash_attention_is_installed
)