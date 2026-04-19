import torch
import warnings

from gpt_lab.utils.common import print0
from gpt_lab.utils.default import DEVICE
from gpt_lab.utils.schemas import TransformerConfig

from typing import Iterable, Optional


# -------------- Positional Encoding utilities -------------- #

def precompute_rope(seq_len: int, d_head: int, base: int = 10000, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    if not device:
        warnings.warn(f'Device not specified for RoPE precomputation. Using default device {DEVICE}.')
        device = DEVICE
    assert d_head % 2 == 0, 'd_head must be even for RoPE'
    channel_range = torch.arange(0, d_head, 2.0, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (channel_range / d_head))
    pos_seq = torch.arange(0, seq_len, dtype=dtype, device=device)

    sinusoid_inp = torch.einsum('i,j->ij', pos_seq, inv_freq)
    rope_cache = torch.stack((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=-1)
    return rope_cache # seq_len x (d_head/2) x 2

def precompute_positional_encoding(n_pos: int, d_model: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    if not device:
        warnings.warn('Device not specified for positional encoding precomputation. Using default device.')
        device = torch.device(DEVICE)
    pos = torch.arange(n_pos, dtype=dtype, device=device)
    i = torch.arange(d_model, dtype=dtype, device=device)

    pos_enc = torch.ger(pos, 1e4 ** (- 2 * (i//2) / d_model))

    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2]) 
    return pos_enc

def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor, pairwise_split: bool = True) -> torch.Tensor:
    assert x.dim() == 4, 'Input tensor x must be of shape (B, T, H, D). Got shape: {}'.format(x.shape)
    D = x.shape[-1]
    assert D % 2 == 0, 'Embedding dimension D must be even for RoPE. Got D: {}'.format(D)

    sin, cos = rope_cache[..., 0], rope_cache[..., 1]
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.device) # 1, T, 1, D//2
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.device) # 1, T, 1, D//2
    if pairwise_split:
        x1, x2 = x[..., ::2], x[..., 1::2]
    else:
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
    batch, n_heads, seq_len, d_head = x.shape
    assert x1.shape == x2.shape == (batch, n_heads, seq_len, d_head // 2), f'Unexpected shapes: x1 {x1.shape}, x2 {x2.shape}'
    _x1 = x1 * cos - x2 * sin
    _x2 = x1 * sin + x2 * cos

    x_out = torch.empty_like(x)
    x_out[..., ::2] = _x1
    x_out[..., 1::2] = _x2
    return x_out


def apply_positional_encoding(x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
    return pos_enc[:,:x.size(-1)]

# -------------- Attention Mask utilities -------------- #

class SelfAttentionMask:
    def __init__(self, pad_idx: int = -100, max_context: int = 512) -> None:
        self.pad_idx = pad_idx
        self.max_context = max_context
        self.base_mask = torch.tril(torch.ones((max_context, max_context), dtype=torch.bool), diagonal=0) # S x S # [ i >= j ]
        self.base_mask.requires_grad = False
    
    def get(self, seq: torch.Tensor, mask_pad_token: bool = True, to_bool: bool = True, is_causal: bool = True) -> torch.Tensor:
        if not mask_pad_token:
            return None
        
        if seq.dim() == 1:
            warnings.warn("Input sequence tensor has dimension 1. Unsqueezing to dimension 2.")
            seq = seq.unsqueeze(0)
        assert seq.dim() == 2, f"Input sequence tensor must be of dimension 2 (B, S). Got {seq.shape}."
        B, S = seq.shape
        assert S <= self.max_context, f"Sequence length {S} exceeds max_context {self.max_context}. If you want to process longer sequences, please update the max_context using `update_max_context` method."

        device = seq.device
        if mask_pad_token:
            pad_mask = (seq != self.pad_idx) # B x S
        
        if is_causal:
            causal_mask = self.base_mask[:S, :S].to(device)
            print0("causal_mask", causal_mask)
        else:
            warnings.warn("Non-causal attention mask is not yet optimized for large sequences.", UserWarning)
            causal_mask = torch.ones((S, S), dtype=torch.bool, device=device)

        attn_mask = (
            pad_mask.unsqueeze(1).unsqueeze(2) &   # keys
            causal_mask.unsqueeze(0).unsqueeze(1)
        ) # B x 1 x S x S

        if not to_bool:
            attn_mask = (~attn_mask).type(torch.float32)
            attn_mask = attn_mask.masked_fill_(attn_mask.bool(), float("-inf"))

        return attn_mask
    
    def __call__(self, seq: torch.Tensor, mask_pad_token: bool = True, to_bool: bool = True, is_causal: bool = True) -> torch.Tensor:
        return self.get(seq, mask_pad_token=mask_pad_token, to_bool=to_bool, is_causal=is_causal)
    
    def update_max_context(self, max_context: int) -> None:
        if max_context > self.max_context:
            self.base_mask = torch.tril(torch.ones((max_context, max_context), dtype=torch.bool))
            self.base_mask.requires_grad = False
        self.max_context = max_context

# -------------- KV Cache utilities -------------- #

class RowState:
    def __init__(self) -> None:
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

class StaticKVCache:
    def __init__(self) -> None:
        pass # TODO: implement static KV cache for non-streaming scenarios

class KVCache:
    def __init__(
            self, 
            ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
            config: Optional[TransformerConfig] = None,
            # offloading: bool = False, # To mimic huggingface.DynamicCache' signature
            # offload_only_non_sliding: bool = False, # To mimic huggingface.DynamicCache' signature
            *args, **kwargs
        ) -> None:
        device = torch.device("meta")
        dtype = torch.float32
        if ddp_cache_data is not None:
            device = ddp_cache_data[0][0].device
            dtype = ddp_cache_data[0][0].dtype
            pass # TODO: load from ddp_cache_data

        self.n_layers = config.n_layers
        self.n_heads = config.n_kv_heads
        self.d_head = config.d_head
        # This cache is dynamic -> no limit on context. Fix later if it is an issue
        # self.max_context = config.max_context # TODO: make it dynamic -> 1 to max_context
        self.batch_size = None

        # easier to manage B, T, H, D shapes
        self.cur_pos = 0

        self.cache = torch.empty(
            (self.n_layers, 2, 0, 0, self.n_heads, self.d_head),
            device=device,
            dtype=dtype
        )   # L, 2, B, T, H, D

    @torch.no_grad()
    def reset(self, *args):
        self.cur_pos = 0

    @property
    def device(self) -> torch.device:
        return self.cache.device

    @property
    def shape(self) -> tuple:
        return self.cache.shape
    
    @property
    def k_shape(self) -> tuple:
        # debug purpose
        return self.cache[:,0].shape
    
    @property
    def v_shape(self) -> tuple:
        # debug purpose
        return self.cache[:,1].shape
    
    @property
    def seqlens(self) -> torch.Tensor:
        # flash attention compatibility
        # temporary fix for meta device
        if self.device == torch.device("meta"):
            return torch.tensor([self.cur_pos] * (self.batch_size if self.batch_size else 1))
        return torch.tensor([self.cur_pos] * (self.batch_size if self.batch_size else 1), device=self.cache[0][0].device)
    
    @torch.no_grad()
    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: make an interator? -> yield self.cache[layer_idx] / prefill next position in seqlen dim
        assert layer_idx < self.shape[0], f"Layer index out of bounds: {layer_idx} >= {self.shape[0]}"
        return self.cache[layer_idx]
    
    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        B, T, H, D = k.shape
        if self.device != k.device:
            self.cache = self.cache.to(k.device)
        
        self.check_sizes(B, T, k.device, k.dtype)
        
        assert k.shape == v.shape, f"Key axnd Value tensors must have the same shape. Got k: {k.shape}, v: {v.shape}."
        assert layer_idx < self.shape[0], f"Layer index out of bounds: {layer_idx} >= {self.shape[0]}"

        if k.dtype != self.cache.dtype:
            self.cache = self.cache.to(dtype=k.dtype)
        
        self.cache[layer_idx,0,:,self.cur_pos:self.cur_pos+T,:,:] = k
        self.cache[layer_idx,1,:,self.cur_pos:self.cur_pos+T,:,:] = v
        return self.cache[layer_idx,0,:,:self.cur_pos+T,:,:], self.cache[layer_idx,1,:,:self.cur_pos+T,:,:]
    
    def check_sizes(self, bs, seqlen, device, dtype) -> None:
        # Initialize cache if not done yet
        if self.batch_size is None:
            self.batch_size = bs
            self.cache = torch.empty((
                self.n_layers, 
                2, 
                self.batch_size if self.batch_size is not None else 0,
                seqlen,
                self.n_heads,
                self.d_head
            ))

        assert self.batch_size == bs, f"Input batch size does not match cache's batch size. Got input: {bs} and cache {self.batch_size}."

        if seqlen + self.cur_pos > self.shape[-3]:
            self.cache = torch.cat([self.cache, 
                                    torch.zeros(self.n_layers, 2, self.batch_size, seqlen, self.n_heads, self.d_head, 
                                                            # device=device, dtype=dtype
                                )], 
                                   dim=-3)

        self.cache.to(device=device, dtype=dtype)


    def advance(self):
        if self.cur_pos == 0:
            self.cur_pos = self.cache.shape[-3]
        else:
            self.cur_pos += 1
    
    def _init_from_ddp_cache(self, ddp_cache_data: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        pass # TODO

    @torch.no_grad()
    def drop_row(self, row_idx: int) -> None:
        """
        Drops a specific row from the KV cache, reducing the sequence length by one.
        Use case: removing outdated entries in streaming scenarios.
        """
        self.cache = torch.cat([self.cache[:, :, :row_idx, :, :, :], self.cache[:, :, row_idx+1:, :, :, :]], dim=2)
        self.shape = (
            self.shape[0],
            self.shape[1],
            self.shape[2] - 1,
            self.shape[3],
            self.shape[4],
            self.shape[5]
        )