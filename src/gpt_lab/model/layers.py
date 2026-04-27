from gpt_lab.utils.common import print0
from gpt_lab.utils.schemas import TransformerConfig
from gpt_lab.model.flash_attn import flash_attn, scaled_dot_product_attention
from gpt_lab.model.utils import apply_rope, has_ve
from gpt_lab.utils.types import AttnImplTypes, NormalizationTypes
from gpt_lab.utils.default import DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional, Callable, get_args
import warnings

def apply_rms_norm(x: torch.Tensor, eps: float = 1e-8, torch_impl: bool = True) -> torch.Tensor:
    if torch_impl:
        return torch.rms_norm(x, normalized_shape=(x.size(-1),))
    else:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x / rms

def apply_layer_norm(x: torch.Tensor, eps: float = 1e-5, torch_impl: bool = True) -> torch.Tensor:
    if torch_impl:
        return torch.layer_norm(x, normalized_shape=(x.size(-1),), eps=eps)
    else:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, unbiased=False, keepdim=True)
        return (x - mean) / (std + eps)
    
def build_norm(normalization: NormalizationTypes, eps: float = 1e-5, torch_impl: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    def norm(x: torch.Tensor) -> torch.Tensor:
        if normalization == 'rms':
            return apply_rms_norm(x, eps=eps, torch_impl=torch_impl)
        elif normalization == 'layer':
            return apply_layer_norm(x, eps=eps, torch_impl=torch_impl)
        else:
            raise ValueError(f'Unknown normalization: {normalization}. Supported normalizations are in {get_args(NormalizationTypes)}.')
    return norm


# -------------- Utility layers definitions -------------- #

class Module(nn.Module):
    def nb_parameters(self) -> int:
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])

    def nb_trainable_parameters(self) -> int:
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])

    def nb_non_trainable_parameters(self) -> int:
        return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])

    def summary(self) -> None:
        print0(f'Number of parameters: {self.nb_parameters():,}')
        print0(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
        print0(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')

    def clean_nan(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

    def clip_gradient(self, max_norm: float) -> None:
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    # def init_weights(self) -> None:
    #     '''Initialize the module weights'''
    #     for module in self.modules():
    #         if hasattr(module, 'init_weights'):
    #             module.init_weights()

class Embedding(Module):
    '''Embedding layer'''
    def __init__(self, config: TransformerConfig, dtype=torch.float32, device=torch.device(DEVICE)) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.d_model, 
            padding_idx=config.pad_id, 
            # max_norm=config.max_norm, 
            # norm_type=config.norm_type, 
            # scale_grad_by_freq=config.scale_grad_by_freq, 
            # sparse=config.sparse or True, 
            # device=config.device,
            # dtype=config.dtype
        )
        # self.embedding = nn.Parameter(
        #     data=torch.randn(config.vocab_size, config.d_model),
        #     requires_grad=True
        # )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class Linear(nn.Linear, Module):
    '''Linear layer'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device = DEVICE, dtype=None) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        # TODO: Reparametrize weights and bias here

    def init_weights(self, std: float = 0.01, method: str ='uniform') -> None:
        assert method in ['uniform', 'normal', 'zero'], 'Method must be "uniform", "normal" or "zero"'
        if method == 'uniform':
            nn.init.uniform_(self.weight, -std, std)
        elif method == 'normal':
            nn.init.normal_(self.weight, 0, std)
        elif method == 'zero':
            nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class TPLinear(Linear):
    '''Tensor Parallel Linear layer'''
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 device: torch.device = DEVICE, dtype=None, tp_spec=None, parallel_axis=None) -> None:
        tp_size = tp_spec.size if tp_spec else None
        assert out_features % tp_size == 0, 'out_features must be divisible by tp_size'
        self.tp_spec = tp_spec
        self.axis = parallel_axis
        if tp_spec is None:
            super().__init__(
                in_features=in_features, 
                out_features=out_features, bias=bias, device=device, dtype=dtype
            )
        
        assert self.axis in ['row', 'column'], 'parallel_axis must be "row" or "column"'

        if parallel_axis == 'row':
            assert out_features % tp_size == 0, f'out_features must be divisible by tp_size for row parallelism. Got out_features={out_features}, tp_size={tp_size}.'
            local_out = out_features // tp_size
            super().__init__(
                in_features=in_features // tp_size, 
                out_features=out_features, bias=bias, device=device, dtype=dtype
            )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if (self.tp_spec is None) or (self.axis == 'column'):
            return super().forward(x)
        
        out = super().forward(x)
        dist.all_reduce(out, group=self.tp_spec.tp_group)
        return out


# --------------      Attention utilities      -------------- #

class CausalSelfAttention(Module):
    def __init__(self, config: TransformerConfig, layer_idx: int = 0) -> None:
        super().__init__()
        assert config.d_model == config.d_head * config.n_heads, f'Dimensions are not correct. dim_model must be equal to d_head * n_heads. Got dim_model={config.d_model}, d_head={config.d_head}, n_heads={config.n_heads}'
        assert config.attn_impl in get_args(AttnImplTypes), f'Attention implementation {config.attn_impl} is not supported. Supported attention implementations are in {get_args(AttnImplTypes)}.'
        assert config.normalization in get_args(NormalizationTypes), f'Normalization {config.normalization} is not supported. Supported normalizations are in {get_args(NormalizationTypes)}.'
        assert config.n_heads % config.n_kv_heads == 0, f'Number of heads must be divisible by number of key-value heads. Got n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}.'
        
        self.d_model = config.d_model
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.norm_before_attn = config.norm_before_attn
        self.dropout_rate = config.dropout

        self.norm = build_norm(config.normalization, eps=config.norm_eps, torch_impl=True)
        self.attn_impl = config.attn_impl
        # TODO: try an implementation with separate q, k, v proj
        self.qkv_proj = Linear(config.d_model, (config.n_heads + 2 * config.n_kv_heads) * config.d_head, bias=False)
        self.o_proj = Linear(config.d_model, config.d_model, bias=False)

        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_heads, bias=False) if has_ve(layer_idx, config.n_layers) else None


    @torch.no_grad()
    def init_weights(self) -> None:
        d_model = self.n_heads * self.d_head
        std = 3**0.5 * d_model**-0.5 
        torch.nn.init.uniform_(self.qkv_proj.weight, -std, std) # weights use Uniform to avoid outliers
        torch.nn.init.zeros_(self.o_proj.weight) # projections are zero

    def forward(self, x: torch.Tensor, value_embeds=None, rope_cache=None,  window_size=None, 
                kv_cache=None, attn_mask=None, return_attn_weights: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, Tq, E = x.size()
        if kv_cache is not None:
            kv_cache.check_sizes(B, Tq, x.device, x.dtype)

        Hq, Hk, D = self.n_heads, self.n_kv_heads, self.d_head
        qkv = self.qkv_proj(x)

        qkv = qkv.view(B, Tq, Hq + 2 * Hk, D)
        
        q = qkv[:, :, :Hq, :].clone()
        k = qkv[:, :, Hq:Hq+Hk, :].clone()
        v = qkv[:, :, Hq+Hk:, :]
        
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if value_embeds is not None:
            value_embeds = value_embeds.view(B, Tq, self.n_kv_heads, self.d_head)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * value_embeds

        if rope_cache is not None:
            q = apply_rope(q, rope_cache)
            k = apply_rope(k, rope_cache)

        if self.norm_before_attn:
            q = self.norm(q)
            k = self.norm(k)

        if (rope_cache is not None) or self.norm_before_attn:
            qkv = torch.cat([q, k, v], dim=-2)

        attn_weights = None
        # interpretability mode - not optimized for memory efficiency
        if return_attn_weights:
            # q, k, v = torch.split(qkv, [Hq, Hk, Hk], dim=-2)
            q, k, v = self.unfused(qkv)
            if kv_cache is not None:
                k, v = kv_cache.update(k, v)
            x, attn_weights = scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask, 
                dropout_p=self.dropout_rate if self.training else .0, 
                is_causal=attn_mask is None, 
                return_attn_weights=return_attn_weights
            )
        
        # inference-efficient mode
        if (not self.training) and (kv_cache is not None):
            q, k, v = self.unfused(qkv)
            k_cache, v_cache = kv_cache.layer(self.layer_idx)
            k_cache, v_cache = k_cache.to(q.device, dtype=q.dtype), v_cache.to(v.device, dtype=q.dtype)
            
            x = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache,
                k, v,
                cache_seqlens=kv_cache.seqlens,
                causal=True,
                window_size=window_size if window_size is not None else (-1, -1),
            )

        # qkv-fused training mode
        if self.attn_impl == 'fused':
            assert Hq == Hk, f'Fused attention implementation does not support GQA. Please set attn_impl to "sdpa" if you want to use GQA. Got Hq={Hq}, Hk={Hk}.'
            qkv = qkv.view(B, Tq, 3, Hq, D).contiguous()  
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout_rate if self.training else 0.0,
                window_size=window_size,
                causal=True
            )

        # qkv-unfused training mode
        elif self.attn_impl == 'sdpa':
            q, k, v = self.unfused(qkv)
            x = flash_attn.flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_rate if self.training else 0.0,
                window_size=window_size,
                causal=True
            )
        else:
            raise ValueError(f'Attention implementation {self.attn_impl} is not supported. Supported attention implementations are in {get_args(AttnImplTypes)}.')
        
        # x: (B, Tq, Hq, D) -> (B, Tq, E)
        x = x.contiguous().view(B, Tq, -1)
        x = self.o_proj(x)
        return x, attn_weights
    
    def unfused(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Hq, Hk, D = self.n_heads, self.n_kv_heads, self.d_head
        q, k, v = torch.split(qkv, [Hq, Hk, Hk], dim=-2)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        return q, k, v
    
# -------------- DenseTransformer layers definitions -------------- #
    
class FeedForward(Module):
    '''Position-Wise Feed Forward Network'''
    def __init__(self, d_in: int, d_latent: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = Linear(d_in, d_latent)
        self.w_2 = Linear(d_latent, d_in)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_2(F.relu(self.w_1(x)))

        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
                
        h += x
        output = apply_rms_norm(h)
        return output
    
class SwigLUFeedForward(Module):
    '''Position-Wise Feed Forward Network with SwiGLU activation'''
    def __init__(self, d_in: int, d_latent: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = Linear(d_in, d_latent * 2)
        self.w_2 = Linear(d_latent, d_in)
        self.dropout_rate = dropout

    @torch.no_grad()
    def init_weights(self) -> None:
        d_in = self.w_1.in_features
        d_latent = self.w_1.out_features // 2
        std = 3**0.5 * d_in**-0.5 * (d_latent / d_in)**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal, and adjust for the increased width of the first layer
       
        self.w_1.init_weights(std=std, method='uniform')
        self.w_2.init_weights(std=std * 0.4, method='uniform')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.w_1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        h = self.w_2(F.silu(x1) * x2)

        if self.training and self.dropout_rate > 0.0:
            h = F.dropout(h, p=self.dropout_rate, training=True)
                
        output = h + x
        # output = apply_rms_norm(h)
        return output
    
    
class DecoderLayer(Module):
    '''Decoder layer'''
    def __init__(self, config: TransformerConfig, layer_idx: int = 0) -> None:
        super().__init__()
        if not config.norm_before_attn:
            warnings.warn('Using "norm_before_attn=False" is not recommended and may lead to training instability.', UserWarning)
        self.norm_before_attn = config.norm_before_attn
        self.norm = build_norm(config.normalization, eps=config.norm_eps, torch_impl=True)
        self.attention = CausalSelfAttention(config=config, layer_idx=layer_idx) 
        
        # self.ffn = FeedForward(d_in=dim_model, d_latent=dim_ffn, dropout=dropout)
        # TODO: try SwigLU + make it configurable
        self.ffn = SwigLUFeedForward(d_in=config.d_model, d_latent=config.d_ffn, dropout=config.dropout) 
        self.dropout_rate = config.dropout

        self.norm = build_norm(config.normalization, eps=1e-8, torch_impl=True)

    def forward(self, x, value_embeds=None, kv_cache=None, rope_cache=None, attn_mask=None, window_size=None, return_attn_weights=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.norm_before_attn:
            norm_x = self.norm(x)
        else:
            norm_x = x
        h, attn_weights = self.attention(norm_x, 
            value_embeds=value_embeds, rope_cache=rope_cache, window_size=window_size, 
            kv_cache=kv_cache, attn_mask=attn_mask, return_attn_weights=return_attn_weights)
        
        h = x + h
        h = h + self.ffn(self.norm(h))

        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
        
        return h, attn_weights
    
    @torch.no_grad()
    def init_weights(self) -> None:
        self.attention.init_weights()

class MixtureOfExpertsLayer(Module):
    '''
    dummy
    Mixture of Experts layer. Note: This is a naive version without load balancing or capacity constraints.
    '''
    def __init__(
            self, 
            dim_model: int, 
            dim_ffn: int, 
            n_experts: int, 
            dropout: float
        ) -> None:
        super().__init__()
        warnings.warn('This is a naive Mixture of Experts layer without load balancing or capacity constraints. Use with caution.', UserWarning)
        self.experts = nn.ModuleList([
            FeedForward(d_in=dim_model, d_latent=dim_ffn, dropout=dropout)
            for _ in range(n_experts)
        ])
        self.router = Linear(dim_model, n_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_scores = F.softmax(self.router(x), dim=-1)  
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  
        router_scores = router_scores.unsqueeze(2)  
        output = torch.sum(expert_outputs * router_scores, dim=-1)  
        return output