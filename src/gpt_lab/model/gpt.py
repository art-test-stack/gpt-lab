from gpt_lab.model.layers import (
    DecoderLayer, 
    Linear, 
    Module, 
    # apply_layer_norm,
    # apply_rms_norm,
    build_norm,
)
from gpt_lab.model.loss import build_loss
from gpt_lab.model.utils import (
    KVCache,
    precompute_rope, 
    precompute_positional_encoding
)
from gpt_lab.tokenizer.tokenizer import build_tokenizer
from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.utils.schemas import (
    get_default_device,
    GenerationConfig,
    GPTConfig, 
    ModelOutput, 
    ModelCompletionOutput,
    TrainingConfig,
    TransformerConfig, 
    TransformerOutput
)
from gpt_lab.utils.default import MODELS_FOLDER, DEVICE
from gpt_lab.utils.common import print0, print0_dict

from gpt_lab.tokenizer.tokenizer import Tokenizer

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from pathlib import Path

import warnings


def build_meta_model(config: TransformerConfig) -> "DenseTransformer":
    with torch.device("meta"):
        model = DenseTransformer(config=config)
    return model

def build_model_from_config(
        config: GPTConfig,
    ) -> "DenseTransformer":
    model = build_meta_model(config.model)
    model = model.to_empty(device=config.device)
    model.init_weights()
    try:
        pe_cache = model._precompute_pos_enc()
        model.pe_cache = pe_cache
    except Exception as e:
        warnings.warn(f"Precomputation of positional encodings failed: {str(e)}", UserWarning)
    
    model.to_empty(device=config.device)
    return model

def has_ve(layer_idx: int, n_layer: int):
    return layer_idx % 2 == (n_layer - 1) % 2


class BaseTransformer(Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
    def init_weights(self):
        raise NotImplementedError("init_weights method must be implemented by subclass")
    
    def resize_token_embeddings(self, new_size: int, resize_head: bool = True) -> None:
        raise NotImplementedError("resize_token_embeddings method must be implemented by subclass")
    
    def number_of_parameters(self) -> int:
        raise NotImplementedError("number_of_parameters method must be implemented by subclass")
    
    def scaling_params(self) -> int:
        raise NotImplementedError("scaling_params method must be implemented by subclass")
    

class DenseTransformer(BaseTransformer):
    def __init__(
            self,
            config: Optional[TransformerConfig] = None,
            pad_vocab_size_to=64
        ) -> None:
        super().__init__()
        # init the model in meta device first
        if config is None:
            config = TransformerConfig()
        self.config = config
        # self._model_as_meta = True # TODO: thinking about it to automatically handle meta init -> forward path
        self.vocab_size = config.vocab_size
        self.window_sizes = config._window_sizes
        self.max_seq_len = config.max_context
        # self.bos_token_id = config.bos_id
        # self.eos_token_id = config.eos_id
        
        self.norm = build_norm(self.config.normalization, eps=self.config.norm_eps, torch_impl=True)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        
        # TODO: change to customed embedding
        # TODO: add padded_vocab_size for more efficient token embedding resizing
        self.embeds = nn.Embedding(
            num_embeddings=padded_vocab_size, 
            embedding_dim=config.d_model, 
            # padding_idx=config.pad_id,
            sparse=False,
        )

        self.blocks: nn.ModuleList[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                dim_model=config.d_model,
                dim_ffn=config.d_ffn, 
                n_heads=config.n_heads, 
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head, 
                dropout=config.dropout,
                layer_idx=layer_idx,
                norm_before_attn=config.norm_before_attn,
                use_gqa=config.use_gqa,
                attn_impl=config.attn_impl,
                normalization=config.normalization,
            ) 
            for layer_idx in range(config.n_layers)
        ])

        kv_dim = config.n_kv_heads * config.d_head

        # TODO: same as karpathy
        # Value embeddings (ResFormer-style): alternating layers, last layer always include
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layers) if has_ve(i, config.n_layers)
        })

        self.lm_head = Linear(config.d_model, padded_vocab_size, bias=False)

        if config.tie_word_embeddings:
            # not sure if this works / maybe have to check dtype - at least
            # self.lm_head.weight = self.blocks .emb.weight.T # E -> V
            # TODO: solve this naive solution
            pass

        self.res_x0 = torch.nn.Parameter(torch.ones(self.config.n_layers), requires_grad=True)
        self.res_h = torch.nn.Parameter(torch.zeros(self.config.n_layers), requires_grad=True)
        
        _pe_cache = self._precompute_pos_enc()
        self.register_buffer("pe_cache", _pe_cache, persistent=False)

        # default loss_fn, can be overridden by trainer or by passing a different loss_fn to the model if needed
        self.loss_fn = build_loss() 


    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.embeds.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        d_model = self.config.d_model
        s = 3**0.5 * d_model**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.blocks:
            # if fused 
            # TODO/NOTE: decide to use .init_weights() -> more modular
            # or init here -> more control for different init for different components
            # so for now, it's a mix of both xD
            block.init_weights()
            torch.nn.init.uniform_(block.ffn.w_1.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.ffn.w_2.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layers = self.config.n_layers
        for i in range(n_layers):
            self.res_h.data[i] = 1.15 - (0.10 * i / max(n_layers - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layers):
            self.res_x0.data[i] = 0.20 - (0.15 * i / max(n_layers - 1, 1))

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.blocks:
            if block.attention.ve_gate is not None:
                torch.nn.init.uniform_(block.attention.ve_gate.weight, 0.0, 0.02)

        # Rotary/positional embeddings
        self.pe_cache = self._precompute_pos_enc()

    def resize_token_embeddings(self, new_size: int, resize_head: bool = True) -> None:
        # TODO: Dummy implementation, to be improved
        weights = self.embeds.weight
        nb_tokens_to_add = new_size - self.vocab_size
        new_emb_weights = torch.empty((nb_tokens_to_add, weights.size(1)), device=weights.device, dtype=weights.dtype)
        
        torch.nn.init.normal_(new_emb_weights.weight, mean=0.0, std=0.8)
        self.embeds.weight = torch.nn.Parameter(torch.cat([weights, new_emb_weights], dim=0), requires_grad=self.training)
        self.vocab_size = new_size
        if resize_head:
            # resize lm head false if gist tokenization: https://arxiv.org/abs/2304.08467
            head_weights = self.lm_head.weight
            new_head_weights = torch.empty((nb_tokens_to_add, head_weights.size(1)), device=head_weights.device, dtype=head_weights.dtype)
            
            torch.nn.init.normal_(new_head_weights, mean=0.0, std=0.001)
            self.lm_head.weight = torch.nn.Parameter(torch.cat([head_weights, new_head_weights], dim=0), requires_grad=self.training)

    def _precompute_pos_enc(self, new_max_context: Optional[int] = None) -> None:
        if not new_max_context:
            new_max_context = self.config.max_context
        device = self.embeds.weight.device
        if self.config.positional_encoding == "rope":
            pe_cache = precompute_rope(
                seq_len=new_max_context * 10,
                d_head=self.config.d_head,
                base=self.config.rope_params.get("rope_theta", 10000),
                # dtype=self.dtype,
                device=device,
            )
        elif self.config.positional_encoding == "positional":
            pe_cache = precompute_positional_encoding(new_max_context, self.config.d_model, device=device)
        else:
            raise ValueError(f"Unknown positional encoding: {self.config.positional_encoding}")

        return pe_cache
    
    @property
    def n_params(self) -> int:
        return self.n_params_per_layer()["total"]
    
    def n_params_per_layer(self) -> Dict[str, int]:
        nb_of_params = dict(
            embeds=sum(p.numel() for p in self.embeds.parameters()),
            blocks=sum(p.numel() for p in self.blocks.parameters()),
            lm_head=sum(p.numel() for p in self.lm_head.parameters()),
            residuals=self.res_x0.numel() + self.res_h.numel(),
            value_embeds=sum(p.numel() for p in self.value_embeds.parameters()),
        )
        nb_of_params["total"] = sum(nb_of_params.values())
        return nb_of_params

    def n_scaling_params(self) -> int:
        nb_of_params = self.n_params_per_layer()
        return sum(nb_of_params[tab] for tab in ["blocks", "lm_head"])
    
    def estimate_flops(self) -> float:
        """Estimate FLOPs per token based on model configuration.

        Adapted from: karpathy/nanochat
        """
        n_params_per_layer = self.n_params_per_layer()
        nparams = n_params_per_layer["total"]
        # Exclude non-matmul params: embeddings and per-layer scalars
        nparams_exclude = sum([
            _nparams for layer, _nparams in n_params_per_layer.items() 
            if layer in ("embeds", "residuals", "value_embeds")])
        
        h, q, t = self.config.n_heads, self.config.d_model // self.config.n_heads, self.config.max_context
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token
    
    def build_optimizer(self, config: TrainingConfig, optim_config_path: Optional[str] = None) -> torch.optim.Optimizer:
        # default config, overriden by optim_config_path if provided
        optim_config = {
            "embeddings":       dict(opt='adamw', lr=config.lr_embeddings,       betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            "head":             dict(opt='adamw', lr=config.lr_head,             betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
            "residual_hiddens": dict(opt='adamw', lr=config.lr_residuals * 0.01, betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.05),
            "residual_inputs":  dict(opt='adamw', lr=config.lr_residuals,        betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            "value_embeds":     dict(opt='adamw', lr=config.lr_embeddings * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),                    
            "transformer":      dict(opt='muon',  lr=config.lr_transformer,      momentum=0.95, ns_steps=5, beta=.9, weight_decay=config.weight_decay),
        }
        optim_config_path = optim_config_path or config.optim_config_path # one or another; idc for now
        
        if optim_config_path is not None:
            optim_config_path = Path(optim_config_path)
            if optim_config_path.is_file() and optim_config_path.suffix in (".yaml", ".yml"):
                import yaml
                with open(optim_config_path, "r") as f:
                    _opt_cfg = yaml.safe_load(f)
                def clean_value(value):
                    if isinstance(value, str):
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return value
                    elif isinstance(value, list):
                        return tuple(clean_value(v) for v in value)
                    else:
                        return value
                
                # clean yaml values (e.g. '1e-2' -> 1e-2, [0.8, 0.995] -> (0.8, 0.995))
                _opt_cfg = {name: {key: clean_value(value) for key, value in group.items()} for name, group in _opt_cfg.items()}

                # update config with values from file -> override defaults from file -> override defaults from code (and keeps defaults for any missing values in the file)
                optim_config = {name: optim_config[name] | _opt_cfg.get("default", {}) | _opt_cfg.get(name, {}) for name in optim_config.keys()}

            else:
                raise FileNotFoundError(f"Optimizer configuration file not found at {optim_config_path}. Please provide a valid path to the optimizer configuration YAML file (ending in .yaml or .yml), or remove the --optim-config-path argument to use the default configuration.")
            
        d_model = self.config.d_model
        # Separate out all parameters into groups
        embedding_params = list(self.embeds.parameters())
        blocks_params = list(self.blocks.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        lm_head_params = list(self.lm_head.parameters())
        residual_h_params = [self.res_h]
        residual_x0_params = [self.res_x0]

        assert len(list(self.parameters())) == len(blocks_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(residual_h_params) + len(residual_x0_params), "Parameter grouping error: some parameters are missing or duplicated in the groups."

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = 1 / math.sqrt(d_model / 768)
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({d_model}/768) = {dmodel_lr_scale:.6f}")

        # Scaling hparams
        for name, group in optim_config.items():
            if name in ("embeddings", "head", "value_embeds"):
                optim_config[name]["lr"] = group["lr"] * dmodel_lr_scale
        optim_config["transformer"]["weight_decay"] *= config.weight_decay_scale

        print0_dict("Optimizer configuration after scaling", optim_config)

        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(params=embedding_params, **optim_config["embeddings"]),
            dict(params=lm_head_params, **optim_config["head"]),
            dict(params=value_embeds_params, **optim_config["value_embeds"]),
            dict(params=residual_h_params, **optim_config["residual_hiddens"]),
            dict(params=residual_x0_params, **optim_config["residual_inputs"]),
        ]

        for shape in sorted({p.shape for p in blocks_params}):
            group_params = [p for p in blocks_params if p.shape == shape]
            param_groups.append(dict(
                params=group_params, **optim_config["transformer"]
            ))

        optimizer = OptimizerFactory(param_groups, distributed=config.dist_info["IS_DDP_INITIALIZED"])
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(
            self, 
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            attn_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            return_attentions: bool = False,
            return_hidden_states: bool = False,
            reduction: str = "mean",
        ) -> ModelOutput:
        assert (past_key_values is None) or (not self.training), "KV cache can not be used during training."
        # assert input_ids.shape[-1] <= self.config.max_context, f"Input sequence length {input_ids.shape[-1]} exceeds max context {self.config.max_context}"
        assert input_ids.dim() == 2, "Input ids should be of shape (batch_size, seq_len)"
        if input_ids.shape[-1] > self.config.max_context:
            warnings.warn(f"Input sequence length {input_ids.shape[-1]} exceeds max context {self.config.max_context}. May cause unexpected behavior if model was initialized with a different max context.")

        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, T = input_ids.size()
        T0 = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "cur_pos"):
                T0 = past_key_values.cur_pos
            elif hasattr(past_key_values, "current_length"):
                T0 = past_key_values.current_length
            elif hasattr(past_key_values, "shape"):
                T0 = past_key_values.shape[3] # assuming shape is (L, 2, B, T, H, D)
            else:
                pass
        x = self.embeds(input_ids)

        if self.config.positional_encoding == "positional_encoding":
            x = x + self.pe_cache[:x.size(2)]

        if self.config.positional_encoding == "rope":
            rope_cache = self.pe_cache[T0:T0+T]
        else:
            rope_cache = None

        x = self.norm(x)
        x0 = x # .clone()
        attentions = []
        hidden_states = [("emb", x)] if return_hidden_states else None
        # assert not torch.isnan(x).any(), "..."
        
        for i, layer in enumerate(self.blocks, 0):
            # TODO: not return attn yet
            return_attn = return_attentions and (i == len(self.blocks) - 1) and False
            x = self.res_h[i] * x + self.res_x0[i] * x0
            x, attn = layer(x, attn_mask=attn_mask,
                rope_cache=rope_cache,
                kv_cache=past_key_values, 
                return_attn_weights=return_attn,
                # TODO: return_attn in special cases only -> interpretability
                # return_attentions=return_attn 
            )
            if return_attn:
                attentions.append(attn)
            if return_hidden_states:
                hidden_states.append((f"layer_{i}", x))

        # if kv_cache is not None:
        #     kv_cache.advance()
        x = self.norm(x)

        softcap = self.config.softcap
        if labels is not None:
            logits = self.lm_head(x)
        else: 
            logits = self.lm_head(x[:, [-1], :])

        logits = logits[..., :self.config.vocab_size] 

        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        # logits = torch.clamp(logits, min=-softcap, max=softcap)
        temperature = 1.0 # TODO: add temperature support to forward method and generation method
        if temperature > 0 and not temperature == 1.0:
            logits = logits / temperature
        
        assert logits.is_contiguous(), "Logits should be contiguous for efficient loss computation"
        loss = None
        if labels is not None:
            assert labels.is_contiguous(), "Labels should be contiguous for efficient loss computation"
            loss = self.loss_fn(logits, labels, reduction=reduction)

        output = ModelOutput(
            logits=logits,
            loss=loss,
            # NOTE: for now useless
            # attentions=output.attentions if attentions else None,
            # log_probs=F.log_softmax(logits, dim=-1) if log_prob else None,
            # probs=F.softmax(logits, dim=-1) if log_prob else None,
            # hidden_states=output.hidden_states,
            past_key_values=past_key_values,
        )
        
        return output
    

class GPTModel:
    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: Tokenizer,
        ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # TODO: Add support for loading Dense vs MoE vs Hybrid models vs DUMAS
        self.main = model
        assert self.main is not None, "Model must be provided"
        assert self.main.vocab_size == self.tokenizer.vocab_size, "Model vocab size must match tokenizer vocab size"
        self.vocab_size = self.main.vocab_size
        self.bos_token_id = self.tokenizer.bos_token_id
    
    @property
    def device(self) -> torch.device:
        return next(self.main.parameters()).device
    
    @property
    def n_params(self) -> int:
        return self.main.number_of_parameters()
    
    def __call__(self, input_ids, labels=None, *args, **kwargs) -> ModelOutput:
        input_ids = input_ids.to(self.device)
        # attn_mask = self.attn_mask(input_ids)
        logits = self.main(
            input_ids, 
            # attn_mask=attn_mask, 
            *args, **kwargs).logits
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.loss_fn(logits, labels)
        return ModelOutput(logits=logits, loss=loss)

    def __repr__(self) -> str:
        return f"GPTModel(config={str(self.config)}, \nmodel={str(self.main)})"
    
    def update_max_context(self, new_max_context: int) -> None:
        assert new_max_context > 0, "New max context must be positive"
        self.config.model.max_context = new_max_context
        self.attn_mask = self.attn_mask.update_max_context(new_max_context)

    def apply_chat_template(self, messages: List[dict], template: str) -> str:
        return self.tokenizer.apply_chat_template(messages, template)

    def forward(
            self, 
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            attentions: bool = False,
            past_key_values: dict | None = None,
            log_prob: bool = False,
            temperature: float = 1.0,
            **kwargs
        ) -> ModelOutput:
        assert (past_key_values is None) or (not self.main.training), "KV cache can not be used during training."

        input_ids = input_ids.to(self.config.device)
        labels = labels.to(self.config.device) if labels is not None else None
        # TODO: ignore attn mask (only used for padding -> ignore padding mask for now)
        # attn_mask = self.attn_mask(input_ids)
        output: TransformerOutput = self.main(
            input_ids, 
            return_attentions=attentions, 
            # attn_mask=attn_mask,
            past_key_values=past_key_values
        )
        if temperature > 0:
            logits = output.logits / temperature
        else:
            logits = output.logits
        
        loss = None
        output = ModelOutput(
            logits=logits,
            loss=loss,
            attentions=output.attentions if attentions else None,
            log_probs=F.log_softmax(logits, dim=-1) if log_prob else None,
            probs=F.softmax(logits, dim=-1) if log_prob else None,
            hidden_states=output.hidden_states,
            past_key_values=past_key_values,
        )
        
        if labels is not None:
            assert output.logits.device == labels.device, f"Logits and labels must be on the same device. Got {output.logits.device} and {labels.device}"
            output.loss = self.loss_fn(output, labels)

        return output
    
    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.Tensor,
            ground_truth: Optional[torch.Tensor] = None,
            generation_config: GenerationConfig | None = None,
            assistant_model = None, # TODO: implement assistant model functionality
        ) -> ModelCompletionOutput | Iterator[ModelCompletionOutput]:
        if assistant_model is not None:
            warnings.warn("Assistant model functionality is not yet implemented. Assistant model provided is just ignored.", UserWarning)
        if generation_config is None:
            warnings.warn("No generation config provided. Using default generation config with provided kwargs.", UserWarning)
        if not generation_config.use_cache:
            warnings.warn("GenerationConfig.use_cache is False. Generation may be slow as prefill will be done for each step.", UserWarning)
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig.model_validate(**generation_config)

        device = self.main.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device).manual_seed(generation_config.seed)

        bos = self.tokenizer.bos_token_id

        input_ids = input_ids.to(device)

        if ground_truth is not None:
            ground_truth = ground_truth.to(device)

        kv_cache = None
        if generation_config.use_cache:
            kv_cache = KVCache(config=self.main.config)

        # Prefill: first forward pass with the full input_ids
        # greedy for now; TODO: implement sampling methods following self.forward method
        logits = self.main(
            input_ids=input_ids,
            past_key_values=kv_cache,
            return_attentions=False,
        ).logits[:,-1,:]
        # TODO: init gpt_lab.model.utils.RowState for generation
        num_generated = 0
        max_length = generation_config.max_length
        batch_size = input_ids.size(0)
        # TODO: dummy implementation for results
        # Initialize with the last token from the first pass
        results = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        while True:
            
            # TODO: add row state for stop conditions/more efficient generation
            
            # TODO: implement sampling methods
            generated_ids = torch.argmax(logits, dim=-1).unsqueeze(-1)
            results = torch.cat([results, generated_ids], dim=-1)
            num_generated += 1
            # results.append(generated_ids.squeeze().tolist())

            if max_length is not None and num_generated >= max_length:
                break
            # results.append(self.tokenizer.decode_batch(generated_ids.squeeze().tolist()))

            # TODO: add forced tokens support

            if kv_cache is not None:
                input_ids = generated_ids
            else:
                input_ids = torch.cat([input_ids, generated_ids], dim=-1)

            logits = self.main(
                input_ids=input_ids,
                past_key_values=kv_cache,
                return_attentions=False,
            ).logits[:,-1,:]
        
        return results.tolist()

    # TODO # PRIOTITY
    def generate_batch(self, prompts, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer.encode_batch(text, prepend_bos=True)
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        generation_config = GenerationConfig(max_length=max_tokens, temperature=temperature,
                                             use_cache=True, seed=seed, num_beams=num_samples, top_k=top_k)
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, generation_config=generation_config):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            model_dir: str | None = None,
        ) -> "GPTModel":
        return cls.load(
            model_name=model_name,
            checkpoint_version="latest",
            model_dir=model_dir
        )

    @classmethod
    def load(
            cls,
            model_name: str,
            ckpt_version: str,
            model_dir: str | None = None,
            device: str | None = None,
        ) -> "GPTModel":
        if model_dir is None:
            model_dir = MODELS_FOLDER
        if not ckpt_version.endswith(".pth"):
            ckpt_version += ".pth"
        config = GPTConfig.from_file(model_name=model_name, model_dir=model_dir)

        model = build_model_from_config(config)

        tokenizer = build_tokenizer(config.tokenizer)
        model_path = config.dirname / ckpt_version
        if not device:
            device = config.device
        if not config.device:
            device = get_default_device()
            config.device = device
        model.load_state_dict(torch.load(model_path, map_location=device))
        return cls(model=model, tokenizer=tokenizer, config=config)
    
    @classmethod
    def from_scratch(
            cls,
            config: GPTConfig
        ) -> "GPTModel":
        config.to_file(mode="pickle")
        model = build_model_from_config(config)
        tokenizer = build_tokenizer(config.tokenizer)
        gpt = cls(model=model, tokenizer=tokenizer)
        # gpt.init_weights()
        return gpt
    
    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
        ) -> "GPTModel":
        # TODO: Implement conversion from Huggingface models for compatibility
        from gpt_lab.model.wrapper import init_mistral_model
        warnings.warn("Loading from Huggingface is experimental and may not work as expected. Only works with Mistral model for now.", UserWarning)
        model, tokenizer, config = init_mistral_model(model_name)
        gpt = cls(
            model=model,
            tokenizer=tokenizer
        )
        return gpt
        # config = GPTConfig.from_huggingface(model_name)
        # hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(config.device)
        # hf_tokenizer = AutoTokenizer.from_pretrained(model_name).to(config.device)

        # gpt = cls(
        #     model=hf_model,
        #     tokenizer=hf_tokenizer,
        #     config=GPTConfig()
        # )
        # return gpt
    
    def save_checkpoint(
            self,
            ckpt_version: str | None = None,
            keep_vars: bool = True,
        ) -> None:
        if ckpt_version is None:
            ckpt_version = "checkpoint.pth"
        if (not ckpt_version.endswith(".pth")) and (not ckpt_version.endswith(".pt")):
            ckpt_version += ".pth"
        model_path = self.config.dirname / ckpt_version
        torch.save(self.main.state_dict(keep_vars=keep_vars), model_path)
    