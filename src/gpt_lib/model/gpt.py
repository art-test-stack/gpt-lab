from gpt_lib.model.layers import (
    DecoderLayer, 
    Linear, 
    Module, 
    # apply_layer_norm,
    # apply_rms_norm,
    build_norm,
)
from gpt_lib.model.loss import build_loss
from gpt_lib.model.utils import (
    KVCache, SelfAttentionMask,
    precompute_rope, 
    precompute_positional_encoding
)
from gpt_lib.tokenizer.tokenizer import build_tokenizer
from gpt_lib.utils.schemas import (
    get_default_device,
    GenerationConfig,
    GPTConfig, 
    ModelOutput, 
    ModelCompletionOutput,
    TransformerConfig, 
    TransformerOutput, 
    OptimizerSpec
)
from gpt_lib.utils.types import Dtypes, Devices, TokenizerTensors
from gpt_lib.utils.default import MODELS_FOLDER, DEVICE

from typing import Any, List, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

import warnings

def init_weights(model: nn.Module) -> nn.Module:
    # Dummy; TODO: Make more flexible weight initialization
    try:
        for name, layer in model.layers.items():
            if hasattr(layer, "init_weights"):
                layer.init_weights()
            else:
                for p in layer.parameters():
                    if p.dim() > 1:
                        nn.init.uniform_(p)
    except:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    return model

def build_meta_model(config: GPTConfig) -> "Transformer":
    with torch.device("meta"):
        model = Transformer(config=config.model, dtype=config.dtype)
    return model

def build_model_from_config(
        config: GPTConfig,
    ) -> "GPTModel":
    model = build_meta_model(config)
    model = init_weights(model)
    # model.load_state_dict(
    #     state_dict=torch.load(
    # )
    try:
        pe_cache = model._precompute_pos_enc()
        model.pe_cache = pe_cache
    except Exception as e:
        warnings.warn(f"Precomputation of positional encodings failed: {str(e)}", UserWarning)
    
    model.to_empty(device=config.device)
    return model

def has_ve(layer_idx: int, n_layer: int):
    return layer_idx % 2 == (n_layer - 1) % 2

class Transformer(Module):
    def __init__(
            self,
            config: TransformerConfig,
            device: str | torch.device = DEVICE,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__()
        # init the model in meta device first
        if isinstance(device, str):
            device = torch.device(device)
        self.config = config
        # self._model_as_meta = True # TODO: thinking about it to automatically handle meta init -> forward path
        self.vocab_size = config.vocab_size
        self.window_sizes = config._window_sizes
        # self.bos_token_id = config.bos_id
        # self.eos_token_id = config.eos_id
        self.device = device
        self.dtype = dtype
        
        self.norm = build_norm(self.config.normalization, eps=self.config.norm_eps, torch_impl=True)
        _pe_cache = self._precompute_pos_enc()

        self.register_buffer("pe_cache", _pe_cache, persistent=False)
        # if self.config.positional_encoding == "rope":
        #     rope_cache = precompute_rope(
        #         seq_len=config.max_context,
        #         d_head=config.d_head,
        #         base=self.config.rope_params.get("rope_theta", 10000),
        #         dtype=dtype,
        #         device=device,
        #     )
        #     self.register_buffer("pe_cache", rope_cache, persistent=False)
        # elif self.config.positional_encoding == "positional":
        #     pos_enc = precompute_positional_encoding(config.max_context, config.d_model, dtype=dtype, device=device)
        #     self.register_buffer("pe_cache", pos_enc, persistent=False)
        # elif self.config.positional_encoding == "alibi":
        #     raise NotImplementedError("ALiBi positional encoding is not yet implemented.")
        # else:
        #     raise ValueError(f"Unknown positional encoding: {self.config.positional_encoding}")

        # TODO: Will change to customed embedding
        embedding = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.d_model, 
            # padding_idx=config.pad_id,
            sparse=False,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleDict(dict(
            emb=embedding,
            blocks=nn.ModuleList([
                DecoderLayer(
                    dim_model=config.d_model,
                    dim_ffn=config.d_ffn, 
                    n_heads=config.n_heads, 
                    n_kv_heads=config.n_kv_heads,
                    d_head=config.d_head, 
                    dropout=config.dropout,
                    layer_idx=layer_idx,
                    norm_before_attn=config.norm_before_attn,
                    enable_gqa=config.enable_gqa,
                    attn_impl=config.attn_impl,
                    normalization=config.normalization,
                ) 
                for layer_idx in range(config.n_layers)
            ])
        ))

        kv_dim = config.n_kv_heads * config.d_head
        # same as karpathy
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layers) if has_ve(i, config.n_layers)
        })

        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # not sure if this works / maybe have to check dtype - at least
            # self.lm_head.weight = self.layers.emb.weight.T # E -> V
            # TODO: solve this naive solution
            pass

        self.w_x0 = torch.nn.Parameter(torch.ones(self.config.n_layers, dtype=dtype), requires_grad=True)
        self.w_res = torch.nn.Parameter(torch.zeros(self.config.n_layers, dtype=dtype), requires_grad=True)

        # TODO: Value embeddings (ResFormer-style): alternating layers, last layer always included
        # head_dim = config.d_model // config.n_heads
        # kv_dim = config.n_kv_heads * head_dim
        # padded_vocab_size = config.vocab_size
        # self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layers) if has_ve(i, config.n_layers)})

    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            return_attentions: bool = False,
            return_hidden_states: bool = False,
        ):
        assert (past_key_values is None) or (not self.training), "KV cache can not be used during training."
        # assert input_ids.shape[-1] <= self.config.max_context, f"Input sequence length {input_ids.shape[-1]} exceeds max context {self.config.max_context}"
        assert input_ids.dim() == 2, "Input ids should be of shape (batch_size, seq_len)"

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
        
        x = self.layers.emb(input_ids)

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
        assert not torch.isnan(x).any(), "..."
        for i, layer in enumerate(self.layers.blocks, 0):
            # TODO: not return attn yet
            return_attn = return_attentions and (i == len(self.layers.blocks) - 1) and False
            x = self.w_res[i] * x + self.w_x0[i] * x0
            x, attn = layer(x, attn_mask=attn_mask,
                # TODO: not yet supported
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
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        # logits = torch.clamp(logits, min=-softcap, max=softcap)
        
        return TransformerOutput(
            logits=logits,
            attentions=attn if return_attentions else None,
            hidden_states=hidden_states if return_hidden_states else None,
            past_key_values=past_key_values
        )

    def resize_token_embeddings(self, new_size: int, resize_head: bool = True) -> None:
        # TODO: Dummy implementation, to be improved
        weights = self.layers.emb.weight
        nb_tokens_to_add = new_size - self.vocab_size
        new_emb_weights = torch.empty((nb_tokens_to_add, weights.size(1)), device=weights.device, dtype=weights.dtype)
        # TODO: change std based on dim
        std = 0.02
        nn.init.uniform_(new_emb_weights, -std, std) 
        self.layers.emb.weight = torch.nn.Parameter(torch.cat([weights, new_emb_weights], dim=0), requires_grad=True)
        self.vocab_size = new_size
        if resize_head:
            # resize lm head false if gist tokenization: https://arxiv.org/abs/2304.08467
            head_weights = self.lm_head.weight
            new_head_weights = torch.empty((nb_tokens_to_add, head_weights.size(1)), device=head_weights.device, dtype=head_weights.dtype)
            nn.init.uniform_(new_head_weights, -std, std)
            self.lm_head.weight = torch.nn.Parameter(torch.cat([head_weights, new_head_weights], dim=0), requires_grad=True)

    def _precompute_pos_enc(self, new_max_context: Optional[int] = None) -> None:
        if not new_max_context:
            new_max_context = self.config.max_context
        if self.config.positional_encoding == "rope":
            pe_cache = precompute_rope(
                seq_len=new_max_context,
                d_head=self.config.d_head,
                base=self.config.rope_params.get("rope_theta", 10000),
                dtype=self.dtype,
                device=self.device,
            )
        elif self.config.positional_encoding == "positional":
            pe_cache = precompute_positional_encoding(new_max_context, self.config.d_model, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unknown positional encoding: {self.config.positional_encoding}")

        return pe_cache
    
    def number_of_parameters(self) -> int:
        nb_of_params = dict(
            emb=sum(p.numel() for p in self.layers.emb.parameters()),
            blocks=sum(p.numel() for p in self.layers.tf.parameters()),
            lm_head=sum(p.numel() for p in self.lm_head.parameters()),
            scalars=self.w_x0.numel() + self.w_res.numel(),
            ve=sum(p.numel() for p in self.ve.parameters()),
        )
        nb_of_params["total"] = sum(nb_of_params.values())
        return nb_of_params
    
    def scaling_params(self) -> int:
        nb_of_params = self.number_of_parameters()
        return sum(nb_of_params[tab] for tab in ["blocks", "lm_head"])

class GPTModel:
    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer: callable,
            config: GPTConfig
        ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        # TODO: Add support for loading Dense vs MoE vs Hybrid models
        self.model = model
        self.loss_fn = build_loss(config.loss)
        assert self.model is not None, "Model must be provided"
        assert self.model.vocab_size == self.tokenizer.vocab_size, "Model vocab size must match tokenizer vocab size"

        self.vocab_size = self.config.model.vocab_size
        self.bos_token_id = self.tokenizer.bos_token_id
    
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    
    def __call__(self, input_ids, labels=None, *args, **kwargs) -> ModelOutput:
        input_ids = input_ids.to(self.device)
        # attn_mask = self.attn_mask(input_ids)
        logits = self.model(
            input_ids, 
            # attn_mask=attn_mask, 
            *args, **kwargs).logits
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.loss_fn(logits, labels)
        return ModelOutput(logits=logits, loss=loss)

    def __repr__(self) -> str:
        return f"GPTModel(config={str(self.config)}, \nmodel={str(self.model)})"
    
    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def to(self, 
            # device: DeviceLikeType | None = ...,
            # dtype: dtype | None = ...,
            # non_blocking: bool = ...
            device = ...,
            dtype = ...,
            non_blocking = ...
        ) -> None:
        self.model = self.model.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.config.device = self.model.device
        # self.config.dtype = self.model.dtype

    def estimate_flops(self) -> float:
        """Estimate FLOPs per token based on model configuration.

        from: karpathy/nanochat https://github.com/karpathy/nanochat/discussions/420#:~:text=def-,estimate_flops,-(self)%3A
        """
        nparams = self.number_of_parameters()
        nparams_emb = self.model.emb.weight.numel()
        l, h, q, t = self.config.model.n_layers, self.config.model.n_heads, self.config.model.d_model // self.config.model.n_heads, self.config.model.max_context
        
        # TODO: consider other components if needed (lm_head, norm, window_size, etc.)
        flops_per_token = 6 * (nparams - nparams_emb) + 12 * l * h * q * t
        return flops_per_token
    
    def update_max_context(self, new_max_context: int) -> None:
        assert new_max_context > 0, "New max context must be positive"
        self.config.model.max_context = new_max_context
        self.attn_mask = self.attn_mask.update_max_context(new_max_context)

    def encode(self, text: str, add_special_tokens: bool = True, return_tensors: TokenizerTensors = "pt") -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors=return_tensors)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True, return_tensors: TokenizerTensors = "pt") -> List[List[int]]:
        # TODO: Change current dummy implementation
        return self.tokenizer.batch_encode(texts, add_special_tokens=add_special_tokens, return_tensors=return_tensors)
    
    def decode_batch(self, token_ids: List[List[int]]) -> List[str]:
        # TODO: Change current dummy implementation
        return [self.tokenizer.decode(ids) for ids in token_ids]

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
        assert (past_key_values is None) or (not self.model.training), "KV cache can not be used during training."

        input_ids = input_ids.to(self.config.device)
        labels = labels.to(self.config.device) if labels is not None else None
        # TODO: ignore attn mask (only used for padding -> ignore padding mask for now)
        # attn_mask = self.attn_mask(input_ids)
        output: TransformerOutput = self.model(
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
            # text: str | List[str],
            # ground_truth: str | List[str] | None = None,
            generation_config: GenerationConfig | None = None,
            assistant_model = None, # TODO: implement assistant model functionality
        ) -> ModelCompletionOutput | Iterator[ModelCompletionOutput]:
        if assistant_model is not None:
            warnings.warn("Assistant model functionality is not yet implemented. Assistant model provided is just ignored.", UserWarning)
        if generation_config is None:
            warnings.warn("No generation config provided. Using default generation config.", UserWarning)
            generation_config = GenerationConfig()
        if not generation_config.use_cache:
            warnings.warn("GenerationConfig.use_cache is False. Generation may be slow as prefill will be done for each step.", UserWarning)
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig.model_validate(**generation_config)

        self.eval()
        # if isinstance(text, str):
        #     text = [text]

        # input_ids = self.tokenizer.batch_encode(text, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        if ground_truth is not None:
            # if isinstance(ground_truth, str):
            #     ground_truth = [ground_truth]
            # label_ids = self.tokenizer.encode(ground_truth, add_special_tokens=True, return_tensors="pt")
            label_ids = label_ids.to(self.device)
        else:
            label_ids = None

        kv_cache = None
        if generation_config.use_cache:
            kv_cache = KVCache(config=self.config.model)
            # kv_cache = kv_cache.maybe_init()


        # Prefill: first forward pass with the full input_ids
        # greedy for now; TODO: implement sampling methods following self.forward method
        logits = self.model(
            input_ids=input_ids,
            past_key_values=kv_cache,
            return_attentions=False,
        ).logits[:,-1,:]
        # TODO: init gpt_lib.model.utils.RowState for generation
        num_generated = 0
        max_length = generation_config.max_length
        batch_size = input_ids.size(0)
        # TODO: dummy implementation for results
        # Initialize with the last token from the first pass
        results = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
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

            logits = self.model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                return_attentions=False,
            ).logits[:,-1,:]
        
        return results.tolist()

    # TODO
    def generate_batch(self):
        pass 
    
    def number_of_parameters(self) -> int:
        try:
            return self.model.nb_parameters()
        except AttributeError:
            return sum([p.numel() for p in self.model.parameters()])

    def init_weights(self) -> None:
        try:
            for name, layer in self.model.layers.items():
                if hasattr(layer, "init_weights"):
                    layer.init_weights()
                else:
                    for p in layer.parameters():
                        if p.dim() > 1:
                            nn.init.uniform_(p)
        except:
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def build_optimizer(self) -> torch.optim.Optimizer:
        # TODO: return optimizer based on config
        opt_config: dict[str, OptimizerSpec] = self.config.trainer._optimizers

        emb_params = list(self.model.emb.parameters())
        tf_params = list(self.model.blocks.parameters())
        params = {
            key: { "params": self.model.__getattr__(key).parameters(), **opt_config[key].kwargs }
            for key in opt_config.keys()
        }
        optimizers = [
            opt_config[key].optimizer_class(**value["kwargs"])
            for key, value in params.items()
        ]
        
    @property
    def tp_plan(self) -> dict:
        # TODO
        try:
            return self.model.tp_plan
        except AttributeError:
            warnings.warn("Model does not have tp_plan attribute. Returning empty dict.", UserWarning)
            return {}
    
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
        gpt = cls(model=model, tokenizer=tokenizer, config=config)
        gpt.init_weights()
        return gpt
    
    @classmethod
    def from_yaml(
            cls,
            yaml_path: str | Path,
        ) -> "GPTModel":
        config = GPTConfig.from_yaml(yaml_path)
        model = build_model_from_config(config)
        tokenizer = build_tokenizer(config.tokenizer)
        gpt = cls(model=model, tokenizer=tokenizer, config=config)
        gpt.init_weights()
        return gpt

    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
        ) -> "GPTModel":
        # TODO: Implement conversion from Huggingface models for compatibility
        from gpt_lib.model.wrapper import init_mistral_model
        warnings.warn("Loading from Huggingface is experimental and may not work as expected. Only works with Mistral model for now.", UserWarning)
        model, tokenizer, config = init_mistral_model(model_name)
        gpt = cls(
            model=model,
            tokenizer=tokenizer,
            config=config
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
        torch.save(self.model.state_dict(keep_vars=keep_vars), model_path)
    