from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt_lab.utils.schemas import (
    GPTConfig,
    TokenizerConfig,
    TransformerConfig,
    ModelOutput,
    TransformerOutput,
    GenerationConfig, ModelCompletionOutput
)
from gpt_lab.utils.logging import log0, log_error
from gpt_lab.tokenizer import Tokenizer
from gpt_lab.model.utils import KVCache
from typing import List, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


def init_mistral_model(model_name="mistralai/Mistral-7B-Instruct-v0.1", device="cpu"):
    # raise NotImplementedError("WIP. Mistral model initialization is not yet supported.")
    from transformers import MistralConfig
    m_pat = model_name.split("/")
    if not m_pat[0] == "mistralai" and len(m_pat) != 2:
        raise ValueError("Model name should start with 'mistralai/'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).to(device)
    config: MistralConfig = model.config
    transformer_config = TransformerConfig(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        d_ffn=config.intermediate_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        d_head=config.head_dim,
        # d_head=config.hidden_size // config.num_attention_heads,
        hidden_act=config.hidden_act,
        norm_eps=config.rms_norm_eps,
        pad_id=config.pad_token_id,
        max_context=config.max_position_embeddings,
        rope_params=config.__dict__.get("rope_parameters", {"rope_theta": config.rope_theta, "rope_type": "default"}),
        window_size=config.sliding_window,
        dropout=getattr(config, "dropout", 0.0)
    )
    tokenizer_config = TokenizerConfig(
        name=model_name,
        source="huggingface",
        vocab_size=config.vocab_size,
        max_context=config.n_positions,

    )
    gpt_config = GPTConfig(
        name=model_name,
        tokenizer=None,  # tokenizer is handled separately
        model=transformer_config,
        objective=None,  # objective can be defined as needed
        dirname=""  # directory can be set as needed
    )
    return model, tokenizer, gpt_config

class HFModelWrapper:
    def __init__(self, model_name="openai-community/gpt2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, text, max_len=50):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_len,
                do_sample=False,  # greedy
                pad_token_id=self.tokenizer.eos_token_id
            )
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred


class Engine:
    """Wrapper around the model and tokenizer to provide a unified interface for inference.
    """
    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            tokenizer: Optional[Tokenizer] = None,
            error_on_none: bool = True,
        ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # TODO: Add support for loading Dense vs MoE vs Hybrid models vs DUMAS
        self.main = model
        if error_on_none:
            if self.main is None:
                log_error("Model is None. Please provide a valid model.", error_type=ValueError, logger=logger)
            if self.tokenizer is None:
                log_error("Tokenizer is None. Please provide a valid tokenizer.", error_type=ValueError, logger=logger)
            if not hasattr(self.main, "config"):
                log_error("Model does not have a config attribute. Please provide a valid model with a config.", error_type=AttributeError, logger=logger)
            if not hasattr(self.main, "vocab_size"):
                log_error("Model does not have a vocab_size attribute. Please provide a valid model with a vocab_size.", error_type=AttributeError, logger=logger)
            if not self.main.vocab_size == self.tokenizer.vocab_size:
                log_error(f"Model vocab size ({self.main.vocab_size}) does not match tokenizer vocab size ({self.tokenizer.vocab_size}). Please provide a compatible model and tokenizer.", error_type=ValueError, logger=logger)
            if not hasattr(self.tokenizer, "bos_token_id"):
                log_error("Tokenizer does not have a bos_token_id attribute. Please provide a valid tokenizer with a bos_token_id.", error_type=AttributeError, logger=logger)
        self.config = self.main.config if (self.main is not None and hasattr(self.main, "config")) else None
        self.vocab_size = self.main.vocab_size if (self.main is not None and hasattr(self.main, "vocab_size")) else None
        self.bos_token_id = self.tokenizer.bos_token_id if (self.tokenizer is not None and hasattr(self.tokenizer, "bos_token_id")) else None
    
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
        if hasattr(self.main, "config"):
            return f"Engine(config={str(self.main.config)}, \nmodel={str(self.main)})"
        return f"Engine(model={str(self.main)})"

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
            log0("Assistant model functionality is not yet implemented. "
                 "Assistant model provided is just ignored.", 
                 level="warning", logger=logger)
        if generation_config is None:
            log0("No generation config provided. Using default generation "
                 "config with provided kwargs.", level="warning", logger=logger)
        if not generation_config.use_cache:
            log0("GenerationConfig.use_cache is False. "
                 "Generation may be slow as prefill will be done for each step.", 
                 level="warning", logger=logger)
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
    def from_huggingface(
            cls,
            model_name: str,
        ) -> Engine:
        # TODO: Implement conversion from Huggingface models for compatibility
        log0(f"Loading model from Huggingface: {model_name}. This is experimental and may not work as expected. " 
             "Only works with Mistral model for now.", level="warning", logger=logger)
        model, tokenizer, config = init_mistral_model(model_name)
        gpt = cls(
            model=model,
            tokenizer=tokenizer
        )
        return gpt
    