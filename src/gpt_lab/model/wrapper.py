import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt_lab.utils.schemas import (
    GPTConfig,
    TokenizerConfig,
    TransformerConfig,
)

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
