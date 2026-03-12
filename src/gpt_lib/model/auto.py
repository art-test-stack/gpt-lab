import math, os
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Optional
from datetime import datetime
import subprocess
import warnings
import torch

from gpt_lib.utils.special_tokens import SpecialTokens
from gpt_lib.utils.schemas import GPTConfig, TransformerConfig, TokenizerConfig
from gpt_lib.utils.default import MODELS_FOLDER, TOKENIZERS_FOLDER
from gpt_lib.tokenizer.tokenizer import Tokenizer
from gpt_lib.model.gpt import GPTModel, build_meta_model


def get_scaling_params(model):
    pass


class AutoGPTConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    basename: str = "ic1"
    dirname: str | Path = MODELS_FOLDER

    # Tokenizer config
    # If None, will be set to vocab size scaling law based on Tao et al. 2O24 (https://arxiv.org/abs/2407.13623)
    tokenizer_model: Optional[str] = None # none, auto, or name
    vocab_size: Optional[int] = None 
    pat_str: Optional[str] = None
    special_tokens: Optional[SpecialTokens] = None

    # Model
    depth: int = 20
    d_head: int = 2**9 # 512
    max_seq_len: int = 2048
    aspect_ratio: int = 2**8
    window_pattern: Optional[str] = None
    window_size: Optional[int] = None
    attn_softcap: Optional[float] = None # not supported yet
    softcap: Optional[float] = None

    # Training 
    total_batch_size: int = -1
    # opt_schema = make a schema for model optimizer given the model' module

    # Auto-scaling targets (if set, will override other parameters to meet these targets based on scaling laws)
    # target_param_data_ratio is overiden by target_flops which is overiden by num_steps
    num_steps: int = -1
    target_time: int = -1 # target time in seconds (default: -1 means is overriden by target flops)
    target_flops: float = -1.0
    target_param_data_ratio: float = 11.0

    def model_post_init(self, context):
        try:
            git_commit = subprocess.run(["git", "log", "-n", "1"], check=True).stdout.decode("utf-8").split("\n")[0].split()[1][:7]
        except Exception as e:
            warnings.warn(f"Couldn't get git commit from current branch. Model will be saved with 'nocommit' suffix. Error: {e}")
            git_commit = "nocommit"
        date = datetime.today().ctime().replace(" ", "_").replace(":", "-")
        self.name = f"{self.basename}_{self.depth}_{git_commit}_{date}"
        if self.vocab_size is not None and self.vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 to ensure all unicode characters are supported.")
        # TODO: check that basename is valid (e.g. no special characters, not too long, etc.)
        if self.basename is not None and (not isinstance(self.basename, str) or len(self.basename) == 0):
            raise ValueError("Basename must be a non-empty string.")
        if TOKENIZERS_FOLDER.exists():
            # check if tokenizer with same name already exists in cache
            existing_tokenizer_path = TOKENIZERS_FOLDER / f"{self.basename}_tok_{int(self.vocab_size/1000)}k_corpu_meta.pkl"
            if existing_tokenizer_path.exists():
                print(f"Warning: Tokenizer with name {self.basename}_tok_{int(self.vocab_size/1000)}k already exists in cache. It will be overwritten.")

    def get_config(self) -> GPTConfig:
        # Initiate instance of optimized GPTConfig to access default values and methods
        base_dim = self.depth * self.aspect_ratio
        tf_config_dict = dict(
            n_layers=12,
            d_head=self.d_head,
            d_model=((base_dim + self.d_head - 1) // self.d_head) * self.d_head, # Round up to nearest multiple of d_head
            window_pattern=self.window_pattern,
            window_size=self.window_size,
        )
        if self.vocab_size is not None:
            tf_config_dict["vocab_size"] = self.vocab_size
        if self.softcap is not None:
            tf_config_dict["softcap"] = self.softcap
        if tf_config_dict["window_pattern"] is not None:
            tf_config_dict["window_pattern"] = self.window_pattern
        
        d12_config = TransformerConfig(**tf_config_dict)

        tokenizer_config_dict = dict(
            name=f"{self.basename}_{vocab_size//1000}k",
            dirname=self.dirname,
            vocab_size=int(vocab_size),
            max_context=self.max_seq_len,
            pat_str=self.pat_str,
            special_tokens=self.special_tokens
        )
        # TODO: make auto config for tokenizer 
        # -> based on a vocab size / split pattern 
        # -> find an appropriate tokenizer given scale? 
        # otherwise -> train a new one
        tokenizer_config = TokenizerConfig(**tokenizer_config_dict)
        # trainer_config = None
        _config = GPTConfig(device="meta").model_dump()
        model_config = _config["model"]
        tokenizer_config = _config["tokenizer"]

        # if device_type == "cuda":
        #     gpu_device_name = torch.cuda.get_device_name(0)
        #     gpu_peak_flops = get_peak_flops(gpu_device_name)
        #     print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
        # else:
        #     gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

        # if self.vocab_size is None:
            
        # TODO: implement auto configuration of model based on dataset size, training steps, and target performance. This is a complex topic and may require extensive experimentation and research to get right. For now, we will just use the provided depth and set other parameters to default values.
        # 1. Model config:
        meta_model = build_meta_model(self.depth)
        model_config = dict()
        base_dim = self.depth * self.aspect_ratio
        model_config["n_layers"] = self.depth
        model_config["d_head"] = self.d_head
        model_config["n_heads"] = model_config["d_model"] // model_config["d_head"]
        
        target_tokens = int(self.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

        if self.total_batch_size == -1:
            D_REF = self.target_param_data_ratio * build_meta_model(12).scaling_params()
            B_REF = 2**19
            self.total_batch_size = 2 ** round(math.log2(B_REF * (target_tokens / D_REF) ** 0.383))
        total_batch_size = args.total_batch_size # user-provided override is possible

        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")


        if self.num_steps > 0:
            self.target_flops = num_flops_per_token * total_batch_size * self.num_steps
            # self.target_param_data_ratio 
        elif self.target_flops > 0:
            self.num_steps = round(self.target_flops / (num_flops_per_token * total_batch_size))
        elif self.target_param_data_ratio > 0:
            num_steps = target_tokens // total_batch_size
        else:
            raise ValueError("At least one of num_steps, target_flops, or target_param_data_ratio must be set to a positive value.")
        
        if self.target_flops > 0:
            vocab_size = .2 * self.target_flops ** .42

        # TODO: check in tokenizer cache for existing tokenizer with ~ vocab_size
        tknzr_config = dict(
            name=f"{self.basename}",
            dirname=self.dirname,
            vocab_size=int(vocab_size),
            max_context=self.max_seq_len,
            pat_str=self.pat_str,
            special_tokens=self.special_tokens
        )
        return GPTConfig(
            name=self.basename,
            dirname=self.dirname,
            model=d12_config,
            tokenizer=TokenizerTrainerConfig(
                name=f"{self.basename}_tok_{int(vocab_size//1000)}k",
                dirname=self.dirname,
                vocab_size=int(vocab_size),
                max_context=self.max_seq_len
            )
        )

    def get_model(self) -> GPTModel:
        config = self.get_config()
        return GPTModel.from_scratch(config)