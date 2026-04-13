import math
import torch
import torch.distributed as dist
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import Any, Dict, List, Literal, Optional, Union, get_args
from pathlib import Path
import os, time
import json, pickle

import warnings
from gpt_lib.utils.common import print0
from gpt_lib.utils.default import (
    DATA_DIR,
    DEVICE,
    MODELS_FOLDER, 
    VOCAB_SIZE, 
    MAX_CONTEXT, 
    NUM_HEADS, 
    NUM_LAYERS, 
    DIM_MODEL, 
    DIM_FFN, 
    DIM_HEAD, 
    WARMUP_ITERS, 
    PAT_STR,
    PatStr,
    TOKENIZERS_FOLDER,
)
from gpt_lib.utils.types import (
    AttnImplTypes,
    Devices,
    Dtypes,
    LossReductionTypes,
    LossTypes,
    NormalizationTypes,
    PositionalEncodingTypes,
    TfTypes,
    TokenizerSources,
    TpModes,
)
from gpt_lib.utils.special_tokens import SpecialTokens


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ParallelismConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False

    mode: TpModes = "dp"
    world_size: int
    dp_size: int = 1
    dp_size: int = 1

    tp_size: int = 1
    tp_rank: int = 0
    # dp_group: dist.ProcessGroup | None = None
    # tp_group: dist.ProcessGroup | None = None

    n_heads_q: Optional[int] = None
    n_heads_kv: Optional[int] = None
    d_head_q: Optional[int] = None

    tp_mode: TpModes = "row"

    @property
    def local_heads_q(self) -> int:
        if self.n_heads_q is None:
            raise ValueError("n_heads_q is not set for TensorParallelConfig")
        return self.n_heads_q // self.tp_size
    
    @property
    def local_heads_kv(self) -> int:
        if self.n_heads_kv is None:
            raise ValueError("n_heads_kv is not set for TensorParallelConfig")
        return self.n_heads_kv // self.tp_size

class TokenizerConfig(BaseModel):
    model_config = ConfigDict(
        json_encoders={Path: str},
    )
    name: str = "ic1_tok"
    dirname: Union[str, Path] = TOKENIZERS_FOLDER
    dircorpus: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None
    vocab_size: int = VOCAB_SIZE
    pat_str: str = "gpt4"
    special_tokens: Optional[SpecialTokens] = Field(default_factory=SpecialTokens)
    source: TokenizerSources = "tiktoken"

    def model_post_init(self, context: Any) -> None:
        if self.pat_str in PAT_STR.keys():
            self.pat_str = PAT_STR.get(self.pat_str)  # Use predefined pattern if pat_str is a key in PAT_STR
        else:
            warnings.warn(f"Using custom pat_str {self.pat_str!r} without validation." \
                          "Make sure it is a valid regex pattern for tokenization.")

        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)
        cleaned_name = self.name.split("/")[-1] # Remove leading/trailing slashes
        if not self.dirname.name == cleaned_name: # add model name to path if not already included
            self.dirname = self.dirname / cleaned_name
        if self.dircorpus is not None and isinstance(self.dircorpus, str):
            self.dircorpus = Path(self.dircorpus) 
        if not self.dirname.exists():
            self.dirname.mkdir(parents=True, exist_ok=False)

    def get_mergeable_ranks(self) -> dict:
        if not self.dirname.exists():
            raise FileNotFoundError(f"Tokenizer directory {self.dirname} does not exist.")
        mergeable_ranks_path = self.dirname / "vocab.pkl"
        if not mergeable_ranks_path.exists():
            raise FileNotFoundError(f"Mergeable ranks file {mergeable_ranks_path} does not exist.")
        with open(mergeable_ranks_path, "rb") as f:
            mergeable_ranks = pickle.load(f)
        print0(f"Loaded mergeable ranks from {mergeable_ranks_path}. Size: {len(mergeable_ranks)}")
        if self.vocab_size == -1:
            self.vocab_size = len(mergeable_ranks) + len(self.special_tokens)
        assert len(mergeable_ranks) + len(self.special_tokens) == self.vocab_size , "Mergeable ranks size does not match vocab size."
        return mergeable_ranks
    
    @classmethod
    def from_directory(cls, name, cachedir: Optional[Union[str, Path]] = None) -> "TokenizerConfig":
        if cachedir is None:
            cachedir = TOKENIZERS_FOLDER
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)
        path: Path = cachedir / name / "config.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No such tokenizer config file: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    
    def save_to_directory(self, directory: Optional[Union[str, Path]] = None):
        if directory is not None:
            if isinstance(directory, str):
                directory = Path(directory)
        else:
            directory = self.dirname
        cleaned_name = self.name.split("/")[-1] # Remove leading/trailing slashes
        if not directory.name == cleaned_name: # add model name to path if not already included
            directory = directory / cleaned_name
        config_path = directory / "config.pkl"
        config_path.mkdir(parents=True, exist_ok=False)

        with open(str(config_path), "wb") as f:
            pickle.dump(self, f)


class TokenizerTrainerConfig(TokenizerConfig):
    model_config = ConfigDict(
        json_encoders={Path: str},
    )
    max_chars: int = -1
    chars_per_doc: int = -1
    merges_per_pass: int = 512 # Only used for fbpe
    num_proc: int = -1
    trainer: Literal["tiktoken", "huggingface", "bpe", "fbpe", "rbpe", "dummy"] = "huggingface"
    show_progress: bool = True
    to_save: bool = True
    
    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.trainer == "tiktoken" and self.pat_str == "":
            warnings.warn("Using tiktoken trainer with an empty pat_str may lead to suboptimal tokenization. Consider using a regex pattern for better tokenization performance.")
        
        if self.max_chars == -1:
            self.max_chars = int(self.vocab_size * 1000 * 2.5) # ~3.5 characters per token on average, adjust as needed based on your corpus
        if self.chars_per_doc == -1:
            self.chars_per_doc = self.max_chars // 1000 # Default to 1000 documents if not specified, adjust as needed
        if self.num_proc <= 0:
            self.num_proc = min(32, (os.cpu_count() or 1) - 1) # Use all available CPUs minus one for training, adjust as needed
    
    def save_to_directory(self, directory: Optional[Union[str, Path]] = None):
        if directory is not None:
            if isinstance(directory, str):
                directory = Path(directory)
        else:
            directory = self.dirname
        config_path = directory / "config.pkl"
        if not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(config_path), "wb") as f:
            pickle.dump(self, f)

        # TODO: consider saving with an other tool
        json_path = TOKENIZERS_FOLDER / "tokenizers.json"
        # df = df[df["name"] != self.name]  # Remove existing entry if it exists

        new_row = {
            "datetime": time.time(),
            "name": self.name,
            "vocab_size": self.vocab_size,
            "special_tokens": len(self.special_tokens.list()),
            "source": self.source,
            "trainer": self.trainer,
            "directory": str(directory),
            "corpus_files": self.dircorpus if isinstance(self.dircorpus, str) else str(self.dircorpus),
            "chars_per_doc": self.chars_per_doc,
            "corpus_nb_chars": self.max_chars,
        }
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(new_row)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        

class DatasetConfig(BaseModel):
    name: str
    hfkwargs: dict = Field(default_factory=dict)
    output_dir: str = "dataset"
    column_name: str = "text"
    postprocess: Optional[str] = None
    upload_name: Optional[str] = None
    shuffle: bool = False
    sorted: bool = True
    max_shards: Optional[int] = None
    streaming: bool = False
    # source: str
    # split: Literal["train", "validation", "test"]
    # seed: Optional[int]
    # shard_size: Optional[int]
    # num_shards: Optional[int]
    # data_dir: Optional[Union[str,Path]] = DATA_DIR
    # num_proc: Optional[int]
    # stream: bool = True

class DataLoaderConfig(BaseModel):
    batch_size: int = 1
    sequence_length: int = 1024
    n_tokenizer_threads: int = -1
    tokenizer_batch_size: int = 128
    buffer_size: int = 10000
    device: str = "cuda"
    use_pin_memory: bool = False

class BaseConfig(BaseModel):
    data_dir: Union[str, Path] = DATA_DIR

    def model_post_init(self, context: Any) -> None:
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

class DownloadConfig(BaseConfig):
    max_retries: int = 5
    retry_delay: int = 5  # in seconds
    num_workers: int = 4
    max_shards: int = 1000

class TransformerConfig(BaseModel):
    # model_config = ConfigDict(frozen=True)
    tf_type: TfTypes = "dense"

    vocab_size: int = VOCAB_SIZE
    max_context: int = MAX_CONTEXT

    positional_encoding: PositionalEncodingTypes = "rope" # Options: "positional", "rope"

    # TODO: Same structure as transformers.RopeParameters for huggingface compatibility
    # https://huggingface.co/docs/transformers/v5.0.0rc1/internal/rope_utils
    rope_params: dict = Field(default_factory=lambda: {"rope_theta": 10_000, "rope_type": "default"})  # Used if positional_encoding is "rope"

    d_model: int = DIM_MODEL
    d_ffn: int = DIM_FFN  # 4 * dim_model
    n_heads: int = NUM_HEADS
    n_kv_heads: Optional[int] = None # GQA
    n_layers: int = NUM_LAYERS
    d_head: int = DIM_HEAD  # dim_model // num_heads
    tie_word_embeddings: bool = True # TODO: implement it in model

    dropout: float = .0 # DROPOUT
    attention_dropout: Optional[float] = None

    norm_before_attn: bool = True
    normalization: NormalizationTypes = "rms"  # Options: "rms", "layer"
    norm_eps: float = 1e-8
    act_func: str = "swiglu" # TODO: make it compatible with model.DenseTransformer implementation

    # TODO: padged attention implementation
    attn_impl: AttnImplTypes = "sdpa"  # Options: "sdpa", "flash_attention", "impl". Not recommended : "impl" if return_weights=False.
    # TODO: # layer_types: Optional[List[TParams]] = None  # e.g., ["standard", "standard", "moe", ...] length must be n_layers
    enable_gqa: bool = False
    
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    # Based on: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
    window_pattern: Optional[str] = "SSSL" # Can only be composed of 'L' and 'S' characters
    window_size: Optional[int] = None  # Size of short windows
    _window_sizes: List[tuple[int, int]] = PrivateAttr(default_factory=list) # TODO later: make it dynamic

    softcap: float = 18.0

    quantization: Optional[str] = None 
    
    def model_post_init(self, context: Any) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.d_head != self.d_model // self.n_heads:
            warnings.warn(f"d_head ({self.d_head}) is not equal to d_model/n_heads ({self.d_model // self.n_heads}). This may lead to unexpected behavior in attention mechanisms.")
        
        self.n_kv_heads = getattr(self, "n_kv_heads", None) or self.n_heads
        self.attention_dropout = self.attention_dropout if self.attention_dropout is not None else self.dropout

        if not self.norm_before_attn:
            warnings.warn("Using post-attention normalization (norm_before_attn=False) may lead to training instability.")
        
        # TODO: handles warnings fallbacks
        # if self.attn_impl == "flash_attention":
        #     if not is_flash_attn3_available_from_kernel():
        #         warnings.warn("FlashAttention 3 kernel is not available. Falling back to standard attention.")
        #         self.attn_impl = "sdpa"
        #     # try:
        #     #     import flash_attn
        #     # except ImportError:
        #     #     warnings.warn("FlashAttention is not installed. Falling back to standard attention.")
        #     #     self.attn_impl = "sdpa"
        # if self.attn_impl == "impl":
        #     warnings.warn("Using 'impl' attention type is not recommended for production use. Only use for experimentation or retrieve attention weights.")
        if self.window_pattern is None:
            self.window_pattern = "L"
        self.window_size = self.window_size or (self.max_context // 4)  # Default short window size is 1/4 of max context
        self._window_sizes = self._compute_window()
        # freeze model_config manually to prevent issues with nested models
        self.model_config["frozen"] = True
        

    def _compute_window(self) -> str:
        pattern = self.window_pattern.upper()
        assert all(c in {'L', 'S'} for c in pattern), "Invalid characters in window_pattern. Only 'L' and 'S' are allowed."

        window_table = {
            'L': (-1, 0), # or (self.max_context, 0) works
            'S': (self.window_size, 0)
        }
        window_sizes = []
        for idx in range(self.n_layers - 1):
            char = pattern[idx % len(pattern)]
            window_sizes.append(window_table[char])
        window_sizes.append((-1, 0))  # Final layer always long
        return window_sizes
    
class DenseTransformerConfig(TransformerConfig):
    pass

class MoETransformerConfig(TransformerConfig):
    tf_type: TfTypes = "moe"
    nb_experts: int = 16
    expert_capacity_factor: float = 1.0

class LossConfig(BaseModel):
    loss_fn: LossTypes = "cross_entropy"
    kwargs: dict = Field(default_factory=dict)
    ignore_index: int = -100
    reduction: LossReductionTypes = "mean"

class GenerationConfig(BaseModel):
    max_length: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    seed: Optional[int] = None
    stream: bool = False
    use_cache: bool = True

    def model_post_init(self, context: Any) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be a positive float.")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be in the range [0.0, 1.0].")
        if self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be a positive integer.")
        if self.seed is None or self.seed < 0:
            self.seed = 42  # Ensure seed is within valid range for torch.manual_seed

class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Training settings
    dist_info: dict = Field(default_factory=dict) # Used for distributed training, populated by get_dist_info()

    # Training hparams
    n_steps: int = 1 # TODO: decide whether step = #forward pass or #tokens seen by model
    n_acc_steps: int = 100
    target_time: float = -1.0 # in seconds, overrides n_steps if > 0
    total_batch_size: int = -1 # Overrides device_batch_size if > 0
    total_tokens: int = -1 # Overrides n_steps if > 0, calculated as total_batch_size * n_steps
    
    device_batch_size: int = 1 

    # Optimization hyperparameters
    optim_config_path: Optional[str] = None # Path to optimizer config file. If not set, will use default config based on model size.
    lr_embeddings: float = .3 
    lr_transformer: float = .02
    lr_head: float = .008
    lr_residuals: float = .5
    weight_decay: float = 0.28
    adamw_weight_decay: Optional[float] = None # ignored for now
    muon_weight_decay: Optional[float] = None # ignored for now
    lr_warmup_steps: int = WARMUP_ITERS
    lr_warmdown_ratio: float = 0.65
    final_lr_ratio: float = 0.05
    batch_lr_scale: float = 1.0
    weight_decay_scale: float = 1.0

    batch_size_scheduling: bool = False # TODO
    bs_warmup_iters: Optional[int] = None # TODO

    # Loggin settings
    n_flops_per_token: Optional[float] = None 
    
    # Dtype settings
    use_amp: bool = False
    fp8: bool = False

    # Evaluatement settings
    eval_bpb_every: int = 250 # Evaluate val bpb every N steps (-1 = disable)
    n_bpb_tokens: int = 80*524288 # Number of tokens to evaluate val loss on
    eval_core_every: int = 2000 # Evaluate CORE metric every N steps (-1 = disable)
    n_core_tokens: int = 500 # Examples per task for CORE metric
    sample_every: int = 2000 # Sample from model every N steps (-1 = disable)
    
    # Checkpoint settings
    save_every: int = -1 # default: -1 (only at the end)
    log_every: int = -1 # default: -1 (only at the end)

    def lr_schedule(self, step: int) -> float:
        n_steps = self.n_steps
        warmup_iters = self.lr_warmup_steps
        warmdown_iters = round(self.lr_warmdown_ratio * n_steps)
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        elif step <= n_steps - warmdown_iters:
            return 1.0
        else:
            progress = (n_steps - step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * self.final_lr_ratio
    
    def muon_momentum_schedule(self, step: int) -> float:
        n_steps = self.n_steps
        warmup_iters = self.lr_warmup_steps
        warmdown_iters = round(self.lr_warmdown_ratio * n_steps)
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        elif step <= n_steps - warmdown_iters:
            return 1.0
        else:
            progress = (n_steps - step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * self.final_lr_ratio
    
    def weight_decay_schedule(self, step: int) -> float:
        return self.weight_decay_scale *  0.5 * (1 + math.cos(math.pi * step / self.n_steps))

class GPTConfig(BaseModel):
    """
    # GPTConfig
    GPTConfig is the configuration class for GPT models. It encapsulates all the necessary settings for
    defining the architecture, tokenizer, and training objectives of a GPT model. It provides methods 
    to save and load configurations. It derives from Pydantic's BaseModel for easy serialization and validation.

    Args:
        name (str): The name of the model.
        tokenizer (TokenizerConfig): Configuration for the tokenizer.
        dir (str | Path): Directory to save/load the model.
        model (TransformerConfig): Configuration for the transformer model.
        loss (LossConfig): Configuration for the training loss.

    ## Methods:
        to_file (str -> None): Save the configuration to a file in the specified format.
        from_file(model_name: str, model_dir: str | Path): Load the model configuration from a file.
        auto_init(auto_config: AutoGPTConfig): Automatically initialize a GPTConfig based on an AutoGPTConfig, inferring missing parameters.
    """
    model_config = ConfigDict(
        json_encoders={Path: str},
        # frozen=True
    )
    name: str = "ic1" # TODO: change it to something more general like 'base_model' -> generate different config to different state model (pretrained, finetuned, etc.)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    dirname: str | Path = MODELS_FOLDER
    model: TransformerConfig = Field(default_factory=TransformerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    # trainer: Optional[TrainingConfig] = Field(default_factory=Optional)
    dtype: Dtypes = "bfloat16"
    device: Devices = DEVICE

    def model_post_init(self, context: Any) -> None:
        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)
        if not self.dirname.name == self.name:
            self.dirname = self.dirname / self.name
        if not self.dirname.exists():
            self.dirname.mkdir(parents=True, exist_ok=True)

        if not hasattr(self.model, "vocab_size"):
            raise ValueError("Model configuration must have a vocab_size attribute.")
        if not hasattr(self.tokenizer, "vocab_size"):
            raise ValueError("Tokenizer configuration must have a vocab_size attribute.")
        if hasattr(self.model, "vocab_size") and hasattr(self.tokenizer, "vocab_size"):
            if self.model.vocab_size != self.tokenizer.vocab_size:
                raise ValueError(f"Model vocab_size ({self.model.vocab_size}) does not match tokenizer vocab_size ({self.tokenizer.vocab_size})")
            
        if not hasattr(self.model, "max_context"):
            raise ValueError("Model configuration must have a max_context attribute.")
        
        self.dtype = getattr(torch, self.dtype)
        self.device = torch.device(self.device)

    def __eq__(self, other: "GPTConfig") -> bool:
        if not isinstance(other, GPTConfig):
            return False
        return self.__dict__ == other.__dict__

    def to_file(self, mode="json") -> None:
        suffix_ = "pickle" if mode == "pickle" else "json"
        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)
        path = self.dirname  / f"config.{suffix_}"
        if mode not in ["json", "python", "pickle"]:
            raise ValueError(f"Unsupported mode: {mode}")
        
        with open(str(path), "wb") as f:
            if mode == "pickle":
                pickle.dump(self, f)
            else:
                json.dump(self.model_dump(mode=mode), f, indent=4)
        # self.dirname = Path(self.dirname)

    @classmethod
    def from_file(cls, model_name: str, model_dir: str | Path = MODELS_FOLDER) -> "GPTConfig":
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        config_path_json = model_dir / model_name / "config.json"
        config_path_pickle = model_dir / model_name / "config.pickle"
        if config_path_json.exists():
            with open(config_path_json, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            return cls.model_validate(config_dict)
        elif config_path_pickle.exists():
            with open(config_path_pickle, "rb") as f:
                config: GPTConfig = pickle.load(f)
            return config
        else:
            raise FileNotFoundError(f"No configuration file found for model {model_name} in {model_dir}")
        
    @classmethod
    def from_yaml(cls, path: str | Path) -> "GPTConfig":
        import yaml
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)


class TransformerOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: torch.Tensor
    attentions: List[torch.Tensor] | None = None
    hidden_states: List[torch.Tensor] | None = None
    past_key_values: Any | dict | None = None

class ModelOutput(TransformerOutput):
    loss: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None

class ModelCompletionOutput(ModelOutput):
    completions: Optional[List[str]] = None
    done: bool = False

class TrainingState(BaseModel):
    step: int = 0
    best_val_loss: float = float("inf")
    early_stopping_counter: int = 0
    loss_train: List[float] = Field(default_factory=list)
    loss_val: List[float] = Field(default_factory=list)
    metrics_train: List[dict] = Field(default_factory=list)


class TrainingResults(BaseModel):
    train_loss: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    steps: List[int] = Field(default_factory=list)

class TrainingMetrics(BaseModel):
    time: List[float] = Field(default_factory=list)
    step: List[int] = Field(default_factory=list)
    tokens: List[int] = Field(default_factory=list)
    epochs: List[int] = Field(default_factory=list)
    accuracy: List[float] = Field(default_factory=list)
    loss: List[float] = Field(default_factory=list)
    val_accuracy: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    best_val_loss: List[float] = Field(default_factory=list)
    core: List[float] = Field(default_factory=list)

    def append(self, state: TrainingState, elapsed_time: float, tokens_processed: int, accuracy: float, val_accuracy: float, core_usage: float) -> None:
        self.time.append(elapsed_time)
        self.step.append(state.step)
        self.tokens.append(tokens_processed)
        self.epochs.append(len(state.train_losses))
        self.accuracy.append(accuracy)
        self.loss.append(state.train_losses[-1] if state.train_losses else float('nan'))
        self.val_accuracy.append(val_accuracy)
        self.val_loss.append(state.val_losses[-1] if state.val_losses else float('nan'))
        self.best_val_loss.append(state.best_val_loss)
        self.core.append(core_usage)


# This is dummy
def get_config_from_huggingface(model_name: str) -> TransformerConfig:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return TransformerConfig(
        tokenizer=tokenizer.encode,
        pad_id=pad_id,
        vocab_size=vocab_size
    )