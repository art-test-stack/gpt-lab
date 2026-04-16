import math

from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Tuple
import warnings

from gpt_lib.utils.default import DATA_DIR, MODELS_FOLDER, TOKENIZERS_FOLDER, PAT_STR
from gpt_lib.utils.special_tokens import SpecialTokens
from gpt_lib.utils.schemas import (
    GPTConfig, 
    TokenizerConfig, 
    TransformerConfig, AttnImplTypes
)
from gpt_lib.utils.common import print0, print0_dict
from gpt_lib.tokenizer.tokenizer import get_closest_tokenizer_size, Tokenizer
from gpt_lib.model.gpt import build_meta_model, GPTModel, DenseTransformer


class AutoGPTConfig(BaseModel):
    basename: str = "ic1"
    dirname: str | Path = MODELS_FOLDER
    name: Optional[str] = None # if None, will be set to {basename}_{depth}_{git_commit}_{date} in model_post_init
    random_seed: int = 42
    dist_info: Optional[dict] = None # if None, will be initialized in model_post_init based on the current distributed environment

    # Tokenizer config
    # If None, will be set to vocab size scaling law based on Tao et al. 2O24 (https://arxiv.org/abs/2407.13623)
    tokenizer_model: Optional[str] = None # none, auto, or name
    train_tokenizer: bool = False
    vocab_size: int = -1 
    pat_str: Optional[str] = None
    special_tokens: Optional[SpecialTokens] = None

    # Model
    depth: int = 20
    aspect_ratio: int = 64
    d_head: int = 2**9 # 512
    d_kv_head: Optional[int] = None # if None, will be set to d_head
    max_seq_len: int = 2048
    window_pattern: Optional[str] = None
    window_size: Optional[int] = None
    attn_softcap: Optional[float] = None # not supported yet
    softcap: Optional[float] = None
    attn_impl: AttnImplTypes = "sdpa" 
    quantization: Optional[str] = None # not supported yet

    # Training 
    n_acc_steps: int = -1
    total_batch_size: int = -1
    device_batch_size: int = 32
    # opt_schema = make a schema for model optimizer given the model' module
    # Auto-scaling targets (if set, will override other parameters to meet these targets based on scaling laws)
    # target_param_data_ratio is overiden by target_flops which is overiden by n_steps
    n_steps: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 11.0

    def model_post_init(self, context):
        if self.name is None:
            # try:
            #     import subprocess
            #     git_commit = subprocess.run(["git", "log", "-n", "1"], capture_output=True, timeout=10, check=True).stdout.decode("utf-8").split("\n")[0].split()[1][:7]
            # except Exception as e:
            #     warnings.warn(f"Couldn't get git commit from current branch. Model will be saved with 'nocommit' suffix. Error: {e}")
            #     git_commit = "nocommit"
            from gpt_lib.utils.report import run_command
            git_commit = run_command("git rev-parse --short HEAD") or "unkcommit"
            from datetime import datetime
            date = datetime.today().ctime().replace(" ", "_").replace(":", "-")
        
            self.name = f"{self.basename}_d{self.depth}_cmt_{git_commit}_dt_{date}"
        if self.vocab_size != -1 and self.vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 to ensure all unicode characters are supported.")
        # TODO: check that basename is valid (e.g. no special characters, etc.)
        if self.basename is not None and (not isinstance(self.basename, str) or len(self.basename) == 0):
            raise ValueError("Basename must be a non-empty string.")
        if self.dist_info is None:
            from gpt_lib.utils.distributed import get_dist_info
            self.dist_info = get_dist_info()
        if self.dirname is None:
            self.dirname = MODELS_FOLDER
        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)


    def generate_gpt_config(self, device) -> GPTConfig:

        special_tokens = SpecialTokens() # TODO: make this configurable

        def _get_tokenizer_pretrained(tname: str, source: str = "tiktoken") -> Tokenizer:
            # TODO: need to be simplified and optimized
            # if a specific tokenizer model is specified, we will use it and ignore the scaling law
            try: 
                _tconfig = TokenizerConfig(name=tname, source=source, vocab_size=-1, special_tokens=special_tokens, pat_str="")
                tokenizer = Tokenizer.from_pretrained(_tconfig)
            except Exception as e:
                _tconfig = TokenizerConfig.from_directory(name=tname)
                _mergeable_ranks = _tconfig.get_mergeable_ranks()
                tokenizer = Tokenizer(
                    mergeable_ranks=_mergeable_ranks,
                    special_tokens=special_tokens,
                    config=_tconfig
                )
            except Exception as e:
                raise ValueError(f"Could not load tokenizer model {self.tokenizer_model} from either tiktoken or local cache. Error: {e}")
            return tokenizer

        def build_meta_model_from_depth(depth: int, vocab_size: int = -1) -> DenseTransformer:
            # Initiate instance of optimized GPTConfig to access default values and methods
            # Use same vocab size for both reference and target model to ensure consistency 
            # in scaling laws
            
            base_dim = depth * self.aspect_ratio
            d_head = self.d_head
            d_model = ((base_dim + d_head - 1) // d_head) * d_head # Round up to nearest multiple of d_head
            n_heads = d_model // d_head
            d_ffn = 4 * d_model # default expansion factor of 4 for FFN dimension
            softcap = self.softcap

            window_pattern = self.window_pattern or "SSSL"
            window_size = self.window_size
            attn_impl = self.attn_impl
            max_seq_len = self.max_seq_len

            config = TransformerConfig(
                tf_type="dense", vocab_size=vocab_size, max_context=max_seq_len, d_model=d_model, d_ffn=d_ffn, 
                n_layers=depth, n_heads=n_heads, n_kv_heads=self.d_kv_head or n_heads, d_head=d_head, 
                window_pattern=window_pattern, window_size=window_size, attn_impl=attn_impl, softcap=softcap
            )

            return build_meta_model(config)

        def compute_optimal_vocab_size(depth: int) -> int:
            """
            Compute optimal vocabulary size based on scaling law from Tao et al. 2024 (https://arxiv.org/abs/2407.13623).
            
            This is a rough estimate and can be tuned based on experiments. The scaling law is based on the number of scaling parameters in the model, which is approximated here as
            $depth * (depth * aspect_ratio) ** 2
            
            If self.train_tokenizer is True: the optimal vocab size is rounded to the nearest 1000 for tokenizer cache efficiency.
            Else: the optimal vocab size is set to the closest in the tokenizer cache to maximize reuse of existing tokenizers.
            
            Args:
                depth (int): The depth of the model (number of layers).
            
            Returns:
                int: The optimal vocabulary size for the model (including special tokens).
            """
            assert (self.tokenizer_model is None) or (self.tokenizer_model == "auto") or (not self.train_tokenizer), "Tokenizer model should not be specified if train_tokenizer is True, since we will be training a new tokenizer from scratch. Please set tokenizer_model to None or 'auto'."
            
            if self.tokenizer_model not in (None, "auto"):
                tokenizer = _get_tokenizer_pretrained(self.tokenizer_model)
                return tokenizer.vocab_size # vocab size = mergeable ranks size + special tokens size
 
            # set vocab size based on scaling law from Tao et al. 2024 (https://arxiv.org/abs/2407.13623)
            # we approximate the values from their paper with a simple scaling law for vocab size based on depth and aspect ratio
            # this is a rough estimate and can be tuned based on experiments.
            # we also approximate the vocab size to the closest in the tokenizer cache to maximize reuse of existing tokenizers
            _mmodel = build_meta_model_from_depth(depth, vocab_size=1)
            n_non_vocab_scaling_params = _mmodel.n_params # vocab size = 1, so n_params ~ Nnv
            power = 0.84
            coeff = .2 / (.08 ** power) / (depth * self.aspect_ratio)
            opt_vocab_size = coeff * (n_non_vocab_scaling_params ** power) # V ~ .2 / d_model * (n_scaling_params / 0.08) ^.84
            del _mmodel # free memory
            print0(f"Number of non-vocabulary scaling parameters for depth {depth}: {n_non_vocab_scaling_params:.2e}")
            if not self.train_tokenizer:
                _, vocab_size = get_closest_tokenizer_size(opt_vocab_size)
            else:
                step = 10 ** (int(math.log10(opt_vocab_size)) - 1)
                vocab_size = round(opt_vocab_size / step) * step # round to nearest log10 for better tokenizer cache efficiency

            if vocab_size < 256:
                raise ValueError(f"Vocab size must be specified and at least 256 to ensure all unicode characters are supported. Computed optimal vocab size based on scaling law is {opt_vocab_size}, but got {self.vocab_size}. Please set vocab_size to a value >= 256.")
                
            return vocab_size + len(special_tokens) # add special tokens to vocab size
        
        vocab_size = self.vocab_size
        if vocab_size == -1:
            vocab_size = compute_optimal_vocab_size(self.depth)
            print0(f"Optimal vocab size based on scaling law for depth {self.depth}: {vocab_size:,.0f}")

        d12_model = build_meta_model_from_depth(12, vocab_size=vocab_size) # reference model for scaling laws
        model = build_meta_model_from_depth(self.depth, vocab_size=vocab_size)
        
        # initiate tokenizer based on vocab size and user config
        if (self.tokenizer_model not in (None, "auto")) or not self.train_tokenizer: 
            tname = self.tokenizer_model or get_closest_tokenizer_size(vocab_size)[0]
            tokenizer = _get_tokenizer_pretrained(tname)
        else: # otherwise train tokenizer
            # choose a pat_str based on vocab size (method/thresholds arbitrary for now)
            # TODO: need to be tuned based on experiments
            if vocab_size < 256:
                raise ValueError(f"Computed vocab size {vocab_size} is too small. Please increase the aspect ratio or set a custom vocab size.")
            elif vocab_size < 64_000:
                pat_str = "gpt2"
            elif vocab_size < 150_000:
                pat_str = "cl100k_base"
            else:
                pat_str = "o200k_base" # for now, we use the same pattern for larger vocab sizes, but ideally we should have a different pattern for very large vocab sizes to ensure good tokenization performance. This is a TODO for future improvement.

            _vs = f"{vocab_size//1000:,}k" if vocab_size < 1e6 else f"{vocab_size/1_000_000:.2f}M"
            _tname = f"{self.basename}_{_vs}"

            from gpt_lib.utils.schemas import TokenizerTrainerConfig
            from gpt_lib.tokenizer.corpus import TokenizerCorpus
            print0(f"Training new tokenizer with vocab size {vocab_size} using pattern {pat_str} on corpus from {DATA_DIR / 'corpus' / self.basename}. This may take a while...")
            corpus = TokenizerCorpus.from_sources(
                corpus_dir=DATA_DIR / "corpus" / self.basename,
                # default sources for now
                max_chars=vocab_size * 4 * 100,
                random_seed=self.random_seed,
            )
            _tok_trainer = TokenizerTrainerConfig(
                name=_tname, dirname=self.dirname, 
                vocab_size=int(vocab_size), max_context=self.max_seq_len, 
                pat_str=PAT_STR.get(pat_str, "gpt2"), special_tokens=special_tokens
            )
            tokenizer = Tokenizer.train_from_iterator(_tok_trainer, iterator=corpus.iterator())

        param_counts = model.n_params_per_layer()

        n_flops_per_token = model.estimate_flops()
        
        # model config:
        d12_n_scaling_params = d12_model.n_scaling_params()
        n_scaling_params = model.n_scaling_params()    

        target_tokens = int(self.target_param_data_ratio * n_scaling_params) # optimal tokens for the model we are about to train

        # ref model training horizon and batch size (µP paper: )
        d12_th = self.target_param_data_ratio * d12_n_scaling_params
        d12_bs = 2**19

        # optimal batch size (https://arxiv.org/abs/2505.13738)
        total_batch_size = self.total_batch_size # total batch size = n tokens per step (1 step = (n forwardbackward . world size^-1 n_acc_steps^-1) . world size . n acc_steps))
        if total_batch_size == -1:
            batch_size_ratio = target_tokens / d12_th
            predicted_batch_size = d12_bs * batch_size_ratio ** 0.383
            total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
            print0(f"AutoGPTConfig computed optimal batch size: {total_batch_size:,} tokens")

        # learning rate correction
        batch_lr_scale = 1.0
        batch_ratio = total_batch_size / d12_bs # η ∝ √(B/B_ref)
        if not batch_ratio == 1.0:
            batch_lr_scale = math.sqrt(batch_ratio)
            print0(f"Scaling learning rate by {batch_lr_scale=:.6f} based on batch size scaling law for total batch size {total_batch_size:,} tokens")
        
        # weight decay correction (https://arxiv.org/abs/2405.13698) λ = λ_ref · √(B/B_ref) · (D_ref/D)
        # TODO: https://arxiv.org/abs/2505.13738
        weight_decay_ratio = math.sqrt(total_batch_size / d12_bs) * (d12_th / target_tokens)
        if not weight_decay_ratio == 1.0:
            print0(f"Scaling weight decay by {weight_decay_ratio=:.6f} for {self.depth=}")
        
        assert self.n_steps > 0 or self.target_param_data_ratio > 0 or self.target_flops > 0 or self.target_time > 0
        if self.n_steps > 0:
            # Override n_steps to a specific value if given
            n_steps = self.n_steps
            print0(f"Using user-provided number of steps: {n_steps:,}. " \
                   f"Hence, n_total_tokens={total_batch_size * n_steps:=,} and training ignores training horizon based on scaling laws. "\
                   "Recommended to set n_steps to -1 to automatically calculate the number of " \
                   "steps based on scaling law targets for training horizon.")
        elif self.target_flops > 0:
            # Calculate the number of steps from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
            n_steps = round(self.target_flops / (n_flops_per_token * total_batch_size))
            print0(f"Calculated number of steps from target FLOPs: {n_steps:,}")
        elif self.target_param_data_ratio > 0:
            # Calculate the number of steps from the target param data ratio (the most common use case)
            n_steps = target_tokens // total_batch_size
            print0(f"Calculated number of steps from target data:param ratio: {n_steps:,}")
        else:
            raise ValueError("No training horizon specified")
        
        n_total_tokens = total_batch_size * n_steps 
        
        n_acc_steps = self.n_acc_steps
        if n_acc_steps == -1: # recommended
            tokens_per_accstep_per_rank = self.device_batch_size * self.max_seq_len
            tokens_per_accstep_per_world = tokens_per_accstep_per_rank * self.dist_info["WORLD_SIZE"]
            assert total_batch_size % tokens_per_accstep_per_world == 0, f"{total_batch_size:=,} must be divisible by tokens per accstep per world {tokens_per_accstep_per_world:=,} for automatic configuration of gradient accumulation steps."
            n_acc_steps = total_batch_size // tokens_per_accstep_per_world
        else:
            assert n_acc_steps >= 0, f"n_acc_steps must be non-negative (except n_acc_steps=-1 for automatic configuration); got {n_acc_steps=}."

            if self.dist_info["LOCAL_RANK"] == 0:
                if n_acc_steps == 0:
                    warnings.warn("Gradient accumulation disabled. Model will be updated every step.")
                else:
                    warnings.warn(f"Using user-provided number of gradient accumulation steps: {n_acc_steps}. This may lead to suboptimal training performance if it does not align well with the training horizon and batch size targets based on scaling laws. Recommended to set n_acc_steps to -1 for automatic configuration based on scaling laws.")

        training_config = dict(
            n_steps=n_steps,
            n_acc_steps=n_acc_steps,
            total_batch_size=total_batch_size,
            batch_lr_scale=batch_lr_scale,
            weight_decay_scale=weight_decay_ratio,
            target_tokens=target_tokens,
            target_param_data_ratio=self.target_param_data_ratio,
            n_flops_per_token=n_flops_per_token,
            n_total_tokens=n_total_tokens,
        )

        meta_config = dict(
            project=self.basename,
            name=self.name,
            dirname=self.dirname /self.basename / self.name,
            model=model,
            tokenizer=tokenizer,
            training_config=training_config,
        )
        # Display the generated configuration for verification
        print0_dict("AutoGPTConfig generated the following tokenizer configuration", tokenizer.config.model_dump())
        print0_dict("AutoGPTConfig generated the following model configuration", model.config.model_dump())
        
        print0_dict("Model Parameter counts", param_counts)
        print0(f"Estimated FLOPS per token: {n_flops_per_token:.2e}")

        return meta_config
    
    