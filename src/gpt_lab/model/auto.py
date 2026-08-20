import math
import warnings

from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

from gpt_lab.utils.default import DATA_DIR, MODELS_FOLDER, TOKENIZERS_FOLDER
from gpt_lab.utils.special_tokens import SpecialTokens
from gpt_lab.utils.schemas import (
    MetaConfig,
    TokenizerTrainerConfig,
    TransformerConfig,
    AttnImplTypes,
)
from gpt_lab.utils.common import print0, print0_dict
from gpt_lab.utils.logging import log0, log_error
from gpt_lab.tokenizer.auto import (
    compute_optimal_vocab_size,
    load_tokenizer,
    resolve_tokenizer,
    round_vocab_size,
    train_new_tokenizer,
)
from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.model.checkpoint import build_meta_model, make_default_run_name

import logging
logger = logging.getLogger(__name__)

class AutoGPTConfig(BaseModel):
    name: str = "ic1" # useless actually lol
    dirname: str | Path = MODELS_FOLDER
    run_name: Optional[str] = None # if None, will be set to {device_name}_{basename}_{depth}_{git_commit}_{date} in model_post_init
    random_seed: int = 42
    dist_info: Optional[dict] = None # if None, will be initialized in model_post_init based on the current distributed environment

    # Tokenizer config
    # If None, will be set to vocab size scaling law based on Tao et al. 2O24 (https://arxiv.org/abs/2407.13623)
    tokenizer_model: Optional[str] = None # none, auto, <name> or clamp
    train_tokenizer: bool = False
    vocab_size: int = -1 
    pat_str: Optional[str] = None
    special_tokens: Optional[SpecialTokens] = None
    tokenizer_dir: str | Path = TOKENIZERS_FOLDER
    tokenizer_trainer: TokenizerTrainerConfig = Field(
        default_factory=TokenizerTrainerConfig
    )

    # Model
    depth: int = 20
    aspect_ratio: int = 64
    d_head: int = 2**9 # 512
    n_kv_heads: Optional[int] = None
    # Deprecated compatibility alias. The old field was named as a dimension
    # but was always consumed as a head count.
    d_kv_head: Optional[int] = Field(default=None, exclude=True)
    max_seq_len: int = 2048
    window_pattern: Optional[str] = None
    window_size: Optional[int] = None
    attn_softcap: Optional[float] = None # not supported yet
    softcap: Optional[float] = 18.0
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
        if self.special_tokens is None:
            self.special_tokens = SpecialTokens()

        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        if self.d_head <= 0:
            raise ValueError("d_head must be positive")

        if self.d_kv_head is not None:
            if self.n_kv_heads is not None and self.n_kv_heads != self.d_kv_head:
                raise ValueError(
                    "n_kv_heads and deprecated d_kv_head disagree; provide only "
                    "n_kv_heads"
                )
            warnings.warn(
                "d_kv_head is deprecated because it represents a head count; "
                "use n_kv_heads instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.n_kv_heads = self.d_kv_head

        if self.n_kv_heads is not None and self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")

        if self.train_tokenizer and self.tokenizer_model not in (None, "auto"):
            raise ValueError(
                "tokenizer_model must be None or 'auto' when train_tokenizer=True"
            )

        if self.dist_info is None:
            from gpt_lab.utils.distributed import get_dist_info
            self.dist_info = get_dist_info()
        if self.run_name is None:
            self.run_name = make_default_run_name(self.depth, self.name, self.dist_info)
        min_vocab_size = 256 + len(self.special_tokens)
        if self.vocab_size != -1 and self.vocab_size < min_vocab_size:
            log_error(
                "Vocab size must include all 256 byte tokens and configured "
                f"special tokens, so it must be at least {min_vocab_size}.",
                logger=logger,
                error_type=ValueError,
            )
        # TODO: check that base name is valid (e.g. no special characters, etc.)
        if self.name is not None and (not isinstance(self.name, str) or len(self.name) == 0):
            log_error("Model name must be a non-empty string.", logger=logger, error_type=ValueError)
        if self.dirname is None:
            self.dirname = MODELS_FOLDER
        if isinstance(self.dirname, str):
            self.dirname = Path(self.dirname)
        if isinstance(self.tokenizer_dir, str):
            self.tokenizer_dir = Path(self.tokenizer_dir)


    def generate_gpt_config(self, device) -> MetaConfig:
        # Models are built on the meta device below; keep this argument for API
        # compatibility with existing callers.
        _ = device
        special_tokens = self.special_tokens
        min_vocab_size = 256 + len(special_tokens)

        def build_transformer_config(depth: int, vocab_size: int) -> TransformerConfig:
            base_dim = depth * self.aspect_ratio
            d_model = (
                (base_dim + self.d_head - 1) // self.d_head
            ) * self.d_head
            n_heads = d_model // self.d_head
            n_kv_heads = (
                self.n_kv_heads
                if self.n_kv_heads is not None
                else n_heads
            )
            if n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"n_heads ({n_heads}) must be divisible by n_kv_heads "
                    f"({n_kv_heads})"
                )

            return TransformerConfig(
                tf_type="dense",
                vocab_size=vocab_size,
                max_context=self.max_seq_len,
                d_model=d_model,
                d_ffn=4 * d_model,
                n_layers=depth,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_head=self.d_head,
                window_pattern=self.window_pattern or "SSSL",
                window_size=self.window_size,
                attn_impl=self.attn_impl,
                softcap=self.softcap if self.softcap is not None else 18.0,
                quantization=self.quantization,
            )

        def build_meta_model_from_depth(
            depth: int,
            vocab_size: int,
        ) -> DenseTransformer:
            return build_meta_model(
                build_transformer_config(depth, vocab_size)
            )

        def choose_pattern(vocab_size: int) -> str:
            if self.pat_str is not None:
                return self.pat_str
            if vocab_size < 64_000:
                return "gpt2"
            if vocab_size < 150_000:
                return "cl100k_base"
            return "o200k_base"

        explicit_tokenizer = self.tokenizer_model not in (None, "auto")
        requested_vocab_size = self.vocab_size

        if explicit_tokenizer:
            tokenizer = load_tokenizer(
                self.tokenizer_model,
                special_tokens=special_tokens,
                tokenizer_dir=self.tokenizer_dir,
            )
            if (
                requested_vocab_size != -1
                and requested_vocab_size != tokenizer.vocab_size
            ):
                raise ValueError(
                    f"Configured vocab_size {requested_vocab_size} does not "
                    f"match tokenizer {self.tokenizer_model!r} vocabulary "
                    f"size {tokenizer.vocab_size}."
                )
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = requested_vocab_size
            if vocab_size == -1:
                probe_model = build_meta_model_from_depth(
                    self.depth,
                    vocab_size=min_vocab_size,
                )
                n_non_vocab_params = probe_model.n_non_vocab_params()
                raw_vocab_size = compute_optimal_vocab_size(
                    n_non_vocab_params=n_non_vocab_params,
                    d_model=probe_model.config.d_model,
                )
                del probe_model

                vocab_size = round_vocab_size(
                    raw_vocab_size,
                    multiple=128,
                    minimum=min_vocab_size,
                )
                print0(
                    "Approach-1 vocabulary estimate for "
                    f"N_nv={n_non_vocab_params:.2e}: {raw_vocab_size:,.0f}; "
                    f"rounded target: {vocab_size:,}"
                )
                if self.n_steps > 0 or self.target_flops > 0:
                    log0(
                        "Automatic vocabulary sizing assumes compute-optimal "
                        "model/data allocation, but an explicit training "
                        "horizon is configured.",
                        logger=logger,
                        level="warning",
                    )

            if self.train_tokenizer:
                pat_str = choose_pattern(vocab_size)
                vocab_label = (
                    f"{vocab_size // 1_000}k"
                    if vocab_size < 1_000_000
                    else f"{vocab_size / 1_000_000:.2f}M"
                )
                tokenizer_name = f"{self.name}_{vocab_label}"
                corpus_dir = DATA_DIR / "corpus" / self.name
                log0(
                    f"Training tokenizer {tokenizer_name!r} with vocabulary "
                    f"size {vocab_size:,} and pattern {pat_str!r} on "
                    f"{corpus_dir}.",
                    logger=logger,
                    level="warning",
                )
                tokenizer = train_new_tokenizer(
                    name=tokenizer_name,
                    vocab_size=vocab_size,
                    pat_str=pat_str,
                    special_tokens=special_tokens,
                    data_dir=corpus_dir,
                    random_seed=self.random_seed,
                    tokenizer_dir=self.tokenizer_dir,
                    trainer_config=self.tokenizer_trainer,
                )
            else:
                tokenizer_name = resolve_tokenizer(
                    self.tokenizer_model,
                    vocab_size,
                    special_tokens,
                )
                tokenizer = load_tokenizer(
                    tokenizer_name,
                    special_tokens=special_tokens,
                    tokenizer_dir=self.tokenizer_dir,
                )
                vocab_size = tokenizer.vocab_size

        if tokenizer.vocab_size != vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size {tokenizer.vocab_size} does not "
                f"match final model vocabulary size {vocab_size}."
            )

        d12_model = build_meta_model_from_depth(12, vocab_size=vocab_size)
        model = build_meta_model_from_depth(self.depth, vocab_size=vocab_size)

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
        
        if not (
            self.n_steps > 0
            or self.target_param_data_ratio > 0
            or self.target_flops > 0
        ):
            raise ValueError(
                "Set n_steps, target_flops, or target_param_data_ratio to a "
                "positive value."
            )
        if self.n_steps > 0:
            # Override n_steps to a specific value if given
            n_steps = self.n_steps
            log0(f"Using user-provided number of steps: {n_steps:,}. " \
                   f"Hence, n_total_tokens={total_batch_size * n_steps:=,} and training ignores training horizon based on scaling laws. "\
                   "Recommended to set n_steps to -1 to automatically calculate the number of " \
                   "steps based on scaling law targets for training horizon.", level="warning", logger=logger)
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
                    log0("Gradient accumulation disabled. Model will be updated every step.", level="warning", logger=logger)
                else:
                    log0(f"Using user-provided number of gradient accumulation steps: {n_acc_steps}. "
                         "This may lead to suboptimal training performance if it does not align well "
                         "with the training horizon and batch size targets based on scaling laws. "
                         "Recommended to set n_acc_steps to -1 for automatic configuration based on "
                         "scaling laws.", level="warning", logger=logger)

        training_config = dict(
            n_steps=n_steps,
            n_acc_steps=n_acc_steps,
            total_batch_size=total_batch_size,
            device_batch_size=self.device_batch_size,
            batch_lr_scale=batch_lr_scale,
            weight_decay_scale=weight_decay_ratio,
            target_tokens=target_tokens,
            target_param_data_ratio=self.target_param_data_ratio,
            n_flops_per_token=n_flops_per_token,
            n_total_tokens=n_total_tokens,
        )
        meta_config = MetaConfig.model_validate(dict(
            name=self.name,
            run_name=self.run_name,
            dirname=self.dirname / self.name / self.run_name,
            model_cfg=model.config,
            tokenizer_cfg=tokenizer.config,
            base_train=training_config,
            # Every distributed rank builds this object, but only rank zero
            # should create the shared metadata file.
            autosave=self.dist_info.get("RANK", 0) == 0,
        ))
        if self.dist_info.get("IS_DDP_INITIALIZED", False):
            import torch.distributed as dist
            dist.barrier()
        # Display the generated configuration for verification
        print0_dict("AutoGPTConfig generated the following tokenizer configuration", tokenizer.config.model_dump())
        print0_dict("AutoGPTConfig generated the following model configuration", model.config.model_dump())
        
        print0_dict("Model Parameter counts", param_counts)
        print0(f"Estimated FLOPS per token: {n_flops_per_token:.2e}")
        del model, tokenizer, d12_model
        return meta_config
