"""
gpt_lab/model/checkpoint.py
Utilities for saving and loading model checkpoints, including the model state dict, optimizer state, and metadata such as general and training configuration.

Models should be saved in the following directory structure:
<CACHE_DIR>/
└── models/
    └── <model_name>/
        └── <model_tag>/
            ├── meta.json                      # immutable (model, tokenizer, git)
            └── <source>/                      # base / sft / rl
                ├── training_config.json       # per-phase config
                ├── checkpoint_state.json      # best bpb/core steps and values
                ├── checkpoint_step_000000/
                │   ├── model.pt
                │   ├── optim_rank0.pt         # optimizer state dict (optim_rank{rank}.pt if sharded, otherwise optim.pt)
                │   ├── optim_rank1.pt
                │   ├── ...                    # more optimizer shards if needed
                │   ├── trainer_state.json     # training state, rng state, data state, best bpb/core steps
                │   └── metrics.json
                ├── checkpoint_step_000100/
                │   └── ...
                └── ...
In the following code, the variables are the following: 
    - cache_dir: Path = <CACHE_DIR> - the base cache directory. Default is given by environment variable GPTLAB_CACHE_DIR or ~/.gpt_lab by default.
    - model_cachedir: Path = <CACHE_DIR>/models - the directory where models are cached. Default is given by environment variable GPTLAB_MODEL_CACHE_DIR or <CACHE_DIR>/models by default.
    - model_dir: Path = <CACHE_DIR>/models/<model_name> - the directory for a specific model. Default is <CACHE_DIR>/models/<model_name>.
    - model_tag: str = <model_tag> - the tag for a specific model run. Default is automatically determined by looking at the most recent checkpoint in the model directory.
    - tag_dir: Path = <CACHE_DIR>/models/<model_name>/<model_tag> - the directory for a specific model tag. Default is automatically determined by looking at the most recent checkpoint in the model directory with the same tag.
    - checkpoint_dir: Path = <CACHE_DIR>/models/<model_name>/<model_tag>/<source> - the directory for a specific checkpoint. Default is automatically determined by looking at the checkpoint step in the checkpoint directory, or the most recent checkpoint if checkpoint step is not specified.
"""
import json
import re
import random, numpy as np
import torch
from typing import Optional, Union, Tuple, Dict, Literal, Mapping, Callable
from pathlib import Path
from gpt_lab.utils.logging import log0, log_error, log_all
from gpt_lab.utils.default import MODELS_FOLDER
from gpt_lab.utils.distributed import get_dist_info
from gpt_lab.utils.schemas import (
    CheckpointState, 
    DataLoaderState, 
    MetaConfig,
    TrainerConfig,
    TrainerState,
    TokenizerConfig, 
    TransformerConfig, 
)
from gpt_lab.utils.system import get_git_info, get_gpu_info, get_system_info
from gpt_lab.tokenizer import build_tokenizer, Tokenizer, TokenizerConfig
from gpt_lab.model.gpt import DenseTransformer

import logging

logger = logging.getLogger(__name__)

_OPT_DEFAULT_NAME: Dict[str, Callable[[int], str]] = dict(
    ddp=lambda x: "optim.pt",
    shard=lambda x: f"optim_rank{x:d}.pt",
) # dict[mode, (rank: int) -> (filename: str)]
_ModelSources = Literal[("base")] # TODO: Add here "sft", "grpo" when implemented

def _solve_path(path: Optional[Union[str, Path]], default: Optional[Path] = None) -> Path:
    if path is None:
        return default
    elif isinstance(path, str):
        return Path(path)
    else:
        return path

def _solve_model_cache_dir(model_cachedir: Optional[Union[str, Path]]) -> Path:
    return _solve_path(model_cachedir, default=MODELS_FOLDER)

def _solve_model_tag(model_dir: Path, model_tag: Optional[Union[str, int]] = None) -> str:
    model_tags = list(sorted(model_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True))
    if not model_tags:
        log_error(f"No model checkpoints found in {model_dir}.", logger=logger, error_type=FileNotFoundError)
    if model_tag is None:
        log0(f"No model tag specified for model {model_dir.name!r}. Automatically selecting the most recent model tag {model_tags[0].name!r}.", level="warning", logger=logger)
        return model_tags[0].name
    elif isinstance(model_tag, int):
        if model_tag >= len(model_tags):
            log_error(f"Invalid model tag index {model_tag} for model directory {model_dir}.\n"
                f"Found {len(model_tags)} model tags in directory {model_dir}.", logger=logger, error_type=IndexError)
        if model_tag < 0:
            model_tag = -(model_tag + 1)
        elif model_tag >= 0:
            model_tag = len(model_tags) - model_tag
        return model_tags[model_tag].name
    elif model_tag in [model_tag.name for model_tag in model_tags]:
        return model_tag
    elif model_tag.startswith("-") and model_tag[1:].isdigit():
        index = int(model_tag[1:]) - 1
        if index < 0 or index >= len(model_tags):
            log_error(f"Invalid model tag index {model_tag!r} for model directory {model_dir}.\n"
                f"Found {len(model_tags)} model tags in directory {model_dir}.", logger=logger, error_type=IndexError)
        return model_tags[index].name
    elif model_tag == "latest":
        return model_tags[0].name
    elif model_tag == "best":
        best_tag = None
        best_bpb = float('inf')
        for tag in model_tags:
            checkpoint_dir = tag / "base"
            if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
                continue
            metrics_file = checkpoint_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            if "bpb" in metrics and metrics["bpb"] < best_bpb:
                best_bpb = metrics["bpb"]
                best_tag = tag
        if best_tag is None:
            log_error(f"Could not find a valid checkpoint for model {model_dir.name!r} with tag {model_tag!r} in directory {model_dir}.", logger=logger, error_type=ValueError)
        return best_tag
    else:
        log_error(f"Could not find checkpoint for model {model_dir.name!r} with tag {model_tag!r} in directory {model_dir}. "
            f"Found tags: {[model_tag.name for model_tag in model_tags]}. Please specify a valid checkpoint run name or index (e.g., '-1' for the most recent tag).", 
            logger=logger, error_type=ValueError)


def _save_state_dict(state_dict, ckpt_dir, filename):
    path = ckpt_dir / filename
    torch.save(state_dict, path)
    log0(f"Saved {filename} to {path}", logger=logger)

def _load_state_dict(ckpt_dir, filename, map_location=None, weight_only=True):
    path = ckpt_dir / filename
    if not path.exists():
        log_error(f"Checkpoint file {path} does not exist.", logger=logger, error_type=FileNotFoundError)
    state_dict = torch.load(path, map_location=map_location, weights_only=weight_only)
    log0(f"Loaded {filename} from {path}.", logger=logger)
    return state_dict

def build_meta_model(config: TransformerConfig) -> "DenseTransformer":
    with torch.device("meta"):
        model = DenseTransformer(config=config)
    return model

def save_meta_config(
        meta_config: Dict, 
        model_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        model_tag: Optional[str] = None,
    ):
    # TODO: not really strong
    tag_dir = _solve_path(model_dir, default=MODELS_FOLDER / meta_config.get("name", model_name) / meta_config.get("model_tag", model_tag))
    meta_cfg = MetaConfig(
        project=meta_config["name"],
        model_tag=meta_config["model_tag"],
        model_cfg=meta_config["model"].config,
        tokenizer_cfg=meta_config["tokenizer"].config,
        git_info=get_git_info(),
    )
    tag_dir.mkdir(parents=True, exist_ok=False)
    _save_state_dict(meta_cfg.model_dump(), tag_dir, filename="meta.json")

    log0(f"Saved meta config in {str(tag_dir)}.", logger=logger)

def load_meta_config(
        name: str, 
        model_tag: Optional[Union[str, int]] = None, 
        model_cachedir: Optional[Union[str, Path]] = None
    ) -> Dict:
    model_cachedir = _solve_model_cache_dir(model_cachedir)
    model_path = model_cachedir / name

    if not model_path.exists():
        log_error(f"Model directory {model_path} does not exist.", logger=logger, error_type=FileNotFoundError)

    model_ckpts = list(sorted(model_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True))
    if not model_ckpts:
        log_error(f"No checkpoints found for model {name!r} in directory {model_path}.", logger=logger, error_type=FileNotFoundError)
    
    model_tag = _solve_model_tag(model_path, model_tag=model_tag)
    tag_dir = model_path / model_tag
    if not tag_dir.exists():
        log_error(f"Checkpoint directory {tag_dir} does not exist.", logger=logger, error_type=FileNotFoundError)
    if not tag_dir.is_dir():
        log_error(f"Checkpoint path {tag_dir} is not a directory.", logger=logger, error_type=ValueError)

    _meta_cfg = _load_state_dict(tag_dir, "meta.json", map_location="cpu", weight_only=False)
    _meta_cfg = MetaConfig.model_validate(_meta_cfg)
    
    log0(f"Loaded meta config from {tag_dir / 'meta.json'}", logger=logger)
    
    meta_cfg = dict(name=_meta_cfg.project, model_tag=_meta_cfg.model_tag, dirname=tag_dir)
    meta_cfg["model_config"] = TransformerConfig.model_validate(_meta_cfg.model_cfg)
    meta_cfg["tokenizer_config"] = TokenizerConfig.model_validate(_meta_cfg.tokenizer_cfg)

    # TODO: not sure to keep this
    meta_cfg["model"] = build_meta_model(meta_cfg["model_config"])
    meta_cfg["tokenizer"] = Tokenizer.from_config(meta_cfg["tokenizer_config"])
    return meta_cfg

def save_trainer_config(trainer_config: TrainerConfig, checkpoint_dir: Union[str, Path]):
    pass

def load_trainer_config(ckpt_dir: Union[str, Path], filename: str = "training_config.json") -> Dict:
    ckpt_dir = _solve_path(ckpt_dir)
    if not ckpt_dir.exists():
        log_error(f"Checkpoint directory {ckpt_dir} does not exist.", logger=logger, error_type=FileNotFoundError)
    if not ckpt_dir.is_dir():
        log_error(f"Checkpoint path {ckpt_dir} is not a directory.", logger=logger, error_type=ValueError)
    
    trainer_cfg = _load_state_dict(ckpt_dir, filename, map_location="cpu", weight_only=False)
    log0(f"Loaded trainer config from {ckpt_dir / filename}", logger=logger)
    return trainer_cfg


def capture_rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }

def set_rng_state(rng_state: Dict[str, Union[bytes, Tuple, None]]):
    torch.random.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])

# adapted from nanochat/checkpoint.py
# https://github.com/karpathy/nanochat/blob/8180e1d/nanochat/checkpoint_manager.py#L23
def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def find_potential_model_tag(model_dir: Union[str, Path], git_info: Optional[Dict] = None) -> str:
    model_dir = _solve_path(model_dir)

    _with_same_commit = git_info is not None
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in model_dir.iterdir() if f.is_dir()]
    if not model_tags:
        log_error(f"No model checkpoints found in {model_dir}.", logger=logger, error_type=FileNotFoundError)

    # 1) normally all model tags are of the form <device>_<name>_d<number>_cmt_..., try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            if _with_same_commit:
                git_match = re.search(r"cmt_([a-f0-9]+)", model_tag)
                if git_match and git_match.group(1) == git_info["commit"][:7]:
                    candidates.append((model_depth, model_tag))
            else:
                candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take without same commit:
    if _with_same_commit:
        log0(f"No model tags with same commit found in {model_dir}, falling back to taking biggest model regardless of commit.", logger=logger)
        return find_potential_model_tag(model_dir, git_info=None)
    # 3) if that also failed, just take the biggest model available:
    model_tags.sort(key=lambda x: model_dir.joinpath(x).stat().st_mtime, reverse=True)
    return model_tags[0]

def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_dir = _solve_path(checkpoint_dir)

    checkpoint_files = [f for f in checkpoint_dir.glob("checkpoint_step_*") if f.is_dir()]
    if not checkpoint_files:
        log_error(f"No model checkpoints found in {checkpoint_dir}.", logger=logger, error_type=FileNotFoundError)
        
    last_step = max(int(p.stem.split("_")[-1]) for p in checkpoint_files)
    return last_step

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layers
    # res_h defaults to 1.0 (identity scaling)
    if "res_h" not in model_data:
        model_data["res_h"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # res_x0 defaults to 0.0 (disabled)
    if "res_x0" not in model_data:
        model_data["res_x0"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def save_checkpoint(
        checkpoint_dir: Union[str, Path], # should be <model_dir>/<model_tag>/<source>/<checkpoint_dir>
        trainer_state: TrainerState,
        checkpoint_state: CheckpointState,
        model_data: Mapping[str, torch.Tensor],
        optimizer_data: Optional[Dict] = None, # should be the state dict
        scaler_state: Optional[Dict] = None,
        mode: Literal["ddp", "shard"] = "ddp",
        dist_info: Optional[Dict] = None,
    ):
    if mode == "shard":
        # TODO: implement sharded checkpointing for FSDP / ZeRO optimizers
        log_error("Shard mode is not implemented yet. Please use ddp mode for now.", logger=logger, error_type=NotImplementedError)
    checkpoint_dir = _solve_path(checkpoint_dir)

    if dist_info is None:
        dist_info = get_dist_info()
    rank = dist_info["RANK"]

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Save the model state parameters
        _save_state_dict(model_data, checkpoint_dir, "model.pt")
        # Save the training state
        _save_state_dict(trainer_state.model_dump(), checkpoint_dir, "trainer_state.json")
        # Save rng state for reproducibility
        _save_state_dict(capture_rng_state(), checkpoint_dir, "rng_state.pt")
        # Save the checkpoint state (best bpb/core steps and values)
        _save_state_dict(checkpoint_state.model_dump(), checkpoint_dir, "checkpoint_state.json")
        # Save scaler state if using mixed precision
        if scaler_state is not None:
            _save_state_dict(scaler_state, checkpoint_dir, "scaler.pt")

    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if (optimizer_data is not None) and ((mode == "shard") or (mode == "ddp" and rank == 0)):
        _save_state_dict(optimizer_data, checkpoint_dir, _OPT_DEFAULT_NAME[mode](rank))

def _get_checkpoint_step(checkpoint_dir, step: Optional[Union[int, str]] = None) -> int:
    if isinstance(step, int) and step >= 0:
        return step
    elif isinstance(step, int) and step < 0:
        return _get_checkpoint_step(checkpoint_dir, step=f"{step:d}")
    elif (step is None) or (step == "latest"):
        return _get_checkpoint_step(checkpoint_dir, step="-1")
    elif isinstance(step, str) and step.startswith("-") and step[1:].isdigit():
        index = int(step[1:]) - 1
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*"))
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"Found checkpoint files: {[p.name for p in checkpoint_files]}")
        print("Selecting checkpoint file based on index: ", index)
        if index < 0 or index >= len(checkpoint_files):
            log_error(f"Invalid checkpoint index {step} for checkpoint directory {checkpoint_dir}. "
                f"Found {len(checkpoint_files)} checkpoints in directory {checkpoint_dir}.", logger=logger, error_type=IndexError)
        step = int(checkpoint_files[index].stem.split("_")[-1])
    elif step == "best":
        checkpoint_state = _load_state_dict(checkpoint_dir, "checkpoint_state.json", map_location="cpu")
        if not checkpoint_state.best_step:
            log_error(f"Checkpoint state in {checkpoint_dir} does not contain a best step for 'best' checkpoint selection.", logger=logger, error_type=ValueError)
        step = checkpoint_state.best_step
    else:
        log_error(f"Invalid step argument: {step}. Must be None, an integer, or a string of the form '-N' where N is a positive integer.", logger=logger, error_type=ValueError)
    return step

def load_checkpoint(
        checkpoint_dir: Union[str, Path], # should be <model_dir>/<model_name>/<model_tag>/<source>/<checkpoint_dir>
        step: Union[int, str] = -1, 
        load_state: str = "model,optim,trainer,rng",
        mode: Literal["ddp", "shard"] = "ddp",
        dist_info: Optional[Dict] = None,
    ):
    # Load the model state
    checkpoint_dir = _solve_path(checkpoint_dir)
    if dist_info is None:
        dist_info = get_dist_info()
    rank = dist_info["RANK"]
    device = dist_info["DEVICE"]

    _states_to_load = set(load_state.strip().split(","))
    step = _get_checkpoint_step(checkpoint_dir, step)
    checkpoint_dir = checkpoint_dir / f"checkpoint_step_{step:d}"
    if not checkpoint_dir.exists():
        log_error(f"Checkpoint directory {checkpoint_dir} does not exist for step {step}.", logger=logger, error_type=FileNotFoundError)
    # Load the model state parameters
    model_data = None
    if "model" in _states_to_load:
        model_data = _load_state_dict(checkpoint_dir, f"model.pt", map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if "optim" in _states_to_load:
        optimizer_data = _load_state_dict(checkpoint_dir, _OPT_DEFAULT_NAME[mode](rank), map_location=device)
    # Load the trainer state
    trainer_state = None
    if "trainer" in _states_to_load:
        trainer_state = _load_state_dict(checkpoint_dir, "trainer_state.json", map_location="cpu", weight_only=False)
    # Load the scaler state if requested
    # TODO: this is a bit hacky, we should probably have a more general way to specify scaler state
    # as it is device-specific and may not always be needed 
    # => we should probably also have a more general way to specify additional states to load/save in the future.
    scaler_state = None
    if "scaler" in _states_to_load:
        scaler_state = _load_state_dict(checkpoint_dir, "scaler.pt", map_location="cpu", weight_only=False)
    # Load the RNG state if requested
    rng_state = None
    if "rng" in _states_to_load:
        rng_state = _load_state_dict(checkpoint_dir, "rng_state.pt", map_location="cpu", weight_only=False)
    return model_data, optimizer_data, trainer_state, scaler_state, rng_state


def build_model(
        model_name: str,
        model_tag: Optional[Union[str, int]] = None,
        model_cachedir: Optional[Union[str, Path]] = None,
        step: Union[int, str] = -1, 
        phase: str = "train",
        source: _ModelSources = "base",
        dist_info: Optional[Dict] = None,
     ) -> tuple[DenseTransformer, Tokenizer, Dict]:
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    if dist_info is None:
        dist_info = get_dist_info()
    device: torch.device = dist_info["DEVICE"]
    model_cachedir = _solve_model_cache_dir(model_cachedir)
    model_tag = _solve_model_tag(model_cachedir / model_name, model_tag=model_tag)
    # get metaconfig
    meta_config = load_meta_config(model_name, model_tag=model_tag, model_cachedir=model_cachedir)
    # get checkpoint data
    ckpt_dir = model_cachedir / model_name / model_tag / source
    model_data, optimizer_data, trainer_state, scaler_state, rng_state = load_checkpoint(ckpt_dir, step, dist_info=dist_info)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_config["model_config"].model_dump()
    # TODO: maybe do it in load_meta_config instead?
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = meta_config["model_config"] # TransformerConfig
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = DenseTransformer(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tkn_cfg = TokenizerConfig.model_validate(meta_config["tokenizer_config"])
    tokenizer = build_tokenizer(tkn_cfg)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.vocab_size == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, trainer_state

def load_model_from_dir(
        model_name: Union[str, Path],
        model_tag: Optional[Union[str, int]] = None, 
        step: Optional[int] =None,
        phase: Literal["train", "eval"] = "train",
        model_cachedir: Optional[Union[str, Path]] = None,
        source: _ModelSources = "base",
        dist_info: Optional[Dict] = None,
    ):
    if dist_info is None:
        dist_info = get_dist_info()
    device = dist_info["DEVICE"]
    model_cachedir = _solve_model_cache_dir(model_cachedir)
    model_dir = model_cachedir / model_name
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_potential_model_tag(model_dir, git_info=get_git_info())
        log0(f"No model tag provided, guessing model tag: {model_tag}.")
    ckpt_dir = Path(model_dir).joinpath(model_tag).joinpath(source)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(ckpt_dir)
    assert step is not None, f"No checkpoints found in {ckpt_dir}"
    # build the model
    model, tokenizer, meta_data = build_model(model_name=model_name, model_tag=model_tag, model_cachedir=model_cachedir, step=step, phase=phase, source=source, dist_info=dist_info)
    log0(f"Loaded model from {ckpt_dir} with step {step}", logger=logger)
    return model, tokenizer, meta_data

def load_optimizer_state(
        model_name: str,
        source: _ModelSources = "base", # for now, only source = base
        model_tag: Optional[Union[str, int]] = None, 
        model_cachedir: Optional[Union[str, Path]] = None, # should not be needed -> this should be CACHE_DIR / models
        step: Optional[int] = None,
        mode: Literal["ddp", "shard"] = "ddp",
        dist_info: Optional[Dict] = None,
    ):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    if dist_info is None:
        dist_info = get_dist_info()
    rank = dist_info["RANK"]
    if mode == "shard":
        log_error("Shard mode is not implemented yet. Please use ddp mode for now.", logger=logger, error_type=NotImplementedError)
    # resolve general checkpoint path
    model_dir = _solve_model_cache_dir(model_cachedir).joinpath(model_name)

    if model_tag is None:
        model_tag = find_potential_model_tag(model_dir, git_info=get_git_info())

    checkpoints_dir = model_dir.joinpath(model_tag).joinpath(source)
    if step is None:
        step = find_last_step(checkpoints_dir)
    
    optimizer_path = (checkpoints_dir
        .joinpath(f"checkpoint_step_{step:06d}")
        .joinpath(_OPT_DEFAULT_NAME[mode](rank)))
    
    if not optimizer_path.exists():
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=dist_info["DEVICE"])
    return optimizer_data


class CheckpointManager:
    """
    A utility class to manage checkpoints during training, 
    including saving and loading model and optimizer states, 
    as well as tracking the best checkpoint based on validation metrics.

    This is just a wrapper around the checkpoint saving and loading functions above,
    that takes model and state pointers to make it easier as an API.
    """
    def __init__(
            self,
            model: str,
            tokenizer: Tokenizer,
            training_config: Dict,
            model_dir: Optional[Union[str, Path]] = None,
            model_tag: Optional[Union[str, int]] = None,
            dist_info: Optional[Dict] = None,
            source: _ModelSources = "base",
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.source = source
        if model_dir is None:
            model_dir = MODELS_FOLDER
        self.model_dir = Path(model_dir)
        self.model_name = model
        if model_tag is None:
            model_tag = find_potential_model_tag(self.model_dir.joinpath(model), git_info=get_git_info())
            log0(f"No model tag provided, guessing model tag: {model_tag}")
        self.model_tag = model_tag
        self.checkpoint_dir = self.model_dir.joinpath(model).joinpath(model_tag)
        self.dist_info = dist_info if dist_info is not None else get_dist_info()
    
    @classmethod
    def resume_from_step(
        cls,
        model_name: str,
        step: int,
        training_config: Dict,
        model_tag: Optional[Union[str, int]] = None,
        model_dir: Optional[Union[str, Path]] = None,
        dist_info: Optional[Dict] = None,
        source: _ModelSources = "base",
    ):
        """Resume training from a specific step."""
        
        meta_data = load_meta_config(model_name, model_tag, model_dir, step, device=dist_info["DEVICE"])

        # return checkpoint_manager