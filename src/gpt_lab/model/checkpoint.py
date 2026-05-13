"""
gpt_lab/model/checkpoint.py
Utilities for saving and loading model checkpoints, including the model state dict, optimizer state, and metadata such as general and training configuration.

Models should be saved in the following directory structure:
<CACHE_DIR>/
└── models/
    └── <model_name>/                          # e.g., "ic1", "gpt2-small", "llama2"
        └── <run_name>/
            ├── meta.pskl                      # immutable (model, tokenizer, git)
            └── <source>/                      # base / sft / rl
                ├── training_config.pkl        # per-phase config
                ├── checkpoint_state.pkl       # best bpb/core steps and values
                ├── checkpoint_step_000000/
                │   ├── model.pt
                │   ├── optim_rank0.pt         # optimizer state dict (optim_rank{rank}.pt if sharded, otherwise optim.pt)
                │   ├── optim_rank1.pt
                │   ├── ...                    # more optimizer shards if needed
                │   ├── trainer_state.pkl      # training state, rng state, data state, best bpb/core steps
                │   └── metrics.pkl
                ├── checkpoint_step_000100/
                │   └── ...
                └── ...
In the following code, the variables are the following: 
    - cache_dir: Path = <CACHE_DIR> - the base cache directory. Default is given by environment variable GPTLAB_CACHE_DIR or ./.gpt_lab by default.
    - model_cachedir: Path = <cache_dir>/models - the directory where models are cached. Default is given by environment variable GPTLAB_MODEL_CACHE_DIR or <CACHE_DIR>/models by default.
    - model_dir: Path = <model_cachedir>/<model_name> - the directory for a specific model. Default is <CACHE_DIR>/models/<model_name>.
    - run_name: str = <run_name> - the tag for a specific model run. Default is automatically determined by looking at the most recent checkpoint in the model directory.
    - run_dir: Path = <model_dir>/<run_name> - the directory for a specific model tag. Default is automatically determined by looking at the most recent checkpoint in the model directory with the same tag.
    - checkpoint_dir: Path = <run_dir>/<source> - the directory for a specific checkpoint. Default is automatically determined by looking at the checkpoint step in the checkpoint directory, or the most recent checkpoint if checkpoint step is not specified.
    - step_dir: Path = <checkpoint_dir>/checkpoint_step_<step:06d> - the directory for a specific checkpoint step. Default is automatically determined by looking at the checkpoint step in the checkpoint directory, or the most recent checkpoint if checkpoint step is not specified.
"""
import re
import random, numpy as np
import torch
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Literal, Mapping, Callable, List
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

_MODEL_DEFAULT_FILENAME: Dict[str, Callable[[int], str]] = dict(
    ddp=lambda x: "model.pt",
    shard=lambda x: f"model.pt",
    # fully_sharded=lambda x: f"model.pt",
) # dict[mode, (rank: int) -> (filename: str)]
_OPT_DEFAULT_FILENAME: Dict[str, Callable[[int], str]] = dict(
    ddp=lambda x: "optim.pt",
    shard=lambda x: f"optim_rank{x:d}.pt",
    # fully_sharded=lambda x: f"optim.pt", 
) # dict[mode, (rank: int) -> (filename: str)]
_STEP_GLOB_PATTERN = "checkpoint_step_*"
_STEP_DIRNAME = lambda step: f"checkpoint_step_{step:06d}"
_ModelSources = Literal[("base")] # TODO: Add here "sft", "grpo" when implemented
_Mode = Literal["ddp", "shard"]
_Step = Union[int, str]  # int, "latest", "best", or "-k" / negative int

class CheckpointMode(str, Enum):
    FULL = "full"                    # rank0 only (DDP)
    SHARDED = "sharded"              # optim state sharded across ranks (FSDP / ZeRO)
    FULLY_SHARDED = "fully_sharded"  # FSDP / ZeRO with full sharding (including optimizer state)

@dataclass
class CheckpointConfig:
    model_name: str
    run_name: Optional[Union[str, int]] = None
    model_cachedir: Optional[Union[str, Path]] = None
    source: _ModelSources = "base"
    dist_info: Optional[Dict] = None
    mode: _Mode = "ddp"
    model_config: Optional[TransformerConfig] = None
    tokenizer_config: Optional[TokenizerConfig] = None
    trainer_config: Optional[TrainerConfig] = None

@dataclass
class CheckpointData:
    model_state: Optional[Mapping[str, torch.Tensor]] = None
    optimizer_state: Optional[Dict] = None
    trainer_state: Optional[TrainerState] = None
    scaler_state: Optional[Dict] = None
    rng_state: Optional[Dict[str, Union[bytes, Tuple, None]]] = None

# private utilities
def _solve_path(path: Optional[Union[str, Path]], default: Optional[Path] = None) -> Path:
    if path is None:
        return default
    elif isinstance(path, str):
        return Path(path)
    else:
        return path

def _solve_model_cache_dir(model_cachedir: Optional[Union[str, Path]]) -> Path:
    return _solve_path(model_cachedir, default=MODELS_FOLDER)

def _solve_run_name(model_dir: Path, run_name: Optional[Union[str, int]] = None, source: _ModelSources = "base") -> str:
    run_names = list(sorted(model_dir.iterdir(), key=_get_model_run_date_from_name, reverse=True))
    if not run_names:
        log_error(f"No model checkpoints found in {model_dir}.", logger=logger, error_type=FileNotFoundError)
    if run_name is None:
        log0(f"No model tag specified for model {model_dir.name!r}. Automatically selecting the most recent model tag {run_names[0].name!r}.", level="warning", logger=logger)
        return run_names[0].name
    elif isinstance(run_name, int):
        if run_name >= len(run_names):
            log_error(f"Invalid model tag index {run_name} for model directory {model_dir}.\n"
                f"Found {len(run_names)} model tags in directory {model_dir}.", logger=logger, error_type=IndexError)
        if run_name < 0:
            run_name = -(run_name + 1)
        elif run_name >= 0:
            run_name = len(run_names) - run_name
        return run_names[run_name].name
    elif run_name in [run_name.name for run_name in run_names]:
        return run_name
    elif run_name.startswith("-") and run_name[1:].isdigit():
        index = int(run_name[1:]) - 1
        if index < 0 or index >= len(run_names):
            log_error(f"Invalid model tag index {run_name!r} for model directory {model_dir}.\n"
                f"Found {len(run_names)} model tags in directory {model_dir}.", logger=logger, error_type=IndexError)
        return run_names[index].name
    elif run_name == "latest":
        return run_names[0].name
    elif run_name == "best":
        best_bpb = float('inf')
        best_run = None
        for _run_name in run_names:
            ckpt_state_path = model_dir / _run_name / source / "checkpoint_state.pkl"
            if ckpt_state_path.exists():
                raw_ckpt_state = _load_state_dict(ckpt_state_path, map_location="cpu", weight_only=False)
                ckpt_state = CheckpointState.model_validate_json(raw_ckpt_state)
                if ckpt_state.best_eval_step is None:
                    log0(f"Checkpoint state in {ckpt_state_path} does not contain a best step for 'best' checkpoint selection. Skipping this checkpoint for best selection.", logger=logger, level="warning")
                if ckpt_state.best_eval_step is not None and ckpt_state.best_eval_value < best_bpb:
                    best_bpb = ckpt_state.best_eval_value
                    best_run = _run_name
        if best_run is None:
            log_error(f"Checkpoint state file {ckpt_state_path} does not exist for model {model_dir.name!r} with tag {run_names[0].name!r}. "
                      f"Cannot determine best checkpoint. Please specify a valid checkpoint run name or index (e.g., '-1' for the most recent tag).", 
                      logger=logger, error_type=FileNotFoundError)
        return best_run.name
        
    else:
        log_error(f"Could not find checkpoint for model {model_dir.name!r} with tag {run_name!r} in directory {model_dir}. "
            f"Found tags: {[run_name.name for run_name in run_names]}. Please specify a valid checkpoint run name or index (e.g., '-1' for the most recent tag).", 
            logger=logger, error_type=ValueError)

def _save_state_dict(state_dict, ckpt_dir, filename=None):
    if filename is not None:
        ckpt_dir = ckpt_dir / filename
    torch.save(state_dict, ckpt_dir)
    log0(f"Saved {filename} to {ckpt_dir}", logger=logger)

def _save_json(data, ckpt_dir, filename=None):
    if filename is not None:
        ckpt_dir = ckpt_dir / filename
    with open(ckpt_dir, "w") as f:
        json.dump(data, f, indent=4)
    log0(f"Saved {filename} to {ckpt_dir}", logger=logger)

def _load_json(ckpt_dir, filename):
    if filename is not None:
        ckpt_dir = ckpt_dir / filename
    if not ckpt_dir.exists():
        log_error(f"JSON file {ckpt_dir} does not exist.", logger=logger, error_type=FileNotFoundError)
    with open(ckpt_dir, "r") as f:
        data = json.load(f)
    log0(f"Loaded {filename} from {ckpt_dir}", logger=logger)
    return data

def _load_state_dict(ckpt_dir, filename=None, map_location=None, weight_only=True):
    if filename is not None:
        ckpt_dir = ckpt_dir / filename
    if not ckpt_dir.exists():
        log_error(f"Checkpoint file {ckpt_dir} does not exist.", logger=logger, error_type=FileNotFoundError)
    state_dict = torch.load(ckpt_dir, map_location=map_location, weights_only=weight_only)
    log0(f"Loaded {filename} from {ckpt_dir}.", logger=logger)
    return state_dict

def _sort_checkpoints(ckpt_dir: Path) -> List[Path]:
    checkpoint_files = list(ckpt_dir.glob(_STEP_GLOB_PATTERN))
    checkpoint_files.sort(key=lambda p: int(p.stem.split("_")[-1].replace("-", "").replace("_", "")), reverse=True)
    return checkpoint_files

def _get_checkpoint_step(checkpoint_dir, step: Optional[Union[int, str]] = None) -> int:
    if isinstance(step, int) and step >= 0:
        return step
    elif isinstance(step, int) and step < 0:
        return _get_checkpoint_step(checkpoint_dir, step=f"{step:d}")
    elif (step is None) or (step == "latest"):
        return _get_checkpoint_step(checkpoint_dir, step="-1")
    elif isinstance(step, str) and step.startswith("-") and step[1:].isdigit():
        index = int(step[1:]) - 1
        checkpoint_files = _sort_checkpoints(checkpoint_dir)
        if index < 0 or index >= len(checkpoint_files):
            log_error(f"Invalid checkpoint index {step!r} for checkpoint directory {checkpoint_dir}. "
                f"Found {len(checkpoint_files)} checkpoints in directory {checkpoint_dir}.", logger=logger, error_type=IndexError)
        step = int(checkpoint_files[index].stem.split("_")[-1])
    elif step == "best":
        # TODO: look at step = -1 checkpoint_state and find the best step from there -> pb: best may not be saved under .../checkpoint_step_*/
        raw = _load_state_dict(checkpoint_dir, "checkpoint_state.pkl", map_location="cpu", weight_only=False)
        checkpoint_state = CheckpointState.model_validate_json(raw)
        if not checkpoint_state.best_eval_step:
            log_error(f"Checkpoint state in {checkpoint_dir} does not contain a best step for 'best' checkpoint selection.", logger=logger, error_type=ValueError)
        step = checkpoint_state.best_eval_step
    else:
        log_error(f"Invalid step argument: {step}. Must be None, an integer, or a string of the form '-N' where N is a positive integer.", logger=logger, error_type=ValueError)
    return step

def _get_model_run_date_from_name(run_name) -> int:
    run_name = run_name.name if isinstance(run_name, Path) else run_name # TODO make it better later
    match = re.search(r"dt_(\d{8}_\d{6})", run_name.replace("-", ""))
    if match:
        date_str = match.group(1)
        date_int = int(date_str.replace("_", "").replace("-", ""))
        return date_int
    else:
        return 0 # 0 because reversed=True

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

# api functions
def make_default_run_name(depth, name, dist_info):
    if dist_info is None:
        dist_info = get_dist_info()
    from gpt_lab.utils.system import run_command
    git_commit = run_command("git rev-parse --short HEAD") or "unkcommit"
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    device_name = (str(dist_info.get("DEVICE_NAME", "unkdevice"))
        .lower().split(" ")[-1].replace(" ", "_").replace("/", "_").replace("-", "_"))
    if name is None:
        name = "model"
    return f"{device_name}_{name}_d{depth}_cmt_{git_commit}_dt_{date}"

def build_meta_model(config: TransformerConfig) -> "DenseTransformer":
    with torch.device("meta"):
        model = DenseTransformer(config=config)
    return model

# NOTE: save meta config is done in MetaConfig.model_post_init
def save_meta_config(
        meta_config: Dict, 
        model_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
    # TODO: not really strong
    run_dir = _solve_path(model_dir, default=MODELS_FOLDER / meta_config.get("name", model_name) / meta_config.get("run_name", run_name))
    meta_cfg = MetaConfig(
        project=meta_config["name"],
        run_name=meta_config["run_name"],
        model_cfg=meta_config["model"].config,
        tokenizer_cfg=meta_config["tokenizer"].config,
        git_info=get_git_info(),
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(meta_cfg.model_dump_json(), run_dir, filename="meta.json")

    log0(f"Saved meta config in {str(run_dir)}.", logger=logger)

def load_meta_config(
        name: str, 
        run_name: Optional[Union[str, int]] = None, 
        model_cachedir: Optional[Union[str, Path]] = None
    ) -> MetaConfig:
    model_cachedir = _solve_model_cache_dir(model_cachedir)
    model_path = model_cachedir / name

    if not model_path.exists():
        log_error(f"Model directory {model_path} does not exist.", logger=logger, error_type=FileNotFoundError)

    model_ckpts = list(sorted(model_path.iterdir(), key=_get_model_run_date_from_name, reverse=True))
    if not model_ckpts:
        log_error(f"No checkpoints found for model {name!r} in directory {model_path}.", logger=logger, error_type=FileNotFoundError)
    
    run_name = _solve_run_name(model_path, run_name=run_name)
    run_dir = model_path / run_name
    if not run_dir.exists():
        log_error(f"Checkpoint directory {run_dir} does not exist.", logger=logger, error_type=FileNotFoundError)
    if not run_dir.is_dir():
        log_error(f"Checkpoint path {run_dir} is not a directory.", logger=logger, error_type=ValueError)

    _meta_cfg = _load_json(run_dir, "meta.json")
    meta_cfg = MetaConfig.model_validate_json(_meta_cfg)
    
    log0(f"Loaded meta config from {run_dir / 'meta.json'}", logger=logger)
    
    # meta_cfg = dict(name=_meta_cfg.project, run_name=_meta_cfg.run_name, dirname=run_dir)
    # meta_cfg["model_config"] = TransformerConfig.model_validate(_meta_cfg.model_cfg)
    # meta_cfg["tokenizer_config"] = TokenizerConfig.model_validate(_meta_cfg.tokenizer_cfg)

    # TODO: not sure to keep this
    # meta_cfg["model"] = build_meta_model(meta_cfg["model_config"])
    # meta_cfg["tokenizer"] = Tokenizer.from_config(meta_cfg["tokenizer_config"])
    return meta_cfg

# adapted from nanochat/checkpoint.py
# https://github.com/karpathy/nanochat/blob/8180e1d/nanochat/checkpoint_manager.py#L23
def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def find_potential_run_name(model_dir: Union[str, Path], git_info: Optional[Dict] = None) -> str:
    model_dir = _solve_path(model_dir)

    _with_same_commit = git_info is not None
    # attempt to guess the model run: take the biggest model available
    run_names = [f for f in model_dir.iterdir() if f.is_dir()]
    if not run_names:
        log_error(f"No model checkpoints found in {model_dir}.", logger=logger, error_type=FileNotFoundError)

    # 1) normally all model tags are of the form <device>_<name>_d<number>_cmt_..., try that first:
    candidates = []
    for run_name in run_names:
        match = re.match(r"d(\d+)", run_name.name)
        if match:
            model_depth = int(match.group(1))
            if _with_same_commit:
                git_match = re.search(r"cmt_([a-f0-9]+)", run_name.name)
                if git_match and git_match.group(1) == git_info["commit"][:7]:
                    candidates.append((model_depth, run_name))
            else:
                candidates.append((model_depth, run_name))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1].name
    # 2) if that failed, take without same commit:
    if _with_same_commit:
        log0(f"No model tags with same commit found in {model_dir}, falling back to taking biggest model regardless of commit.", logger=logger)
        return find_potential_run_name(model_dir, git_info=None)
    # 3) if that also failed, just take the biggest model available:
    run_names.sort(key=lambda x: model_dir.joinpath(x).stat().st_mtime, reverse=True)
    return run_names[0].name

def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_dir = _solve_path(checkpoint_dir)

    checkpoint_files = [f for f in checkpoint_dir.glob(_STEP_GLOB_PATTERN) if f.is_dir()]
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
        model: torch.nn.Module,
        checkpoint_dir: Union[str, Path], # should be <model_dir>/<run_name>/<source>/<checkpoint_dir>
        step: _Step,
        trainer_state: TrainerState,
        checkpoint_state: CheckpointState,
        optimizer: Optional[torch.optim.Optimizer] = None, # should be the state dict
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        mode: Literal["ddp", "shard"] = "ddp",
        dist_info: Optional[Dict] = None,
    ):
    if mode == "shard":
        # TODO: implement sharded checkpointing for FSDP / ZeRO optimizers
        log_error("Shard mode is not implemented yet. Please use ddp mode for now.", logger=logger, error_type=NotImplementedError)
    checkpoint_dir = _solve_path(checkpoint_dir)

    step_dir = checkpoint_dir.joinpath(_STEP_DIRNAME(step))

    if dist_info is None:
        dist_info = get_dist_info()
    rank = dist_info["RANK"]

    if rank == 0:
        step_dir.mkdir(parents=True, exist_ok=True)
        # Save the model state parameters
        _save_state_dict(model.state_dict(), step_dir, "model.pt")
        # Save the training state
        _save_json(trainer_state.model_dump(), step_dir, filename="trainer_state.json")
        # Save rng state for reproducibility
        _save_state_dict(capture_rng_state(), step_dir, "rng_state.pt")
        # Save the checkpoint state (best bpb/core steps and values)
        _save_state_dict(checkpoint_state.model_dump(), step_dir, "checkpoint_state.pkl")
        # Save scaler state if using mixed precision
        if scaler is not None:
            _save_state_dict(scaler.state_dict(), step_dir, "scaler.pt")

    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if (optimizer is not None) and ((mode == "shard") or (mode == "ddp" and rank == 0)):
        _save_state_dict(optimizer.state_dict(), step_dir, _OPT_DEFAULT_FILENAME[mode](rank))

def load_checkpoint(
        checkpoint_dir: Union[str, Path], # should be <model_dir>/<model_name>/<run_name>/<source>/<checkpoint_dir>
        step: Union[int, str] = -1, 
        load_model: bool = True,
        load_optim: bool = True,
        load_trainer: bool = True,
        load_scaler: bool = True,
        load_rng: bool = True,
        mode: Literal["ddp", "shard"] = "ddp",
        dist_info: Optional[Dict] = None,
    ) -> CheckpointData:
    # Load the model state
    checkpoint_dir = _solve_path(checkpoint_dir)
    if dist_info is None:
        dist_info = get_dist_info()
    rank = dist_info["RANK"]
    device = dist_info["DEVICE"]

    step = _get_checkpoint_step(checkpoint_dir, step)
    step_dir = checkpoint_dir / _STEP_DIRNAME(step)
    if not step_dir.exists():
        log_error(f"Checkpoint directory {step_dir} does not exist for step {step}.", logger=logger, error_type=FileNotFoundError)
    # Load the model state parameters
    model_data = None
    if load_model:
        model_data = _load_state_dict(step_dir, f"model.pt", map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optim:
        optimizer_data = _load_state_dict(step_dir, _OPT_DEFAULT_FILENAME[mode](rank), map_location=device)
    # Load the trainer state
    trainer_state = None
    if load_trainer:
        trainer_state = TrainerState.model_validate(_load_json(step_dir, "trainer_state.json"))
    # Load the scaler state if requested
    # TODO: this is a bit hacky, we should probably have a more general way to specify scaler state
    # as it is device-specific and may not always be needed 
    # => we should probably also have a more general way to specify additional states to load/save in the future.
    scaler_state = None
    if load_scaler:
        if not (step_dir / "scaler.pt").exists():
            log0("Scaler state file scaler.pt not found in step directory. Skipping loading scaler state.", logger=logger, level="warning")
        else:
            scaler_state = _load_state_dict(step_dir, "scaler.pt", map_location="cpu", weight_only=False)
    # Load the RNG state if requested
    rng_state = None
    if load_rng:
        rng_state = _load_state_dict(step_dir, "rng_state.pt", map_location="cpu", weight_only=False)
    return CheckpointData(
        model_state=model_data,
        optimizer_state=optimizer_data,
        trainer_state=trainer_state,
        scaler_state=scaler_state,
        rng_state=rng_state
    )

def build_model(
        model_name: str,
        run_name: Optional[Union[str, int]] = None,
        model_cachedir: Optional[Union[str, Path]] = None,
        step: Union[int, str] = -1, 
        phase: str = "train",
        source: _ModelSources = "base",
        dist_info: Optional[Dict] = None,
     ) -> tuple[DenseTransformer, Tokenizer, CheckpointData, Optional[TrainerConfig]]:
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    if dist_info is None:
        dist_info = get_dist_info()
    device: torch.device = dist_info["DEVICE"]
    model_cachedir = _solve_model_cache_dir(model_cachedir)
    run_name = _solve_run_name(model_cachedir / model_name, run_name=run_name)
    # get metaconfig
    meta_config = load_meta_config(model_name, run_name=run_name, model_cachedir=model_cachedir)
    trainer_config = None
    # get checkpoint data
    ckpt_dir = model_cachedir / model_name / run_name / source
    if phase == "train":
        trainer_config = TrainerConfig.model_validate(_load_state_dict(ckpt_dir, "training_config.pkl", map_location="cpu", weight_only=False))
    load_scaler = (dist_info.get("dtype", False) == torch.float16) and (device.type == "cuda")
    checkpoint_data = load_checkpoint(ckpt_dir, step, dist_info=dist_info, load_model=True, load_optim=True, load_trainer=True, load_scaler=load_scaler, load_rng=True)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        checkpoint_data.model_state = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in checkpoint_data.model_state.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in checkpoint_data.model_state.items()}
    model_config_kwargs = meta_config.model_cfg.model_dump()
    # TODO: maybe do it in load_meta_config instead?
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = meta_config.model_cfg # TransformerConfig
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = DenseTransformer(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.precompute_pos_enc()
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = build_tokenizer(meta_config.tokenizer_cfg)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.vocab_size == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    # last
    set_rng_state(checkpoint_data.rng_state)
    return model, tokenizer, checkpoint_data, trainer_config

def load_model_from_dir(
        model_name: Union[str, Path],
        run_name: Optional[Union[str, int]] = None, 
        step: Optional[int] =None,
        phase: Literal["train", "eval"] = "train",
        model_cachedir: Optional[Union[str, Path]] = None,
        source: _ModelSources = "base",
        dist_info: Optional[Dict] = None,
    ) -> Tuple[DenseTransformer, Tokenizer, CheckpointData, Optional[TrainerConfig]]:
    if dist_info is None:
        dist_info = get_dist_info()
    device = dist_info["DEVICE"]
    model_cachedir = _solve_model_cache_dir(model_cachedir)
    model_dir = model_cachedir / model_name
    if run_name is None:
        # guess the model tag by defaulting to the largest model
        run_name = find_potential_run_name(model_dir, git_info=get_git_info())
        log0(f"No model run provided, guessing model run: {run_name!r}.")
    ckpt_dir = Path(model_dir).joinpath(run_name).joinpath(source)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(ckpt_dir)
    assert step is not None, f"No checkpoints found in {ckpt_dir}"
    # build the model
    model, tokenizer, ckpt_data, trainer_config = build_model(model_name=model_name, run_name=run_name, model_cachedir=model_cachedir, step=step, phase=phase, source=source, dist_info=dist_info)
    log0(f"Loaded model from {ckpt_dir} with step {step}", logger=logger)
    return model, tokenizer, ckpt_data, trainer_config

def load_optimizer_state(
        model_name: str,
        source: _ModelSources = "base", # for now, only source = base
        run_name: Optional[Union[str, int]] = None, 
        model_cachedir: Optional[Union[str, Path]] = None, # should not be needed -> this should be CACHE_DIR / models
        step: Optional[int] = None,
        mode: Literal["ddp", "shard"] = "ddp",
        dist_info: Optional[Dict] = None,
    ) -> Optional[Dict]:
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    if dist_info is None:
        dist_info = get_dist_info()
    rank = dist_info["RANK"]
    if mode == "shard":
        log_error("Shard mode is not implemented yet. Please use ddp mode for now.", logger=logger, error_type=NotImplementedError)
    # resolve general checkpoint path
    model_dir = _solve_model_cache_dir(model_cachedir).joinpath(model_name)

    if run_name is None:
        run_name = find_potential_run_name(model_dir, git_info=get_git_info())

    checkpoints_dir = model_dir.joinpath(run_name).joinpath(source)
    if step is None:
        step = find_last_step(checkpoints_dir)
    
    optimizer_path = (checkpoints_dir
        .joinpath(_STEP_DIRNAME(step))
        .joinpath(_OPT_DEFAULT_FILENAME[mode](rank)))
    
    if not optimizer_path.exists():
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=dist_info["DEVICE"])
    return optimizer_data

# general class api to simply checkpoint management 
class CheckpointManager:
    """
    Thin stateful wrapper around the module-level checkpoint helpers.
    Owns one training source directory (base / sft / rl) and tracks
    the best checkpoint by bpb.
    """
    def __init__(
        self,
        model_name: str,
        model_run: Optional[Union[str, int]] = None,
        model_cachedir: Optional[Union[str, Path]] = None,
        source: _ModelSources = "base",
        dist_info: Optional[Dict] = None,
        mode: _Mode = "ddp",
    ) -> None:
        self.model_name = model_name
        self.model_run = model_run
        self.model_cachedir = _solve_model_cache_dir(model_cachedir)
        self.run_name = _solve_run_name(self.model_cachedir / self.model_name, run_name=model_run)
        self.source_dir = self.model_cachedir / self.model_name / self.run_name / source
        
        self.dist_info = dist_info or get_dist_info()
        self.mode = mode
        self._rank = self.dist_info["RANK"]
        self._device = self.dist_info["DEVICE"]
 
        self._ckpt_state: Optional[CheckpointState] = self._load_checkpoint_state()
    
    def save_training_config(self, trainer_config: TrainerConfig) -> None:
        if self._rank != 0:
            return
        self.source_dir.mkdir(parents=True, exist_ok=True)
        _save_state_dict(trainer_config, self.source_dir, "training_config.pkl")
        log0(f"Saved training config to {self.source_dir / 'training_config.pkl'}", logger=logger)

    def load_training_config(self) -> Optional[TrainerConfig]:
        path = self.source_dir / "training_config.pkl"
        if not path.exists():
            log0(f"No training config found at {path}.", logger=logger, level="warning")
            return None
        raw = _load_state_dict(self.source_dir, "training_config.pkl")
        return TrainerConfig.model_validate(raw)
    
    def save(
        self,
        step: _Step,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        trainer_state: Optional[TrainerState] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Path:
        step = _get_checkpoint_step(self.source_dir, step)
        step_dir = self.source_dir / _STEP_DIRNAME(step)
        if self._ckpt_state is None:
            self._ckpt_state = CheckpointState(best_eval_step=step, best_eval_value=float('inf'))
        save_checkpoint(
            model=model,
            checkpoint_dir=self.source_dir,
            step=step,
            trainer_state=trainer_state,
            optimizer=optimizer,
            checkpoint_state=self._ckpt_state,
            scaler=scaler,
            mode=self.mode,
            dist_info=self.dist_info,
        )
        return step_dir
 
    def load_states(
        self,
        step: _Step = "latest",
        load_model: bool = True,
        load_optim: bool = True,
        load_rng: bool = True,
        load_scaler: bool = True,
    ) -> CheckpointData:
        step = _get_checkpoint_step(self.source_dir, step)
        ckpt_data = load_checkpoint(self.source_dir, step=step, dist_info=self.dist_info,
            load_model=load_model, load_optim=load_optim, load_rng=load_rng, load_scaler=load_scaler)
        return ckpt_data 
    
    def load(self, step: _Step = "latest", phase: Literal["train", "eval"] = "train") -> Tuple[DenseTransformer, Tokenizer, CheckpointData, Optional[TrainerConfig]]:
        step = _get_checkpoint_step(self.source_dir, step)
        model, tokenizer, ckpt_data, trainer_config = build_model(
            model_name=self.model_name,
            run_name=self.run_name,
            model_cachedir=self.model_cachedir,
            step=step,
            phase=phase,
            source=self.source_dir.name,
            dist_info=self.dist_info,
        )
        return model, tokenizer, ckpt_data, trainer_config
 
    def mark_best_bpb(self, step: int, value: float) -> None:
        """
        bpb eval is lower is the best
        """
        if self._rank != 0:
            return
        if self._ckpt_state is None:
            log0(f"No existing checkpoint state found. Marking step {step:,} as best with bpb {value:.4f}.", logger=logger, level="warning")
            self._ckpt_state = CheckpointState(best_eval_step=step, best_eval_value=value)
        elif value < self._ckpt_state.best_eval_value:
            self._ckpt_state.best_eval_step = step
            self._ckpt_state.best_eval_value = value
            self._save_checkpoint_state()
            log0(f"New best checkpoint - step {step:,}, bpb {value:.4f}", logger=logger)

    def mark_best_core(self, step: int, value: float) -> None:
        """
        core eval is higher is the best
        """
        if self._rank != 0:
            return
        if self._ckpt_state is None:
            log0(f"No existing checkpoint state found. Marking step {step:,} as best with core {value:.4f}.", logger=logger, level="warning")
            self._ckpt_state = CheckpointState(best_core_step=step, best_core_value=value)
        elif value > self._ckpt_state.best_core_value:
            self._ckpt_state.best_core_step = step
            self._ckpt_state.best_core_value = value
            self._save_checkpoint_state()
            log0(f"New best checkpoint - step {step:,}, core {value:.4f}", logger=logger)

    @property
    def config(self) -> CheckpointConfig:
        meta_config = load_meta_config(self.model_name, run_name=self.run_name, model_cachedir=self.model_cachedir)
        trainer_config = self.load_training_config()
        return CheckpointConfig(
            model_name=meta_config.name,
            run_name=meta_config.run_name,
            model_cachedir=meta_config.dirname,
            dist_info=self.dist_info,
            mode=self.mode,
            model_config=meta_config.model_cfg,
            tokenizer_config=meta_config.tokenizer_cfg,
            trainer_config=trainer_config
        )
 
    def list_steps(self) -> List[int]:
        dirs = sorted(
            (p for p in self.source_dir.glob(_STEP_GLOB_PATTERN) if p.is_dir()),
            key=lambda p: int(p.name.split("_")[-1]),
        )
        return [int(d.name.split("_")[-1]) for d in dirs]
 
    def latest_step(self) -> Optional[int]:
        if not self.has_checkpoint():
            return None
        return find_last_step(self.source_dir)
 
    def has_checkpoint(self) -> bool:
        return any(self.source_dir.glob(_STEP_GLOB_PATTERN))
 
    def _step_dir(self, step: int) -> Path:
        return self.source_dir / _STEP_DIRNAME(step)
 
    @property
    def _ckpt_state_path(self) -> Path:
        return self.source_dir / "checkpoint_state.pkl"
 
    def _save_checkpoint_state(self) -> None:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        _save_state_dict(self._ckpt_state.model_dump(), self.source_dir, "checkpoint_state.pkl")
 
    def _load_checkpoint_state(self) -> Optional[CheckpointState]:
        if not self._ckpt_state_path.exists():
            return None
        raw_ckpt_state = _load_state_dict(self._ckpt_state_path, map_location="cpu", weight_only=False)
        return CheckpointState.model_validate_json(raw_ckpt_state)