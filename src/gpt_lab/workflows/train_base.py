"""
# Model Base Training Script

Full recipe for training, evaluating and monitoring a base GPT model with auto-configuration based on model depth and scaling laws training horizon targets (FLOPs, param-to-data ratio, training time).
This script is meant to be a starting point for training new models on new datasets, and can be adapted for more specific use cases.
It is mainly adapted from the sources given below, but the overall structure is more modular or adapted for *my* use, permitting more customization and experimentation with different setups.

## Usage

Use `gpt-lab train --help` or `gpt-lab train resume --help`.
Distributed entry point: `torchrun -m gpt_lab.cli train ...`.
See the gpt_lab.cli.main module docstring for commands and examples.

## Aknowledgements:
This code is inspired by and adapted from the following sources:
- nanochat by @karpathy (https://github.com/karpathy/nanochat)
- plainLM by @Niccolo-Ajroldi (https://github.com/Niccolo-Ajroldi/plainLM)
- The Hugging Face Transformers library (https://github.com/huggingface/transformers)
- The nanotron library (https://github.com/nanotron/nanotron)
- Hugging face's jobs for training models on GPUs

Author: Arthur Testard (arthur.testard.pro@gmail.com)
Please cite this work if the code is helpful to you.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gpt_lab.utils.paths import get_cache_dir

if TYPE_CHECKING:
    from gpt_lab.utils.schemas import OptimizerConfig
    from gpt_lab.utils.distributed import ComputeContext


def train_base(
    *,
    optim_config: OptimizerConfig | None = None,
    model_init='auto',
    model_name='ic1',
    run_name=None,
    model_dir=None,
    random_seed=42,
    device='auto',
    board='wandb',
    board_dir=None,
    ds_name='climbmix-base',
    save_on_best=False,
    eval_bpb_every=250,
    n_bpb_tokens=41943040,
    eval_core_every=2000,
    n_core_tokens=500,
    sample_every=0,
    save_every=-1,
    log_every=250,
    monitor_grad_norms=False,
    device_batch_size=32,
    use_nanochat_dataloader=False,
    max_seq_len=2048,
    tokenizer_model=None,
    vocab_size=-1,
    pat_str=None,
    train_tokenizer=False,
    depth=12,
    aspect_ratio=64,
    d_head=128,
    n_kv_heads=None,
    tie_embeddings="embs;h_embs;head",
    window_pattern=None,
    window_size=None,
    softcap=18.0,
    attn_softcap=None,
    attn_impl='sdpa',
    num_steps=-1,
    target_flops=-1.0,
    target_param_data_ratio=11.0,
    target_time=-1.0,
    n_acc_steps=-1,
    total_batch_size=-1,
    warmup_steps=40,
    warmdown_ratio=0.65,
    final_lr_frac=0.05,
    checkpoint_step=None,
    checkpoint_dir=None,
) -> None:
    """Train from Python keywords using a loaded base optimizer profile.

    Fresh runs require optim_config; resume restores the saved architecture,
    trainer settings, and optimizer state. The CLI only loads files and forwards
    keywords. The profile supplies base values, which scaling laws adjust once.
    """
    if model_init not in ("auto", "resume"):
        raise ValueError("model_init must be auto or resume")
    if model_init == "auto" and optim_config is None:
        raise ValueError("Fresh training requires a loaded OptimizerConfig")
    if optim_config is not None:
        from gpt_lab.utils.schemas import OptimizerConfig
        if not isinstance(optim_config, OptimizerConfig):
            raise TypeError("optim_config must be a loaded OptimizerConfig")
    model_dir = str(model_dir or get_cache_dir() / "models")
    parameters = locals().copy()
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from gpt_lab.utils.common import get_banner, print0_dict
    from gpt_lab.utils.logging import init_logger, log0
    from gpt_lab.utils.distributed import cleanup_dist_groups, get_device_type, init_dist_groups, broadcast_model
    from gpt_lab.utils.system import get_git_info, get_gpu_info, get_system_info
    from gpt_lab.utils.schemas import TrainerConfig

    from gpt_lab.tokenizer import Tokenizer

    from gpt_lab.model.checkpoint import CheckpointManager, build_meta_model, load_meta_config

    import logging
    init_logger()
    logger = logging.getLogger(__name__)
    get_banner(to_print=True)
    if checkpoint_dir is not None:
        resolved = resolve_checkpoint(checkpoint_dir, checkpoint_step)
        model_name, model_dir, run_name, checkpoint_step = (
            resolved["model_name"], resolved["model_dir"], resolved["run_name"], resolved["checkpoint_step"]
        )
        parameters.update(resolved)
    board_args = parameters | {"optim_config": optim_config.model_dump() if optim_config is not None else None}

    log0(f"Initializing model base training on mode {model_init!r}", logger=logger)

    # ------------------------------------------------------------------------------
    # SETUP ENVIRONEMENT
    # ------------------------------------------------------------------------------

    device_type = get_device_type() if device == "auto" else device
    comp_info = init_dist_groups(device_type=device_type, random_seed=random_seed)
    board_instance = None
    try:
        is_master_process = comp_info.rank == 0

        device = comp_info.device
        is_resumed = model_init == "resume"
        print0_dict("Environment setup", vars(comp_info) | {"rank": comp_info.rank, "world_size": comp_info.world_size})

        git_info = get_git_info()
        gpu_info = get_gpu_info()
        sys_info = get_system_info()

        board_args = board_args | {"git_info": git_info, "gpu_info": gpu_info, "sys_info": sys_info}

        print0_dict("Git info", git_info)
        print0_dict("GPU info", gpu_info)
        print0_dict("System info", sys_info)

        # ------------------------------------------------------------------------------
        # GET MODEL CONFIG
        # ------------------------------------------------------------------------------

        if model_init == "auto":
            meta_config = build_auto_config(parameters, device, comp_info)
            base_training_config = meta_config.base_train

        elif is_resumed:
            # load config from checkpoint and override with CLI args if specified
            meta_config = load_meta_config(
                name=model_name,
                run_name=run_name,
                model_cachedir=model_dir
            )
            base_training_config = meta_config.base_train

        # ------------------------------------------------------------------------------
        # INIT MODEL BY LOADING CHECKPOINT OR FROM SCRATCH
        # ------------------------------------------------------------------------------

        ckpt_manager = CheckpointManager(
            model_name=meta_config.name,
            model_run=meta_config.run_name,
            model_cachedir=model_dir,
            comp_info=comp_info,
            optimizer_state_layout="sharded" if comp_info.is_distributed else "full",
        )
        model = build_meta_model(meta_config.model_cfg)
        tokenizer = Tokenizer.from_config(meta_config.tokenizer_cfg)
        if model_init == "auto":
            model = model.to_empty(device=device)
            model.init_weights()
            broadcast_model(model, comp_info)
        elif is_resumed:
            log0("Resuming training from checkpoint.", logger=logger)
            model, tokenizer, ckpt_data, trainer_config = ckpt_manager.load(
                step=checkpoint_step if checkpoint_step is not None else "latest",
                phase="train",
            )
            if optim_config is not None:
                saved_profile = getattr(trainer_config, "optim_config", None)
                if saved_profile is None or optim_config.group_configs() != saved_profile.group_configs():
                    raise ValueError("optim_config conflicts with the saved optimizer profile; resume without an optimizer override")
        # Check just model embeddings and hope others are fine
        assert model.embeds.weight.device.type == comp_info.device_type, "Model parameters are not on the correct device after initialization."

        # ------------------------------------------------------------------------------
        # DATASET, DATALOADERS
        # ------------------------------------------------------------------------------

        from gpt_lab.data.loader import build_dataloader

        resume_state = None
        if is_resumed:
            print("ckpt_data.trainer_state", ckpt_data.trainer_state)
            if ckpt_data and hasattr(ckpt_data.trainer_state, "train_loader_state") and ckpt_data.trainer_state.train_loader_state is not None:
                resume_state = ckpt_data.trainer_state.train_loader_state
            else:
                log0("No dataloader state found in checkpoint data. Resuming without dataloader state.", logger=logger, level="warning")

        loader_common_kwargs = dict(
            name=ds_name,
            tokenizer=tokenizer,
            column="text",
            seq_len=model.config.max_context,
            base_url=None, # no downloading here, just load local shards
            shard_limit=None,
            max_shards=None, # TODO: configured based on configs/data.yaml
            batch_size=base_training_config.get("device_batch_size", device_batch_size),
            comp_info=comp_info,
            use_nanochat=use_nanochat_dataloader,
        )
        # TODO: add option to configure buffer size
        train_loader = build_dataloader(split="train", resume_state=resume_state, **loader_common_kwargs)
        val_loader = build_dataloader(split="val", **loader_common_kwargs)

        # ------------------------------------------------------------------------------
        # TRAINER CONFIG
        # ------------------------------------------------------------------------------

        if is_resumed:
            assert type(trainer_config) == TrainerConfig, "Trainer config loaded from checkpoint is not of type TrainerConfig. Please check the checkpoint data and trainer config."
            # TODO: add option to override some trainer config parameters from CLI args even when resuming (eg: evaluation frequency, save frequency, etc.)
        else:
            trainer_config = build_trainer_config(parameters, comp_info, base_training_config)
            ckpt_manager.save_training_config(trainer_config)
        print0_dict("Trainer config", trainer_config.model_dump())

        # ------------------------------------------------------------------------------
        # OPTIMIZER
        # ------------------------------------------------------------------------------

        optimizers = model.build_optimizer(trainer_config, optim_config=optim_config)
        if is_resumed and ckpt_data and ckpt_data.optimizer_state is not None:
            log0("Loading optimizer state from checkpoint data.", logger=logger)
            optimizers.load_state_dict(ckpt_data.optimizer_state)

        # ------------------------------------------------------------------------------
        # INIT BOARD
        # ------------------------------------------------------------------------------

        from gpt_lab.utils.board import Board, DummyBoard

        if is_master_process:
            board_args["dirname"] = ckpt_manager.source_dir
            board_instance = Board(
                board_type=board,
                # entity_name=None, # TODO: add option for wandb entity
                project=f"trainbase_{meta_config.name}",
                run=meta_config.run_name,
                config=board_args | {"meta_config": meta_config.model_dump(), "training_config": trainer_config.model_dump(), "model_card": model.config.model_dump()},
                board_dir=board_dir,
                resume=is_resumed,
            )
        else:
            board_instance = DummyBoard()

        # ------------------------------------------------------------------------------
        # TRAINER: TRAINING, EVALUATION, CHECKPOINTING LOOPS
        # ------------------------------------------------------------------------------

        from gpt_lab.train.trainer import Trainer

        trainer = Trainer(
            model=model, tokenizer=tokenizer, optimizer=optimizers,
            train_loader=train_loader, val_loader=val_loader,
            config=trainer_config, board=board_instance, checkpoint_manager=ckpt_manager,
            resume_state=ckpt_data.trainer_state if is_resumed else None,
            best_state=ckpt_data.checkpoint_state if is_resumed else None,
        )
        if is_resumed and ckpt_data.scaler_state is not None:
            if trainer.scaler is None:
                raise ValueError(
                    "Checkpoint contains GradScaler state but the resumed trainer "
                    "did not create a scaler. Check GPTLAB_DTYPE."
                )
            trainer.scaler.load_state_dict(ckpt_data.scaler_state)
        trainer.train()

    finally:
        try:
            if board_instance is not None:
                board_instance.close()
        finally:
            cleanup_dist_groups()


def resolve_checkpoint(checkpoint_dir, checkpoint_step=None) -> dict:
    """Resolve a run/base/step directory without importing the CLI."""
    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if not (path / "meta.json").is_file():
        if path.name.startswith("checkpoint_step_"):
            step = int(path.name.removeprefix("checkpoint_step_"))
            if checkpoint_step is not None and checkpoint_step != step:
                raise ValueError("--checkpoint-step conflicts with the explicit checkpoint directory")
            checkpoint_step = step
            path = path.parent
        if path.name == "base":
            path = path.parent
    if not (path / "meta.json").is_file():
        raise ValueError("Checkpoint directory must identify a run with meta.json, its base directory, or a checkpoint_step_N directory")
    return dict(model_name=path.parent.name, run_name=path.name,
                model_dir=str(path.parent.parent), checkpoint_step=checkpoint_step)


def build_auto_config(parameters: dict, device, comp_info: ComputeContext):
    """Construct the existing auto configuration independently of training."""
    from gpt_lab.model.auto import AutoGPTConfig

    return AutoGPTConfig(
        # metadata
        name=parameters["model_name"],
        run_name=parameters["run_name"],
        dirname=parameters["model_dir"],
        random_seed=parameters["random_seed"],
        comp_info=comp_info,
        max_seq_len=parameters["max_seq_len"],
        # tokenizer
        tokenizer_model="auto" if parameters["train_tokenizer"] else parameters.get("tokenizer_model"),
        train_tokenizer=parameters["train_tokenizer"],
        vocab_size=parameters["vocab_size"],
        pat_str=parameters["pat_str"],
        # model architecture
        depth=parameters["depth"],
        aspect_ratio=parameters["aspect_ratio"],
        d_head=parameters["d_head"],
        n_kv_heads=parameters["n_kv_heads"],
        tie_embeddings=parameters["tie_embeddings"],
        window_pattern=parameters["window_pattern"],
        window_size=parameters["window_size"],
        softcap=parameters["softcap"],
        attn_softcap=parameters["attn_softcap"],
        attn_impl=parameters["attn_impl"], # for now, only support 'sdpa' and 'fused'
        # training horizon targets
        n_steps=parameters["num_steps"],
        target_flops=parameters["target_flops"],
        target_param_data_ratio=parameters["target_param_data_ratio"],
        # training params
        n_acc_steps=parameters["n_acc_steps"],
        device_batch_size=parameters["device_batch_size"],
        total_batch_size=parameters["total_batch_size"],
    ).generate_gpt_config(device)


def build_trainer_config(parameters: dict, comp_info: ComputeContext, base_training_config: dict):
    """Translate CLI and scaling-law settings into the existing TrainerConfig."""
    from gpt_lab.utils.schemas import TrainerConfig

    return TrainerConfig(
        optim_config=parameters["optim_config"],
        lr_warmup_steps=parameters["warmup_steps"],
        lr_warmdown_ratio=parameters["warmdown_ratio"],
        final_lr_ratio=parameters["final_lr_frac"],
        target_time=parameters["target_time"],
        comp_info=comp_info,
        eval_bpb_every=parameters["eval_bpb_every"],
        n_bpb_tokens=parameters["n_bpb_tokens"],
        eval_core_every=parameters["eval_core_every"],
        n_core_tokens=parameters["n_core_tokens"],
        sample_every=parameters["sample_every"],
        save_every=parameters["save_every"],
        log_every=parameters["log_every"],
        monitor_grad_norms=parameters["monitor_grad_norms"],
        save_on_best=parameters["save_on_best"],
        # training horizon args from meta config
        **base_training_config
    )
