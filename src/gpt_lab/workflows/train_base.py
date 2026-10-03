"""
# Model Base Training Script

Full recipe for training, evaluating and monitoring a base GPT model with auto-configuration based on model depth and scaling laws training horizon targets (FLOPs, param-to-data ratio, training time).
This script is meant to be a starting point for training new models on new datasets, and can be adapted for more specific use cases.
It is mainly adapted from the sources given below, but the overall structure is more modular or adapted for *my* use, permitting more customization and experimentation with different setups.

## Usage

Use `gpt-lab train base auto --help` or `gpt-lab train base resume --help`.
Distributed entry point: `torchrun -m gpt_lab.cli train base ...`.
See docs/cli.md for full examples.

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

from argparse import Namespace


def run(args: Namespace) -> None:
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
    from gpt_lab.cli.commands.base import resolve_checkpoint
    args = resolve_checkpoint(args)
    board_args = {key: value for key, value in vars(args).items() if not key.startswith("_")}

    log0(f"Initializing model base training on mode {args.model_init!r}", logger=logger)

    # ------------------------------------------------------------------------------
    # SETUP ENVIRONEMENT
    # ------------------------------------------------------------------------------

    device_type = get_device_type() if args.device == "auto" else args.device
    dist_info = init_dist_groups(device_type=device_type)
    board = None
    try:
        is_master_process = dist_info["RANK"] == 0

        device = dist_info["DEVICE"]
        is_resumed = args.model_init == "resume"
        print0_dict("Environment setup", dist_info)

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

        if args.model_init == "auto":
            meta_config = build_auto_config(args, device, dist_info)
            base_training_config = meta_config.base_train

        elif is_resumed:
            # load config from checkpoint and override with CLI args if specified
            meta_config = load_meta_config(
                name=args.model_name,
                run_name=args.run_name,
                model_cachedir=args.model_dir
            )
            base_training_config = meta_config.base_train

        # ------------------------------------------------------------------------------
        # INIT MODEL BY LOADING CHECKPOINT OR FROM SCRATCH
        # ------------------------------------------------------------------------------

        ckpt_manager = CheckpointManager(
            model_name=meta_config.name,
            model_run=meta_config.run_name,
            model_cachedir=args.model_dir,
            dist_info=dist_info,
            mode="shard" if dist_info["IS_DDP_INITIALIZED"] else "ddp",
        )
        model = build_meta_model(meta_config.model_cfg)
        tokenizer = Tokenizer.from_config(meta_config.tokenizer_cfg)
        if args.model_init == "auto":
            model = model.to_empty(device=device)
            model.init_weights()
            broadcast_model(model, dist_info)
        elif is_resumed:
            log0("Resuming training from checkpoint.", logger=logger)
            model, tokenizer, ckpt_data, trainer_config = ckpt_manager.load(
                step=args.checkpoint_step if args.checkpoint_step is not None else "latest",
                phase="train",
            )
        # Check just model embeddings and hope others are fine
        assert model.embeds.weight.device.type == dist_info["DEVICE_TYPE"], "Model parameters are not on the correct device after initialization."

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
            name=args.ds_name,
            tokenizer=tokenizer,
            column="text",
            seq_len=model.config.max_context,
            base_url=None, # no downloading here, just load local shards
            shard_limit=None,
            max_shards=None, # TODO: configured based on configs/data.yaml
            batch_size=base_training_config.get("device_batch_size", args.device_batch_size),
            dist_info=dist_info,
            use_nanochat=args.use_nanochat_dataloader,
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
            trainer_config = build_trainer_config(args, dist_info, base_training_config)
            ckpt_manager.save_training_config(trainer_config)
        print0_dict("Trainer config", trainer_config.model_dump())

        # ------------------------------------------------------------------------------
        # OPTIMIZER
        # ------------------------------------------------------------------------------

        optim_cfg_path = getattr(args, "optim_config_path", None)
        optimizers = model.build_optimizer(trainer_config, optim_config_path=optim_cfg_path)
        if is_resumed and ckpt_data and ckpt_data.optimizer_state is not None:
            log0("Loading optimizer state from checkpoint data.", logger=logger)
            optimizers.load_state_dict(ckpt_data.optimizer_state)

        # ------------------------------------------------------------------------------
        # INIT BOARD
        # ------------------------------------------------------------------------------

        from gpt_lab.utils.board import Board, DummyBoard

        if is_master_process:
            board_args["dirname"] = ckpt_manager.source_dir
            board = Board(
                board_type=args.board,
                # entity_name=None, # TODO: add option for wandb entity
                project=f"trainbase_{meta_config.name}",
                run=meta_config.run_name,
                config=board_args | {"meta_config": meta_config.model_dump(), "training_config": trainer_config.model_dump(), "model_card": model.config.model_dump()},
                board_dir=args.board_dir,
                resume=is_resumed,
            )
        else:
            board = DummyBoard()

        # ------------------------------------------------------------------------------
        # TRAINER: TRAINING, EVALUATION, CHECKPOINTING LOOPS
        # ------------------------------------------------------------------------------

        from gpt_lab.train.trainer import Trainer

        trainer = Trainer(
            model=model, tokenizer=tokenizer, optimizer=optimizers,
            train_loader=train_loader, val_loader=val_loader,
            config=trainer_config, board=board, checkpoint_manager=ckpt_manager,
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
            if board is not None:
                board.close()
        finally:
            cleanup_dist_groups()


def build_auto_config(args: Namespace, device, dist_info: dict):
    """Construct the existing auto configuration independently of training."""
    from gpt_lab.model.auto import AutoGPTConfig

    return AutoGPTConfig(
        # metadata
        name=args.model_name,
        run_name=args.run_name,
        dirname=args.model_dir,
        random_seed=args.random_seed,
        dist_info=dist_info,
        max_seq_len=args.max_seq_len,
        # tokenizer
        tokenizer_model="auto" if args.train_tokenizer else args.tokenizer_model,
        train_tokenizer=args.train_tokenizer,
        vocab_size=args.vocab_size,
        pat_str=args.pat_str,
        # model architecture
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        d_head=args.d_head,
        n_kv_heads=args.n_kv_heads,
        window_pattern=args.window_pattern,
        window_size=args.window_size,
        softcap=args.softcap,
        attn_softcap=args.attn_softcap,
        attn_impl=args.attn_impl, # for now, only support 'sdpa' and 'fused'
        # training horizon targets
        n_steps=args.num_steps,
        target_flops=args.target_flops,
        target_param_data_ratio=args.target_param_data_ratio,
        # training params
        n_acc_steps=args.n_acc_steps,
        device_batch_size=args.device_batch_size,
        total_batch_size=args.total_batch_size,
    ).generate_gpt_config(device)


def build_trainer_config(args: Namespace, dist_info: dict, base_training_config: dict):
    """Translate CLI and scaling-law settings into the existing TrainerConfig."""
    from gpt_lab.utils.schemas import TrainerConfig

    lr_scale = base_training_config.get("batch_lr_scale", 1.0)
    weight_decay_scale = base_training_config.get("weight_decay_scale", 1.0)
    return TrainerConfig(
        lr_embeddings=args.lr_embeddings * lr_scale,
        lr_transformer=args.lr_transformer * lr_scale,
        lr_head=args.lr_head * lr_scale,
        lr_residuals=args.lr_residuals * lr_scale,
        weight_decay=args.weight_decay * weight_decay_scale,
        lr_warmup_steps=args.warmup_steps,
        lr_warmdown_ratio=args.warmdown_ratio,
        final_lr_ratio=args.final_lr_frac,
        target_time=args.target_time,
        dist_info=dist_info,
        optim_config_path=args.optim_config_path,
        eval_bpb_every=args.eval_bpb_every,
        n_bpb_tokens=args.n_bpb_tokens,
        eval_core_every=args.eval_core_every,
        n_core_tokens=args.n_core_tokens,
        sample_every=args.sample_every,
        save_every=args.save_every,
        log_every=args.log_every,
        monitor_grad_norms=args.monitor_grad_norms,
        save_on_best=args.save_on_best,
        # training horizon args from meta config
        **base_training_config
    )
