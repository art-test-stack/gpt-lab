"""
# Model Base Training Script

Full recipe for training, evaluating and monitoring a base GPT model with auto-configuration based on model depth and scaling laws training horizon targets (FLOPs, param-to-data ratio, training time). 
This script is meant to be a starting point for training new models on new datasets, and can be adapted for more specific use cases.
It is mainly adapted from the sources given below, but the overall structure is more modular or adapted for *my* use, permitting more customization and experimentation with different setups. 

## Usage

How to run it from root directory of the repo:

- Train a new model from scratch with auto-config based on depth:
    python -m scripts.train_base auto --model-name my_model --depth=12

- Train a new moodel on distributed setup with auto-config:
    torchrun --nproc_per_node=8 -m scripts.train_base auto --model-name my_model --depth=12

- Train on a CPU-only machine (not recommended) or Macbook with MPS backend:
    python -m scripts.train_base auto --model-name my_model --depth=4 --max-seq-len=256 --device-batch-size=1 --total-batch-size=256

- Look at all options for sub-command in {auto, arch, custom} ('arch' and 'custom' not implemented yet):
    python -m scripts.train_base <sub-command> --help

[!NOTE]
Recommended: run with `--optim-config-path=configs/optim.yaml` argument.

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

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from gpt_lab.utils.common import get_banner, print0, print0_dict
from gpt_lab.utils.default import DATA_DIR, MODELS_FOLDER
from gpt_lab.utils.distributed import cleanup_dist_groups, get_device_type, init_dist_groups
from gpt_lab.utils.report import get_git_info, get_gpu_info, get_system_info
from gpt_lab.utils.schemas import GPTConfig, TrainingConfig

from gpt_lab.model.auto import AutoGPTConfig
from gpt_lab.model.gpt import DenseTransformer

from argparse import ArgumentParser


if __name__ == "__main__":
    get_banner(to_print=True)
    parser = ArgumentParser(description="Train base model.")
    subparsers = parser.add_subparsers(dest="model_init")
    
    # Common arguments
    def get_common_arguments(prs: ArgumentParser):
        prs.add_argument("--model-name", type=str, default="ic1", help="Model name")
        prs.add_argument("--model-dir", type=str, default=str(MODELS_FOLDER), help="Cache directory to save model checkpoints and logs.")
        prs.add_argument("--weight-decay", type=float, default=0.28, help="(default: 0.28) Weight decay for optimizer.")
        prs.add_argument("--max-seq-len", type=int, default=2048, help="(default: 2048) Maximum sequence length for training.")
        prs.add_argument("--random-seed", type=int, default=42, help="(default: 42) Random seed for model initialization")
        # TODO: separate random seed for model initialization and training (data shuffling, dropout, etc.)
        # prs.add_argument("--random-seed", type=int, default=42, help="(default: 42) Random seed for model initialization")
        prs.add_argument("--optim-config-path", type=str, default=None, help="(default: None) Path to optimizer config file. If not set, will use default config based on model size.")
        prs.add_argument("--device", type=str, default="auto", help="(default: auto) Device to train on. If 'auto', will detect best device available.")
        prs.add_argument("--board", default="wandb", type=str, choices=("tensorboard", "wandb", "dummy"), help="(default: wandb) Board log directory (options: 'tensorboard', 'wandb', 'dummy').")
        prs.add_argument("--board-dir", type=str, default=None, help="(default: None) Directory to save board logs. If not set, will use default cache directory.")
        prs.add_argument("--ds-config-path", type=str, default="configs/data.yaml", help="(default: configs/data.yaml) Path to datasets config file. If not set, will use default config.")
        prs.add_argument("--ds-name", type=str, default="climbmix-base", help="(default: climbmix-base) Name of the dataset to train on (must be in config YAML file).")
        prs.add_argument("--resume-from-ckpt", type=str, default=None, help="(default: None) Path to checkpoint to resume training from. If not set, will train from scratch.")
        return prs
    
    # Auto-config subparser
    auto_parser = subparsers.add_parser("auto", help="Auto-config model based on depth." \
        "This is the recommended way to train a new model, as it will automatically determine " \
        "the model architecture and training hyperparameters based on the specified depth and " \
        "scaling laws training horizon targets (FLOPs, param-to-data ratio, training time). " \
        "Commonly, when options '-1' are available, it means that the parameter will be automatically " \
        "determined based on scaling laws and training horizon targets. For example, if '--num-steps' is " \
        "set to -1, the number of training steps will be automatically determined based on the target FLOPs, " \
        "target param-to-data ratio, and target training time (if specified) for the model with the given depth.")
    auto_parser = get_common_arguments(auto_parser)

    ## Tokenizer arguments
    auto_parser.add_argument("--tokenizer-model", type=str, default=None, help="(default: None) Tokenizer model to use for auto-configured models. If not set, will use vocab size scaling law to determine tokenizer config.")
    auto_parser.add_argument("--vocab-size", type=int, default=-1, help="(default: -1) Vocabulary size for auto-configured models. If not set, will be determined by vocab size scaling law based on model depth.")
    auto_parser.add_argument("--pat-str", type=str, default=None, help="(default: None) Split pattern for pre-tokenization if training a new-tokenizer. Options are 'gpt2, 'gpt4', 'cl100k_base', 'o200k_base', or directly the pattern string. If not set, will default to 'gpt2' pattern.")
    auto_parser.add_argument("--train-tokenizer", action="store_true", help="(default: False) Whether to train a new tokenizer from scratch.")

    ## Model arguments
    auto_parser.add_argument("--depth", type=int, default=12, help="(default: 12) Number of model layers.")
    auto_parser.add_argument("--aspect-ratio", type=float, default=64, help="(default: 64) Aspect ratio for auto-configured models.")
    auto_parser.add_argument("--d-head", type=int, default=128, help="(default: 128) Dimension of each attention head for auto-configured models. If not set, will be determined by aspect ratio and model depth.")
    auto_parser.add_argument("--d-kv-head", type=int, default=None, help="(default: None) Dimension of each key/value attention head for enable GQA. If not set, will be set to d_head.")
    auto_parser.add_argument("--window-pattern", type=str, default=None, help="(default: None) Window pattern for sliding attention window. String of 'S' and 'L'. If 'None', will be later set as 'SSSL'.")
    auto_parser.add_argument("--window-size", type=str, default=None, help="(default: None) Window size for pattern smalls (S).")
    auto_parser.add_argument("--softcap", type=float, default=18.0, help="(default: 12.0) Soft cap for model logits to prevent overflow.")
    auto_parser.add_argument("--attn-softcap", type=float, default=None, help="(default: 12.0) Soft cap for attention scores to prevent overflow.")
    auto_parser.add_argument("--attn-impl", type=str, default="sdpa", help="(default: sdpa) Attention implementation to use for auto-configured models. Options are 'sdpa' and 'fused'. Both shoulf exhibit same results but 'fused' should be slightly faster (if runned under cuda device).")

    ## Training arguments
    auto_parser.add_argument("--num-steps", type=int, default=-1, help="(default: -1) Number of training steps (overrides num-epochs if > 0).")
    auto_parser.add_argument("--target-flops", type=float, default=-1., help="(default: -1.) Target FLOPS for auto-configured models.")
    auto_parser.add_argument("--target-param-data-ratio", type=float, default=11., help="(default: 11.) Target parameter-to-data ratio for auto-configured models.")
    auto_parser.add_argument("--target-time", type=float, default=-1., help="(default: -1.) Target training time in seconds for auto-configured models. This parameter overrides num-steps. (Default: -1, meaning not used)")
    auto_parser.add_argument("--use-amp", action="store_true", help="(default: False) Whether to use automatic mixed precision (AMP) during training for auto-configured models.")
    auto_parser.add_argument("--fp8", action="store_true", help="(default: False) Whether to use FP8 precision for auto-configured models. This is an experimental feature and may not be stable. Use with caution.")

    ## Optimization
    auto_parser.add_argument("--n-acc-steps", type=int, default=-1, help="(default: -1) Number of gradient accumulation steps to perform before each optimizer step (-1 automatically sets; 0 disables). Reccomended: -1.")
    auto_parser.add_argument("--device-batch-size", type=int, default=32, help="(default: 32) Batch size for each device during training. Batch size define further effective batch size as device_batch_size * max_seq_len * n_acc_steps.")
    auto_parser.add_argument("--total-batch-size", type=int, default=-1, help="(default: -1) Total batch size across all devices for auto-configured models. If set, will override device batch size as device_batch_size = total_batch_size // (world_size * n_acc_steps). `total_batch_size`=-1 is thus recommended for invariant steps by tokens.")
    auto_parser.add_argument("--lr-embeddings", type=float, default=.3, help="(default: 0.3) Learning rate for embedding layer. If not set, will be the same as learning rate for other layers.")
    auto_parser.add_argument("--lr-transformer", type=float, default=.02, help="(default: 0.02) Learning rate for transformer blocks for auto-configured models.")
    auto_parser.add_argument("--lr-head", type=float, default=.008, help="(default: 0.008) Learning rate for head layer. If not set, will be the same as learning rate for other layers.")
    auto_parser.add_argument("--lr-residuals", type=float, default=.5, help="(default: 0.5) Learning rate for residual connections for auto-configured models.")
    auto_parser.add_argument("--warmup-steps", type=int, default=40, help="(default: 40) Number of warmup steps for learning rate scheduler.")
    auto_parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="(default: 0.65) Ratio of training steps to warm down the learning rate at the end of training for auto-configured models.")
    auto_parser.add_argument("--final-lr-frac", type=float, default=0.05, help="(default: 0.05) Final learning rate as a fraction of the initial learning rate for auto-configured models.")

    ## Evaluation
    auto_parser.add_argument("--eval-bpb-every", type=int, default=250, help="(default: 250) Evaluate val bpb every N steps (-1 = last, 0 = disable, N > 0 = every N steps).")
    auto_parser.add_argument("--n-bpb-tokens", type=int, default=80*524288, help="(default: 80*524288) Number of tokens to evaluate val loss on.")
    auto_parser.add_argument("--eval-core-every", type=int, default=2000, help="(default: 2000) Evaluate CORE metric every N steps (-1 = last, 0 = disable, N > 0 = every N steps).")
    auto_parser.add_argument("--n-core-tokens", type=int, default=500, help="(default: 500) Examples per task for CORE metric")
    auto_parser.add_argument("--sample-every", type=int, default=0, help="(default: 2000) Sample from model every N steps (-1 = last, 0 = disable, N > 0 = every N steps).")
    auto_parser.add_argument("--save-every", type=int, default=-1, help="(default: -1) Save checkpoints every N steps (-1 = last, 0 = disable, N > 0 = every N steps).")
    auto_parser.add_argument("--log-every", type=int, default=250, help="(default: -1) Log metrics every N steps (-1 = last, 0 = disable, N > 0 = every N steps).")
    auto_parser.add_argument("--monitor-grad-norms", action="store_true", help="(default: False) Whether to monitor gradient norms during training. If set, will log the norm of the gradients of each parameter to the board at each training step.")

    # TODO: Arch-config subparser
    arch_parser = subparsers.add_parser("arch", help="Config model architecture based on model name (eg: gpt2, llama2, mixtral).")
    arch_parser = get_common_arguments(arch_parser)

    # TODO: Custom-config subparser
    custom_parser = subparsers.add_parser("cust", help="Custom-config model with specified parameters.")
    custom_parser = get_common_arguments(custom_parser)
    custom_parser.add_argument("--batch-size", type=int, default=-1, help="(default: -1) Batch size for training.")
    custom_parser.add_argument("--learning-rate", type=float, default=1e-4, help="(default: 1e-4) Learning rate for optimizer.")
    
    args = parser.parse_args()
    board_args = vars(args).copy()

    # ------------------------------------------------------------------------------
    # SETUP ENVIRONEMENT
    # ------------------------------------------------------------------------------

    device_type = get_device_type() if args.device == "auto" else args.device
    dist_info = init_dist_groups(device_type=device_type)
    is_master_process = dist_info["RANK"] == 0

    device = dist_info["DEVICE"]

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
        mconfig = AutoGPTConfig(
            # metadata
            basename=args.model_name,
            dirname=args.model_dir,
            random_seed=args.random_seed,
            dist_info=dist_info,
            # tokenizer
            tokenizer_model="auto" if args.train_tokenizer else args.tokenizer_model,
            vocab_size=args.vocab_size,
            # model architecture
            depth=args.depth,
            aspect_ratio=args.aspect_ratio,
            max_seq_len=args.max_seq_len,
            d_head=args.d_head,
            d_kv_head=args.d_kv_head,
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
        )
        meta_config = mconfig.generate_gpt_config(device)
        model: DenseTransformer = meta_config["model"]
        tokenizer = meta_config["tokenizer"]
        base_training_config = meta_config["training_config"]

    elif args.model_init == "arch":
        raise NotImplementedError("Arch-model configuration is not implemented yet. Please use 'auto' mode for now.")
    
    elif args.model_init == "cust":
        raise NotImplementedError("Custom model configuration is not implemented yet. Please use 'auto' mode for now.")
        gpt_config = GPTConfig(
            basename=args.model_name,
            dirname=args.model_dir,
            n_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size
        )

    else:
        raise ValueError(f"Model initialization mode {args.model_init!r} is not known. Please choose among 'auto', 'custom', etc.")
    
    # ------------------------------------------------------------------------------
    # INIT MODEL
    # ------------------------------------------------------------------------------

    # TODO: add option to resume training from checkpoint
    model = model.to_empty(device=device)    
    model.init_weights()
    assert model.embeds.weight.device.type == dist_info["DEVICE_TYPE"], "Model parameters are not on the correct device after initialization."

    # ------------------------------------------------------------------------------
    # DATASET, DATALOADERS
    # ------------------------------------------------------------------------------

    # TODO: 
    # manage datasets
    # init shard manager
    # from gpt_lab.data.sharder import ShardManager
    # shard_manager = ShardManager(data_dir=DATA_DIR, name="base")
    # init tokenizing data loaders
    from gpt_lab.data.loader import build_dataloader

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
    )
    # TODO: add option to configure buffer size
    train_loader = build_dataloader(split="train", **loader_common_kwargs)
    val_loader = build_dataloader(split="val", **loader_common_kwargs)

    # ------------------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------------------

    lr_scale = base_training_config.get("batch_lr_scale", 1.0)
    weight_decay_scale = base_training_config.get("weight_decay_scale", 1.0)
    trainer_config = TrainingConfig(
        lr_embeddings=args.lr_embeddings * lr_scale,
        lr_transformer=args.lr_transformer * lr_scale,
        lr_head=args.lr_head * lr_scale,
        lr_residuals=args.lr_residuals * lr_scale,
        weight_decay=args.weight_decay * weight_decay_scale,
        lr_warmup_steps=args.warmup_steps,
        lr_warmdown_ratio=args.warmdown_ratio,
        final_lr_frac=args.final_lr_frac,
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
        use_amp=args.use_amp,
        monitor_grad_norms=args.monitor_grad_norms,
        **base_training_config
    )
    print0_dict("Trainer config", trainer_config.model_dump())

    optimizers = model.build_optimizer(trainer_config)

    # ------------------------------------------------------------------------------
    # INIT BOARD
    # ------------------------------------------------------------------------------

    from gpt_lab.utils.board import Board, DummyBoard

    if is_master_process:
        board_args["dirname"] = meta_config["dirname"]
        board = Board(
            board_type=args.board,
            # entity_name=None, # TODO: add option for wandb entity
            project=f"trainbase_{meta_config['project']}",
            run=meta_config['name'],
            config=board_args | {"meta_config": meta_config, "training_config": base_training_config, "model_card": model.config.model_dump()},
            board_dir=args.board_dir,
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
        config=trainer_config, board=board, checkpoint_dir=meta_config["dirname"] 
    )
    trainer.train()

    # ------------------------------------------------------------------------------
    # LOG REPORT
    # ------------------------------------------------------------------------------

    from gpt_lab.utils.report import get_report

    # get_report().log(section="Base model training", data=[
    #     board_args, # CLI args
    #     { # stats about the training setup
    #         "Number of parameters": model.n_params,
    #         "Number of FLOPs per token": f"{trainer_config.n_flops_per_token:e}",
    #         "Calculated number of iterations": trainer_config.n_steps,
    #         "Number of training tokens": trainer_config.total_batch_size,
    #         "Tokens : Scaling params ratio": trainer_config.total_batch_size * trainer_config.n_steps / model.n_scaling_params(),
    #         "DDP world size": dist_info["WORLD_SIZE"],
    #         "warmup_steps": trainer_config.warmup_steps,
    #         "warmdown_ratio": trainer_config.warmdown_ratio,
    #         "final_lr_frac": trainer_config.final_lr_frac,
    #     },
    #     { # stats about training outcomes
    #         "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
    #         "Final validation bpb": val_bpb,
    #         "CORE metric estimate": results.get("core_metric", None),
    #         "MFU %": f"{mfu:.2f}%",
    #         "Total training flops": f"{flops_so_far:e}",
    #         "Total training time": f"{total_training_time/60:.2f}m",
    #         "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    #     }
    # ])

    # cleanup
    board.close() # wandb run finish
    cleanup_dist_groups()