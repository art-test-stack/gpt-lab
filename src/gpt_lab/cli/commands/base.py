from argparse import ArgumentParser, Namespace

from gpt_lab.cli.options import cache_dir, checkpoint_step


def add_common_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--model-name', type=str, default='ic1', help='Model name')
    parser.add_argument('--run-name', type=str, default=None, help='Run name for checkpointing and logging. Auto generates a name when omitted; resume selects the latest run. Resume also accepts latest, best, or -N.')
    parser.add_argument('--model-dir', type=str, default=str(cache_dir() / 'models'), help='Cache directory to save model checkpoints and logs.')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for model initialization')
    parser.add_argument('--optim-config-path', type=str, default=None, help='YAML overrides for the bundled base optimizer profile; values are scaled automatically.')
    parser.add_argument('--device', type=str, default='auto', help="Device to train on. If 'auto', will detect best device available. Recommended to keep as 'auto' unless you have specific needs.")
    parser.add_argument('--board', default='wandb', type=str, choices=('tensorboard', 'wandb', 'dummy'), help="Board log directory (options: 'tensorboard', 'wandb', 'dummy').")
    parser.add_argument('--board-dir', type=str, default=None, help='Directory to save board logs. If not set, will use default cache directory.')
    parser.add_argument('--ds-config-path', type=str, default='configs/data.yaml', help='Legacy config path, retained but unused; training reads local shards under --ds-name.')
    parser.add_argument('--ds-name', type=str, default='climbmix-base', help='Dataset directory name under the data cache; requires prepared local shards.')
    parser.add_argument('--save-on-best', action='store_true', help="Save a checkpoint when evaluation improves the best metric.")
    parser.add_argument('--eval-bpb-every', type=int, default=250, help='Evaluate val bpb every N steps (-1 = last, 0 = disable, N > 0 = every N steps).')
    parser.add_argument('--n-bpb-tokens', type=int, default=80 * 524288, help='Number of tokens to evaluate val loss on.')
    parser.add_argument('--eval-core-every', type=int, default=2000, help='Evaluate CORE metric every N steps (-1 = last, 0 = disable, N > 0 = every N steps).')
    parser.add_argument('--n-core-tokens', type=int, default=500, help='Examples per task for CORE metric')
    parser.add_argument('--sample-every', type=int, default=0, help='Sample from model every N steps (-1 = last, 0 = disable, N > 0 = every N steps).')
    parser.add_argument('--save-every', type=int, default=-1, help='Save checkpoints every N steps (-1 = last, 0 = disable, N > 0 = every N steps).')
    parser.add_argument('--log-every', type=int, default=250, help='Log metrics every N steps (-1 = last, 0 = disable, N > 0 = every N steps).')
    parser.add_argument('--monitor-grad-norms', action='store_true', help='Whether to monitor gradient norms during training. If set, will log the norm of the gradients of each parameter to the board at each training step.')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Batch size for each device during training. Batch size define further effective batch size as device_batch_size * max_seq_len * n_acc_steps.')
    parser.add_argument('--use-nanochat-dataloader', action='store_true', help='Whether to use the nanochat dataloader instead of the default dataloader.')


def add_auto_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length for training.')
    parser.add_argument('--tokenizer-model', type=str, default=None, help='Tokenizer model to use for auto-configured models. If not set, will use vocab size scaling law to determine tokenizer config.')
    parser.add_argument('--vocab-size', type=int, default=-1, help='Vocabulary size for auto-configured models. If not set, will be determined by vocab size scaling law based on model depth.')
    parser.add_argument('--pat-str', type=str, default=None, help="Split pattern for pre-tokenization if training a new-tokenizer. Options are 'gpt2, 'gpt4', 'cl100k_base', 'o200k_base', or directly the pattern string. If not set, will default to 'gpt2' pattern.")
    parser.add_argument('--train-tokenizer', action='store_true', help='Whether to train a new tokenizer from scratch.')
    parser.add_argument('--truncate-tokenizer', action='store_true', help='Legacy flag, retained but unused by the training pipeline.')
    parser.add_argument('--depth', type=int, default=12, help='Number of model layers.')
    parser.add_argument('--aspect-ratio', type=float, default=64, help='Aspect ratio for auto-configured models.')
    parser.add_argument('--d-head', type=int, default=128, help='Dimension of each attention head for auto-configured models. If not set, will be determined by aspect ratio and model depth.')
    parser.add_argument('--n-kv-heads', '--d-kv-head', dest='n_kv_heads', type=int, default=None, help='Number of key/value heads for GQA; None uses the query head count. --d-kv-head is a deprecated alias.')
    parser.add_argument('--tie-embeddings', type=str, default='embs;h_embs;head', help='Embedding groups: comma ties, semicolon separates. Names: embs, head, h_embs. Standalone h_embs: ties all value tables; bare h_embs leaves them separate. Quote the string in the shell.')
    parser.add_argument('--window-pattern', type=str, default=None, help="Window pattern for sliding attention window. String of 'S' and 'L'. If 'None', will be later set as 'SSSL'.")
    parser.add_argument('--window-size', type=str, default=None, help='Window size for pattern smalls (S).')
    parser.add_argument('--softcap', type=float, default=18.0, help='Soft cap for model logits to prevent overflow.')
    parser.add_argument('--attn-softcap', type=float, default=None, help='Soft cap for attention scores to prevent overflow.')
    parser.add_argument('--attn-impl', type=str, default='sdpa', help="Attention implementation to use for auto-configured models. Options are 'sdpa' and 'fused'. Both shoulf exhibit same results but 'fused' should be slightly faster (if runned under cuda device).")
    parser.add_argument('--num-steps', type=int, default=-1, help='Number of training steps (overrides num-epochs if > 0).')
    parser.add_argument('--target-flops', type=float, default=-1.0, help='Target FLOPS for auto-configured models.')
    parser.add_argument('--target-param-data-ratio', type=float, default=11.0, help='Target parameter-to-data ratio for auto-configured models.')
    parser.add_argument('--target-time', type=float, default=-1.0, help='Target training time in seconds for auto-configured models. This parameter overrides num-steps. ')
    parser.add_argument('--fp8', action='store_true', help='Legacy experimental flag, retained but unused by the training pipeline.')
    parser.add_argument('--n-acc-steps', type=int, default=-1, help='Number of gradient accumulation steps to perform before each optimizer step (-1 automatically sets; 0 disables). Recommended: -1.')
    parser.add_argument('--total-batch-size', type=int, default=-1, help='Override the total tokens per optimizer step across devices; -1 uses the scaling law. With automatic accumulation, must be divisible by device_batch_size * max_seq_len * world_size.')
    parser.add_argument('--warmup-steps', type=int, default=40, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--warmdown-ratio', type=float, default=0.65, help='Ratio of training steps to warm down the learning rate at the end of training for auto-configured models.')
    parser.add_argument('--final-lr-frac', type=float, default=0.05, help='Final learning rate as a fraction of the initial learning rate for auto-configured models.')


def add_resume_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--checkpoint-step', type=checkpoint_step, default=None, help='Step of checkpoint to resume from. If not set, will try to resume from the latest checkpoint in checkpoint directory. Otherwise, with -1, -2, etc., will try to resume from the last, second to last, etc. checkpoint in checkpoint directory.')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Explicit run directory (with meta.json), base directory, or checkpoint_step_N directory. Overrides model-dir/model-name/run-name; a step directory selects that exact step.')


def register(parser) -> None:
    modes = parser.add_subparsers(dest='model_init', required=True, title='initialization modes')
    auto = modes.add_parser('auto', help='Train from scratch with depth-based scaling laws.',
                            prog=parser.prog,
                            description='Train from scratch by default. -1 selects automatic scaling for supported horizon/batch options. May load or train a tokenizer and contact the selected metrics board.',
                            epilog='Resume: gpt-lab train resume --help. Distributed: torchrun --standalone --nproc_per_node=8 -m gpt_lab.cli train ...')
    add_common_arguments(auto)
    add_auto_arguments(auto)
    auto.set_defaults(_handler='gpt_lab.cli.commands.base:run')
    resume = modes.add_parser('resume', help='Restore a run, optimizer, and dataloader checkpoint.',
                              description='Resume expensive training. The latest run and latest checkpoint are discovered automatically. Saved model/trainer configuration takes precedence over common training defaults.')
    add_common_arguments(resume)
    add_resume_arguments(resume)
    resume.set_defaults(_handler='gpt_lab.cli.commands.base:run')


def resolve_checkpoint(args: Namespace) -> Namespace:
    """Compatibility adapter for checkpoint path resolution."""
    if not getattr(args, "checkpoint_dir", None):
        return Namespace(**vars(args))
    from gpt_lab.workflows.train_base import resolve_checkpoint as resolve_path
    return Namespace(**(vars(args) | resolve_path(args.checkpoint_dir, args.checkpoint_step)))


def run(args: Namespace) -> None:
    from gpt_lab.utils.schemas import OptimizerConfig
    from gpt_lab.workflows.train_base import train_base

    ignored = {"_handler", "optim_config_path", "ds_config_path", "truncate_tokenizer", "fp8"}
    parameters = {key: value for key, value in vars(args).items() if key not in ignored}
    profile = None
    if args.model_init == "auto" or args.optim_config_path is not None:
        profile = OptimizerConfig.from_file(args.optim_config_path)
    train_base(optim_config=profile, **parameters)
