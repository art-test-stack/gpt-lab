
from gpt_lib.model.auto import AutoGPTConfig
from gpt_lib.model.gpt import GPTConfig
from gpt_lib.utils.schemas import GPTConfig, TokenizerConfig

from gpt_lib.utils.default import DATA_DIR, MODELS_FOLDER

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train base model.")
    subparsers = parser.add_subparsers(dest="model-init")
    
    # Common arguments
    parser.add_argument("--model-name", type=str, default="ic1", help="Model name")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_FOLDER), help="Cache directory to save model checkpoints and logs.")
    parser.add_argument("--num-steps", type=int, default=-1, help="Number of training steps (overrides num-epochs if > 0).")
    parser.add_argument("--batch-size", type=int, default=-1, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length for training.")
    parser.add_argument("--vocab-size", type=int, default=30000, help="Vocabulary size for tokenizer.")
    parser.add_argument("--target-param-data-ratio", type=float, default=1e-6, help="Target parameter-to-data ratio for auto-configured models.")
    parser.add_argument("--aspect-ratio", type=float, default=4.0, help="Aspect ratio (d_model/d_head) for auto-configured models.")
    parser.add_argument("--random-seed", type=int, defualt=42, help="Random seed for model initialization")

    # Auto-config subparser
    auto_parser = subparsers.add_parser("auto", help="Auto-config model based on depth.")
    auto_parser.add_argument("--depth", type=int, default=12, help="Number of model laeyrs")
    auto_parser.add_argument("--aspect-ratio", type=float, default=256, help="")
    auto_parser.add_argument("--window-pattern", type=str, default=None, help="Window pattern for sliding attention window. String of 'S' and 'L'")
    auto_parser.add_argument("--window-size", type=str, default=None, help="Window size for pattern smalls (S).")
    auto_parser.add_argument("--softcap", type=float, default=None, help="Soft cap for model logits to prevent overflow.")
    auto_parser.add_argument("--attn-softcap", type=float, default=None, help="Soft cap for attention scores to prevent overflow. Not supported yet.")
    auto_parser.add_argument("--num-steps", type=int, default=-1, help="")
    auto_parser.add_argument("--target-flops", type=float, default=-1., help="")
    auto_parser.add_argument("--target-param-data-ratio", type=float, default=11., help="")
    auto_parser.add_argument("--total-batch-size", type=int, default=-1, help="Total batch size across all devices for auto-configured models.")

    # Custom-config subparser
    custom_parser = subparsers.add_parser("custom", help="Custom-config model with specified parameters.")
    
    args = parser.parse_args()

    if args.model_init == "auto":
        mconfig = AutoGPTConfig(
            basename=args.model_name,
            dirname=args.model_dir,
            num_steps=args.num_steps,
            target_flops=args.target_flops,
            target_param_data_ratio=args.target_param_data_ratio,
            aspect_ratio=args.aspect_ratio,
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size
        )
        gpt_config = mconfig.get_config()

    elif args.model_init == "custom":
        raise NotImplementedError("Custom model configuration is not implemented yet. Please use 'auto' mode for now.")
        gpt_config = GPTConfig(
            basename=args.model_name,
            dirname=args.model_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size
        )

    else:
        raise ValueError(f"Model initialization mode {args.model_init!r} is not known. Please choose among 'auto', 'custom', etc.")
    
    model = GPTModel.from_scratch(gpt_config)