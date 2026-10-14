"""Fixed estimated-compute sweep using truncated pretrained tokenizers. See README.md."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from gpt_lab.evaluate.bpb import compute_bpb
from gpt_lab.model.checkpoint import CheckpointManager, build_meta_model
from gpt_lab.tokenizer.tokenizer import Tokenizer
from gpt_lab.train.trainer import Trainer
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.schemas import (
    DataLoaderState,
    TrainerConfig, TransformerConfig,
)
from gpt_lab.utils.special_tokens import SpecialTokens


SHARING = {
    "fully-shared": "embs,head,value_embs:",
    "input-value": "embs,value_embs;head",
    "output-value": "embs;head,value_embs",
    "shared-values-tied-head": "embs,head;value_embs:",
    "shared-values": "embs;head;value_embs:",
    "independent-values-tied-head": "embs,head;value_embs",
    "independent": "embs;head;value_embs",
}
ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-text", type=Path, help="local UTF-8 training file")
    p.add_argument("--eval-text", type=Path, help="separate local UTF-8 held-out file")
    p.add_argument("--base-tokenizer", choices=["gpt2", "cl100k_base", "o200k_base"], default="cl100k_base",
                   help="pretrained tiktoken encoding whose merge ranks are truncated")
    p.add_argument("--vocab-sizes", type=positive_int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--sharing", choices=list(SHARING), nargs="+", default=list(SHARING))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train-flops", type=float, default=1e9, help="estimated training FLOPs per run")
    p.add_argument("--depth", type=positive_int, default=4)
    p.add_argument("--width", type=positive_int, default=64)
    p.add_argument("--heads", type=positive_int, default=4)
    p.add_argument("--seq-len", type=positive_int, default=64)
    p.add_argument("--batch-size", type=positive_int, default=2)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--output", type=Path, default=ROOT / "experiments/results/value_vocab.jsonl")
    p.add_argument("--dry-run", action="store_true", help="load/truncate tokenizers and plan; no model training/evaluation")
    p.add_argument("--smoke", action="store_true", help="built-in ASCII text, depth=4 width=16 context=8 batch=1 budget=3e6")
    return p


def plan(model, budget, batch_tokens):
    flops = model.estimate_flops()
    steps = math.floor(budget / (flops * batch_tokens))
    if steps < 1:
        raise ValueError("FLOP budget cannot fund one batch; increase --train-flops or reduce batch/model size")
    counts = model.n_params_per_layer()
    return {
        "n_non_vocab_params": model.n_non_vocab_params(),
        # Unique counts follow the repository's ownership order: input, head, values.
        "input_embedding_params": counts["embeds"],
        "output_embedding_params": counts["lm_head"],
        "value_embedding_params": counts["value_embeds"],
        "output_table_params": model.lm_head.weight.numel(),
        "value_table_count": len({id(ve.weight) for ve in model.value_embeds.values()}),
        "total_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "padded_vocab_size": model.embeds.num_embeddings,
        "flops_per_token": flops,
        "planned_training_steps": steps,
        "planned_training_tokens": steps * batch_tokens,
        "planned_training_flops": steps * batch_tokens * flops,
        "compute_shortfall": budget - steps * batch_tokens * flops,
    }


class CyclingBatches:
    """Sequential next-token batches, wrapping the same training stream."""
    def __init__(self, ids, batch_size, seq_len, device):
        self.ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.B, self.seq_len = batch_size, seq_len

    def __iter__(self):
        size = self.B * self.seq_len
        offset = 0
        while True:
            positions = (torch.arange(size + 1, device=self.ids.device) + offset) % len(self.ids)
            tokens = self.ids[positions]
            yield (tokens[:-1].view(self.B, self.seq_len).contiguous(),
                   tokens[1:].view(self.B, self.seq_len).contiguous(),
                   DataLoaderState(epoch=offset // len(self.ids)))
            offset += size


def evaluation_batches(ids, seq_len, device):
    """Include every target once, including a short final chunk; ids start with BOS."""
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    for start in range(0, len(ids) - 1, seq_len):
        end = min(start + seq_len, len(ids) - 1)
        yield tokens[start:end][None].contiguous(), tokens[start + 1:end + 1][None].contiguous(), None


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(args):
    if args.smoke:
        args.depth, args.width, args.heads = 4, 16, 2
        args.seq_len, args.batch_size, args.train_flops = 8, 1, 3e6
        if args.train_text or args.eval_text:
            raise ValueError("--smoke uses built-in text; omit text paths")
        train_text = "Small language models learn from repeated words and sentences.\n" * 40
        eval_text = "A small model can predict words. Held out sentences test its learning.\n"
    else:
        if not args.train_text or not args.eval_text:
            raise ValueError("provide both --train-text and --eval-text, or use --smoke")
        if args.train_text.resolve() == args.eval_text.resolve():
            raise ValueError("training and held-out files must differ")
        train_text = args.train_text.read_bytes().decode("utf-8")
        eval_text = args.eval_text.read_bytes().decode("utf-8")
    if not train_text or not eval_text or train_text == eval_text:
        raise ValueError("training/held-out texts must be nonempty and different")
    if not math.isfinite(args.train_flops) or args.train_flops <= 0:
        raise ValueError("--train-flops must be finite and positive")
    if args.depth < 3:
        raise ValueError("use depth >= 3 to distinguish shared from independent value tables")
    if args.width % args.heads or (args.width // args.heads) % 2:
        raise ValueError("width must divide heads with an even head dimension for RoPE")
    minimum_vocab = 256 + len(SpecialTokens().list())
    if min(args.vocab_sizes) < minimum_vocab:
        raise ValueError(f"vocab sizes must be >= {minimum_vocab} (bytes plus special tokens)")
    if any(len(values) != len(set(values)) for values in (args.vocab_sizes, args.sharing, args.seeds)):
        raise ValueError("sweep axes must not contain duplicates")
    base_tokenizer = Tokenizer.from_pretrained(args.base_tokenizer, source="tiktoken")
    if max(args.vocab_sizes) > base_tokenizer.vocab_size:
        raise ValueError(
            f"requested vocabulary exceeds {args.base_tokenizer}'s {base_tokenizer.vocab_size} tokens; "
            "choose a larger --base-tokenizer or reduce --vocab-sizes"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comp = ComputeContext(device=torch.device(args.device), compute_dtype=torch.float32)
    # Exclusive creation preserves previous sweeps and partial results after failure.
    with args.output.open("x") as results, tempfile.TemporaryDirectory(prefix="value-vocab-") as work:
        for vocab in args.vocab_sizes:
            tokenizer = Tokenizer.truncated_from_pretrained(args.base_tokenizer, vocab, source="tiktoken")
            if tokenizer.vocab_size != vocab:
                raise ValueError("cached truncated tokenizer size differs from requested vocabulary")
            train_ids = tokenizer.encode(train_text, prepend_bos=True)
            eval_ids = tokenizer.encode(eval_text, prepend_bos=True)
            for text, ids in ((train_text, train_ids), (eval_text, eval_ids)):
                if tokenizer.decode(ids[1:]) != text:
                    raise ValueError("tokenizer round-trip changed text; use a supported corpus/tokenizer")
            token_bytes = tokenizer.token_bytes
            eval_bytes = int(token_bytes[eval_ids[1:]].sum())
            if eval_bytes != len(eval_text.encode("utf-8")):
                raise ValueError("token byte counts do not match held-out UTF-8 bytes")
            for sharing, seed in itertools.product(args.sharing, args.seeds):
                config = TransformerConfig(
                    vocab_size=tokenizer.vocab_size, n_layers=args.depth, d_model=args.width,
                    d_ffn=4 * args.width, n_heads=args.heads, n_kv_heads=args.heads,
                    d_head=args.width // args.heads, max_context=args.seq_len,
                    tie_embeddings=SHARING[sharing], window_pattern="L", attn_impl="sdpa",
                )
                model = build_meta_model(config)
                row = dict(
                    status="planned" if args.dry_run else "completed",
                    requested_vocab_size=vocab, actual_vocab_size=tokenizer.vocab_size,
                    base_tokenizer=args.base_tokenizer, base_vocab_size=base_tokenizer.vocab_size,
                    tokenizer_name=tokenizer.config.name, tokenizer_method="merge-rank-prefix",
                    sharing=sharing, tie_embeddings=SHARING[sharing], seed=seed,
                    requested_training_flops=args.train_flops,
                    model_config=config.model_dump(mode="json"),
                    train_sha256=text_hash(train_text), eval_sha256=text_hash(eval_text),
                    eval_bytes=eval_bytes, eval_tokens=len(eval_ids) - 1,
                    device=args.device, precision="float32", world_size=1,
                    batch_size=args.batch_size, smoke=args.smoke,
                    training_steps=0, training_tokens=0, training_flops=0,
                    eval_loss=None, eval_bpb=None,
                    **plan(model, args.train_flops, args.batch_size * args.seq_len),
                )
                if not args.dry_run:
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    model.to_empty(device=comp.device)
                    model.init_weights()
                    train_config = TrainerConfig(
                        comp_info=comp, n_steps=row["planned_training_steps"], n_acc_steps=1,
                        device_batch_size=args.batch_size, total_batch_size=args.batch_size * args.seq_len,
                        n_flops_per_token=row["flops_per_token"], eval_bpb_every=0,
                        eval_core_every=0, sample_every=0, save_every=0, save_on_best=False,
                    )
                    run_name = f"{vocab}-{sharing}-{seed}"
                    (Path(work) / "sweep" / run_name).mkdir(parents=True)
                    manager = CheckpointManager("sweep", model_run=run_name,
                                                model_cachedir=work, comp_info=comp)
                    trainer = Trainer(model, tokenizer,
                                      CyclingBatches(train_ids, args.batch_size, args.seq_len, comp.device),
                                      None, config=train_config, checkpoint_manager=manager)
                    was_gc_enabled = gc.isenabled()
                    try:
                        # Avoid compilation startup cost for these serial, small experiments.
                        with torch.compiler.set_stance("force_eager"):
                            trainer.train()
                    finally:
                        # Trainer freezes tracked objects; release them between sweep runs.
                        gc.unfreeze()
                        if was_gc_enabled:
                            gc.enable()
                    row["training_tokens"] = trainer.state.n_tokens
                    row["training_steps"] = trainer.state.step + 1
                    row["training_flops"] = row["training_tokens"] * row["flops_per_token"]
                    model.eval()
                    metrics = compute_bpb(model, evaluation_batches(eval_ids, args.seq_len, comp.device),
                                          math.ceil((len(eval_ids) - 1) / args.seq_len), token_bytes, comp)
                    row["eval_bpb"] = metrics["bpb"]
                    # compute_bpb's debug loss averages batch means; weight the short tail correctly.
                    row["eval_loss"] = metrics["bpb"] * math.log(2) * eval_bytes / row["eval_tokens"]
                    if not math.isfinite(row["eval_bpb"]):
                        raise ValueError("nonfinite evaluation BPB")
                    del trainer
                results.write(json.dumps(row, allow_nan=False) + "\n")
                results.flush()
                print(f"{row['status']}: V={vocab}/{tokenizer.vocab_size} {sharing} seed={seed} steps={row['planned_training_steps']} BPB={row['eval_bpb']}")
                del model
    return args.output


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
