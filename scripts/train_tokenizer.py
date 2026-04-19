if not __name__ == "__main__":
    raise ImportError("This script is intended to be run as a standalone program.")

from gpt_lab.tokenizer.tokenizer import Tokenizer
from gpt_lab.tokenizer.bpe import bpe, bpe_fast
from gpt_lab.utils.schemas import TokenizerTrainerConfig
from gpt_lab.tokenizer.corpus import TokenizerCorpus
from typing import Union, Generator


from pathlib import Path
import argparse, pickle
import time

parser = argparse.ArgumentParser(description="Train BPE tokenizer on a given corpus or evaluate BPE tokenizer against baselines.")
parser.add_argument("--vocab-size", type=int, default=32_000, help="Vocabulary size for BPE tokenizer (default: 30,000).")
parser.add_argument('--max-chars', type=int, default=-1, help='Maximum characters to train or evaluate on (default: 10B)')
parser.add_argument("--chars-per-doc", type=int, default=-1, help="Maximum number of characters per document to use from the corpus for training or evaluation (default: 10,000).")
parser.add_argument("--name", type=str, default="ic1_tokenizer", help="Name of the tokenizer (default: 'ic1_tokenizer').")
parser.add_argument("--corpus-path", type=str, default="./.data/corpus.txt", help="Path to the corpus file (default: './.data/corpus.txt').")
parser.add_argument("--write-corpus", action="store_true", help="Flag to indicate training mode.")
parser.add_argument("--fast", action="store_true", help="Flag to indicate using fast implementation.")
parser.add_argument("--num-proc", type=int, default=-1, help="Number of processes to use for training (default: -1).")
parser.add_argument("--trainer", type=str, default="tiktoken", choices=["tiktoken", "huggingface", "bpe", "fbpe", "rbpe", "dummy"], help="Which BPE training implementation to use (default: 'tiktoken').")
parser.add_argument("--corpus-seed", type=int, default=42, help="Random seed for corpus sampling (default: 42).")
# parser.add_argument("--nb_special_tokens", type=int, default=16, help="Number of special tokens to reserve in the tokenizer vocabulary (default: 16).")
args = parser.parse_args()

corpus_path = Path(args.corpus_path)
config = TokenizerTrainerConfig(
    max_chars=args.max_chars,
    chars_per_doc=args.chars_per_doc,
    vocab_size=args.vocab_size,
    name=args.name,
    num_proc=args.num_proc, 
    trainer=args.trainer,
    dircorpus=corpus_path,
)

print(f"Tokenizer training configuration: {config}")
if args.write_corpus:
    corpus = TokenizerCorpus.write_from_sources(
        corpus_dir=corpus_path,
        sources=None,
        chars_per_doc=config.chars_per_doc,
        max_chars=config.max_chars,
        random_seed=args.corpus_seed
    )
else:
    corpus = TokenizerCorpus.from_path(corpus_path)

print(f"Loading corpus from {corpus_path}")
print(f"Corpus size: {corpus_path.stat().st_size / 1e6:.2f} MB")

t0 = time.time()
tokenizer = Tokenizer.train_from_iterator(
    text_iterator=corpus.iterator(),
    config=config
)
t1 = time.time()

# Test the tokenizer on some sample inputs
simple_test_samples = [
    "Hello, world!",
    "This is a test of the BPE tokenizer.",
    "The quick brown fox jumps over the lazy dog.",
    "I am fine, thank you!",
    "GPT models are powerful for natural language processing tasks."
]
token_ids = [tokenizer.encode(sample) for sample in simple_test_samples]
assert all(token_ids), "Tokenizer failed to encode some test samples."
assert all(tokenizer.decode(token_id) == sample for sample, token_id in zip(simple_test_samples, token_ids)), "Tokenizer failed to round-trip encode-decode some test samples."
comp_ratios = []
for sample, token_id in zip(simple_test_samples, token_ids):
    print(f"Sample: {sample}")
    print(f"Tokens: {' | '.join([tokenizer.decode([t_id]) for t_id in token_id])}")
    print(f"Token IDs: {' | '.join([str(t_id) for t_id in token_id])}")
    print(f"Decoded: {tokenizer.decode(token_id)}")
    ratio = len(token_id) / len(sample)
    comp_ratios.append(ratio)
    print(f"Compression ratio: {ratio:.2f}")
    print("-" * 50)


print(f"Training took {t1 - t0:.2f} seconds.")
print(f"Average compression ratio on test samples: {sum(comp_ratios) / len(comp_ratios):.2f}")

# Analyze the learned vocabulary
vocab = tokenizer.token_to_id
import random
rd_idx = random.randint(0, len(vocab) - 10)

print("Vocab size", len(vocab))
print("Sample vocab:", list(vocab.items())[rd_idx:rd_idx+10])

print("Resume training or evaluation with the above tokenizer configuration and implementation choice, and compare against baselines if needed.")
print("Implementation | Vocabulary size | Num proc | Corpus size | Training time | Compression ratio")
print(f"{config.trainer} | {config.vocab_size:,} | {config.num_proc} | {corpus_path.stat().st_size / 1e6:.2f} MB | {t1 - t0:.2f} seconds | {sum(comp_ratios) / len(comp_ratios):.2f}")


print(f"Test token 32: {tokenizer.decode([32])!r}")