"""Tokenizer training using the current TokenizerConfig and corpus APIs."""

from argparse import Namespace
from collections.abc import Iterable, Iterator
from pathlib import Path
import time


def limit_characters(texts: Iterable[str], max_chars: int, chars_per_doc: int) -> Iterator[str]:
    """Keep legacy character limits as characters, including Unicode text."""
    remaining = max_chars
    for text in texts:
        if chars_per_doc >= 0:
            text = text[:chars_per_doc]
        if remaining >= 0:
            text = text[:remaining]
            remaining -= len(text)
        if text:
            yield text
        if remaining == 0:
            break


def build_config(args: Namespace):
    from gpt_lab.utils.schemas import TokenizerConfig, TokenizerTrainerConfig

    return TokenizerConfig(
        name=args.name,
        vocab_size=args.vocab_size,
        trainer=TokenizerTrainerConfig(
            source=args.trainer,
            num_proc=args.num_proc,
            dircorpus=Path(args.corpus_path),
        ),
    )


def read_text_file(path: Path) -> Iterator[str]:
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            yield line.rstrip('\n')


def run(args: Namespace) -> None:
    if args.trainer != 'huggingface':
        raise ValueError(f'Trainer {args.trainer!r} is not implemented by the current tokenizer API; use --trainer huggingface')
    if args.max_chars < -1 or args.chars_per_doc < -1:
        raise ValueError('Character limits must be -1 (unlimited) or nonnegative')
    if args.write_corpus and args.max_chars <= 0:
        raise ValueError('--write-corpus requires a positive --max-chars to bound source sampling')

    from gpt_lab.tokenizer import Tokenizer
    from gpt_lab.tokenizer.corpus import TokenizerCorpus

    config = build_config(args)
    path = Path(args.corpus_path)
    print(f'Tokenizer training configuration: {config}')
    if args.write_corpus:
        # The current source writer budgets bytes. UTF-8 needs at most four
        # bytes per character. Apply the exact legacy character limits below.
        corpus = TokenizerCorpus.write_from_sources(
            corpus_dir=path, sources=None,
            max_bytes=4 * args.max_chars,
            bytes_per_doc=4 * (args.chars_per_doc if args.chars_per_doc > 0 else args.max_chars),
            random_seed=args.corpus_seed,
        )
        texts = corpus.iterator()
    elif path.is_file():
        texts = read_text_file(path)
    else:
        texts = TokenizerCorpus.from_path(path).iterator()

    start = time.time()
    tokenizer = Tokenizer.train_from_iterator(
        text_iterator=limit_characters(texts, args.max_chars, args.chars_per_doc),
        config=config,
    )
    samples = ['Hello, world!', 'This is a test of the BPE tokenizer.',
               'The quick brown fox jumps over the lazy dog.', 'I am fine, thank you!',
               'GPT models are powerful for natural language processing tasks.']
    ratios = []
    for sample in samples:
        tokens = tokenizer.encode(sample)
        assert tokens and tokenizer.decode(tokens) == sample, 'Tokenizer failed to round-trip encode-decode'
        ratios.append(len(tokens) / len(sample))
        print(f'Sample: {sample}\nToken IDs: {tokens}\nDecoded: {tokenizer.decode(tokens)}')
    print(f'Training took {time.time() - start:.2f} seconds.')
    print(f'Average compression ratio on test samples: {sum(ratios) / len(ratios):.2f}')
    print(f'Vocab size: {len(tokenizer.mergeable_ranks)}')
