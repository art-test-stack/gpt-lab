from argparse import ArgumentParser, Namespace


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size for BPE tokenizer.')
    parser.add_argument('--max-chars', type=int, default=-1, help='Maximum characters used for training; -1 reads the full existing corpus. Required positive for --write-corpus.')
    parser.add_argument('--chars-per-doc', type=int, default=-1, help='Maximum number of characters per document to use from the corpus for training or evaluation.')
    parser.add_argument('--name', type=str, default='ic1_tokenizer', help='Name of the tokenizer.')
    parser.add_argument('--corpus-path', type=str, default='./.data/corpus.txt', help='UTF-8 text file or TokenizerCorpus directory; --write-corpus writes a directory at this path.')
    parser.add_argument('--write-corpus', action='store_true', help='Generate a corpus directory from the default source mixture (network and disk intensive). Samples up to four UTF-8 bytes per character before training limits are applied.')
    parser.add_argument('--fast', action='store_true', help='Legacy flag, retained but unused. Select the backend with --trainer.')
    parser.add_argument('--num-proc', type=int, default=-1, help='Number of processes to use for training.')
    parser.add_argument('--trainer', type=str, default='tiktoken', choices=['tiktoken', 'huggingface', 'bpe', 'fbpe', 'rbpe', 'dummy'], help='Legacy backend choices/default are retained; only huggingface is currently implemented.')
    parser.add_argument('--corpus-seed', type=int, default=42, help='Random seed for corpus sampling.')


def register(subparsers) -> None:
    parser = subparsers.add_parser('tokenizer', help='Train a BPE tokenizer (expensive)', description='Train a BPE tokenizer (expensive). Reads a local text file or corpus directory; --write-corpus downloads and writes the default source mixture. Only the huggingface training backend is currently implemented.')
    children = parser.add_subparsers(required=True, title='commands')
    command = children.add_parser('train', help='Train a BPE tokenizer (expensive)', description='Train a BPE tokenizer (expensive). Reads a local text file or corpus directory; --write-corpus downloads and writes the default source mixture. Only the huggingface training backend is currently implemented.')
    add_arguments(command)
    command.set_defaults(_handler='gpt_lab.cli.commands.tokenizer:run')


def run(args: Namespace) -> None:
    from gpt_lab.workflows.train_tokenizer import run as execute
    execute(args)
