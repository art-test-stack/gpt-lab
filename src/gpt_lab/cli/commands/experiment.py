from argparse import ArgumentParser, Namespace

from gpt_lab.cli.options import cache_dir, positive_int


_BASELINES = ["gpt2", "cl100k_base", "o200k_base"]

def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default is 42.')
    parser.add_argument('--num-procs', type=positive_int, default=None, help='Number of processes to use for tokenizer training. Defaults to the number of CPU cores available, capped at 32 to avoid overloading the system.')
    parser.add_argument('--baselines', type=str, default=','.join(_BASELINES), help=f"Comma-separated list of baseline tokenizers to compare with. Default is {','.join(_BASELINES)}.")
    parser.add_argument('--vocab-sizes', type=str, default='50000,70000,100000,200000', help='Comma-separated list of vocabulary sizes to train tokenizers with.')
    parser.add_argument('--pat-strs', type=str, default=None, help='Comma-separated list of pattern string names to use for tokenizer training. If not specified, defaults to using the GPT-2 pattern string.')
    parser.add_argument('--write-corpus', action='store_true', help='Flag to indicate training mode (write corpus). If not set, the script will attempt to load an existing corpus from disk.')
    parser.add_argument('--corpus-dir', type=str, default=None, help="Corpus directory; omitted uses <cache>/data/corpus/<result-directory-name>.")
    parser.add_argument('--corpus-sizes-gb', type=str, default=None, help='Comma-separated positive corpus sizes in GiB (1024**3 bytes). Required for a new experiment; resume can use saved metadata.')
    parser.add_argument('--compare-truncated-baselines', action='store_true', help='Whether to compare trained tokenizers with truncated versions of baseline tokenizers.')
    parser.add_argument('--corpus-temperature-alpha', type=float, default=None, help='Optional temperature parameter to control the randomness of the corpus generation. Higher values will result in a more diverse corpus, while lower values will make it more focused on the most common samples. This can be useful for testing how the tokenizer performs with different levels of corpus diversity.')
    parser.add_argument('--resume', action='store_true', help='Whether to resume from existing results file. If set, the script will attempt to load existing results from the specified results path and continue from there, skipping any experiments that have already been completed. This can be useful for long-running experiments that may be interrupted or for iteratively adding new configurations without re-running everything.')
    parser.add_argument('--result-dir', type=str, default=str(cache_dir() / 'tokenizers' / 'scaling_tokenizer_results'), help="Result directory; existing results are backed up with a numeric suffix unless --resume is supplied.")
    parser.add_argument('--save-every', type=positive_int, default=10, help='Number of runs to execute before saving intermediate results to disk. This can help prevent data loss in case of interruptions and allows for monitoring progress during long experiments. Default is 10.')
    parser.add_argument('--board', default='dummy', help="Legacy monitoring selection, retained but unused by this experiment.")


def register(subparsers) -> None:
    parser = subparsers.add_parser('experiment', help='Experimental and expensive corpus/vocabulary scaling sweep', description='Experimental and expensive corpus/vocabulary scaling sweep. May download training and evaluation data and baseline tokenizers. Writes results; --resume skips completed runs.')
    children = parser.add_subparsers(required=True, title='commands')
    command = children.add_parser('tokenizer-scaling', help='Experimental and expensive corpus/vocabulary scaling sweep', description='Experimental and expensive corpus/vocabulary scaling sweep. May download training and evaluation data and baseline tokenizers. Writes results; --resume skips completed runs.')
    add_arguments(command)
    command.set_defaults(_handler='gpt_lab.cli.commands.experiment:run')


def run(args: Namespace) -> None:
    from gpt_lab.workflows.tokenizer_scaling import run as execute
    execute(args)
