from argparse import ArgumentParser, Namespace



def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--ds-name', default='climbmix', help='Dataset key in the YAML config.')
    parser.add_argument('--config-path', default='configs/data.yaml', help='Dataset YAML path (relative to the current working directory).')
    parser.add_argument('--mode', choices=['bucket', 'dataset', 'local'], default='bucket', help='Output destination. bucket and dataset upload; local writes the data cache.')
    parser.add_argument('--repo-id', type=str, default=None, help='Hugging Face bucket/repository ID; required for remote modes.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory name below data/ (defaults to output_dir in the dataset config)')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming; otherwise retain the dataset config setting.')
    parser.add_argument('--max-shards', type=int, help='Override the dataset maximum shard count.')
    parser.add_argument('--chars-per-shard', type=int, default=256000000, help='Target characters per Parquet shard.')
    parser.add_argument('--row-group-size', type=int, default=1024, help='Documents per Parquet row group.')
    parser.add_argument('--max-in-flight', type=int, default=8, help='Maximum concurrent shard writes/uploads.')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum upload attempts.')
    parser.add_argument('--retry-timeout', type=int, default=10, help='Initial retry backoff in seconds.')


def register(subparsers) -> None:
    parser = subparsers.add_parser('data', help='Expensive dataset download and Parquet resharding', description='Expensive dataset download and Parquet resharding. Bucket/dataset modes upload to Hugging Face using HF_TOKEN; local mode writes the cache. Existing shard names may be overwritten.')
    children = parser.add_subparsers(required=True, title='commands')
    command = children.add_parser('reshard', help='Expensive dataset download and Parquet resharding', description='Expensive dataset download and Parquet resharding. Bucket/dataset modes upload to Hugging Face using HF_TOKEN; local mode writes the cache. Existing shard names may be overwritten.')
    add_arguments(command)
    command.set_defaults(_handler='gpt_lab.cli.commands.data:run')


def run(args: Namespace) -> None:
    if args.mode != 'local' and not args.repo_id:
        raise ValueError('--repo-id is required in bucket and dataset modes')
    from gpt_lab.workflows.reshard import run as execute
    execute(args)
