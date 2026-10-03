from argparse import ArgumentParser, Namespace

from gpt_lab.cli.options import positive_int


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic', help='Synthetic documents or local token files.')
    parser.add_argument('--loader', choices=['glab', 'glab-on-the-fly', 'glab-tokenized', 'nanochat', 'all'], default='all', help='glab selects both gpt-lab variants; all also includes nanochat.')
    parser.add_argument('--batch-size', '--batch_size', type=positive_int, default=8, help='Sequences per batch.')
    parser.add_argument('--seq-len', '--seq_len', type=positive_int, default=2048, help='Tokens per sequence.')
    parser.add_argument('--num-batches', '--num_batches', type=positive_int, default=20, help='Number of batches to time (after 2 warm-up batches)')
    parser.add_argument('--buffer-size', '--buffer_size', type=positive_int, default=1000, help='Token buffer budget for gpt-lab; document buffer count for nanochat')
    parser.add_argument('--num-docs', '--num_docs', type=positive_int, default=20000, help='Number of synthetic documents')
    parser.add_argument('--device', default='cpu', help='torch device, e.g. cpu or cuda')
    parser.add_argument('--bos-id', '--bos_id', type=int, default=1, help='BOS token id (for nanochat loader)')
    parser.add_argument('--bin', default=None, help='Path to pretokenized .bin file')
    parser.add_argument('--idx', default=None, help='Path to pretokenized .idx file')


def register(subparsers) -> None:
    parser = subparsers.add_parser('benchmark', help='Benchmark dataloader implementations', description='Benchmark dataloader implementations. Synthetic mode is CPU/offline by default; real mode reads local uint32 .bin tokens and uint64 .idx boundaries. May allocate substantial RAM.')
    children = parser.add_subparsers(required=True, title='commands')
    command = children.add_parser('dataloaders', help='Benchmark dataloader implementations', description='Benchmark dataloader implementations. Synthetic mode is CPU/offline by default; real mode reads local uint32 .bin tokens and uint64 .idx boundaries. May allocate substantial RAM.')
    add_arguments(command)
    command.set_defaults(_handler='gpt_lab.cli.commands.benchmark:run')


def run(args: Namespace) -> None:
    if args.mode == 'real' and (args.bin is None or args.idx is None):
        raise ValueError('--mode real requires --bin and --idx')
    from gpt_lab.workflows.dataloaders import run as execute
    execute(args)
