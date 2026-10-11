"""Cache deletion needs only the standard library."""

from argparse import Namespace
from pathlib import Path
import shutil

from gpt_lab.cli.options import cache_dir


SUBDIRS = {'tokenizer': 'tokenizers', 'data': 'data', 'corpus': 'data/corpus', 'models': 'models'}


def register(subparsers) -> None:
    description = 'Destructive cache deletion. Prompts for confirmation unless --force is supplied.'
    parser = subparsers.add_parser('cache', help=description, description=description)
    children = parser.add_subparsers(required=True, title='commands')
    clear = children.add_parser('clear', help=description, description=description)
    clear.add_argument('--cache-dir', '--cache_dir', dest='cache_dir', default=str(cache_dir()), help='Cache root to clear.')
    clear.add_argument('--subdirs', nargs='*', default=[], choices=[*SUBDIRS, 'all'], help='Areas to clear. Omitted, empty, or all clears the whole cache.')
    clear.add_argument('--force', action='store_true', help='Delete without the confirmation prompt.')
    clear.set_defaults(_handler='gpt_lab.cli.commands.cache:run')


def run(args: Namespace) -> int:
    root = Path(args.cache_dir).expanduser()
    if not root.is_dir():
        print(f'Cache directory does not exist: {root}')
        return 0
    # Resolve the selected cache root, but never follow subdirectory symlinks.
    root = root.resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError('Refusing to delete a filesystem root or home directory')
    paths = [root] if not args.subdirs or 'all' in args.subdirs else [root / SUBDIRS[s] for s in dict.fromkeys(args.subdirs)]
    for path in paths:
        if path != root and not path.parent.resolve().is_relative_to(root):
            raise ValueError(f'Cache area escapes the cache root: {path}')
    print('Clearing: ' + ', '.join(map(str, paths)))
    if not args.force and input('Are you sure? This action cannot be undone. (y/N): ').lower() != 'y':
        print('Cache clearing cancelled.')
        return 1
    for path in paths:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    print('Cache cleared successfully.')
    return 0
