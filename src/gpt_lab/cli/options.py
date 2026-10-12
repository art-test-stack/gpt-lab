"""Shared, standard-library-only parser helpers."""

import argparse
import sys

from gpt_lab.utils.paths import get_cache_dir as cache_dir


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, default_subcommand=None, **kwargs):
        self.default_subcommand = default_subcommand
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args=None, namespace=None):
        if self.default_subcommand is not None:
            args = list(sys.argv[1:] if args is None else args)
            if not args or args[0].startswith("-"):
                args.insert(0, self.default_subcommand)
        return super().parse_known_args(args, namespace)


def checkpoint_step(value: str) -> int | str:
    if value in ("latest", "best"):
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer step, -N, latest, or best") from None


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number
