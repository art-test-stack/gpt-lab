"""Shared, standard-library-only parser helpers."""

import argparse

from gpt_lab.utils.paths import get_cache_dir as cache_dir


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)


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
