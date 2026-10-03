"""Shared, standard-library-only parser helpers."""

import argparse
import os
from pathlib import Path


def cache_dir() -> Path:
    """Compute the cache default without importing utils or probing the disk.

    Actual filesystem resolution belongs to command execution. In particular,
    Path.resolve() must not be used here because it probes symlinks.
    """
    value = os.environ.get("GPTLAB_CACHE_DIR")
    if value is not None:
        return Path(os.path.abspath(os.path.expanduser(value)))
    return Path.home() / ".cache" / "gpt_lab"


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
