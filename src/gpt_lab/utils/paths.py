"""Shared cache paths; importing this module does not create directories."""

import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Read the cache root without probing the filesystem or resolving symlinks."""
    value = os.environ.get("GPTLAB_CACHE_DIR")
    if value is not None:
        return Path(os.path.abspath(os.path.expanduser(value)))
    return Path.home() / ".cache" / "gpt_lab"


CACHE_DIR = get_cache_dir()
BOARD_DIR = CACHE_DIR / "runs"
DATA_DIR = CACHE_DIR / "data"
OUTPUT_FOLDER = CACHE_DIR / "output"
MODELS_FOLDER = CACHE_DIR / "models"
TOKENIZERS_FOLDER = CACHE_DIR / "tokenizers"
