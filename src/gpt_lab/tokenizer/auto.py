"""Automatic vocabulary sizing and tokenizer orchestration helpers.

The scaling-law calculation in this module is intentionally pure. Model
construction remains in :mod:`gpt_lab.model`, which owns the architecture and
can therefore supply the real hidden size and non-vocabulary parameter count.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from gpt_lab.tokenizer.base import _BaseTokenizer
from gpt_lab.tokenizer.tokenizer import Tokenizer, get_closest_tokenizer_size
from gpt_lab.utils.default import TOKENIZERS_FOLDER
from gpt_lab.utils.schemas import TokenizerConfig, TokenizerTrainerConfig
from gpt_lab.utils.special_tokens import SpecialTokens


# Exact Approach-1 coefficients from the reference implementation for
# Tao et al. (2024), "Scaling Laws with Vocabulary".
_K_NON_VOCAB = math.exp(-2.4846510161625193)
_K_VOCAB = math.exp(-1.589031299255507)
_ALPHA_NON_VOCAB = 0.5
_ALPHA_VOCAB = 0.4163622634135234


def compute_optimal_vocab_size(
    n_non_vocab_params: int,
    d_model: int,
) -> float:
    """Estimate the Approach-1 compute-optimal vocabulary size.

    This eliminates the compute budget from the fitted relationships

    ``N_nv = k_nv * C**alpha_nv`` and
    ``N_v = k_v * C**alpha_v``,

    then converts vocabulary parameters to vocabulary entries with
    ``V = N_v / d_model``.

    The estimate assumes that model parameters and training data are allocated
    compute-optimally. Selection of a cached tokenizer, integer rounding, and
    special-token handling deliberately happen outside this function.
    """
    if n_non_vocab_params <= 0:
        raise ValueError("n_non_vocab_params must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")

    gamma = _ALPHA_VOCAB / _ALPHA_NON_VOCAB
    optimal_vocab_params = _K_VOCAB * (
        n_non_vocab_params / _K_NON_VOCAB
    ) ** gamma
    return optimal_vocab_params / d_model


def round_vocab_size(
    vocab_size: float,
    *,
    multiple: int = 128,
    minimum: int = 256,
) -> int:
    """Round a vocabulary target to a supported integer model vocabulary."""
    if not math.isfinite(vocab_size) or vocab_size <= 0:
        raise ValueError("vocab_size must be a finite positive number")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if minimum <= 0:
        raise ValueError("minimum must be positive")

    rounded = int(math.floor(vocab_size / multiple + 0.5)) * multiple
    return max(rounded, minimum)


def resolve_tokenizer(
    name: Optional[str],
    vocab_size: int,
    special_tokens: SpecialTokens,
) -> str:
    """Resolve an explicit name or the closest tokenizer for a total size.

    ``Tokenizer.vocab_size`` includes special tokens, whereas the tokenizer
    size registry records mergeable ranks. The special-token count is removed
    before comparing the target with registry entries.
    """
    if name not in (None, "auto"):
        return name

    n_mergeable = vocab_size - len(special_tokens)
    if n_mergeable < 256:
        raise ValueError(
            "A byte-level tokenizer must retain at least 256 mergeable byte "
            f"tokens; got {n_mergeable}."
        )
    return get_closest_tokenizer_size(n_mergeable)[0]


def load_tokenizer(
    name: str,
    *,
    special_tokens: SpecialTokens,
    tokenizer_dir: str | Path = TOKENIZERS_FOLDER,
) -> _BaseTokenizer:
    """Load a local project tokenizer or a supported pretrained tokenizer.

    A present-but-invalid local tokenizer is treated as corruption and its
    error is allowed to propagate; only a genuinely absent local config falls
    through to pretrained source discovery.
    """
    tokenizer_dir = Path(tokenizer_dir)
    try:
        local_config = TokenizerConfig.from_directory(name, cachedir=tokenizer_dir)
    except FileNotFoundError:
        local_config = None

    if local_config is not None:
        if local_config.special_tokens != special_tokens:
            raise ValueError(
                f"Local tokenizer {name!r} uses different special tokens from "
                "the requested configuration."
            )
        return Tokenizer.from_disk(name, cachedir=tokenizer_dir)

    return Tokenizer.from_pretrained(name, special_tokens=special_tokens)


def train_new_tokenizer(
    *,
    name: str,
    vocab_size: int,
    pat_str: str,
    special_tokens: SpecialTokens,
    data_dir: str | Path,
    random_seed: int,
    tokenizer_dir: str | Path = TOKENIZERS_FOLDER,
    trainer_config: Optional[TokenizerTrainerConfig] = None,
) -> Tokenizer:
    """Train and optionally persist a tokenizer using a local tokenizer config."""
    from gpt_lab.tokenizer.corpus import TokenizerCorpus

    trainer_config = (
        trainer_config.model_copy(deep=True)
        if trainer_config is not None
        else TokenizerTrainerConfig()
    )
    if trainer_config.source != "huggingface":
        raise NotImplementedError(
            "Automatic tokenizer training currently supports only the "
            "Hugging Face trainer backend."
        )

    config = TokenizerConfig(
        name=name,
        dirname=tokenizer_dir,
        source="local",
        vocab_size=vocab_size,
        pat_str=pat_str,
        special_tokens=special_tokens,
        trainer=trainer_config,
    )

    corpus_kwargs = {
        "corpus_dir": data_dir,
        "random_seed": random_seed,
    }
    if trainer_config.max_bytes > 0:
        corpus_kwargs["max_bytes"] = trainer_config.max_bytes
    if trainer_config.bytes_per_doc > 0:
        corpus_kwargs["bytes_per_doc"] = trainer_config.bytes_per_doc

    corpus = TokenizerCorpus.from_sources(**corpus_kwargs)
    tokenizer = Tokenizer.train_from_iterator(
        text_iterator=corpus.iterator(),
        config=config,
    )
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"Trained tokenizer has vocabulary size {tokenizer.vocab_size}, "
            f"expected {vocab_size}."
        )
    return tokenizer


def build_or_load_tokenizer(
    name: Optional[str],
    vocab_size: int,
    train_tokenizer: bool,
    base_name: str,
    pat_str: str,
    special_tokens: SpecialTokens,
    data_dir: str | Path,
    random_seed: int,
    dirname: Optional[str | Path] = None,
    *,
    tokenizer_dir: Optional[str | Path] = None,
    trainer_config: Optional[TokenizerTrainerConfig] = None,
) -> _BaseTokenizer:
    """Compatibility dispatcher around the explicit load and train paths.

    ``dirname`` is the legacy name for the tokenizer cache directory.
    ``tokenizer_dir`` is preferred, and conflicting values are rejected.
    """
    if dirname is not None and tokenizer_dir is not None:
        if Path(dirname) != Path(tokenizer_dir):
            raise ValueError("dirname and tokenizer_dir refer to different paths")
    tokenizer_dir = tokenizer_dir or dirname or TOKENIZERS_FOLDER

    if train_tokenizer:
        if name not in (None, "auto"):
            raise ValueError(
                "tokenizer_model must be None or 'auto' when training a new "
                "tokenizer"
            )
        return train_new_tokenizer(
            name=base_name,
            vocab_size=vocab_size,
            pat_str=pat_str,
            special_tokens=special_tokens,
            data_dir=data_dir,
            random_seed=random_seed,
            tokenizer_dir=tokenizer_dir,
            trainer_config=trainer_config,
        )

    resolved_name = resolve_tokenizer(name, vocab_size, special_tokens)
    return load_tokenizer(
        resolved_name,
        special_tokens=special_tokens,
        tokenizer_dir=tokenizer_dir,
    )
