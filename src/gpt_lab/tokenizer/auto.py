"""
Phase 4 (auto): tokenizer orchestration helpers.

Introduced in Phase 4 of the tokenizer refactor. This module centralizes
tokenizer selection, optimal-vocab computations and build-or-load
orchestration so `AutoGPTConfig` can remain thin and model-config driven.

Note: This module may import from model config utilities (scaling laws
require architecture information). Model config MUST NOT import from
this module to avoid cycles; the dependency is one-way.
"""
from __future__ import annotations

import math
from typing import Optional

from gpt_lab.tokenizer.tokenizer import Tokenizer, get_closest_tokenizer_size
from gpt_lab.utils.special_tokens import SpecialTokens
from gpt_lab.utils.schemas import TokenizerConfig, TokenizerTrainerConfig
from gpt_lab.utils.logging import log0, log_error


def compute_optimal_vocab_size(depth: int, aspect_ratio: int, train_tokenizer: bool, tokenizer_model: Optional[str], special_tokens: SpecialTokens, get_closest=get_closest_tokenizer_size) -> int:
    """Compute optimal vocab size using the project's scaling-law approximation.

    Returns the vocabulary size including special tokens.
    """
    # Build a small helper to create the meta model for the depth
    def build_meta_model_from_depth(d: int, vocab_size: int = -1):
        # Import here to avoid circular dependency with gpt_lab.model.auto
        from gpt_lab.utils.schemas import TransformerConfig
        from gpt_lab.model.checkpoint import build_meta_model
        
        config = TransformerConfig(
            tf_type="dense",
            vocab_size=vocab_size,
            max_context=2048,
            d_model=(d * aspect_ratio),
            d_ffn=4 * (d * aspect_ratio),
            n_layers=d,
            n_heads=1,
            d_head=1,
        )
        return build_meta_model(config)

    assert (tokenizer_model is None) or (tokenizer_model == "auto") or (not train_tokenizer)

    if tokenizer_model not in (None, "auto"):
        tokenizer = Tokenizer.from_pretrained(tokenizer_model)
        return tokenizer.vocab_size

    _mmodel = build_meta_model_from_depth(depth, vocab_size=1)
    n_non_vocab_scaling_params = _mmodel.n_params
    power = 0.84
    coeff = .2 / (.08 ** power) / (depth * aspect_ratio)
    opt_vocab_size = coeff * (n_non_vocab_scaling_params ** power)
    del _mmodel
    log0(f"Number of non-vocabulary scaling parameters for depth {depth}: {n_non_vocab_scaling_params:.2e}")

    if not train_tokenizer:
        _, vocab_size = get_closest(opt_vocab_size)
    else:
        step = 10 ** (int(math.log10(opt_vocab_size)) - 1)
        vocab_size = round(opt_vocab_size / step) * step

    if vocab_size < 256:
        raise ValueError("Computed optimal vocab size is <256; increase model size or set vocab_size explicitly.")

    return int(vocab_size) + len(special_tokens.list())


def resolve_tokenizer(name: Optional[str], vocab_size: int, special_tokens: SpecialTokens) -> str:
    """Return a tokenizer name to use given an explicit name or a vocab size.

    If `name` is provided and not 'auto', return it. Otherwise choose the
    closest cached tokenizer name for `vocab_size`.
    """
    if name not in (None, "auto"):
        return name
    return get_closest_tokenizer_size(vocab_size)[0]


def build_or_load_tokenizer(tname: Optional[str], vocab_size: int, train_tokenizer: bool, base_name: str, pat_str: str, special_tokens: SpecialTokens, data_dir, random_seed: int, dirname=None):
    """Orchestrate loading or training of a tokenizer.

    - If `not train_tokenizer`, attempt to load a pretrained tokenizer.
    - Else, train a new tokenizer using the corpus and `TokenizerTrainerConfig`.
    Returns a `Tokenizer` instance.
    """
    if not train_tokenizer:
        name_or_choice = tname or resolve_tokenizer(tname, vocab_size, special_tokens)
        try:
            return Tokenizer.from_pretrained(name_or_choice)
        except Exception as e:
            log0(f"Error loading tokenizer {name_or_choice}: {e}", level="warning")
            # Try to construct from config/disk
            try:
                cfg = TokenizerConfig(name=name_or_choice, source="tiktoken", vocab_size=vocab_size, special_tokens=special_tokens, pat_str=pat_str)
                return Tokenizer.from_config(cfg)
            except Exception as e2:
                log0(f"Fallback to local tokenizer load failed: {e2}", level="warning")
                cfg2 = TokenizerConfig.from_directory(name_or_choice)
                mergeable = cfg2.get_mergeable_ranks()
                return Tokenizer(mergeable_ranks=mergeable, special_tokens=special_tokens.list(), config=cfg2)

    # Train a new tokenizer
    from gpt_lab.tokenizer.corpus import TokenizerCorpus
    
    _tname = base_name
    trainer_cfg = TokenizerTrainerConfig() # cfg should be adapted
    cfg = TokenizerConfig(name=_tname, source="huggingface", vocab_size=vocab_size, pat_str=pat_str, special_tokens=special_tokens, trainer=trainer_cfg)
    corpus = TokenizerCorpus.from_sources(corpus_dir=data_dir, max_bytes=vocab_size * 4 * 100, random_seed=random_seed)
    tokenizer = Tokenizer.train_from_iterator(text_iterator=corpus.iterator(), config=cfg)
    return tokenizer
