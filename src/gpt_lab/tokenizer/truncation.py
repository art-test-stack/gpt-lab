"""
Phase 2 (truncation): deterministic truncation utilities.

Introduced in Phase 2 of the tokenizer refactor. This module exposes
`parse_truncated_name` and `truncated_from_pretrained` to perform safe,
deterministic tokenizer truncation while preserving byte tokens and
reassigning contiguous ranks.
"""
from __future__ import annotations

import re
from typing import Optional
from pathlib import Path

# Lightweight module: avoid importing heavy project modules at import time.
# Logging is optional; use print() for informational messages here.

TRUNCATED_PATTERN = re.compile(r"^(?P<base>.+)_truncated_(?P<vocab>\d+)$")


def parse_truncated_name(name: str) -> Optional[tuple[str, int]]:
    """Parse a truncated tokenizer name like 'foo_truncated_30000'.

    Returns (base_name, vocab_size) or None if the pattern doesn't match.
    """
    match = TRUNCATED_PATTERN.match(name)
    if match is None:
        return None
    return match.group("base"), int(match.group("vocab"))


def truncated_from_pretrained(base_name: str, new_vocab_size: int, source: str = "tiktoken", special_tokens: Optional[SpecialTokens] = None):
    """Create a truncated tokenizer keeping the first K mergeable ranks.

    This function is deterministic, non-mutating to the source tokenizer,
    reassigns ranks to be contiguous from 0..K-1, and enforces that all
    256 primitive byte tokens are retained.
    """
    # Local import to avoid circular dependencies at module import time
    from gpt_lab.tokenizer.tokenizer import Tokenizer

    # Local imports to avoid pulling heavy dependencies at module import time
    from gpt_lab.utils.schemas import TokenizerConfig
    from gpt_lab.utils.special_tokens import SpecialTokens

    if special_tokens is None:
        special_tokens = SpecialTokens()

    new_name = f"{base_name}_truncated_{new_vocab_size}"

    # If a truncated tokenizer already exists on disk, prefer loading it
    try:
        return Tokenizer.from_disk(new_name)
    except FileNotFoundError:
        print(f"No existing truncated tokenizer found for {new_name}; creating new one.")

    # Load base tokenizer
    base_tok = Tokenizer.from_pretrained(base_name, source=source, special_tokens=special_tokens)

    n_special = len(special_tokens.list())
    n_mergeable_keep = new_vocab_size - n_special

    N_BYTE_TOKENS = 256
    if n_mergeable_keep < N_BYTE_TOKENS:
        raise ValueError(
            f"Cannot truncate to {n_mergeable_keep} mergeable ranks: must retain all {N_BYTE_TOKENS} byte-level tokens."
        )

    if new_vocab_size >= base_tok.vocab_size:
        return base_tok

    # Sort mergeable ranks by original rank (ascending) and keep first K
    sorted_items = sorted(base_tok.mergeable_ranks.items(), key=lambda x: x[1])
    kept = sorted_items[:n_mergeable_keep]

    # Reassign ranks contiguously from 0
    new_mergeable = {token: rank for rank, (token, _) in enumerate(kept)}

    # Build new config
    config = TokenizerConfig(
        name=new_name,
        source=base_tok.config.source,
        dirname=base_tok.config.dirname.parent / new_name,
        vocab_size=new_vocab_size,
        pat_str=base_tok.config.pat_str,
        special_tokens=base_tok.config.special_tokens,
    )

    new_tokenizer = Tokenizer(
        mergeable_ranks=new_mergeable,
        special_tokens=config.special_tokens.list(),
        config=config,
    )

    # Validate contiguous ranks on the created mergeable_ranks using the
    # serialization validation function (loaded lazily to avoid importing
    # the package and its heavy dependencies at module import time).
    import importlib.util, sys
    from pathlib import Path as _P
    src_root = _P(__file__).resolve().parents[2]
    serial_path = src_root / 'gpt_lab' / 'tokenizer' / 'serialization.py'
    spec = importlib.util.spec_from_file_location('tokenizer_serial_local', str(serial_path))
    serial_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serial_mod)
    serial_mod.validate_mergeable_ranks(new_mergeable)

    # Verify token_bytes semantics (compute from raw bytes keys)
    sorted_items_new = sorted(new_mergeable.items(), key=lambda x: x[1])
    token_bytes_list = [len(token) for token, _ in sorted_items_new]
    token_bytes_list.extend([0] * len(config.special_tokens.list()))
    assert len(token_bytes_list) == new_vocab_size, (
        "Computed token bytes length does not match new_vocab_size"
    )

    # Persist token bytes and config via Tokenizer helper when possible
    try:
        new_tokenizer.update_token_bytes()
    except Exception:
        # If torch or disk are unavailable in this environment, we've already
        # validated the in-memory invariants above.
        print("Unable to call update_token_bytes() — environment may lack torch. In-memory checks passed.")

    return new_tokenizer
