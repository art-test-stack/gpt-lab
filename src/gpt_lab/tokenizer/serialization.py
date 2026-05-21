"""
Phase 1 (serialization): deterministic msgpack-based tokenizer persistence.

Introduced in Phase 1 of the tokenizer refactor. This module provides
functions to save/load/validate `mergeable_ranks` in a deterministic,
portable, and fingerprinted format using `msgpack` and `sha256`.
"""
from __future__ import annotations

import msgpack
from hashlib import sha256
from pathlib import Path
from typing import Dict

def save_mergeable_ranks(path: Path, mergeable_ranks: Dict[bytes, int]) -> str:
    """Save mergeable_ranks deterministically (rank-sorted) to `path`.

    Returns the sha256 hex fingerprint of the written payload.
    """
    sorted_items = sorted(mergeable_ranks.items(), key=lambda x: x[1])
    # ensure insertion order is by rank
    sorted_dict = {k: v for k, v in sorted_items}

    payload = msgpack.packb({"version": 1, "mergeable_ranks": sorted_dict}, use_bin_type=True)
    fingerprint = sha256(payload).hexdigest()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)

    return fingerprint


def load_mergeable_ranks(path: Path) -> Dict[bytes, int]:
    """Load mergeable_ranks from a msgpack file written by `save_mergeable_ranks`.

    Raises ValueError on unsupported version or malformed payload.
    """
    raw = path.read_bytes()
    data = msgpack.unpackb(raw, raw=False)

    version = data.get("version")
    if not isinstance(version, int):
        raise ValueError("Tokenizer file 'version' must be an integer.")
    if version != 1:
        raise ValueError(f"Unsupported tokenizer file version: {version}. Expected 1.")

    raw_map = data.get("mergeable_ranks")
    if raw_map is None:
        raise ValueError("Missing 'mergeable_ranks' in tokenizer file.")

    # When unpacked with raw=False keys are bytes
    mergeable_ranks: Dict[bytes, int] = {k: v for k, v in raw_map.items()}
    return mergeable_ranks


def validate_mergeable_ranks(mergeable_ranks: Dict[bytes, int]) -> None:
    """Validate rank semantics (non-empty, byte keys, integer contiguous ranks).

    Raises AssertionError with informative messages on failure.
    """
    ranks = list(mergeable_ranks.values())

    assert len(ranks) > 0, "Tokenizer cannot have empty mergeable_ranks."
    assert all(isinstance(k, (bytes, bytearray)) and len(k) > 0 for k in mergeable_ranks.keys()), (
        "All mergeable rank keys must be non-empty bytes."
    )
    assert all(isinstance(r, int) for r in ranks), "All mergeable rank values must be integers."
    assert all(0 <= r < 2 ** 31 for r in ranks), "All ranks must satisfy 0 <= rank < 2**31."
    assert len(set(ranks)) == len(ranks), "Duplicate ranks in mergeable_ranks."
    assert min(ranks) == 0, "Ranks do not start at 0."
    assert max(ranks) == len(ranks) - 1, "Ranks are not contiguous."


def validate_no_special_token_overlap(mergeable_ranks: Dict[bytes, int], special_tokens: Dict[str, int]) -> None:
    """Ensure no special token (string) collides with a mergeable rank key (bytes).

    Raises AssertionError if overlap detected.
    """
    mergeable_set = set(mergeable_ranks.keys())
    special_token_bytes = {t.encode("utf-8") for t in special_tokens.keys()}
    overlap = mergeable_set & special_token_bytes
    assert not overlap, (
        f"Special tokens overlap with mergeable ranks: {[b.decode('utf-8', errors='replace') for b in overlap]}"
    )
