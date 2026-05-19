from __future__ import annotations

import os, json
from pathlib import Path
from typing import Iterable, Dict

from gpt_lab.utils.logging import log_all, log0, log_error
from gpt_lab.utils.schemas import TokenizerConfig, TokenizerTrainerConfig
from gpt_lab.utils.special_tokens import SpecialTokens
from gpt_lab.tokenizer.base import _BaseTokenizer

import logging

logger = logging.getLogger(__name__)

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, pre_tokenizers, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
except Exception as e:
    log0(f"Failed to import HuggingFace tokenizers library: {e}. " \
         "HuggingFace tokenizer functionality will be unavailable. " \
         "To use HuggingFace tokenizers, please install the 'tokenizers' library via pip.", 
         level="warning", logger=logger)
    HFTokenizer = None

class HuggingFaceTokenizerWrapper(_BaseTokenizer):
    """Light wrapper around HuggingFace `tokenizers` tokenizer.

    Provides a compatible subset of the previous wrapper used by
    Tokenizer.from_pretrained and Tokenizer.from_directory.
    """
    def __init__(self, tokenizer, config: TokenizerConfig):
        self.main = tokenizer
        self.config = config

    @property
    def special_tokens(self):
        special_tokens_map = self.main.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    @classmethod
    def from_pretrained(cls, hf_path: str):
        if HFTokenizer is None:
            log_error("tokenizers library is required to load HuggingFace tokenizer", logger=logger, error_type=ImportError)
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        config = TokenizerConfig(
            name=hf_path,
            source="huggingface",
            vocab_size=tokenizer.get_vocab_size(),
            pat_str=None,
            special_tokens=SpecialTokens(),
        )
        return cls(tokenizer, config=config)

    @classmethod
    def from_directory(cls, tokenizer_dir: str):
        if HFTokenizer is None:
            log_error("tokenizers library is required to load HuggingFace tokenizer", logger=logger, error_type=ImportError)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        config = TokenizerConfig(
            name=tokenizer_dir,
            source="local",
            vocab_size=tokenizer.get_vocab_size(),
            pat_str=None,
            special_tokens=SpecialTokens(),
        )
        return cls(tokenizer, config=config)

    def id_to_token(self, id):
        return self.main.id_to_token(id)

    def encode(self, text, add_special_tokens=False):
        return self.main.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self.main.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.main.save(tokenizer_path)


def train_huggingface_from_iterator(text_iterator: Iterable[str], config: TokenizerConfig) -> Dict[bytes, int]:
    """Train a HuggingFace BPE tokenizer and return mergeable_ranks mapping.

    Returns a dict mapping byte-strings to ranks (integers).
    """
    tr_config = getattr(config, "trainer", None)
    if tr_config is None:
        log_error("TokenizerConfig must have a 'trainer' attribute with training parameters for HuggingFace tokenizer training.", logger=logger, error_type=ValueError)
    if HFTokenizer is None:
        log_error("tokenizers library is required for HuggingFace trainer", logger=logger, error_type=ImportError)
    
    tknzr = HFTokenizer(
        BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False
        )
    )
    tknzr.normalizer = None
    pattern = Regex(config.pat_str)
    tknzr.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=pattern, behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tknzr.decoder = decoders.ByteLevel()
    tknzr.post_processor = None
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()

    # Prefer training-specific params container when available
    vocab_size_no_special = config.vocab_size - len(config.special_tokens.list())
    trainer = BpeTrainer(
        vocab_size=vocab_size_no_special,
        show_progress=True,
        min_frequency=0,
        initial_alphabet=initial_alphabet,
        special_tokens=[],
    )
    trainer.show_progress = tr_config.show_progress 
    tknzr.train_from_iterator(iterator=text_iterator, trainer=trainer)

    merges = json.loads(tknzr.to_str())["model"]["merges"]

    def merge_to_bytes(merge):
        left, right = merge
        left = left.replace("Ġ", " ")
        right = right.replace("Ġ", " ")
        return left.encode("utf-8") + right.encode("utf-8")

    mergeable_ranks = {merge_to_bytes(merge): rank + 256 for rank, merge in enumerate(merges)}
    # Add single-byte tokens
    mergeable_ranks.update({bytes([i]): i for i in range(256) if bytes([i]) not in mergeable_ranks})
    return mergeable_ranks
