from __future__ import annotations

import torch
import random, csv

import pickle
import warnings
import json as _json

from gpt_lab.tokenizer.base import _BaseTokenizer
from gpt_lab.tokenizer.serialization import (
    save_mergeable_ranks,
    load_mergeable_ranks,
    validate_mergeable_ranks,
    validate_no_special_token_overlap,
)
from pathlib import Path

from gpt_lab.utils.schemas import TokenizerConfig
from gpt_lab.utils.default import TOKENIZERS_FOLDER
from gpt_lab.utils.special_tokens import SpecialTokens
from gpt_lab.utils.logging import log0, log_error

import tiktoken
from gpt_lab.tokenizer.hf import HuggingFaceTokenizerWrapper, train_huggingface_from_iterator

from typing import Callable, Iterable, List, Optional, Union, Tuple, Dict
import logging

logger = logging.getLogger(__name__)
from gpt_lab.tokenizer.truncation import parse_truncated_name

# ------------------------------------------------------------
# FACTORY FUNCTION TO BUILD TOKENIZER FROM CONFIG
# ------------------------------------------------------------

def _get_tokenizer_sizes_in_cache() -> Dict[str, int]:
    """Get the vocab sizes of all tokenizers in the cache based on the provided tokenizer name."""
    tiktoken_encs = { 
        name: len(tiktoken.get_encoding(name)._mergeable_ranks) 
        for name in ("gpt2", "cl100k_base", "o200k_base") 
    }
    tokenizer_cache = TOKENIZERS_FOLDER / "tokenizers.csv"
    df_tok_cache = {}
    if tokenizer_cache.exists():
        # make a cache for this
        with open(tokenizer_cache, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name")
                vocab = row.get("vocab_size")
                if name and vocab:
                    df_tok_cache[name] = int(vocab) 
        
    tokenizer_sizes = {**tiktoken_encs, **df_tok_cache}
    return tokenizer_sizes

def get_higher_closest_tokenizer_size(vocab_size: int) -> Tuple[str, int]:
    """Get the next higher tokenizer size from the cache based on the provided vocab size."""
    tokenizer_sizes = _get_tokenizer_sizes_in_cache()
    higher_tokenizers = { name: size for name, size in tokenizer_sizes.items() if size >= vocab_size }
    if not higher_tokenizers:
        raise ValueError(f"No tokenizer found with vocab size ≥ {vocab_size}.")
    tok_name, closest_size = min(
        higher_tokenizers.items(), 
        key=lambda x: x[1]
    )
    return tok_name, closest_size

def get_closest_tokenizer_size(vocab_size: int) -> Tuple[str, int]:
    """Get the closest tokenizer size from the cache based on the provided vocab size."""
    # TODO: i removed tokenizer cache manager in a previous commit
    # have to make it back in order to make this works with new tokenizers
    tokenizer_sizes = _get_tokenizer_sizes_in_cache()
    tok_name, closest_size = min(
        tokenizer_sizes.items(), 
        key=lambda x: abs(x[1] - vocab_size)
    )
    return tok_name, closest_size
     
def build_tokenizer(config: TokenizerConfig) -> Callable:
    return Tokenizer.from_config(config)

# ------------------------------------------------------------
# DUMMY TOKENIZER INSTANCE (for quick tests/dev)
# ------------------------------------------------------------

class DummyTokenizer(_BaseTokenizer):
    def __init__(self, config: Optional[TokenizerConfig] = None):
        if config is None:
            config = TokenizerConfig(
                name="dummy",
                source="dummy",
                vocab_size=256,
                pat_str=None,
                special_tokens=SpecialTokens()
            )
        super().__init__(config)
        n_merges = config.vocab_size - len(config.special_tokens)
        self.mergeable_ranks = { bytes([i]): i for i in range(min(256, n_merges)) }
        
        if n_merges > 256:
            for i in range(256, n_merges): # make merges deterministic at least lol
                merge1 = bytes([i // 256])
                merge2 = bytes([i % 256])
                self.mergeable_ranks[merge1+merge2] = i

        self.special_tokens = config.special_tokens
        self.bos_token_id = config.vocab_size 
    
    def encode(self, text, *args, **kwargs):
        assert isinstance(text, str), "Input text must be a string"
        
        length_encoded = random.randint(1, len(text) - 1)
        token_ids = [random.randint(0, self.vocab_size - 1) for _ in range(length_encoded)]
        return token_ids

    def decode(self, tokens, *args, **kwargs):
        return "".join([chr(t) for t in tokens])

# ------------------------------------------------------------
# MAIN TOKENIZER CLASS 
#   - train with huggingface, 
#   - encode & decode with tiktoken
#   - falldown to HFTokenizer if pretrained mode = huggingface
# ------------------------------------------------------------

class Tokenizer(_BaseTokenizer):
    """Wrapper class for different tokenizer implementations 
    ## Use cases include:
        - Encoding with Tiktoken API (faster)
        - Loading TikToken tokenizer
        - Loading a custom trained BPE tokenizer from local directory (must have merges + pattern + special tokens)
        - Loading a pretrained tokenizer from HuggingFace Hub (slower, use HuggingFace api for encoding/decoding)
        - Training HuggingFace tokenizer from corpus and convert it in Tiktoken implementation

    Args:
        mergeable_ranks (dict[bytes, int]): The mergeable ranks for the tokenizer
        special_tokens (list[str]): The list of special tokens for the tokenizer
        config (gpt_lab.utils.schemas.TokenizerConfig): Configuration settings for the tokenizer
    """
    def __init__(
            self, 
            # enc: Callable, 
            mergeable_ranks: dict[bytes, int],
            special_tokens: list[str],
            config: TokenizerConfig
        ):
        super().__init__(config=config)
        special_tokens = { sp: rank + len(mergeable_ranks) for rank, sp in enumerate(special_tokens) }
        self.mergeable_ranks = mergeable_ranks
        self.main = tiktoken.Encoding(
            name=config.name,
            pat_str=config.pat_str,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int]
            special_tokens=special_tokens, # Only add special tokens to the encoding, not to the mergeable ranks, to avoid conflicts
            explicit_n_vocab=config.vocab_size
        )
        self.special_tokens = special_tokens
        self.config = config
        self.bos_token_id = self.encode_special(config.special_tokens.bos)

    @classmethod
    def from_pretrained(cls, name: str, source: Optional[str] = None, special_tokens: Optional[SpecialTokens] = None):
        if special_tokens is None:
            special_tokens = SpecialTokens()
        # Phase 3: handle truncated names early to allow creating or loading
        # pre-truncated tokenizers. This must run before any source-dispatch logic.
        truncated = parse_truncated_name(name)
        if truncated is not None:
            base_name, vocab_size = truncated
            try:
                return cls.from_disk(name)
            except FileNotFoundError:
                # Expected: not cached yet. Build from base tokenizer.
                return cls.truncated_from_pretrained(
                    base_name,
                    vocab_size,
                    source=source or "tiktoken",
                    special_tokens=special_tokens,
                )
            # Do not catch other exceptions here; allow them to propagate.
        if source is None:
            # Build expected-missing exception tuple dynamically, adding any
            # tiktoken-specific exception if it can be discovered at runtime.
            EXPECTED_MISSING = (FileNotFoundError, KeyError)
            try:
                import tiktoken
                try:
                    # probe tiktoken for its missing-encoding exception type
                    tiktoken.get_encoding("__NONEXISTENT_ENCODING__")
                except Exception as e:
                    EXPECTED_MISSING = tuple(set(EXPECTED_MISSING) | {type(e)})
            except Exception:
                # tiktoken not available in this environment; proceed with defaults
                pass

            for source in ("tiktoken", "huggingface", "local"):
                try:
                    return cls.from_pretrained(name, source=source, special_tokens=special_tokens)
                except EXPECTED_MISSING as e:
                    logger.debug(f"Source {source!r} not applicable for {name!r}: {e}")
                    continue
                except Exception:
                    logger.debug(
                        f"Unexpected error loading {name!r} from {source!r}:",
                        exc_info=True,
                    )
                    raise

            raise ValueError(f"Failed to load tokenizer {name!r} from all sources.")
        elif source == "tiktoken":
            enc = tiktoken.get_encoding(name)
            mergeable_ranks = enc._mergeable_ranks
            pat_str = enc._pat_str
            config = TokenizerConfig(
                name=name,
                source=source,
                vocab_size=len(mergeable_ranks) + len(special_tokens),
                pat_str=pat_str,
                special_tokens=special_tokens
            )
            return cls(mergeable_ranks, special_tokens.list(), config)
        elif source == "huggingface":
            return HuggingFaceTokenizerWrapper.from_pretrained(name)
        elif source == "local":
            return HuggingFaceTokenizerWrapper.from_directory(name)

    @classmethod
    def from_config(cls, config: TokenizerConfig):
        """Load a pretrained tokenizer from tiktoken/HuggingFace Hub/local directory based on the provided configuration."""
        if config.source == "tiktoken":
            enc = tiktoken.get_encoding(config.name)
            mergeable_ranks = enc._mergeable_ranks
            special_tokens = config.special_tokens.list()
            pat_str = enc._pat_str
        elif config.source == "huggingface":
            # TODO: convert HuggingFace tokenizer to tiktoken encoding, 
            # -> extracting merges and vocab from the HuggingFace tokenizer 
            # -> creating a new tiktoken encoding with those merges and vocab
            # -> need to handle special tokens 
            # -> need to extract hgf pretokenizer + string pattern
            return HuggingFaceTokenizerWrapper.from_pretrained(config.name)
        elif config.source == "local":
            mergeable_ranks = config.get_mergeable_ranks()
            special_tokens = config.special_tokens.list()
            pat_str = config.pat_str
        elif config.source == "dummy":
            log0("Using DummyTokenizer, this is not a real tokenizer and should only be used for testing purposes.", level="warning", logger=logger)
            return DummyTokenizer(config)
        else:
            raise ValueError(f"Unsupported tokenizer source: {config.source}")
        
        config.vocab_size = len(mergeable_ranks) + len(special_tokens)
        config.pat_str = pat_str
        return cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            config=config
        )

    def get_bos_token_id(self):
        return self.bos_token_id
    
    @classmethod
    def train_from_iterator(
            cls,
            text_iterator: Iterable[str],
            config: TokenizerConfig,
        ):
        special_tokens = config.special_tokens.list()
        vocab_size_no_special = config.vocab_size - len(special_tokens)
        # TODO: make the other tokenizers for comparison; lines +1 and +2 below are temporary
        tp_trainer = config.trainer 
        if tp_trainer is None:
            log_error("TokenizerConfig.trainer is not set. Please set the trainer explicitly in TokenizerConfig.", logger=logger, error_type=UserWarning)
        if tp_trainer.source != "huggingface":
            msg = f"Training tokenizer with trainer {tp_trainer.source!r} is not implemented yet. Please use 'huggingface' trainer for now."
            log_error(msg, error_type=NotImplementedError, logger=logger)
        # TODO: make pretokenizer here -> options: 1. gpt2, 2. custom
        if tp_trainer.source == "tiktoken":
            from tiktoken._educational import bpe_train
            log0("Training tokenizer with tiktoken is a TODO for future improvement.", level="warning", logger=logger)
            # TODO: WIP, not tested yet
            mergeable_ranks = bpe_train(data=text_iterator, vocab_size=vocab_size_no_special, pat_str=config.pat_str)
        elif tp_trainer.source == "huggingface":
            # Delegate HuggingFace training logic to tokenizer.hf module
            mergeable_ranks = train_huggingface_from_iterator(text_iterator, config)

        # TODO: add other trainer options (bpe, rust bpe, fast bpe...)
        # The following options are placeholders for future impl.
        elif tp_trainer.source in ["bpe", "fbpe", "rbpe"]:
            raise NotImplementedError(f"Tokenizer training mode {tp_trainer.source!r} is not yet implemented. Please use 'huggingface' mode.")
        elif tp_trainer.source == "bpe":
            # naive python implementation of byte-level BPE, not optimized for large corpora, but serves as a reference
            from gpt_lab.tokenizer.bpe import bpe
            _, mergeable_ranks = bpe()
        elif tp_trainer.source == "fbpe":
            from gpt_lab.tokenizer.bpe import bpe_fast
            trainer = ...
        elif tp_trainer.source == "rbpe":
            from rbpe import bpe
            ...
        elif tp_trainer.source == "dummy":
            log0("Using DummyTokenizer for training, this is not a real tokenizer and should only be used for testing purposes.", level="warning", logger=logger)
            return cls(DummyTokenizer(config), config)
        else:
            msg = f"Tokenizer trainer {tp_trainer.source!r} is not supported."
            log_error(msg, error_type=NotImplementedError, logger=logger)
        tokenizer = cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            config=config
        )
        to_save_flag = tp_trainer.to_save if tp_trainer is not None else getattr(config, "to_save", False)
        if to_save_flag:
            tokenizer.save_to_directory()
        return tokenizer

    @classmethod
    def from_disk(cls, name: str, cachedir: Optional[Union[str, Path]] = None):
        if cachedir is None:
            cachedir = TOKENIZERS_FOLDER
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)
        # dirname = cachedir / name
        config = TokenizerConfig.from_directory(name, cachedir=cachedir)

        directory = Path(config.dirname)
        msgpack_path = directory / "mergeable_ranks.msgpack"
        pkl_path = directory / "vocab.pkl"

        if msgpack_path.exists():
            mergeable_ranks = load_mergeable_ranks(msgpack_path)
        elif pkl_path.exists():
            warnings.warn(
                f"Loading tokenizer from legacy pickle format at {pkl_path}. "
                "Re-save this tokenizer to migrate to the msgpack format.",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(pkl_path, "rb") as f:
                mergeable_ranks = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"No tokenizer vocab file found in {directory}. Expected {msgpack_path} or {pkl_path}."
            )

        # Validate mergeable ranks and ensure no overlap with special tokens
        validate_mergeable_ranks(mergeable_ranks)
        special_tokens_map = {tok: 0 for tok in config.special_tokens.list()}
        validate_no_special_token_overlap(mergeable_ranks, special_tokens_map)

        log0(
            f"Loaded tokenizer config from {name} with vocab size "
            f"{len(mergeable_ranks) + len(config.special_tokens)}",
            logger=logger,
        )
        return cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=config.special_tokens.list(),
            config=config
        )
    
    @classmethod
    def truncated_from_pretrained(cls, name: str, new_vocab_size: int, source: str = "tiktoken", special_tokens: Optional[SpecialTokens] = None) -> Tokenizer:
        """Delegate truncation to tokenizer.truncation.truncated_from_pretrained (Phase 2).

        Signature preserved for backward compatibility.
        """
        if special_tokens is None:
            special_tokens = SpecialTokens()
        from gpt_lab.tokenizer.truncation import truncated_from_pretrained as _trunc

        return _trunc(name, new_vocab_size, source=source, special_tokens=special_tokens)
    
    @classmethod
    def get_closest_truncated_from_pretrained(cls, tokenizer: Tokenizer, target_vocab_size: int) -> Tokenizer:
        # Find the closest vocab size that is less than or equal to the target
        closest_vocab_size = min(
            vocab_size for vocab_size in tokenizer.config.vocab_sizes
            if vocab_size <= target_vocab_size
        )
        return cls.truncated_from_pretrained(tokenizer.config.name, closest_vocab_size)
    
    def update_token_bytes(self):
        # Recompute token_bytes directly from mergeable_ranks keys to avoid decode()/string roundtrip
        if not hasattr(self, "mergeable_ranks") or self.mergeable_ranks is None:
            raise RuntimeError("mergeable_ranks is missing when updating token bytes")

        sorted_items = sorted(self.mergeable_ranks.items(), key=lambda x: x[1])
        token_bytes_list = [len(token) for token, _ in sorted_items]
        token_bytes_list.extend([0] * len(self.special_tokens))
        new_token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device="cpu")

        old_vocab_size = getattr(self, "_token_bytes", torch.tensor([])).numel() if hasattr(self, "_token_bytes") else 0
        self._token_bytes = new_token_bytes
        log0(f"Updated token bytes after truncation from {old_vocab_size:,} to {self.vocab_size:,}", logger=logger)
        # Save token_bytes to disk
        token_bytes_path = Path(self.config.dirname) / "token_bytes.pt"
        torch.save(self.token_bytes, token_bytes_path)
        # Persist tokenizer config/metadata
        try:
            self.config.save_to_directory()
        except Exception:
            log0("Warning: failed to save config.pkl after truncation; token_bytes saved.", level="warning", logger=logger)

    def save_to_directory(self, directory: Optional[Union[str, Path]] = None):
        # Save the tokenizer's merges and vocab to the specified directory
        if directory is None:
            directory = self.config.dirname
        if isinstance(directory, str):
            directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # We volontary confuse the vocab and merges terminology here, 
        # since in our implementation the mergeable ranks dict contains 
        # all the tokens (single byte + merged tokens) and their corresponding ids, 
        # which is essentially the vocab of the tokenizer. 
        # We don't have a separate merges dict since the mergeable ranks already 
        # encodes the merges in the order they were added during training.
        # Persist mergeable ranks in deterministic msgpack format and compute fingerprint
        msgpack_path = directory / "mergeable_ranks.msgpack"
        fingerprint = save_mergeable_ranks(msgpack_path, self.mergeable_ranks)

        # Save token bytes tensor
        token_bytes_path = directory / "token_bytes.pt"
        torch.save(self.token_bytes, token_bytes_path)
        
        # Write a lightweight JSON descriptor alongside the pickle config for readability
        config_json = {
            "name": self.config.name,
            "vocab_size": self.config.vocab_size,
            "pat_str": self.config.pat_str,
            "mergeable_ranks_sha256": fingerprint,
            "source": self.config.source,
        }
        json_path = directory / "tokenizer_config.json"
        with open(json_path, "w") as jf:
            _json.dump(config_json, jf, indent=2)

        # Keep legacy pickle-based config for backwards compatibility
        try:
            self.config.save_to_directory()
        except Exception:
            # If saving the pickle-config fails, still keep msgpack and json
            log0("Warning: failed to save config.pkl; msgpack and json were written.", level="warning", logger=logger)

        log0(f"Saved tokenizer mergeable ranks to {msgpack_path}", logger=logger)

    def encode_special(self, token: str) -> int:
        return self.special_tokens[token]
        
    def encode(
            self, 
            text: Union[str, List[str]],
            *args, **kwargs
        ) -> Union[List[int], List[List[int]]]:
        # NOTE: maybe it would be better to unfused both str and list encoding into separate methods to avoid confusion 
        # and potential bugs with the different options (e.g. prepend_bos, unsqueeze) that may not be compatible with both modes?
        prepend_bos = kwargs.pop("prepend_bos", False)
        unsqueeze = kwargs.pop("unsqueeze", False)
        if isinstance(text, str):
            token_ids = self.main.encode_ordinary(text)
            if prepend_bos:
                token_ids = [self.bos_token_id] + token_ids
            if unsqueeze:
                token_ids = [token_ids]
        elif isinstance(text, list):
            num_threads = kwargs.get("num_threads", 8)
            token_ids = self.main.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_bos:
                token_ids = [[self.bos_token_id] + seq for seq in token_ids]
            if unsqueeze:
                log0(f"Unsqueeze option is not typically used for batch encoding, as it adds an extra dimension that may not be necessary. Use with caution. Encoder input is already a batch of {len(text)} sequences.", level="warning", logger=logger)
        else:
            text_type = f"List[{type(text[0])}]" if isinstance(text, list) else text_type
            msg = f"Tokenizer.encode expected 'str' or 'List[str]', got {text_type!r}."
            log_error(msg, error_type=TypeError, logger=logger)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.main.decode(token_ids)
    
    def decode_single_token_bytes(self, token_id: int) -> bytes:
        if token_id in self.special_tokens.values():
            # Special tokens are not part of the mergeable ranks and do not have corresponding byte sequences
            log0(f"Token ID {token_id} is a special token and does not have a corresponding byte sequence. Returning empty bytes.", level="warning", logger=logger)
            return b""
        return self.main.decode_single_token_bytes(token_id)
    
    def __call__(self, text, *args, **kwds):
        return self.encode(text, *args, **kwds)