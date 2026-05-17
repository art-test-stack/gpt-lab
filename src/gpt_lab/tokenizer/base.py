from gpt_lab.utils.schemas import TokenizerConfig
from gpt_lab.utils.logging import log0
from pathlib import Path
from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)
# ------------------------------------------------------------
# BASE TOKENIZER INTERFACE (for consistency)
# ------------------------------------------------------------

class _BaseTokenizer:
    """Base tokenizer class defining the common interface for all tokenizers.
    
    This class is not meant to be used directly, but rather to be inherited by specific tokenizer implementations.
    
    Should implement the following methods:
    - encode: to convert text to token ids
    - decode: to convert token ids back to text
    - encode_special: to encode special tokens to their corresponding ids"""
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config
        self.special_tokens = None
        self.mergeable_ranks = None
        try:
            self.token_bytes = self.get_token_bytes()
        except Exception as e:
            log0(f"Failed to get token bytes during initialization: {e}. " \
                  f"This may cause issues with optimizers that rely on token byte lengths. "\
                "You can try calling get_token_bytes() manually after initialization to see the full error message and debug the issue.", 
                level="warning", logger=logger)

    def get_vocab(self):
        return {**self.mergeable_ranks, **self.special_tokens}
    
    @property
    def vocab_size(self):
        "vocab_size value icludes both mergeable ranks and special tokens"
        return len(self.mergeable_ranks) + len(self.special_tokens)
    
    @property
    def n_special_tokens(self):
        return len(self.special_tokens)
    
    @property
    def n_ranks(self):
        return len(self.mergeable_ranks)
         
    def get_token_bytes(self):
        token_bytes_path = Path(self.config.dirname) / "token_bytes.pt"
        if getattr(self, "token_bytes", None) is not None:
            return self.token_bytes

        if token_bytes_path.exists():
            token_bytes = torch.load(token_bytes_path)
            log0(f"Loaded token_bytes from {token_bytes_path}", logger=logger)
        else:
            # Compute byte lengths directly from mergeable_ranks keys (which are bytes)
            mergeable = self.mergeable_ranks or {}
            # Sort by rank to produce deterministic ordering
            sorted_items = sorted(mergeable.items(), key=lambda x: x[1])
            token_bytes_list = [len(token) for token, _ in sorted_items]
            # Special tokens are always zero-length for token_bytes
            token_bytes_list.extend([0] * len(self.special_tokens))
            token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device="cpu")
            with open(token_bytes_path, "wb") as f:
                torch.save(token_bytes, f)
            log0(f"Saved token_bytes to {token_bytes_path}", logger=logger)

        self.token_bytes = token_bytes
        return token_bytes

    def __call__(self, text, *args, **kwds):
        return self.encode(text, *args, **kwds)