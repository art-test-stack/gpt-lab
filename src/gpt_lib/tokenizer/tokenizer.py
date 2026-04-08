from gpt_lib.utils.schemas import TokenizerConfig, TokenizerTrainerConfig
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
import torch
from typing import Callable, Iterable, List, Optional, Union, Tuple
import pickle
from pathlib import Path
import random, warnings, json, os
from gpt_lib.utils.default import TOKENIZERS_FOLDER
from gpt_lib.utils.special_tokens import SpecialTokens

from tokenizers import Tokenizer as HFTokenizer


# ------------------------------------------------------------
# FACTORY FUNCTION TO BUILD TOKENIZER FROM CONFIG
# ------------------------------------------------------------
    
def get_closest_tokenizer_size(vocab_size: int) -> Tuple[str, int]:
    """Get the closest tokenizer size from the cache based on the provided vocab size."""
    tiktoken_encs = { name: len(tiktoken.get_encoding(name)._mergeable_ranks) for name in ("gpt2", "cl100k_base", "o200k_base") }
    tokenizer_cache = TOKENIZERS_FOLDER / "tokenizers.csv"
    if tokenizer_cache.exists():
        import pandas as pd
        df_tok_cache = pd.read_csv(tokenizer_cache)[["name", "vocab_size"]].drop_duplicates()
        df_tok_cache = {row["name"]: row["vocab_size"] for _, row in df_tok_cache.iterrows()}
    else:
        df_tok_cache = {}
    tokenizer_sizes = {**tiktoken_encs, **df_tok_cache}
    tok_name, closest_size = min(tokenizer_sizes.items(), key=lambda x: abs(x[1] - vocab_size))

    return tok_name, closest_size
     
def build_tokenizer(config: TokenizerConfig) -> Callable:
    return Tokenizer.from_pretrained(config)

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
        self.token_to_id = None

    def get_vocab(self):
        return {**self.token_to_id, **self.special_tokens}
    
    @property
    def vocab_size(self):
        return len(self.token_to_id) + len(self.special_tokens)
    
    @property       
    def token_bytes(self):
        token_bytes_path = Path(self.config.dirname) / "token_bytes.pt"
        if self.token_bytes_cache is not None:
            return self.token_bytes_cache
        
        if token_bytes_path.exists():
            token_bytes = torch.load(token_bytes_path)
            print(f"Loaded token_bytes from {token_bytes_path}")
        else:
            vocab_size = self.vocab_size
            special_set = set(self.special_tokens)
            token_strings = [self.decode([token_id]) for token_id in range(vocab_size)]
            token_bytes = []
            for token_id in range(vocab_size):
                token_str = token_strings[token_id] # the Python string representation of this token
                if token_str in special_set:
                    token_bytes.append(0) # special characters are not counted
                else:
                    id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
                    token_bytes.append(id_bytes)
            token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
            with open(token_bytes_path, "wb") as f:
                torch.save(token_bytes, f)
            print(f"Saved token_bytes to {token_bytes_path}")
        self.token_bytes_cache = token_bytes
        return token_bytes

    def __call__(self, text, *args, **kwds):
        return self.encode(text, *args, **kwds)
    
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
        self.token_to_id = { bytes([i]): i for i in range(min(256, n_merges)) }
        
        if n_merges > 256:
            for i in range(256, n_merges):
                merge1 = bytes([i // 256])
                merge2 = bytes([i % 256])
                self.token_to_id[merge1+merge2] = i

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
# HUGGINGFACE TOKENIZER WRAPPER (for some utilities)
# ------------------------------------------------------------


class HuggingFaceTokenizerWrapper(_BaseTokenizer):
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer: HFTokenizer, config: TokenizerConfig):
        super().__init__(config)
        self.main = tokenizer
    
    @property
    def special_tokens(self):
        special_tokens_map = self.main.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        config = TokenizerConfig(
            name=hf_path,
            source="huggingface",
            vocab_size=tokenizer.get_vocab_size(),
            pat_str=None, # TODO: extract pattern from HuggingFace tokenizer if possible, otherwise use a default one
            special_tokens=tokenizer.get_added_tokens_decoder()
        )
        return cls(tokenizer, config=config)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        config = TokenizerConfig(
            name=tokenizer_dir,
            source="local",
            vocab_size=tokenizer.get_vocab_size(),
            pat_str=None,
            special_tokens=SpecialTokens(), # tokenizer.get_added_tokens_decoder()
        )
        return cls(tokenizer, config=config)
    
    def id_to_token(self, id):
        return self.main.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.main.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.main.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode_ordinary(self, text, *args, **kwargs):
        # encode a single string without adding special tokens
        return self._encode_one(text, *args, **kwargs)
    
    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.main.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.main.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# ------------------------------------------------------------
# MAIN TOKENIZER CLASS 
#   - train with huggingface, 
#   - encode & decode with tiktoken
#   - falldown to HFTokenizer if pretrained mode = huggingface
# ------------------------------------------------------------

class Tokenizer(_BaseTokenizer):
    """ Wrapper class for different tokenizer implementations 
    ## Use cases include:
        - Encoding with Tiktoken API (faster)
        - Loading TikToken tokenizer
        - Loading a custom trained BPE tokenizer from local directory (must have merges + pattern + special tokens)
        - Loading a pretrained tokenizer from HuggingFace Hub (slower, use HuggingFace api for encoding/decoding)
        - Training HuggingFace tokenizer from corpus and convert it in Tiktoken implementation

    Args:
        mergeable_ranks (dict[bytes, int]): The mergeable ranks for the tokenizer
        special_tokens (list[str]): The list of special tokens for the tokenizer
        config (gpt_lib.utils.schemas.TokenizerConfig): Configuration settings for the tokenizer
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
        self.token_to_id = mergeable_ranks
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
        self.token_bytes_cache = None

    def encode_special(self, token: str) -> int:
        return self.special_tokens[token]

    @classmethod
    def from_pretrained(cls, config: TokenizerConfig):
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
            warnings.warn("Using DummyTokenizer, this is not a real tokenizer and should only be used for testing purposes.")
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
            config: TokenizerTrainerConfig
        ):
        special_tokens = config.special_tokens.list()
        vocab_size_no_special = config.vocab_size - len(special_tokens)
        # TODO: make the other tokenizers for comparison; lines +1 and +2 below are temporary
        if not config.trainer == "huggingface":
            raise NotImplementedError("Training with other configuration than 'huggingface' is not implemented yet.")
        # TODO: make pretokenizer here -> options: 1. gpt2, 2. custom
        if config.trainer == "tiktoken":
            from tiktoken._educational import bpe_train
            warnings.warn("Training tokenizer with tiktoken is a TODO for future improvement.")
            # TODO: WIP, not tested yet
            mergeable_ranks = bpe_train(data=text_iterator, vocab_size=vocab_size_no_special, pat_str=config.pat_str)
        elif config.trainer == "huggingface":
            from tokenizers import decoders, pre_tokenizers, Regex
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            
            # from huggingface_hub import logging
            # logging.disable_progress_bars()
            tknzr = HFTokenizer(
                BPE(
                    byte_fallback=True,
                    unk_token=None,
                    fuse_unk=False
                ))
            tknzr.normalizer = None
            pattern = Regex(config.pat_str)
            tknzr.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(pattern=pattern, behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
            ])
            tknzr.decoder = decoders.ByteLevel()
            tknzr.post_processor = None
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
            
            trainer = BpeTrainer(
                vocab_size=vocab_size_no_special, 
                show_progress=True,
                min_frequency=0,
                initial_alphabet=initial_alphabet,
                special_tokens=[]
            )
            trainer.show_progress = config.show_progress
            tknzr.train_from_iterator(iterator=text_iterator, trainer=trainer)

            # os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
            # print("Tokenizer state", tknzr.model.__getstate__().keys())
            merges = json.loads(tknzr.to_str())["model"]["merges"]
            def merge_to_bytes(merge):
                left, right = merge
                # Handle the special case of the space token, 
                # which is represented as "Ġ" in the HuggingFace tokenizer
                left = left.replace("Ġ", " ") 
                right = right.replace("Ġ", " ")
                return left.encode("utf-8") + right.encode("utf-8")
            mergeable_ranks = { 
                merge_to_bytes(merge): rank + 256
                for rank, merge in enumerate(merges) 
            }
            # mergeable_ranks = { 
            #     left.encode("utf-8") + right.encode("utf-8"): rank + 256
            #     for rank, (left, right) in enumerate(merges) 
            # }
            mergeable_ranks.update({ bytes([i]): i for i in range(256) if i not in mergeable_ranks }) # Add single byte tokens to mergeable ranks

        # TODO: add other trainer options (bpe, rust bpe, fast bpe...)
        # The following options are placeholders for future impl.
        elif config.trainer in ["bpe", "fbpe", "rbpe"]:
            raise NotImplementedError(f"Tokenizer training mode {config.trainer!r} is not yet implemented. Please use 'huggingface' mode.")
        elif config.trainer == "bpe":
            # naive python implementation of byte-level BPE, not optimized for large corpora, but serves as a reference
            from gpt_lib.tokenizer.bpe import bpe
            _, mergeable_ranks = bpe()
        elif config.trainer == "fbpe":
            from gpt_lib.tokenizer.bpe import bpe_fast
            trainer = ...
        elif config.trainer == "rbpe":
            from rbpe import bpe
            ...
        elif config.trainer == "dummy":
            warnings.warn("Using DummyTokenizer for training, this is not a real tokenizer and should only be used for testing purposes.")
            return cls(DummyTokenizer(config), config)
        else:
            raise ValueError(f"Unsupported tokenizer trainer: {config.trainer}")
        tokenizer = cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            # special_tokens=config.special_tokens.list(),
            config=config
        )

        if config.to_save:
            tokenizer.save_to_directory()
        return tokenizer
    
    @property
    def token_bytes(self):
        token_bytes_path = Path(self.config.dirname) / "token_bytes.pt"
        if self.token_bytes_cache is not None:
            return self.token_bytes_cache
        
        if token_bytes_path.exists():
            token_bytes = torch.load(token_bytes_path)
            print(f"Loaded token_bytes from {token_bytes_path}")
        else:
            vocab_size = self.vocab_size
            special_set = set(self.special_tokens)
            token_strings = [self.decode([token_id]) for token_id in range(vocab_size)]
            token_bytes = []
            for token_id in range(vocab_size):
                token_str = token_strings[token_id] # the Python string representation of this token
                if token_str in special_set:
                    token_bytes.append(0) # special characters are not counted
                else:
                    id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
                    token_bytes.append(id_bytes)
            token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
            with open(token_bytes_path, "wb") as f:
                torch.save(token_bytes, f)
            print(f"Saved token_bytes to {token_bytes_path}")
        self.token_bytes_cache = token_bytes
        return token_bytes

    @classmethod
    def from_disk(cls, name: str, cachedir: Optional[Union[str, Path]] = None):
        if cachedir is None:
            cachedir = TOKENIZERS_FOLDER
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)
        # dirname = cachedir / name
        config = TokenizerConfig.from_directory(name, cachedir=cachedir)
        mergeable_ranks = config.get_mergeable_ranks()
        # vocab_path = dirname / "vocab.pkl"
        # with open(vocab_path, "rb") as vf:
        #     mergeable_ranks = pickle.load(vf)
        return cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=config.special_tokens.list(),
            config=config
        )
    
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
        self.config.save_to_directory()

        vocab_path = directory / "vocab.pkl" 
        with open(vocab_path, "wb") as vf:
            pickle.dump(self.token_to_id, vf)

        
    def encode(
            self, 
            text: Union[str, List[str]],
            *args, **kwargs
        ) -> Union[List[int], List[List[int]]]:
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
                warnings.warn(f"Unsqueeze option is not typically used for batch encoding, as it adds an extra dimension that may not be necessary. Use with caution. Encoder input is already a batch of {len(text)} sequences.")
        else:
            text_type = f"List[{type(text[0])}]" if isinstance(text, list) else text_type
            raise TypeError(f"Tokenizer.encode expected 'str' or 'List[str]', got {text_type!r}.")
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.main.decode(token_ids)
    
    def __call__(self, text, *args, **kwds):
        return self.encode(text, *args, **kwds)