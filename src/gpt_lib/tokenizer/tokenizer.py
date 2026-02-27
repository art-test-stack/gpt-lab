from gpt_lib.utils.schemas import TokenizerConfig, TokenizerTrainerConfig
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
import torch
from typing import Callable, Iterable, List, Optional, Union
import pickle
from pathlib import Path
import os
import random, warnings, json
from gpt_lib.utils.special_tokens import SpecialTokens
from gpt_lib.utils.default import TOKENIZERS_FOLDER

class DummyTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
          
        self.bos_token_id = config.vocab_size 
    
    def encode(self, text, padding=None, to_torch=False, *args, **kwargs):
        assert isinstance(text, str), "Input text must be a string"
        
        length_encoded = random.randint(1, len(text) - 1)
        encoded = [random.randint(0, self.vocab_size - 1) for _ in range(length_encoded)]
        if padding == "max_length":
            while len(encoded) < self.config.max_context:
                encoded.append(0)
        elif padding == "longest":
            pass
        encoded = encoded[-self.config.max_context:]
        if to_torch:
            return torch.tensor(encoded)
        return encoded

    def decode(self, tokens, *args, **kwargs):
        return "".join([chr(t) for t in tokens])
     
     
def build_tokenizer(config: TokenizerConfig) -> Callable:
    return Tokenizer.from_pretrained(config)

# def load_tokenizer(config: TokenizerConfig) -> Callable:
#     """ Load a tokenizer based on the provided configuration """
#     if config.source == "tiktoken":
#         tokenizer = TikTokenizer(config)
#     elif config.source == "bpe":
#         tokenizer = ByteLevelBPE.from_directory(config)
#     elif config.source == "rust_bpe":
#         tokenizer = RustByteLevelBPE.from_directory(config.dir)
#     elif config.source == "huggingface":
#         tokenizer = HFTokenizer.from_pretrained(config.name)
#     else:
#         raise ValueError(f"Unsupported tokenizer type: {config.source}")
    
#     return tokenizer


class Tokenizer:
    """ Wrapper class for different tokenizer implementations 
    ## Use cases include:
        - Encoding with Tiktoken API (faster)
        - Loading TikToken tokenizer
        - Loading a custom trained TikToken tokenizer from local directory (must have merges + pattern + special tokens)
        - Loading a pretrained tokenizer from HuggingFace Hub
        - Loading a custom trained Byte-Level BPE tokenizer from local directory
        - Training Byte-Level BPE tokenizer (python implementation) from corpus
        - Training Byte-Level BPE tokenizer (Rust implementation) from corpus
        - Training HuggingFace tokenizer from corpus
        - Training HuggingFace tokenizer from corpus and convert it in Tiktoken implementation

    Args:
        tokenizer (Callable): The tokenizer instance to wrap
        config (Settings): Configuration settings for the tokenizer
    """
    def __init__(
            self, 
            # enc: Callable, 
            mergeable_ranks: dict[bytes, int],
            special_tokens: list[str],
            config: TokenizerConfig
        ):
        special_tokens = { sp: rank + len(mergeable_ranks) for rank, sp in enumerate(special_tokens) }
        self.token_to_id = mergeable_ranks
        # print("Vocab size (including special tokens):", len(self.token_to_id))
        # print("Len mergeable ranks (excluding special tokens):", len(mergeable_ranks))
        # print("Len special tokens:", len(special_tokens))
        
        self.enc = tiktoken.Encoding(
            name=config.name,
            pat_str=config.pat_str,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int]
            special_tokens=special_tokens, # Only add special tokens to the encoding, not to the mergeable ranks, to avoid conflicts
            explicit_n_vocab=config.vocab_size
        )
        # self.token_to_id.update(special_tokens)
        self.special_tokens = special_tokens
        self.config = config
        self.bos_token_id = self.encode_special(config.special_tokens.bos)

    def encode_special(self, token: str) -> int:
        return self.special_tokens[token]

    def get_vocab(self):
        return {**self.token_to_id, **self.special_tokens}
    
    @classmethod
    def from_pretrained(cls, config: TokenizerConfig):
        """Load a pretrained tokenizer from tiktoken/HuggingFace Hub/local directory based on the provided configuration."""
        if config.source == "tiktoken":
            enc = tiktoken.get_encoding(config.name)
            mergeable_ranks = enc.mergeable_ranks
            special_tokens = enc.special_tokens
            pat_str = enc.pat_str
        elif config.source == "huggingface":
            # TODO: convert HuggingFace tokenizer to tiktoken encoding, 
            # -> extracting merges and vocab from the HuggingFace tokenizer 
            # -> creating a new tiktoken encoding with those merges and vocab
            # -> need to handle special tokens 
            # -> need to extract hgf pretokenizer + string pattern
            return HFTokenizer.from_pretrained(config.name)
        elif config.source == "local":
            mergeable_ranks = config.get_mergeable_ranks()
            special_tokens = config.special_tokens.list()
            pat_str = config.pat_str
        elif config.source == "dummy":
            warnings.warn("Using DummyTokenizer, this is not a real tokenizer and should only be used for testing purposes.")
            return DummyTokenizer(config)
        else:
            raise ValueError(f"Unsupported tokenizer source: {config.source}")
        
        config.pat_str = pat_str
        return cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            config=config
        )
    
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
            # TODO: WIP, not tested yet
            mergeable_ranks = bpe_train(data=text_iterator, vocab_size=vocab_size_no_special, pat_str=config.pat_str)
        elif config.trainer == "huggingface":
            from tokenizers import decoders, pre_tokenizers, Regex
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
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
            tknzr.train_from_iterator(iterator=text_iterator, trainer=trainer)
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
        config.save_to_directory()
        tokenizer = cls(
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            # special_tokens=config.special_tokens.list(),
            config=config
        )
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

        vocab_path = directory / "vocab.pkl" 
        with open(vocab_path, "wb") as vf:
            pickle.dump(self.token_to_id, vf)

        
    def encode(
            self, 
            text: Union[str, List[str]],
            # num_threads: int = 8,
            *args, **kwargs
        ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            token_ids = self.enc.encode_ordinary(text)
        elif isinstance(text, list):
            token_ids = self.enc.encode_ordinary_batch(text)
        else:
            raise TypeError(f"Expected str or list of str, got {type(text)}")
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids)