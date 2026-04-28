from gpt_lab.utils.schemas import TokenizerConfig
from typing import List
import regex as re

from tokenizers import NormalizedString, PreTokenizedString
from concurrent.futures import ThreadPoolExecutor, as_completed


class SimplePreTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.split_pattern = config.pat_str
        self.special_tokens = config.special_tokens.list()
        # self.tokenizer_type = config.tokenizer_type
        self.strip_spaces = False
        self.whitespace_tokens = {
            " ": "Ġ", 
            "\t": config.special_tokens.tab, 
            "\n": config.special_tokens.new_line
        }
        self.byte_level = True # Enable byte-level handling

    def _split(
            self, 
            text: str, 
            drop_special_tokens: bool = False
        ) -> List[str]:
        if text == "":
            return []

        special_tokens = self.special_tokens
        if len(special_tokens) > 0:
            safe_tokens = [re.escape(t) for t in special_tokens]
            reg = f"({'|'.join(safe_tokens)}|{self.split_pattern})"
        else: 
            reg = self.split_pattern

        parts = re.findall(reg, text, flags=re.UNICODE)
        parts: List[str] = [p for p in parts if p != ""]

        tokens: List[str] = []
        buffer = ""
        def _classify(part):
            if part.isspace():
                return ("space", self.whitespace_tokens.get(part, "Ġ"))
            elif part in special_tokens:
                return ("special", part)
            else:
                return ("text", part)
        # Classify parts in parallel, preserve order with executor.map
        max_workers = min(32, len(parts))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            classified = list(ex.map(_classify, parts))
            for kind, val in classified:
                if kind == "space":
                    buffer = val
                elif kind == "special":
                    if not drop_special_tokens:
                        tokens.append(val)
                    buffer = ""
                else:  # "text"
                    tokens.append(buffer + val)
                buffer = ""

        if self.strip_spaces:
            tokens = [t.lstrip() for t in tokens if t.lstrip() != ""]
    
         
        return tokens

    def split(self, i: int, normalized_str: NormalizedString, drop_special_tokens: bool = False) -> List[NormalizedString]:
        text = normalized_str.get_str()
        tokens = self._split(text, drop_special_tokens)
        tokens = [NormalizedString(t) for t in tokens]
        # TODO: Handle byte-level here?
        return tokens

    def batch_split(self, texts: List[str], drop_special_tokens: bool = False) -> List[List[str]]:
        tokens = []
        if not texts:
            return []

        tokens: List[List[str]] = [None] * len(texts)
        max_workers = min(32, len(texts))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._split, text, drop_special_tokens): i for i, text in enumerate(texts)}
        for fut in as_completed(futures):
            idx = futures[fut]
            tokens[idx] = fut.result()
        return tokens

    def __call__(self, text: str | List[str], drop_special_tokens: bool = False) -> List[str] | List[List[str]]:
        if isinstance(text, str):
            return self.split(text, drop_special_tokens)
        return self.batch_split(text, drop_special_tokens)
    
    def pre_tokenize(self, pretok: PreTokenizedString) -> None:
        """In case of using the tokenizers library"""
        pretok.split(self.split)
