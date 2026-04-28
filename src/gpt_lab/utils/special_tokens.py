from typing import List
from pydantic import BaseModel


class SpecialTokens(BaseModel):
    # # pad: str = "<|padding|>"
    # # oov: str = "<|out-of-vocabulary|>"
    # tab: str = "<|tab|>"
    # new_line: str = "<|new-line|>"
    bos: str = "<|bos|>"
    # # eos: str = "<|eos|>"
    # boa: str = "<|start-of-assistant|>"
    # eoa: str = "<|end-of-assistant|>"
    # bou: str = "<|start-of-user|>"
    # eou: str = "<|end-of-user|>"
    # bot: str = "<think>"
    # eot: str = "</think>"
    # py_start: str = "<|python|>"
    # py_end: str = "<|end-python|>"
    
    def list(self) -> List[str]:
        """
        Returns a list of all special tokens.
        """
        return [v for v in self.__dict__.values() if v is not None]
    
    def dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def __len__(self):
        return len(self.list())