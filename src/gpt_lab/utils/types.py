from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, Literal
from functools import lru_cache
import torch

from gpt_lab.utils.import_utils import is_torch_greater_or_equal

TParams = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

# Type Literals
Devices = Literal["cpu", "cuda", "mps", "xpu", "meta"]
TokenizerSources = Literal["tiktoken", "local", "huggingface", "dummy"]
TokenizerTensors = Literal["pt", "np", "tf", "jax"]
NormalizationTypes = Literal["rms", "layer"]
AttnImplTypes = Literal["sdpa", "fused", "torch", "raw"] # raw = return attention weights^^
PositionalEncodingTypes = Literal["positional", "rope"]
TfTypes = Literal["dense", "moe"]
Dtypes = Literal["float32", "float16", "bfloat16"]
OptimizerNames = Literal[ "adam", "adamw", "muon", "shampoo", "adahessian"]
LossTypes = Literal["cross_entropy", "kl_divergence"]
LossReductionTypes = Literal["none", "mean", "sum"]
TpModes = Literal["row", "column"]

# str_to_torch_dtype = {
#     "bool": torch.bool,
#     "uint8": torch.uint8,
#     "int8": torch.int8,
#     "int16": torch.int16,
#     "float16": torch.float16,
#     "bfloat16": torch.bfloat16,
#     "int32": torch.int32,
#     "float32": torch.float32,
#     "float64": torch.float64,
#     "int64": torch.int64,
#     "float8_e4m3": torch.float8_e4m3fn,
#     "float8_e5m2": torch.float8_e5m2,
# }

# if is_torch_greater_or_equal("2.3.0"):
#     str_to_torch_dtype["uint16"] = torch.uint16
#     str_to_torch_dtype["uint32"] = torch.uint32
#     str_to_torch_dtype["uint64"] = torch.uint64

# Dtypes = Literal[tuple(str_to_torch_dtype.keys())]