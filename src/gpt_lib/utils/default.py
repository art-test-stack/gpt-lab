import torch
import os
from pathlib import Path
from typing import Tuple, Dict, Literal
from gpt_lib.utils.special_tokens import SpecialTokens

# ----------- PROCESSOR -----------

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

NUM_THREADS = os.cpu_count()

# ------------- DATA -------------

IS_TIKTOKEN = False # TODO: parse as arg

SPECIAL_TOKENS = SpecialTokens()

# CACHE_DIR = Path.home() / ".gpt_lib"
CACHE_DIR = Path(".gpt_lib") # easier for testing; can be overridden by env var or arg
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BOARD_DIR = CACHE_DIR / "runs"
DATA_DIR = CACHE_DIR / "data"
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = CACHE_DIR / "output"
MODELS_FOLDER = CACHE_DIR / "models"
TOKENIZERS_FOLDER = CACHE_DIR / "tokenizers"
VOCAB_SIZE = 32_000
MAX_TOKEN_LENGTH = 32

# TOKEN_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # GPT 4 SPLIT
# https://arxiv.org/pdf/2402.01035
_pat_str_eng_contr = dict(
    gpt2=r"""'s|'t|'re|'ve|'m|'ll|'d""",
    gpt4=r"""'(?i:[sdmt]|ll|ve|re)""", # r""""?i:'s|'t|'re|'ve|'m|'ll|'d""" 
)
_pat_str_words = dict(
    gpt2=r""" ?\p{L}+""",
    gpt4=r"""[^\r\n\p{L}\p{N}]?+\p{L}+""",
) 
_pat_str_digits = dict(
    gpt2=r""" ?\p{N}+""",
    gpt4=r"""\p{N}{1,3}""",
    nanochat=r"""\p{N}{1,2}""",
)
_pat_str_non_alpha = dict(
    gpt2=r""" ?[^\s\p{L}\p{N}]+""",
    gpt4=r""" ?[^\s\p{L}\p{N}]++[\r\n]*"""
)
_pat_str_line_breaks = r"""\s*[\r\n]+"""
_pat_str_trailing_spaces = r"""\s+(?!\S)"""
_pat_str_whitespace = r"""\s+"""

# PAT_STR_GPT2 = r"""('s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_STR_GPT2 = rf"""{_pat_str_eng_contr['gpt2']}|{_pat_str_words['gpt2']}|{_pat_str_digits['gpt2']}|{_pat_str_non_alpha['gpt2']}|{_pat_str_trailing_spaces}|{_pat_str_whitespace}"""
# PAT_STR_GPT4 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
PAT_STR_GPT4 = rf"""{_pat_str_eng_contr['gpt4']}|{_pat_str_words['gpt4']}|{_pat_str_digits['gpt4']}|{_pat_str_non_alpha['gpt4']}|{_pat_str_line_breaks}|{_pat_str_trailing_spaces}|{_pat_str_whitespace}"""
# PAT_STR_punct = r"""(?\\p{L}+)|\\p{N}{1,3}|?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\s+(?!\S)|\s+"""
PAT_STR_punct = rf"""{_pat_str_words['gpt2']}|{_pat_str_digits['gpt4']}|{_pat_str_non_alpha['gpt4']}|{_pat_str_line_breaks}|{_pat_str_trailing_spaces}|{_pat_str_whitespace}""" # pattern for pre-tokenization that focuses on punctuation, used for ablation
PAT_STR_nanochat = rf"""{_pat_str_eng_contr['gpt4']}|{_pat_str_words['gpt4']}|{_pat_str_digits['nanochat']}|{_pat_str_non_alpha['gpt4']}|{_pat_str_line_breaks}|{_pat_str_trailing_spaces}|{_pat_str_whitespace}"""
# copied from tiktoken.get_encoding("cl100k_base")._pat_str and tiktoken.get_encoding("o200k_base")._pat_str
PAT_STR_cl100k_base = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
PAT_STR_o200k_base = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
PAT_STR: Dict[str, str] = dict(
    gpt2=PAT_STR_GPT2,
    gpt4=PAT_STR_GPT4,
    punct=PAT_STR_punct,
    cl100k_base=PAT_STR_cl100k_base,
    o200k_base=PAT_STR_o200k_base,
    nanochat=PAT_STR_nanochat
)
PatStr = Literal[tuple(PAT_STR.keys())]

# ------------- DRIVE -------------

SAVE_ON_DRIVE = True
DRIVE_FILE = ""
SAVE_ON_WANDB = True

# ------------- MODEL -------------

VOCAB_SIZE = 32_000
MAX_CONTEXT = 64

NUM_HEADS = 2
NUM_LAYERS = 2

DIM_MODEL = 128
DIM_FFN = 4 * DIM_MODEL

DIM_HEAD = DIM_MODEL // NUM_HEADS

DROPOUT = .1

# ------------- TRAIN -------------

BATCH_SIZE = 128
PRETRAINING_VAL_RATIO = 1e-3

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_ITERS = 2_000

WEIGHT_DECAY = .1
DECAY_ITERS = 100_000

BETA_1 = .9
BETA_2 = .95

EPSILON = 1e-8

VALIDATION_STEP = 50

RANDOM_SEED = 42


plt_config = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "savefig.bbox": "tight",
    "savefig.format": "pdf"
}

from matplotlib import pyplot as plt
plt.style.use(plt_config)

pt = 1./72.27
golden = (1 + 5 ** 0.5) / 2

width = 337.33545 * pt
height = width / golden