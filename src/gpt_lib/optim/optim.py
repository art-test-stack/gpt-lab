from gpt_lib.optim.adamw import AdamW
from gpt_lib.optim.muon import Muon
from gpt_lib.optim.shampoo import Shampoo
from gpt_lib.optim.adam import Adam
from gpt_lib.optim.adahessian import Adahessian

from typing import Literal

opt_class = {
    "adamw": AdamW,
    "muon": Muon,
    "shampoo": Shampoo,
    "adam": Adam,
    "adahessian": Adahessian,
}

OptimizerNames = Literal[tuple(opt_class.keys())]