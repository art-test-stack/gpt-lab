from gpt_lib.train.base import BaseTrainer
from gpt_lib.model.gpt import GPTModel
from gpt_lib.utils.schemas import TrainingConfig
from gpt_lib.utils.default import DEVICE

import torch
from torch import nn
from typing import Iterable

class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Iterable,
        val_dataset: Iterable,
        test_dataset: Iterable,
        config: TrainingConfig,
        device: torch.device = DEVICE,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=config,
            device=device,
            dtype=dtype,
        )