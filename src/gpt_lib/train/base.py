from gpt_lib.model.gpt import GPTModel
from gpt_lib.utils.schemas import TrainingConfig, TrainingState, TrainingMetrics
from gpt_lib.train.optimizer import AdamW
from gpt_lib.utils.board import Board

from gpt_lib.utils.default import DEVICE

import torch
from torch import nn
from torch.utils.data import DataLoader

import math
import numpy as np
import time, pickle
# import time, pickle, wandb
from typing import Callable, Iterable, Literal
from pathlib import Path

from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def __init__(
            self,
            model: GPTModel,
            train: Iterable,
            val_dataset: Iterable,
            test_dataset: Iterable,
            config: TrainingConfig,
            board_type: Literal["wandb", "tensorboard", "none"] = "none",
            device: torch.device = DEVICE,
            dtype: torch.dtype = torch.float32,
        ):
        ...


    def init_board(self, board_type: Literal["wandb", "tensorboard", "none"] = "none") -> None:
        self.board = Board(place=board_type)
    
    def evaluate(self, dataset: Iterable) -> TrainingMetrics:
        raise NotImplementedError
    
    def get_lr_multiplier(self, it):
        warmup_iters = round(self.config.warmup_ratio * it)
        warmdown_iters = round(self.config.warmdown_ratio * it)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= it - warmdown_iters:
            return 1.0
        else:
            progress = (it - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * self.config.final_lr_frac

    def _training_step(
            self,
            model,
        ):
        pass

    def _validation_step(
            self,
            model,
        ):
        pass

    def train(self):
        # init data loader

        # loop on time/flops

        # iter training steps 

        # Fast fail: abort if loss is exploding or NaN
        # if math.isnan(train_loss_f) or train_loss_f > 100:
        #     print("FAIL")
        #     exit(1)

        # iter on validation steps
        pass

    
    def save_model(self, path: Path) -> None:
        raise NotImplementedError
    
    def load_model(self, path: Path) -> None:
        raise NotImplementedError
    
    def save_metrics(self) -> None:
        raise NotImplementedError
    
    def load_metrics(self, path: Path) -> None:
        raise NotImplementedError
