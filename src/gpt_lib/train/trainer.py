from gpt_lib.model.gpt import GPTModel
from gpt_lib.utils.schemas import TrainingConfig, TrainingState, TrainingMetrics
from gpt_lib.train.optimizer import AdamW

from gpt_lib.train.base import BaseTrainer

from gpt_lib.utils.default import DEVICE

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import time, pickle
# import time, pickle, wandb
from typing import Callable, Iterable, Optional, Union
from pathlib import Path


class Trainer(BaseTrainer):
    @classmethod
    def from_checkpoint(
        self,
        model_name: str,
        version: str,
        ckpt_dir: Optional[Union[str,Path]]
    ):
        model = GPTModel.from_pretrained(
            model_name=model_name,
            model_dir=ckpt_dir
        )
        config = model.config.trainer
        # train_dataset =

    @classmethod
    def from_scratch(
        self,
        model_name: str
    ):
        ...

    def __init__(
            self, 
            model: GPTModel, # tokenizer should be integrated in the model
            train_dataset: Iterable, # TODO: must prepare dataset with tokenizer
            val_dataset: Iterable,
            test_dataset: Iterable,
            config: TrainingConfig | None = None,
            device: torch.device = DEVICE,
            dtype: torch.dtype = torch.float32,
            optimizer: torch.optim.Optimizer | None = None, # optional optimizer builder
        ):
        super().__init__(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

        self.config: TrainingConfig = config if config is not None else model.config.trainer 
        # self.model = model.to(device=device, dtype=dtype)
        self.model = model
        self.model_config = self.model.config
        # TODO: 
        # TRAINER INIT MAP:
        # 1. load model 
        # 2. load training steps
        # if training crashed -> from checkpoint dir (need every thing to be the same)
        # if sft -> from base model 
        # if dpo / grpo -> from base model + reward model -> GRPO

        if not optimizer:
            optimizer = model.build_optimizer() # to make
        self.optimizer = optimizer

        self.train_set = train_dataset
        self.val_set = val_dataset
        self.test_set = test_dataset

        self.metrics = TrainingMetrics()
        self.state = TrainingState()
        self.time = .0
        self.iter = 0
        self.tokens = 0
        self.model = model
        self.epochs = 0
        self.loss = float('inf')
        self.accuracy = .0
        self.val_loss = .0
        self.val_accuracy = .0
        self.best_val_loss = float('inf')
            
        self.max_sequence_length = self.model.max_content

        self.device = device
        self.metrics = {
            "time": [],
            "step": [],
            "tokens": [],
            "epochs": [],
            "accuracy": [],
            "loss": [],
            "val_accuracy": [],
            "val_loss": [],
            "best_val_loss": []
        }

        # if SAVE_ON_WANDB:
        #     wandb.init(
        #         project="michel-gpt-training",
        #         config={
        #             "learning_rate": self.optimizer.learning_rate,
        #             "architecture": "Transformer",
        #             "dataset": "many",
        #             "epochs": WARMUP_ITERS
        #         }
        #     )


    def save_metrics(self) -> None:

        self.metrics['time'].append(time.time() - self.time)
        self.metrics["iter"].append(self.iter)
        self.metrics["tokens"].append(self.tokens)
        self.metrics["epochs"].append(self.epochs)
        self.metrics["accuracy"].append(self.accuracy)
        self.metrics["loss"].append(self.loss)
        self.metrics["val_accuracy"].append(self.val_accuracy)
        self.metrics["val_loss"].append(self.val_loss)
        self.metrics["best_val_loss"].append(self.best_val_loss)

        if not OUTPUT_FOLDER.exists():
            OUTPUT_FOLDER.mkdir()

        pickle.dump(self.metrics, open(OUTPUT_FOLDER.joinpath('metrics.pkl'), 'wb'))
        self.time = time.time()


    def load_metrics(self, path: Path) -> None:
        if not path.exists():
            return
        
        self.metrics_history = pickle.load(open(OUTPUT_FOLDER.joinpath('metrics.pkl'), 'rb'))
        self.iter = self.metrics["iter"][-1]
        self.metrics = self.metrics["tokens"][-1]
        self.epochs = self.metrics["epochs"][-1]
        self.accuracy = self.metrics["accuracy"][-1]
        self.loss = self.metrics["loss"][-1]
        self.val_accuracy = self.metrics["val_accuracy"][-1]
        self.val_loss = self.metrics["val_loss"][-1]
        self.best_val_loss = np.min(self.metrics["val_loss"])


    def save_model(self, path: Path) -> None:
        if not path.exists():
            path.mkdir()
        self.model.save_checkpoint(path.joinpath("model.pt"))
        self.optimizer.save_checkpoint(path.joinpath("optimizer.pt"))
        
        if SAVE_ON_DRIVE:
            pass


    def load_model(self, path: Path) -> None:
        if not path.exists():
            return
        
        self.model.load_state_dict(torch.load(path.joinpath("model.pt"), map_location=DEVICE))
        self.optimizer.load_state_dict(torch.load(path.joinpath("optimizer.pt"), map_location=DEVICE))    

    def find_previous_session(self):
        pass
    
    def fit(self):

        self.time = time.time()

        train_set = DataLoader(
            dataset=self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_set = DataLoader(
            dataset=self.val_set,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # while True:
        for _ in range(5):
            losses = []

            for _, batch in enumerate(train_set, 0):
                
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                mask = torch.ones_like(batch).to(DEVICE)

                pred = self.model(x=x, mask=mask)
                loss = self.loss_function(y, pred) / len(x)

                loss.backward()


            self.model.clean_nan()

            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

            epoch_loss = np.average(losses)
            self.losses.append(epoch_loss)

            if self.iter % VALIDATION_STEP == 0:
                self._validation_step(val_set)

    def _training_loop(self):
        pass

    def _training_step(self):
        pass


    def _validation_step(self):
        pass

