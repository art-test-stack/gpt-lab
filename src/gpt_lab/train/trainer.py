"""
Trainer: Main training loop with checkpointing, evaluation, and metrics tracking.

Key responsibilities:
- Training loop with gradient accumulation
- Checkpoint save/load
- Validation and evaluation
- Metrics tracking and logging
- Device management and memory optimization
"""

from typing import Optional, Callable, Union, Literal, Dict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import random
import numpy as np

import time
import pickle
import gc
import warnings
import math
import logging
from packaging.version import parse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler

from gpt_lab.utils.board import Board
from gpt_lab.utils.default import CACHE_DIR, MODELS_FOLDER
from gpt_lab.utils.distributed import _DTYPE_MAP, get_dist_info
from gpt_lab.utils.common import print0, print0_dict
from gpt_lab.utils.logging import log_error, log_critical, log0
from gpt_lab.utils.schemas import (
    CheckpointState,
    COREMetrics,
    DataLoaderState,
    EvalMetrics, 
    TrainerConfig, 
    TrainerMetrics, 
    TrainerState
)
from gpt_lab.evaluate.bpb import compute_bpb
from gpt_lab.evaluate.core import evaluate_core
from gpt_lab.model.wrapper import Engine
from gpt_lab.model.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint, make_default_run_name

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration & State
# ============================================================================

# @dataclass
# class TrainerState:
#     """Complete state for trainer resumption."""
#     global_step: int = 0
#     global_tokens: int = 0
#     num_epochs: int = 0
#     best_val_loss: float = float('inf')
#     total_training_time: float = 0.0
#     smooth_train_loss: float = 0.0
#     train_loader_state: Optional[DataLoaderState] = None
    
    # Dataloader state for resumption

class DummyContext:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self, *args, **kwargs): pass
    def __exit__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwds):
        self.__enter__()
        return self

# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Training orchestrator with distributed support, checkpointing, and evaluation.
    
    Features:
    - Multi-GPU training with DDP support
    - Gradient accumulation
    - Mixed precision training (AMP)
    - Checkpoint save/load with full resumption
    - Validation and evaluation loops
    - Comprehensive metrics tracking
    - Memory optimization with gc management

    Example:
    ```python
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        trainer.train()
    ```
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_loader,
        val_loader,
        config: Optional[TrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        board: Optional[Board] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        lr_schedule: Optional[Callable] = None,
        muon_momentum_schedule: Optional[Callable] = None,
        weight_decay_schedule: Optional[Callable] = None,
        scaler: Optional[GradScaler] = None, # only used if config.compute_dtype is float16 and for resume training
        resume_state: Optional[TrainerState] = None,
        best_state: Optional[CheckpointState] = None,
    ):
        """
        Initialize Trainer.
        
        Args:
            model: The model to train
            tokenizer: Tokenizer for evaluation
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            optimizer: Optimizer (or None to use model.build_optimizer())
            board: Logging board (wandb, tensorboard, etc.)
            checkpoint_dir: Where to save checkpoints
            lr_schedule: Learning rate schedule function (step -> lr multiplier). Default: TrainerConfig.lr_schedule
            muon_momentum_schedule: Momentum schedule for Muon optimizer (step -> momentum multiplier). Default: TrainerConfig.muon_momentum_schedule
            weight_decay_schedule: Weight decay schedule for Muon optimizer (step -> weight decay multiplier). Default: TrainerConfig.weight_decay_schedule
            scaler: Gradient scaler for mixed precision training. Default: None (will be created if config.compute_dtype is float16)
        """
        self.training_type = "base" # for now we only have base training, TODO: extend to sft, grpo, etc.
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Config
        if config is None:
            config = TrainerConfig(n_steps=1)
            log0(
                "No training config provided. Using default config with 1 step. "
                "Please provide a TrainerConfig instance.", level="warning", logger=logger
            )
        self.config = config
        
        if checkpoint_manager is None:
            model_name = self.model.__class__.__name__
            depth = getattr(self.model.config, "n_layers", "unknown")
            model_run = make_default_run_name(model_name, depth, self.config.dist_info)
            checkpoint_manager = CheckpointManager(
                model_name=self.model.__class__.__name__,
                model_run=model_run,
                source=self.training_type, 
                dist_info=self.config.dist_info,
                mode="shard" if self.config.dist_info.get("IS_DDP_INITIALIZED", False) else "ddp", # naming is a bit dummy
            )
        self.ckpt_manager = checkpoint_manager 
        self.dirname = self.ckpt_manager.source_dir
        # TODO: make an error if dir exists and training is not resuming
        self.dirname.mkdir(parents=True, exist_ok=True)
        
        # Board
        if board is None:
            board = Board(board_type="dummy")
        self.board = board
        
        # Optimizer
        if optimizer is None:
            if hasattr(model, "build_optimizer"):
                optimizer = model.build_optimizer(config)
            else:
                raise ValueError(
                    "No optimizer provided and model has no build_optimizer() method. "
                    "Please provide an optimizer."
                )
        self.optimizer = optimizer
        
        # State and metrics
        self.state = resume_state or TrainerState()
        self.ckpt_state = best_state or CheckpointState()
        self.metrics = TrainerMetrics()     # TODO
        self.eval_metrics = EvalMetrics()   # TODO
        self.core_metrics = COREMetrics()   # TODO
        
        # Mixed precision
        self.dtype = self.config.dist_info["compute_dtype"]
        self.use_amp = self.dtype in ["float16", "bfloat16"]
        def amp_context():
            if self.use_amp:
                return autocast(
                    device_type=self.config.dist_info["DEVICE_TYPE"],
                    dtype=_DTYPE_MAP[self.dtype]
                )
            else:
                return DummyContext()
            
        def disable_fp8_context():
            if self.config.fp8:
                # TODO: this is a placeholder impl.
                from gpt_lab.model.fp8 import DisableFP8
                return DisableFP8
            else:
                return DummyContext
        self.train_context = amp_context
        self.val_context = disable_fp8_context()

        self.scaler = None
        if scaler is not None:
            self.scaler = scaler
        elif self.dtype == "float16":
            self.scaler = GradScaler()
        
        # LR and other schedules
        self.lr_schedule = lr_schedule or config.lr_multiplier_schedule
        self.muon_momentum_schedule = muon_momentum_schedule or config.muon_momentum_schedule
        self.weight_decay_schedule = weight_decay_schedule or config.weight_decay_schedule

        # Device config
        self.device_type = config.dist_info.get("DEVICE_TYPE", "cpu")
        self.device = torch.device(config.dist_info.get("DEVICE", "cpu"))
        self._get_sync_fn()
        self._add_model_hook_for_grad_monitoring()

    def _get_sync_fn(self):
        """Set up device synchronization function."""
        if self.device_type == "cuda":
            self.synchronize = torch.cuda.synchronize
            self.get_max_memory = torch.cuda.max_memory_allocated
        else:
            self.synchronize = lambda: None
            self.get_max_memory = lambda: 0

    def _add_model_hook_for_grad_monitoring(self):
        if not self.config.monitor_grad_norms or getattr(self, "_hooked", False):
            return
        self._grad_stats = {}

        def _hook(name):
            if not name.endswith("weight"):
                return lambda grad: None  # only monitor weight gradients for now
            param_name = name.replace('.', '/').replace('/weight', '')
            def hook_fn(grad):
                self._grad_stats[param_name] = {
                    "rms": grad.square().mean().sqrt().detach(),
                    "mean": grad.mean().detach(),
                    "abs_mean": grad.abs().mean().detach(),
                }
            return hook_fn

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(_hook(name))

        self._hooked = True # avoid hooking multiple times

    def _rm_model_hooks(self):
        """Remove all hooks from model parameters."""
        for param in self.model.parameters():
            param._backward_hooks = {}
        self._hooked = False

    def log_gradients(self):
        "called between loss.backward() and optimizer.step() to log gradient norms and means"
        if not hasattr(self, "_grad_stats"):
            return

        logs = {}
        for name, stats in self._grad_stats.items():
            logs[f"grad_rms/{name}"] = stats["rms"].item()
            logs[f"grad_mean/{name}"] = stats["mean"].item()
            logs[f"grad_abs_mean/{name}"] = stats["abs_mean"].item()

        self.board.log(logs, step=self.state.step)
        self._grad_stats.clear()  # important: reset for next step

    def train(self):
        """
        Main training loop.
        
        Performs gradient accumulation, validation, checkpointing, and logging.
        """
        print0("=" * 70)
        print0(f"Starting training for {self.config.n_steps:,} steps -- {self.config.total_batch_size:,} tokens per step.")
        print0("=" * 70)
        
        # Extracte main constants from config
        step = self.state.step + int(self.state.step > 0) # start from the next step to avoid repeating the last step if resuming
        n_steps = self.config.n_steps
        n_flops_per_token = self.config.n_flops_per_token
        total_batch_size = self.config.total_batch_size
        n_acc_steps = self.config.n_acc_steps
        
        # Compile model if using PyTorch 2.0+
        if parse(torch.__version__) >= parse("2.0"):
            self._compiled_model = torch.compile(self.model, dynamic=False)
        else:
            self._compiled_model = self.model

        train_iter = iter(self.train_loader)
        x, y, dataloader_state = next(train_iter) # prefetch

        # Prepare for training
        self._compiled_model.train()
        smooth_loss = self.state.smooth_train_loss
        ema_beta = 0.9
        
        total_dt = []  # For ETA calculation
        
        while step < n_steps:
            self.state.step = step
            last_step = (step == n_steps - 1)
            flops_so_far = n_flops_per_token * total_batch_size * step
            self.synchronize()
            
            # ================================================================
            # Validation on 'val_loader' every 'eval_every' steps
            # ================================================================
            
            if (
                (self.config.eval_bpb_every == -1 and last_step) or # always eval at last step
                (self.config.eval_bpb_every > 0 and step > 0 and # eval every eval_bpb_every steps except the first step
                (last_step or step % self.config.eval_bpb_every == 0)) 
            ):
                start_bpb_eval = time.time()
                self._compiled_model.eval()
                eval_steps = self.config.n_bpb_tokens // (self.config.device_batch_size * self.model.config.max_context * self.config.dist_info["WORLD_SIZE"])
                with self.val_context(self._compiled_model):
                    val_res = compute_bpb(
                        self._compiled_model, 
                        self.val_loader(), 
                        eval_steps,
                        dist_info=self.config.dist_info,
                        token_bytes=self.tokenizer.token_bytes
                    )
                print0(f"Step {step:05d} | "\
                       f"Validation bpb: {val_res['bpb']:.6f} | "\
                    f"Validation loss: {val_res['loss']:.6f}")
                
                dt_bpb_eval = time.time() - start_bpb_eval

                if (
                    (self.ckpt_state.best_eval_value is None) or 
                    (val_res['bpb'] < self.ckpt_state.best_eval_value)
                ):
                    self.ckpt_state.best_eval_value = val_res['bpb']
                    self.ckpt_state.best_eval_step = step
                    if (
                        self.config.save_on_best and 
                        not ((self.config.save_every > 0 and step > 0 and step % self.config.save_every == 0)) # already saving this step
                        ):
                        self.save_checkpoint()
                
                log_dict = {
                    "eval/loss": val_res['loss'],
                    "eval/bpb": val_res['bpb'],
                    "eval/best_bpb": self.ckpt_state.best_eval_value,
                    "eval/step_time_ms": dt_bpb_eval * 1000,  # Convert to milliseconds
                }
                self.board.log(log_dict, step=step)
                self.eval_metrics.append(log_dict, step=step)
                self._compiled_model.train()

            # ================================================================
            # Validation on CORE metric every 'core_eval_every' steps
            # ================================================================
            
            results = {}
            if (
                (self.config.eval_core_every == -1 and last_step) or
                (self.config.eval_core_every > 0 and (
                last_step or (step > 0 and step % self.config.eval_core_every == 0)))
            ):
                self.model.eval()
                with self.val_context(self.model):
                    results = evaluate_core(
                        self.model,
                        self.tokenizer,
                        self.device,
                        max_per_task=self.config.n_core_tokens,
                    )
                max_throughput = results.get("core/max_per_task", 0) * self.config.dist_info.get("WORLD_SIZE", 1) / results.get("core/step_time_ms", 1e-3) * 1000
                print0(f"Step {step:05d}/{n_steps:05d} | "
                       f"CORE: {results['core/core']:.4f} | "
                       f"Accuracy: {results['core/accuracy']:.4f} | "
                       f"Max per task: {int(results['core/max_per_task'])} | "
                       f"Step time: {results.get('core/step_time_ms', 0):.2f}ms | "
                       f"Max throughput: {max_throughput:,.0f} tok/s")
                log_dict = {
                    "core/core": results["core/core"],
                    "core/accuracy": results["core/accuracy"],
                    "core/max_per_task": results["core/max_per_task"],
                    "core/step_time_ms": results.get("core/step_time_ms", 0),
                }
                for task_label, task_results in results.get("all_core_results", {}).items():
                    for task_metric, task_value in task_results.items():
                        log_dict[f"core/{task_label}/{task_metric}"] = task_value
                if (
                    (self.ckpt_state.best_core_value is None) or 
                    (results["core/core"] > self.ckpt_state.best_core_value)
                ):
                    self.ckpt_state.best_core_value = results["core/core"]
                    self.ckpt_state.best_core_step = step
                self.board.log(log_dict, step=step)
                self.core_metrics.append(results, step=step)
                self.model.train()
            
            # ================================================================
            # Sample some outputs every 'sample_every' steps
            # ================================================================

            if (
                self.config.dist_info["RANK"] == 0 and 
                ((self.config.sample_every == -1 and last_step) or
                (self.config.sample_every > 0 and 
                (last_step or (step > 0 and step % self.config.sample_every == 0))))
            ):
                self.model.eval()
                prompts = [
                    "The capital of France is",
                    "The chemical symbol of gold is",
                    "If yesterday was Friday, then tomorrow will be",
                    "The opposite of hot is",
                    "The planets of the solar system are:",
                    "My favorite color is",
                    "If 5*x + 3 = 13, then x is",
                ]
                engine = Engine(self.model, self.tokenizer) # use orig_model to avoid recompilation
                with self.val_context(self.model):
                    samples = engine.generate_batch(prompts, num_samples=1, max_tokens=16, temperature=0)
                print0(self.tokenizer.decode(samples[0]))
                self.model.train()

            # ================================================================
            # Training step with gradient accumulation
            # ================================================================

            step_start_time = time.time()
            loss_accum = 0.0
            
            for _ in range(n_acc_steps):
                self.state.train_loader_state = dataloader_state
                with self.train_context():
                    loss = self._compute_loss(x, y)
                loss = loss / n_acc_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.detach().item()
                if math.isnan(loss_accum) or math.isinf(loss_accum):
                    torch.save(x, self.dirname / "bad_batch.pt")
                    log_error("⛔️ BAD ACCUMULATED LOSS DETECTED !\n" \
                        f"Loss is NaN or Inf at {step=}.\n" \
                        f"Dataloader state: {dataloader_state.__dict__}.\n" \
                        f"Model inputs shape: {x.shape}, values: {x}\n" \
                        f"Model targets shape: {y.shape}, values: {y}\n" \
                        f"Accumulated loss: {loss_accum}",
                        error_type=ValueError, logger=logger  
                    )
                x, y, dataloader_state = next(train_iter)
            
            if dataloader_state is not None:
                if isinstance(dataloader_state, DataLoaderState):
                    self.state.n_epochs = dataloader_state.epoch
                else:
                    self.state.n_epochs = dataloader_state.get("epoch", 0)
            
            lrm, muon_momentum, weight_decay = self._apply_optim_hparam_scheduler(step)

            if self.config.monitor_grad_norms and self.config.dist_info["RANK"] == 0:
                self.log_gradients()

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                if self.config.dist_info["IS_DDP_INITIALIZED"]:
                    for g in self.scaler._found_inf__per_device:
                        dist.all_reduce(g, op=dist.ReduceOp.MAX)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self._compiled_model.zero_grad(set_to_none=True)

            self.synchronize()
            step_end_time = time.time()
            
            # ================================================================
            # Logging and metrics
            # ================================================================
            
            step_dt = step_end_time - step_start_time
            
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_accum
            debiased_smooth_loss = smooth_loss / (1 - ema_beta ** (step + 1))
            
            # Calculate throughput
            eff_global_tokens = (
                x.numel() * n_acc_steps *
                (self.config.dist_info.get("WORLD_SIZE", 1) if self.config.dist_info else 1)
            )
            tokens_per_sec = eff_global_tokens / step_dt

            step_flops_per_sec = n_flops_per_token * tokens_per_sec
            mfu = step_flops_per_sec / self.config.dist_info["gpu_peak_flops"] * 100
            
            total_dt.append(step_dt)
            self.state.n_tokens += eff_global_tokens
            self.state.smooth_train_loss = smooth_loss
            self.state.total_training_time += step_dt
            
            # Log every log_every steps
            if step % self.config.log_every == 0:
                pct_done = 100 * step / n_steps
                
                # ETA calculation
                if len(total_dt) > 10:
                    avg_step_time = sum(total_dt[-10:]) / 10
                    eta_seconds = (n_steps - step) * avg_step_time
                    eta_str = f" | ETA: {eta_seconds/60:.1f}m"
                else:
                    eta_str = ""

                if math.isnan(debiased_smooth_loss) or math.isinf(debiased_smooth_loss):
                    log_error(f"Loss is NaN or Inf at {step=}, {debiased_smooth_loss=}. Check previous logs for details.", error_type=ValueError, logger=logger)
                
                print0(
                    f"Step {step:05d}/{n_steps:05d} ({pct_done:5.1f}%) | "
                    f"loss: {debiased_smooth_loss:.6f} | "
                    f"lrm: {lrm:.2e} | "
                    f"dt: {step_dt*1000:.2f}ms | "
                    f"tok/s: {tokens_per_sec:,.0f}"
                    f"{eta_str}"
                )
                log_dict = {
                    "epochs": self.state.n_epochs,
                    "train/loss": debiased_smooth_loss,
                    "train/raw_loss": loss_accum,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_time_ms": step_dt * 1000,
                    "train/total_tokens": self.state.n_tokens,
                    "train/flops_per_sec": step_flops_per_sec,
                    "train/mfu": mfu,
                    "lrm": lrm,
                    "muon_momentum": muon_momentum,
                    "weight_decay": weight_decay,
                    "train/total_training_flops": flops_so_far,
                    "train/total_training_time": self.state.total_training_time,
                    "train/eta_seconds": eta_seconds if len(total_dt) > 10 else float("inf"),
                }

                self.metrics.append(log_dict, step)
                self.board.log(log_dict, step=step)
            
            # ================================================================
            # Checkpointing
            # ================================================================
            if (self.config.save_every > 0 and step > 0 and step % self.config.save_every == 0):
                self.save_checkpoint()
            
            if (sum(total_dt) > self.config.target_time * 60) and self.config.target_time > 0:
                print0(f"Reached target time of {self.config.target_time} minutes. Stopping training.")
                break
            
            # ================================================================
            # Cleanup
            # ================================================================
            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif step % 5000 == 0:
                gc.collect()
            
            step += 1

        del self._compiled_model  # Clean up compiled model if it exists
        self._compiled_model = None
        
        print0("=" * 70)
        print0(f"Training completed!")
        print0(f"Total tokens: {self.state.n_tokens:,}")
        print0(f"Total time: {self.state.total_training_time/60:.1f} minutes")
        print0("=" * 70)
        
        # Final checkpoint
        self.save_checkpoint(tag="final")

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        output = self._compiled_model(x, y)
        if hasattr(output, "loss"):
            return output.loss
        else:
            B, T, V = output.shape
            return torch.nn.functional.cross_entropy(
                output.view(B * T, V),
                y.view(B * T),
            )

    def _apply_optim_hparam_scheduler(self, step: int):
        """Update learning rate according to schedule."""

        lrm = self.lr_schedule(step)
        muon_momentum = self.muon_momentum_schedule(step)
        weight_decay = self.weight_decay_schedule(step)
        
        self.optimizer.update_hyperparams(lrm=lrm, muon_momentum=muon_momentum, weight_decay=weight_decay)
        return lrm, muon_momentum, weight_decay

    def save_checkpoint(self):
        """
        Save checkpoint with model, optimizer, and state.
        
        Args:
            tag: Identifier for this checkpoint (e.g., "latest", "best", "step_1000")
        """
        if not self.config.dist_info.get("RANK", 0) == 0:
            return # Only master process saves checkpoints
        
        self.ckpt_manager.save(step=self.state.step, model=self.model, optimizer=self.optimizer, scaler=self.scaler, trainer_state=self.state)

    @classmethod
    def from_checkpoint(
        cls,
        model_name: str,
        model_run: str,
        step: Union[int, str] = "latest",
        cache_dir: Optional[Union[str, Path]] = None,
        board: Optional[Board] = None,
        dist_info: Optional[Dict] = None,
    ):
        ckpt_manager = CheckpointManager(
            model_name=model_name,
            model_run=model_run,
            source="base", # TODO: make this dynamic based on training type
            dist_info=dist_info or get_dist_info(),
            mode="shard", # TODO: make this dynamic based on how checkpoints were saved
            cache_dir=cache_dir,
        )
        model, tokenizer, ckpt_data, trainer_config = ckpt_manager.load(step=step, phase="train")
        opt = model.build_optimizer(trainer_config)

        trainer = cls(
            model=model,
            tokenizer=tokenizer,
            train_loader=None,
            val_loader=None,
            config=trainer_config,
            optimizer=opt,
            board=board,
            lr_schedule=None,
            muon_momentum_schedule=None, 
            weight_decay_schedule=None,
            resume_state=ckpt_data.trainer_state,
            checkpoint_manager=ckpt_manager,
        )
        log0(f"Resumed trainer from checkpoint at step {trainer.state.step} with best eval bpb {trainer.ckpt_state.best_eval_value:.6f} at step {trainer.ckpt_state.best_eval_step}.", logger=logger)
        log0(f"Trainer instance created has no 'train_loader' or 'val_loader'. Please set these manually before calling 'trainer.train()'. Maybe consider using 'trainer.state.dataloader_state'", level="warning", logger=logger)

        return trainer