"""
Trainer: Main training loop with checkpointing, evaluation, and metrics tracking.

Key responsibilities:
- Training loop with gradient accumulation
- Checkpoint save/load
- Validation and evaluation
- Metrics tracking and logging
- Device management and memory optimization
"""

from typing import Optional, Callable, Union, Dict
from pathlib import Path
from contextlib import ExitStack, contextmanager, nullcontext

import numpy as np

import time
import gc
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
from gpt_lab.utils.common import print0
from gpt_lab.utils.logging import log_error, log_critical, log0, log_dict
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
from gpt_lab.model.checkpoint import CheckpointManager, make_default_run_name

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
            model_run = make_default_run_name(depth, model_name, self.config.dist_info)
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
        
        # Device and mixed precision. get_dist_info() stores a torch.dtype, but
        # accept legacy string values when resuming older checkpoints.
        self.device_type = config.dist_info.get("DEVICE_TYPE", "cpu")
        self.device = torch.device(config.dist_info.get("DEVICE", "cpu"))
        raw_dtype = self.config.dist_info["compute_dtype"]
        self.dtype = _DTYPE_MAP.get(raw_dtype, raw_dtype)
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported compute dtype: {raw_dtype!r}")
        self.use_amp = self.dtype in (torch.float16, torch.bfloat16)

        def amp_context():
            if self.use_amp:
                return autocast(
                    device_type=self.device_type,
                    dtype=self.dtype,
                )
            return nullcontext()

        @contextmanager
        def val_context(model):
            with ExitStack() as stack:
                stack.enter_context(amp_context())
                if self.config.fp8:
                    from gpt_lab.model.fp8 import DisableFP8
                    stack.enter_context(DisableFP8(model))
                yield model

        self.train_context = amp_context
        self.val_context = val_context

        self.scaler = scaler
        if self.scaler is None and self.dtype == torch.float16:
            if self.device_type != "cuda":
                raise ValueError("float16 training is only supported on CUDA")
            self.scaler = GradScaler(self.device_type)

        actual_batch_size = getattr(self.train_loader, "B", None)
        if (
            actual_batch_size is not None
            and actual_batch_size != self.config.device_batch_size
        ):
            raise ValueError(
                "TrainerConfig.device_batch_size does not match the training "
                f"loader: {self.config.device_batch_size} != {actual_batch_size}"
            )
        
        # LR and other schedules
        self.lr_schedule = lr_schedule or config.lr_multiplier_schedule
        self.muon_momentum_schedule = muon_momentum_schedule or config.muon_momentum_schedule
        self.weight_decay_schedule = weight_decay_schedule or config.weight_decay_schedule

        self._get_sync_fn()

    def _get_sync_fn(self):
        """Set up device synchronization function."""
        if self.device_type == "cuda":
            self.synchronize = torch.cuda.synchronize
            self.get_max_memory = torch.cuda.max_memory_allocated
        else:
            self.synchronize = lambda: None
            self.get_max_memory = lambda: 0

    def log_gradients(self):
        """Log accumulated pre-reduction gradient statistics.

        Statistics are aggregated across rank-local accumulated gradients with
        one small collective. This avoids a hook on every parameter for every
        microbatch while still reporting all ranks, rather than only rank zero's
        final microbatch.
        """
        names = []
        stats = []
        numels = []
        for name, param in self.model.named_parameters():
            if not name.endswith("weight") or param.grad is None:
                continue
            grad = param.grad.detach().float()
            names.append(name.replace('.', '/').removesuffix('/weight'))
            stats.append(torch.stack((
                grad.square().sum(),
                grad.sum(),
                grad.abs().sum(),
            )))
            numels.append(param.numel())

        if not stats:
            return

        stats_tensor = torch.stack(stats)
        world_size = self.config.dist_info.get("WORLD_SIZE", 1)
        if self.config.dist_info.get("IS_DDP_INITIALIZED", False):
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        logs = {}
        if self.config.dist_info.get("RANK", 0) == 0:
            # One device synchronization for the complete statistics table,
            # rather than one .item() synchronization per metric.
            stats_cpu = stats_tensor.detach().cpu()
            for name, row, numel in zip(names, stats_cpu, numels):
                denominator = numel * world_size
                logs[f"grad_rms/{name}"] = (
                    row[0] / denominator
                ).sqrt().item()
                logs[f"grad_mean/{name}"] = (row[1] / denominator).item()
                logs[f"grad_abs_mean/{name}"] = (
                    row[2] / denominator
                ).item()
            self.board.log(logs, step=self.state.step)

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

        world_size = self.config.dist_info.get("WORLD_SIZE", 1) if self.config.dist_info else 1
        
        # Compile model if using PyTorch 2.0+
        if parse(torch.__version__) >= parse("2.0"):
            self._compiled_model = torch.compile(self.model, dynamic=False)
        else:
            self._compiled_model = self.model

        train_iter = iter(self.train_loader)

        rank = self.config.dist_info["RANK"]

        print(f"[rank={rank}] before barrier", flush=True)
        if self.config.dist_info["IS_DDP_INITIALIZED"]:
            dist.barrier()

        print(f"[rank={rank}] after barrier / before prefetch", flush=True)

        x, y, dataloader_state = next(train_iter) # prefetch

        print(f"[rank={rank}] after prefetch", flush=True)
        log_dict("First dataloader_state", dataloader_state.__dict__, logger=logger)

        # Prepare for training
        self._compiled_model.train()
        smooth_loss = self.state.smooth_train_loss
        ema_beta = 0.9
        
        total_dt = []  # For ETA calculation
        
        while step < n_steps:
            self.state.step = step
            last_step = (step == n_steps - 1)
            should_log_step = (
                (self.config.log_every == -1 and last_step)
                or (
                    self.config.log_every > 0
                    and step % self.config.log_every == 0
                )
            )
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
                tokens_per_eval_step = (
                    self.config.device_batch_size
                    * self.model.config.max_context
                    * self.config.dist_info["WORLD_SIZE"]
                )
                eval_steps = max(
                    1,
                    math.ceil(self.config.n_bpb_tokens / tokens_per_eval_step),
                )
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
                
                board_dict = {
                    "eval/loss": val_res['loss'],
                    "eval/bpb": val_res['bpb'],
                    "eval/best_bpb": self.ckpt_state.best_eval_value,
                    "eval/step_time_ms": dt_bpb_eval * 1000,  # Convert to milliseconds
                }
                self.board.log(board_dict, step=step)
                self.eval_metrics.append(board_dict, step=step)
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
                board_dict = {
                    "core/core": results["core/core"],
                    "core/accuracy": results["core/accuracy"],
                    "core/max_per_task": results["core/max_per_task"],
                    "core/step_time_ms": results.get("core/step_time_ms", 0),
                }
                for task_label, task_results in results.get("all_core_results", {}).items():
                    for task_metric, task_value in task_results.items():
                        board_dict[f"core/{task_label}/{task_metric}"] = task_value
                if (
                    (self.ckpt_state.best_core_value is None) or 
                    (results["core/core"] > self.ckpt_state.best_core_value)
                ):
                    self.ckpt_state.best_core_value = results["core/core"]
                    self.ckpt_state.best_core_step = step
                self.board.log(board_dict, step=step)
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
                # print0(self.tokenizer.decode(samples[0]))
                self.model.train()

            # ================================================================
            # Training step with gradient accumulation
            # ================================================================

            loss_accum = torch.zeros((), device=self.device, dtype=torch.float32)

            eff_global_tokens = 0

            self.synchronize()
            step_start_time = time.perf_counter()
            for _ in range(n_acc_steps):
                eff_global_tokens += x.numel() * world_size

                self.state.train_loader_state = dataloader_state
                with self.train_context():
                    loss = self._compute_loss(x, y)
                loss = loss / n_acc_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum.add_(loss.detach().float())

                if torch.isnan(loss_accum) or torch.isinf(loss_accum):
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

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                if self.config.dist_info["IS_DDP_INITIALIZED"]:
                    for g in self.scaler._found_inf_per_device(self.optimizer).values():
                        dist.all_reduce(g, op=dist.ReduceOp.MAX)

            if self.config.monitor_grad_norms and should_log_step:
                self.log_gradients()

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self._compiled_model.zero_grad(set_to_none=True)

            self.synchronize()
            step_end_time = time.perf_counter()
            
            # ================================================================
            # Logging and metrics
            # ================================================================
            
            step_dt_tensor = torch.tensor(step_end_time - step_start_time, device=self.device)

            if self.config.dist_info["IS_DDP_INITIALIZED"]:
                dist.all_reduce(step_dt_tensor, op=dist.ReduceOp.MAX)

            step_dt = step_dt_tensor.item()

            loss_accum = loss_accum.item()
            
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_accum
            debiased_smooth_loss = smooth_loss / (1 - ema_beta ** (step + 1))
            
            # Calculate throughput
            if eff_global_tokens != total_batch_size:
                log0(f"eff_global_tokens != total_batch_size, got {eff_global_tokens=} and {total_batch_size=}", logger=logger, level="warning")

            tokens_per_sec = eff_global_tokens / step_dt

            step_flops_per_sec = n_flops_per_token * tokens_per_sec
            mfu = step_flops_per_sec / self.config.dist_info["gpu_peak_flops"] * 100 / world_size
            
            total_dt.append(step_dt)
            self.state.n_tokens += eff_global_tokens
            self.state.smooth_train_loss = smooth_loss
            self.state.total_training_time += step_dt
            
            # Log every log_every steps
            if should_log_step:
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
                board_dict = {
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

                self.metrics.append(board_dict, step)
                self.board.log(board_dict, step=step)
            
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
        self.save_checkpoint()

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
        self.ckpt_manager.save(
            step=self.state.step,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            trainer_state=self.state,
            checkpoint_state=self.ckpt_state,
        )

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
        dist_info = dist_info or get_dist_info()
        ckpt_manager = CheckpointManager(
            model_name=model_name,
            model_run=model_run,
            source="base", # TODO: make this dynamic based on training type
            dist_info=dist_info,
            mode="shard" if dist_info.get("IS_DDP_INITIALIZED", False) else "ddp",
            model_cachedir=cache_dir,
        )
        model, tokenizer, ckpt_data, trainer_config = ckpt_manager.load(step=step, phase="train")
        opt = model.build_optimizer(trainer_config)
        if ckpt_data.optimizer_state is not None:
            opt.load_state_dict(ckpt_data.optimizer_state)

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
            best_state=ckpt_data.checkpoint_state,
            checkpoint_manager=ckpt_manager,
        )
        if ckpt_data.scaler_state is not None:
            if trainer.scaler is None:
                raise ValueError(
                    "Checkpoint contains GradScaler state but the resumed "
                    "trainer did not create a scaler."
                )
            trainer.scaler.load_state_dict(ckpt_data.scaler_state)
        log0(f"Resumed trainer from checkpoint at step {trainer.state.step} with best eval bpb {trainer.ckpt_state.best_eval_value:.6f} at step {trainer.ckpt_state.best_eval_step}.", logger=logger)
        log0(f"Trainer instance created has no 'train_loader' or 'val_loader'. Please set these manually before calling 'trainer.train()'. Maybe consider using 'trainer.state.dataloader_state'", level="warning", logger=logger)

        return trainer
