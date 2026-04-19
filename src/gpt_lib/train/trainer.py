"""
Trainer: Main training loop with checkpointing, evaluation, and metrics tracking.

Key responsibilities:
- Training loop with gradient accumulation
- Checkpoint save/load
- Validation and evaluation
- Metrics tracking and logging
- Device management and memory optimization
"""

from typing import Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
import pickle
import gc
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler

from gpt_lib.utils.schemas import TrainingConfig, TrainingMetrics, EvalMetrics, COREMetrics
from gpt_lib.utils.board import Board
from gpt_lib.utils.common import print0, print0_dict
from gpt_lib.evaluate.bpb import compute_bpb
from gpt_lib.evaluate.core import evaluate_core
from gpt_lib.model.gpt import GPTModel
from gpt_lib.utils.default import CACHE_DIR, MODELS_FOLDER


# ============================================================================
# Configuration & State
# ============================================================================

@dataclass
class TrainerState:
    """Complete state for trainer resumption."""
    global_step: int = 0
    global_tokens: int = 0
    num_epochs: int = 0
    best_val_loss: float = float('inf')
    total_training_time: float = 0.0
    smooth_train_loss: float = 0.0
    
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
    ```python
    Example:
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
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        board: Optional[Board] = None,
        checkpoint_dir: Optional[Path] = None,
        lr_schedule: Optional[Callable] = None,
        muon_momentum_schedule: Optional[Callable] = None,
        weight_decay_schedule: Optional[Callable] = None
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
            lr_schedule: Learning rate schedule function (step -> lr multiplier). Default: TrainingConfig.lr_schedule
            muon_momentum_schedule: Momentum schedule for Muon optimizer (step -> momentum multiplier). Default: TrainingConfig.muon_momentum_schedule
            weight_decay_schedule: Weight decay schedule for Muon optimizer (step -> weight decay multiplier). Default: TrainingConfig.weight_decay_schedule
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Config
        if config is None:
            config = TrainingConfig(n_steps=1)
            warnings.warn(
                "No training config provided. Using default config with 1 step. "
                "Please provide a TrainingConfig instance."
            )
        self.config = config
        
        self.checkpoint_dir = checkpoint_dir or (MODELS_FOLDER / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.state = TrainerState()
        self.metrics = TrainingMetrics()
        self.eval_metrics = EvalMetrics()
        self.core_metrics = COREMetrics()
        
        # Mixed precision
        self.use_amp = config.use_amp
        def amp_context():
            if self.use_amp:
                return autocast(device_type=self.config.dist_info["DEVICE_TYPE"], dtype=torch.bfloat16)
            else:
                return DummyContext()
        def disable_fp8_context():
            if self.config.fp8:
                # TODO: this is a placeholder impl.
                from gpt_lib.model.fp8 import DisableFP8
                return DisableFP8
            else:
                return DummyContext
        self.train_context = amp_context()
        self.val_context = disable_fp8_context()
        self.scaler = GradScaler() if self.use_amp else None
        
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
        """Add backward hooks to model parameters to monitor gradient norms."""
        if self.config.monitor_grad_norms and not getattr(self, "_hooked", False):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.register_hook(
                        lambda grad, name=name: self.board.log({
                            f"grad_norm/{name.replace('.', '/').replace('/weight', '')}": grad.norm().mean().item()}, 
                            step=self.state.global_step))
        self._hooked = True # avoid hooking multiple times

    def _rm_model_hooks(self):
        """Remove all hooks from model parameters."""
        for param in self.model.parameters():
            param._backward_hooks = {}
        self._hooked = False

    def train(self):
        """
        Main training loop.
        
        Performs gradient accumulation, validation, checkpointing, and logging.
        """
        print0("=" * 70)
        print0(f"Starting training for {self.config.n_steps:,} steps -- {self.config.total_batch_size:,} tokens per step.")
        print0("=" * 70)
        
        # Extracte main constants from config
        step = self.state.global_step
        n_steps = self.config.n_steps
        n_flops_per_token = self.config.n_flops_per_token
        total_batch_size = self.config.total_batch_size
        n_acc_steps = self.config.n_acc_steps
        
        # Compile model if using PyTorch 2.0+
        if torch.__version__ >= "2.0":
            self._model = torch.compile(self.model, dynamic=False)
        _model = self._model

        train_iter = iter(self.train_loader)
        x, y, dataloader_state = next(train_iter) # prefetch

        # Prepare for training
        self._model.train()
        smooth_loss = self.state.smooth_train_loss
        ema_beta = 0.9
        
        self.state.total_training_time = 0.0
        total_dt = []  # For ETA calculation
        
        while step < n_steps:
            self.state.global_step = step
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
                _model.eval()
                eval_steps = self.config.n_bpb_tokens // (self.config.device_batch_size * self.model.config.max_context * self.config.dist_info["WORLD_SIZE"])
                with self.val_context(_model):
                    val_res = compute_bpb(
                        _model, 
                        self.val_loader, 
                        eval_steps,
                        dist_info=self.config.dist_info,
                        token_bytes=self.tokenizer.token_bytes
                    )
                print0(f"Step {step:05d} | Validation bpb: {val_res['bpb']:.6f} | Validation loss: {val_res['loss']:.6f}")
                
                dt_bpb_eval = start_bpb_eval - time.time()

                if val_res['bpb'] < self.state.best_val_loss:
                    self.state.best_val_loss = val_res['bpb']
                    self.save_checkpoint(tag="best")
                
                log_dict = {
                    "eval/loss": val_res['loss'],
                    "eval/bpb": val_res['bpb'],
                    "eval/best_bpb": self.state.best_val_loss,
                    "eval/step_time_ms": dt_bpb_eval * 1000,  # Convert to milliseconds
                }
                self.board.log(log_dict, step=step)
                self.eval_metrics.append(log_dict, step)
                _model.train()

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
                print0(f"Step {step:05d}/{n_steps:05d} | CORE: {results['core/core']:.4f} | Accuracy: {results['core/accuracy']:.4f} | Max per task: {int(results['core/max_per_task'])} | Step time: {results.get('core/step_time_ms', 0):.2f}ms | Max throughput: {max_throughput:,.0f} tok/s")
                log_dict = {
                    "core/core": results["core/core"],
                    "core/accuracy": results["core/accuracy"],
                    "core/max_per_task": results["core/max_per_task"],
                    "core/step_time_ms": results.get("core/step_time_ms", 0),
                }
                for task_label, task_results in results.get("all_core_results", {}).items():
                    for task_metric, task_value in task_results.items():
                        log_dict[f"core/{task_label}/{task_metric}"] = task_value
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
                engine = GPTModel(self.model, self.tokenizer) # use orig_model to avoid recompilation
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

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.detach().item()
                x, y, dataloader_state = next(train_iter)

            lrm, muon_momentum, weight_decay = self._apply_optim_hparam_scheduler(step)

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                if self.config.dist_info["IS_DDP_INITIALIZED"]:
                    for g in self.scaler._found_inf__per_device:
                        dist.all_reduce(g, op=dist.ReduceOp.MAX)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            _model.zero_grad(set_to_none=True)

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
            
            total_dt.append(step_dt)
            self.state.global_tokens += eff_global_tokens
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
                
                print0(
                    f"Step {step:05d}/{n_steps:05d} ({pct_done:5.1f}%) | "
                    f"loss: {debiased_smooth_loss:.6f} | "
                    f"lrm: {lrm:.2e} | "
                    f"dt: {step_dt*1000:.2f}ms | "
                    f"tok/s: {tokens_per_sec:,.0f}"
                    f"{eta_str}"
                )
                log_dict = {
                    "epochs": self.state.num_epochs,
                    "train/loss": debiased_smooth_loss,
                    "train/raw_loss": loss_accum,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_time_ms": step_dt * 1000,
                    "train/total_tokens": self.state.global_tokens,
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
                self.save_checkpoint(tag=f"step_{step}")
            
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

        del self._model  # Clean up compiled model if it exists
        
        print0("=" * 70)
        print0(f"Training completed!")
        print0(f"Total tokens: {self.state.global_tokens:,}")
        print0(f"Total time: {self.state.total_training_time/60:.1f} minutes")
        print0("=" * 70)
        
        # Final checkpoint
        self.save_checkpoint(tag="final")

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        output = self._model(x, y)
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

    def save_checkpoint(self, tag: str = "latest"):
        """
        Save checkpoint with model, optimizer, and state.
        
        Args:
            tag: Identifier for this checkpoint (e.g., "latest", "best", "step_1000")
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{tag}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), checkpoint_path / "scaler.pt")
        
        with open(checkpoint_path / "state.pkl", "wb") as f:
            pickle.dump(asdict(self.state), f)
        
        with open(checkpoint_path / "metrics.pkl", "wb") as f:
            pickle.dump(self.metrics.model_dump(), f)
        
        with open(checkpoint_path / "eval_metrics.pkl", "wb") as f:
            pickle.dump(self.eval_metrics.model_dump(), f)
        
        with open(checkpoint_path / "core_metrics.pkl", "wb") as f:
            pickle.dump(self.core_metrics.model_dump(), f)

        if tag == "latest":
            print0(f"Saved checkpoint to {checkpoint_path!r}.")

    def load_checkpoint(self, checkpoint_path: Union[Path, str], device: Optional[torch.device] = None):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load checkpoint onto (defaults to trainer's device)
        """
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print0(f"Checkpoint path {checkpoint_path!r} does not exist. Starting fresh.")
            return
        device = device or self.device
        
        self.model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location=device))
        self.optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location=device))
        
        if self.scaler is not None and (checkpoint_path / "scaler.pt").exists():
            self.scaler.load_state_dict(torch.load(checkpoint_path / "scaler.pt", map_location=device))
        
        if (checkpoint_path / "state.pkl").exists():
            with open(checkpoint_path / "state.pkl", "rb") as f:
                state_dict = pickle.load(f)
            self.state = TrainerState(**state_dict)
            print0_dict(asdict(self.state), title=f"Loaded checkpoint from {checkpoint_path!r}.")
        
        if (checkpoint_path / "metrics.pkl").exists():
            with open(checkpoint_path / "metrics.pkl", "rb") as f:
                metrics_dict = pickle.load(f)
            self.metrics = TrainingMetrics(**metrics_dict)
        
        if (checkpoint_path / "eval_metrics.pkl").exists():
            with open(checkpoint_path / "eval_metrics.pkl", "rb") as f:
                eval_metrics_dict = pickle.load(f)
            self.eval_metrics = EvalMetrics(**eval_metrics_dict)

        if (checkpoint_path / "core_metrics.pkl").exists():
            with open(checkpoint_path / "core_metrics.pkl", "rb") as f:
                core_metrics_dict = pickle.load(f)
            self.core_metrics = COREMetrics(**core_metrics_dict)
        
        # TODO: make HUGE warning if loaded checkpoint's git commit does not match current code's git commit
        