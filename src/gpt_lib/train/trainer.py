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
from torch.amp import autocast, GradScaler

from gpt_lib.utils.schemas import TrainingConfig, TrainingMetrics
from gpt_lib.utils.board import Board
from gpt_lib.utils.common import print0, print0_dict
from gpt_lib.evaluate.bpb import compute_bpb
from gpt_lib.evaluate.core import evaluate_core
from gpt_lib.model.gpt import GPTModel


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

@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    steps: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    epochs: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    learning_rate: list = field(default_factory=list)
    throughput_tokens_per_sec: list = field(default_factory=list)
    time_per_step_ms: list = field(default_factory=list)


class DummyContext:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self, *args, **kwargs): pass
    def __exit__(self, *args, **kwargs): pass
    

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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        
        # Config
        if config is None:
            config = TrainingConfig(n_steps=1)
            warnings.warn(
                "No training config provided. Using default config with 1 step. "
                "Please provide a TrainingConfig instance."
            )
        self.config = config
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
        
        # Mixed precision
        self.use_amp = config.use_amp
        def amp_context():
            if self.use_amp:
                return autocast(device=self.device, dtype=torch.bfloat16)
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
        self.lr_schedule = lr_schedule or config.lr_schedule
        self.muon_momentum_schedule = muon_momentum_schedule or config.muon_momentum_schedule
        self.weight_decay_schedule = weight_decay_schedule or config.weight_decay_schedule

        # Device config
        self.device_type = config.dist_info.get("device_type", "cpu")
        self.device = torch.device(config.dist_info.get("device", "cpu"))
        self._get_sync_fn()

    def _get_sync_fn(self):
        """Set up device synchronization function."""
        if self.device_type == "cuda":
            self.synchronize = torch.cuda.synchronize
            self.get_max_memory = torch.cuda.max_memory_allocated
        else:
            self.synchronize = lambda: None
            self.get_max_memory = lambda: 0

    def train(self):
        """
        Main training loop.
        
        Performs gradient accumulation, validation, checkpointing, and logging.
        """
        print0("=" * 70)
        print0(f"Starting training for {self.config.n_steps:,} steps")
        print0("=" * 70)
        
        # Extracte main constants from config
        step = self.state.global_step
        n_steps = self.config.n_steps
        n_flops_per_token = self.config.n_flops_per_token
        total_batch_size = self.config.total_batch_size
        
        # Compile model if using PyTorch 2.0+
        if torch.__version__ >= "2.0":
            self._model = torch.compile(self.model, dynamic=False)
        _model = self._model
        train_iter = next(self.train_loader)

        # Prepare for training
        self._model.train()
        smooth_loss = self.state.smooth_train_loss
        ema_beta = 0.9
        
        self.state.total_training_time = 0.0
        step_times = []  # For ETA calculation
        
        while step < n_steps:
            last_step = (step == n_steps - 1)
            flops_so_far = n_flops_per_token * total_batch_size * step
            self.synchronize()
            step_start_time = time.time()
            
            # ================================================================
            # Validation on 'val_loader' every 'eval_every' steps
            # ================================================================
            if (
                self.config.eval_bpb_every > 0 and
                (step == 0 or step % self.config.eval_bpb_every == 0)
            ):
                _model.eval()
                eval_steps = self.config.n_bpb_tokens // (self.config.device_batch_size * self.model.config.max_context * self.config.dist_info["world_size"])
                with self.val_context(_model):
                    val_res = compute_bpb(
                        _model, 
                        self.val_loader, 
                        eval_steps,
                        dist_info=self.config.dist_info,
                        token_bytes=self.tokenizer.token_bytes
                    )
                print0(f"Step {step:05d} | Validation bpb: {val_res['bpb']:.6f} | Validation loss: {val_res['loss']:.6f}")
                
                if val_res['bpb'] < self.state.best_val_loss:
                    self.state.best_val_loss = val_res['bpb']
                    self.save_checkpoint(tag="best")
                
                self.board.log({
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": self.state.total_training_time,
                    "val/loss": val_res['loss'],
                    "val/bpb": val_res['bpb'],
                    "best_val_bpb": self.state.best_val_loss,
                })
                _model.train()

            # ================================================================
            # Validation on CORE metric every 'core_eval_every' steps
            # ================================================================
            
            results = {}
            if (
                self.config.eval_core_every > 0 and 
                (last_step or (step > 0 and step % self.config.eval_core_every == 0))
            ):
                self.model.eval()
                with self.val_context(self.model):
                    results = evaluate_core(
                        self.model,
                        self.tokenizer,
                        self.device,
                        max_per_task=self.config.n_core_tokens,
                    )
                print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
                self.board.log({
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                })
                self.model.train()
            
            # ================================================================
            # Sample some outputs every 'sample_every' steps
            # ================================================================

            if (
                self.config.sample_every > 0 and 
                self.config.dist_info["rank"] == 0 and 
                (last_step or (step > 0 and step % self.config.sample_every == 0))
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
                for prompt in prompts:
                    tokens = self.tokenizer(prompt, prepend_bos=True)
                    with self.val_context(self.model):
                        sample = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                    print0(self.tokenizer.decode(sample[0]))
                self.model.train()

            # ================================================================
            # Training step with gradient accumulation
            # ================================================================
            loss_accum = 0.0
            grad_accum_steps = self.config.gradient_accumulation_steps
            
            for _ in range(grad_accum_steps):
                inputs, targets, dataloader_state = next(train_iter)
                self.state.train_loader_state = dataloader_state
                
                with self.train_context():
                    loss = self._compute_loss(inputs, targets)
                loss = loss / grad_accum_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                loss_accum += loss.detach().item()

            self._update_lr(step)
            
            # Optimizer step
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # ================================================================
            # Logging and metrics
            # ================================================================
            self.synchronize()
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_accum
            debiased_smooth_loss = smooth_loss / (1 - ema_beta ** (step + 1))
            
            # Calculate throughput
            total_tokens = (
                inputs.numel() * grad_accum_steps *
                (self.config.dist_info.get("world_size", 1) if self.config.dist_info else 1)
            )
            tokens_per_sec = total_tokens / step_time
            
            step_times.append(step_time)
            self.state.global_step = step
            self.state.global_tokens += total_tokens
            self.state.smooth_train_loss = smooth_loss
            self.state.total_training_time += step_time
            
            # Log every log_every steps
            if step % self.config.log_every == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pct_done = 100 * step / n_steps
                
                # ETA calculation
                if len(step_times) > 10:
                    avg_step_time = sum(step_times[-10:]) / 10
                    eta_seconds = (n_steps - step) * avg_step_time
                    eta_str = f" | ETA: {eta_seconds/60:.1f}m"
                else:
                    eta_str = ""
                
                print0(
                    f"Step {step:05d}/{n_steps:05d} ({pct_done:5.1f}%) | "
                    f"loss: {debiased_smooth_loss:.6f} | "
                    f"lr: {current_lr:.2e} | "
                    f"dt: {step_time*1000:.2f}ms | "
                    f"tok/s: {tokens_per_sec:,.0f}"
                    f"{eta_str}"
                )
                
                log_dict = {
                    "step": step,
                    "loss": debiased_smooth_loss,
                    "learning_rate": current_lr,
                    "tokens_per_sec": tokens_per_sec,
                    "step_time_ms": step_time * 1000,
                    "epoch": self.state.num_epochs,
                    "total_tokens": self.state.global_tokens,
                }
                
                self.metrics.steps.append(step)
                self.metrics.tokens.append(self.state.global_tokens)
                self.metrics.epochs.append(self.state.num_epochs)
                self.metrics.train_loss.append(debiased_smooth_loss)
                self.metrics.learning_rate.append(current_lr)
                self.metrics.throughput_tokens_per_sec.append(tokens_per_sec)
                self.metrics.time_per_step_ms.append(step_time * 1000)
                
                self.board.log(log_dict)
            
            # ================================================================
            # Checkpointing
            # ================================================================
            if (self.config.save_every > 0 and step > 0 and step % self.config.save_every == 0):
                self.save_checkpoint(tag=f"step_{step}")
            
            if sum(step_times) > self.config.target_time * 60:
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

    def _compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        output = self._model(inputs, targets)
        if hasattr(output, "loss"):
            return output.loss
        else:
            B, T, V = output.shape
            return torch.nn.functional.cross_entropy(
                output.view(B * T, V),
                targets.view(B * T),
            )

    def _update_lr(self, step: int):
        """Update learning rate according to schedule."""
        if not hasattr(self.config, 'get_lr_schedule'):
            return
        
        lr = self.lr_schedule(step)
        muon_momentum = self.muon_momentum_schedule(step)
        weight_decay = self.weight_decay_schedule(step)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lr
            if param_group["type"] == "muon":
                param_group["momentum"] = muon_momentum
                param_group["weight_decay"] = weight_decay

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
            pickle.dump(asdict(self.metrics), f)
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
        