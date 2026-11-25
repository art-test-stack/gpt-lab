"""Study evaluation and dashboard diagnostics for the Muon arms."""

import json
import math
from collections import deque
from contextlib import contextmanager, nullcontext
from statistics import median
from time import perf_counter

from gpt_lab.evaluate.bpb import compute_bpb
from gpt_lab.train.trainer import Trainer
from gpt_lab.utils.common import print0


STUDY_EVAL_FILE = "study-eval.enabled"


class ExperimentTrainer(Trainer):
    def __init__(self, *, study_eval=None, monitor_qk_norms=False, guard_mfu=False, **kwargs):
        super().__init__(**kwargs)
        self.monitor_qk_norms = monitor_qk_norms
        self.mfu_window = deque(maxlen=10) if guard_mfu else None

        run_dir = self.ckpt_manager.source_dir.parent
        policy = run_dir / STUDY_EVAL_FILE
        self.study_eval = study_eval is not None and (study_eval or policy.is_file())
        self.endpoint_bpb_path = run_dir / "climbmix-full.json"
        self.core_updates = ()
        interval = self.config.eval_core_every
        if study_eval is not None and (interval == -1 or interval > 0):
            if self.config.n_core_tokens <= 0:
                raise ValueError("Training CORE requires a positive --n-core-tokens for the study")
            if self.study_eval:
                horizon = self.config.n_steps
                self.core_updates = tuple((quarter * horizon + 3) // 4 for quarter in (1, 2, 3))
                # Let the base loop dispatch CORE; the override filters to the saved horizon's quarters.
                self.config = self.config.model_copy(update={"eval_core_every": 1})
        if study_eval and self.config.comp_info.rank == 0:
            policy.touch()

    def train(self):
        if not self.study_eval or self.state.completed_steps < self.config.n_steps:
            super().train()
            checkpoint = self.ckpt_manager.source_dir / f"checkpoint_step_{self.state.step:06d}"
            print0(f"Final checkpoint: {checkpoint.resolve()}")

        interval = self.config.eval_bpb_every
        if (
            self.study_eval and (interval == -1 or interval > 0)
            and self.state.completed_steps == self.config.n_steps
            and not self.endpoint_bpb_path.is_file()
        ):
            self._evaluate_full_bpb()

    def evaluate_bpb(self):
        if self.study_eval:
            interval = self.config.eval_bpb_every
            completed = self.state.completed_steps
            if interval <= 0 or completed == self.config.n_steps or completed % interval:
                return
        with self._evaluation_timing("bpb"):
            super().evaluate_bpb()

    def evaluate_core(self):
        completed = self.state.completed_steps
        if not self.study_eval or (completed < self.config.n_steps and completed in self.core_updates):
            with self._evaluation_timing("core"):
                super().evaluate_core()

    def _evaluate_full_bpb(self):
        model = self.model
        training = model.training
        model.eval()
        try:
            with self._evaluation_timing("bpb"), self.val_context(model):
                result = compute_bpb(
                    model, self.val_loader(include_partial=True), None,
                    comp_info=self.config.comp_info, token_bytes=self.tokenizer.token_bytes,
                )
        finally:
            model.train(training)

        step = self.state.completed_steps
        metrics = {"eval/bpb": result["bpb"], "eval/loss": result["loss"], "eval/full_endpoint_update": step}
        print0(f"Step {step:05d} | Full validation bpb: {result['bpb']:.6f} | Validation loss: {result['loss']:.6f}")
        if self.config.comp_info.rank == 0:
            tmp = self.endpoint_bpb_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(metrics, allow_nan=False) + "\n")
                tmp.replace(self.endpoint_bpb_path)
            finally:
                tmp.unlink(missing_ok=True)
        self.board.log(metrics, step=step)

    def _evaluation_timing(self, phase):
        return nullcontext()

    def save_checkpoint(self):
        if self.study_eval and self.state.completed_steps == self.config.n_steps and self.config.comp_info.rank == 0:
            self.endpoint_bpb_path.unlink(missing_ok=True)
        return super().save_checkpoint()

    def update_metrics(self, step_metrics, total_dt):
        metrics = super().update_metrics(step_metrics, total_dt)
        completed = self.state.step + 1
        window = self.mfu_window
        if window is None or completed <= 10:
            return metrics

        window.append(metrics["train/mfu"])
        if completed % 10 == 0 and len(window) == 10:
            median_mfu = median(window)
            metrics["train/mfu_window_median"] = median_mfu
            if not math.isfinite(median_mfu) or median_mfu <= 40:
                metrics["train/mfu_guard_tripped"] = 1
                metrics["train/completed_steps"] = completed
                self.board.log(metrics, step=self.state.step)
                raise RuntimeError(
                    f"MFU guard stopped after update {completed}: median MFU over updates "
                    f"{completed - 9}–{completed} was {median_mfu:.2f}%, below the required >40%."
                )
        return metrics

    def optimizer_step(self, should_log_step):
        if not self.monitor_qk_norms:
            return super().optimizer_step(should_log_step)

        completed = self.state.step + 1
        endpoint = self.config.n_steps
        if self.config.stop_after_step is not None:
            endpoint = min(endpoint, self.config.stop_after_step)

        strategy = self.optimizer.backend.strategies["muon"]
        strategy.monitor = completed in (1, endpoint) or completed % 10 == 0
        super().optimizer_step(should_log_step)
        if strategy.monitor:
            metrics = strategy.collect_norms()
            if metrics is not None:
                metrics["qk/completed_update"] = completed
                self.board.log(metrics, step=self.state.step)


class BenchmarkTrainer(ExperimentTrainer):
    def __init__(self, *, worker_start, **kwargs):
        super().__init__(**kwargs)
        self.worker_start = worker_start

    def _timing(self, phase, completed, elapsed):
        print("rope-muon-timing " + json.dumps({
            "rank": self.config.comp_info.rank,
            "completed_update": completed,
            "phase": phase,
            "seconds": elapsed,
        }), flush=True)

    @contextmanager
    def _measure(self, phase, completed):
        self.synchronize()
        start = perf_counter()
        yield
        self.synchronize()
        self._timing(phase, completed, perf_counter() - start)

    def train_step(self, train_iter, batch, should_log_step):
        next_batch, metrics = super().train_step(train_iter, batch, should_log_step)
        completed = self.state.step + 1
        if completed == 1:
            self._timing("worker_start_through_first_update", completed, perf_counter() - self.worker_start)
        self._timing("complete_update", completed, metrics["step_time"])
        return next_batch, metrics

    def optimizer_step(self, should_log_step):
        with self._measure("optimizer", self.state.step + 1):
            super().optimizer_step(should_log_step)

    def _evaluation_timing(self, phase):
        return self._measure(phase, self.state.completed_steps)

    def save_checkpoint(self):
        with self._measure("checkpoint", self.state.completed_steps):
            return super().save_checkpoint()
