"""Study evaluation and dashboard diagnostics for the Muon arms."""

import json
import math
from collections import deque
from contextlib import contextmanager
from statistics import median
from time import perf_counter

from gpt_lab.train.trainer import Trainer


class ExperimentTrainer(Trainer):
    def __init__(self, *, study_eval=False, monitor_qk_norms=False, guard_mfu=False, **kwargs):
        super().__init__(**kwargs)
        self.study_eval = study_eval
        self.monitor_qk_norms = monitor_qk_norms
        self.mfu_window = deque(maxlen=10) if guard_mfu else None

    def _evaluate_routine(self, last_step):
        if not self.study_eval:
            return super()._evaluate_routine(last_step)

        completed = self.state.completed_steps
        if completed == 0:
            return
        interval = self.config.eval_bpb_every
        if interval > 0 and completed % interval == 0:
            self.evaluate_bpb()
        horizon = self.config.n_steps
        if completed in ((horizon + 3) // 4, (2 * horizon + 3) // 4, (3 * horizon + 3) // 4):
            self.evaluate_core()

    def _full_endpoint_includes_core(self):
        return not self.study_eval

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
        strategy.monitor = completed in (1, endpoint) or completed % 50 == 0
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

    def evaluate_bpb(self):
        with self._measure("bpb", self.state.completed_steps):
            super().evaluate_bpb()

    def evaluate_core(self):
        with self._measure("core", self.state.completed_steps):
            super().evaluate_core()

    def save_checkpoint(self):
        with self._measure("checkpoint", self.state.completed_steps):
            return super().save_checkpoint()
