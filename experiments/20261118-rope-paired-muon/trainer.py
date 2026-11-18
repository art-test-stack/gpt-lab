"""Dashboard diagnostics for the three Muon arms."""

import json
from contextlib import contextmanager
from time import perf_counter

from gpt_lab.train.trainer import Trainer


class ExperimentTrainer(Trainer):
    def optimizer_step(self, should_log_step):
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
            super().save_checkpoint()
