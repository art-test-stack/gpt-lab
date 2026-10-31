"""Production CLI with matched-initial-state checks and a final selection report."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-initial", type=Path, required=True)
    args, training = parser.parse_known_args()
    if training[:1] == ["--"]:
        training = training[1:]
    import torch
    from gpt_lab.cli import main as cli
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.model.checkpoint import capture_rng_state
    from experiments.optimizer_comparison.check_h100 import fingerprint
    from experiments.optimizer_comparison.recipe import model_and_horizon, storage_summary
    original_train, original_prepare = Trainer.train, Trainer.prepare_batch
    initial = {}

    def prepare(self, batch):
        result = original_prepare(self, batch)
        if "first_payload" not in initial:
            initial["first_payload"] = fingerprint(result[0])
            initial["data_position"] = result[1].model_dump() if result[1] else None
            rank_file = f"initial_rank_{self.config.comp_info.rank}.json"
            expected_path = args.expected_initial/rank_file
            if args.output != args.expected_initial:
                expected = json.loads(expected_path.read_text())
                if expected != initial:
                    raise RuntimeError("Initial model/RNG/data mismatch; stopping before first update")
            (args.output/rank_file).write_text(json.dumps(initial, indent=2)+"\n")
        return result

    def train(self):
        args.output.mkdir(parents=True, exist_ok=True)
        storage = storage_summary(self.model)
        initial.update(model=fingerprint(self.model.state_dict()), rng=fingerprint(capture_rng_state()))
        _, horizon = model_and_horizon()
        assert self.config.n_steps == horizon["horizon"]
        assert self.config.batch_lr_scale == horizon["base_training"]["batch_lr_scale"]
        assert self.config.weight_decay_scale == horizon["base_training"]["weight_decay_scale"]
        assert self.config.stop_after_step is None
        assert self.config.cuda_graph_mode == "disabled" and self.config.fp8_ffn_forward
        assert (self.config.n_acc_steps, self.config.device_batch_size, self.config.total_batch_size) == (1, 32, 524288)
        original_train(self)
        assert self.state.completed_steps == self.config.n_steps
        assert self.state.n_tokens == horizon["rounded_tokens"]
        assert self.eval_metrics.step[-1] == self.state.completed_steps
        assert self.core_metrics.step[-1] == self.state.completed_steps
        selections = dict(best_bpb=self.ckpt_state.best_eval_step,
                          best_core=self.ckpt_state.best_core_step,
                          final_resumable=self.state.step)
        for index in selections.values():
            assert (self.ckpt_manager.source_dir/f"checkpoint_step_{index:06d}").is_dir()
        if self.config.comp_info.rank == 0:
            report = dict(sha=os.environ["GPTLAB_SELECTED_SHA"], status="finished",
                          completed_updates=self.state.completed_steps, actual_tokens=self.state.n_tokens, horizon=horizon, parameter_storage=storage,
                          checkpoints={key:str(self.ckpt_manager.source_dir/f"checkpoint_step_{value:06d}") for key,value in selections.items()},
                          best_metrics=self.ckpt_state.model_dump(), config=self.config.model_dump(mode="json", exclude={"comp_info"}))
            (args.output/"selected_result.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
    Trainer.prepare_batch, Trainer.train = prepare, train
    try:
        cli(["train", "auto", *training])
    finally:
        Trainer.prepare_batch, Trainer.train = original_prepare, original_train


if __name__ == "__main__":
    main()
