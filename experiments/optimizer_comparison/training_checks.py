"""Record actual run state without imposing a frozen experiment recipe."""
import json
import os


def install(output):
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.model.checkpoint import capture_rng_state
    from experiments.optimizer_comparison.check_h100 import fingerprint
    original_train, original_prepare = Trainer.train, Trainer.prepare_batch
    initial = {}

    def prepare(self, batch):
        result = original_prepare(self, batch)
        if "first_payload" not in initial:
            initial["first_payload"] = fingerprint(result[0])
            initial["data_position"] = result[1].model_dump() if result[1] else None
            (output/f"initial_rank_{self.config.comp_info.rank}.json").write_text(json.dumps(initial, indent=2)+"\n")
        return result

    def train(self):
        output.mkdir(parents=True, exist_ok=True)
        rng = capture_rng_state()
        initial.update(model=fingerprint(self.model.state_dict()),
                       rng=fingerprint({key: rng[key] for key in ("torch", "cuda")}))
        original_train(self)
        if self.config.comp_info.rank == 0:
            report = dict(sha=os.environ["GPTLAB_SELECTED_SHA"], status="finished",
                completed_updates=self.state.completed_steps, actual_tokens=self.state.n_tokens,
                parameter_count=sum(p.numel() for p in self.model.parameters()),
                scaling_parameters=self.model.n_scaling_params(),
                horizon=self.config.n_steps, total_batch_size=self.config.total_batch_size,
                n_acc_steps=self.config.n_acc_steps, best_metrics=self.ckpt_state.model_dump(),
                config=self.config.model_dump(mode="json", exclude={"comp_info"}))
            (output/"selected_result.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
    Trainer.prepare_batch, Trainer.train = prepare, train
