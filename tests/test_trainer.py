from gpt_lab.utils.distributed import ComputeContext
import pytest
import torch
from gpt_lab.model.wrapper import Engine
from gpt_lab.train.trainer import Trainer

from gpt_lab.utils.schemas import (
    GPTConfig, 
    LossConfig, 
    TokenizerConfig, 
    TransformerConfig,
)
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

from gpt_lab.utils.schemas import TrainerConfig


class _CaptureBoard:
    def __init__(self):
        self.records = []

    def log(self, values, step=None):
        self.records.append((values, step))


def _make_trainer(tmp_path, compute_dtype=torch.bfloat16, monitor=False):
    model = torch.nn.Linear(2, 2, bias=False)
    board = _CaptureBoard()
    config = TrainerConfig(
        n_steps=1,
        n_acc_steps=1,
        total_batch_size=4,
        device_batch_size=2,
        log_every=1,
        monitor_grad_norms=monitor,
        comp_info=ComputeContext(device=torch.device("cpu"), compute_dtype=compute_dtype),
    )
    trainer = Trainer(
        model=model,
        tokenizer=None,
        train_loader=SimpleNamespace(B=2),
        val_loader=None,
        config=config,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        board=board,
        checkpoint_manager=SimpleNamespace(source_dir=tmp_path),
    )
    return trainer, model, board


@pytest.mark.fast
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, "bfloat16"])
def test_trainer_enables_amp_for_torch_and_legacy_dtypes(tmp_path, compute_dtype):
    trainer, model, _ = _make_trainer(tmp_path, compute_dtype=compute_dtype)
    inputs = torch.ones(2, 2)

    with trainer.train_context():
        train_output = model(inputs)
    with trainer.val_context(model):
        val_output = model(inputs)

    assert trainer.use_amp
    assert trainer.dtype == torch.bfloat16
    assert train_output.dtype == torch.bfloat16
    assert val_output.dtype == torch.bfloat16


@pytest.mark.fast
def test_gradient_monitor_reads_accumulated_parameter_gradients(tmp_path):
    trainer, model, board = _make_trainer(tmp_path, monitor=True)
    model.weight.grad = torch.tensor([[1.0, -3.0], [1.0, -3.0]])

    trainer.log_gradients()

    values, step = board.records[-1]
    assert step == 0
    assert values["grad_rms/weight"] == pytest.approx(5.0 ** 0.5)
    assert values["grad_mean/weight"] == pytest.approx(-1.0)
    assert values["grad_abs_mean/weight"] == pytest.approx(2.0)
    assert not model.weight._backward_hooks

@pytest.mark.fast
@pytest.mark.parametrize("step,previous_best,save_on_best,should_save", [
    (5, None, True, True),
    (10, 2.0, True, False),  # The scheduled checkpoint will save this step.
    (5, 1.0, True, False),
    (5, 2.0, False, False),
])
def test_evaluate_bpb_tracks_metrics_and_best_checkpoint(
    tmp_path, monkeypatch, step, previous_best, save_on_best, should_save,
):
    trainer, model, board = _make_trainer(tmp_path)
    model.config = SimpleNamespace(max_context=2)
    trainer._compiled_model = torch.nn.Linear(2, 2)
    trainer.tokenizer = SimpleNamespace(token_bytes=torch.ones(2))
    trainer.val_loader = Mock(return_value=iter(()))
    trainer.config = trainer.config.model_copy(update={
        "n_bpb_tokens": 9, "save_every": 10, "save_on_best": save_on_best,
    })
    trainer.state.step = step
    trainer.ckpt_state.best_eval_value = previous_best
    trainer.ckpt_state.best_eval_step = 1
    trainer.save_checkpoint = Mock()

    def evaluate(*args, **kwargs):
        assert not trainer._compiled_model.training
        assert torch.is_autocast_enabled("cpu")
        return {"bpb": 1.5, "loss": 2.0}

    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr("gpt_lab.train.trainer.compute_bpb", evaluator)

    trainer.evaluate_bpb()

    trainer.val_loader.assert_called_once_with()
    evaluator.assert_called_once_with(
        trainer._compiled_model, trainer.val_loader.return_value, 3,
        comp_info=trainer.config.comp_info,
        token_bytes=trainer.tokenizer.token_bytes,
    )
    improved = previous_best is None or previous_best > 1.5
    best = 1.5 if improved else previous_best
    assert trainer.ckpt_state.best_eval_value == best
    assert trainer.ckpt_state.best_eval_step == (step if improved else 1)
    assert trainer.save_checkpoint.call_count == int(should_save)
    assert trainer._compiled_model.training
    values, logged_step = board.records[-1]
    assert logged_step == step
    assert values["eval/best_bpb"] == best
    assert trainer.eval_metrics.step == [step]
    assert trainer.eval_metrics.bpb == [1.5]
    assert trainer.eval_metrics.loss == [2.0]
    assert trainer.eval_metrics.step_time_ms == [values["eval/step_time_ms"]]


@pytest.mark.fast
@pytest.mark.parametrize("previous_best", [None, 1.0])
def test_evaluate_core_tracks_metrics_and_best_score(tmp_path, monkeypatch, previous_best):
    trainer, model, board = _make_trainer(tmp_path)
    trainer.state.step = 5
    trainer.ckpt_state.best_core_value = previous_best
    trainer.ckpt_state.best_core_step = 1
    results = {
        "core/core": 0.6,
        "core/accuracy": 0.7,
        "core/max_per_task": 2,
        "core/step_time_ms": 10,
        "all_core_results": {"task": {"accuracy": 0.8}},
    }

    def evaluate(*args, **kwargs):
        assert not model.training
        assert torch.is_autocast_enabled("cpu")
        return results

    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr("gpt_lab.train.trainer.evaluate_core", evaluator)

    trainer.evaluate_core()

    evaluator.assert_called_once_with(
        model, trainer.tokenizer, trainer.device,
        max_per_task=trainer.config.n_core_tokens,
    )
    assert model.training
    assert trainer.ckpt_state.best_core_value == (0.6 if previous_best is None else 1.0)
    assert trainer.ckpt_state.best_core_step == (5 if previous_best is None else 1)
    values, step = board.records[-1]
    assert step == 5
    assert values["core/core"] == 0.6
    assert values["core/task/accuracy"] == 0.8
    assert trainer.core_metrics.step == [5]
    assert trainer.core_metrics.core == [0.6]
    assert trainer.core_metrics.accuracy == [0.7]
    assert trainer.core_metrics.results == [results["all_core_results"]]


@pytest.mark.fast
def test_evaluate_samples_uses_original_model_and_generation_settings(tmp_path, monkeypatch):
    trainer, model, _ = _make_trainer(tmp_path)
    trainer._compiled_model = torch.nn.Linear(2, 2)

    def generate(prompts, **kwargs):
        assert not model.training
        assert torch.is_autocast_enabled("cpu")
        assert len(prompts) == 7
        assert prompts[0] == "The capital of France is"

    engine = Mock()
    engine.generate_batch.side_effect = generate
    engine_factory = Mock(return_value=engine)
    monkeypatch.setattr("gpt_lab.train.trainer.Engine", engine_factory)

    trainer.evaluate_samples()

    engine_factory.assert_called_once_with(model, trainer.tokenizer)
    engine.generate_batch.assert_called_once()
    assert engine.generate_batch.call_args.kwargs == {
        "num_samples": 1, "max_tokens": 16, "temperature": 0,
    }
    assert model.training


class TestModelTrainer:
    model_name = "test-model"
    pad_token_id = 0
    tmpdirname = tempfile.mkdtemp()
    tokenizer_config = TokenizerConfig(
        vocab_size=1000,
        max_context=16,
        name="simple-tokenizer",
        source="dummy"
    )
    model_config = TransformerConfig(
        vocab_size=1000,
        pad_id=pad_token_id,
        max_context=16,
        d_model=16,
        d_ffn=64,
        n_heads=4,
        n_layers=4,
        d_head=4,
        dropout=0.1
    )
    loss_config = LossConfig(
        loss_fn="cross_entropy",
        ignore_index=pad_token_id,
        kwargs={"reduction": "mean"}
    )
    config = GPTConfig(
        name=model_name,
        tokenizer=tokenizer_config,
        model=model_config,
        loss=loss_config,
        dirname=tmpdirname
    )

    # def setup_method(self):
    #     self.model = Engine.from_scratch(self.config)
    #     self.trainer = Trainer(
    #         model=self.model,
    #         train_dataset=[],
    #         val_dataset=[],
    #         test_dataset=[],
    #     )
    
    # WIP
    # @pytest.mark.fast
    # def test_trainer_initialization(self):
    #     assert self.trainer.model == self.model
    #     assert self.trainer.config == self.model.config.trainer
    #     assert self.trainer.device.type == "cpu"
    #     assert self.trainer.dtype == torch.float32
    
    
    # @pytest.mark.fast
    # def test_loss_decrease_over_epochs(self):
    #     # This is a placeholder test. In a real scenario, you would train the model for a few epochs
    #     # and check if the loss decreases. Here, we just check if the fit method can be called without error.
    #     try:
    #         self.trainer.fit()
    #     except NotImplementedError:
    #         pass  # Since fit is not implemented, we just ensure it raises NotImplementedError
