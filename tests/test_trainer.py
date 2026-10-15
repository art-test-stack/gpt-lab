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

from gpt_lab.utils.schemas import DataLoaderState, TrainerConfig
from gpt_lab.model.checkpoint import load_checkpoint, save_checkpoint


class _CaptureBoard:
    def __init__(self):
        self.records = []

    def log(self, values, step=None):
        self.records.append((values, step))


def _make_trainer(tmp_path, compute_dtype=torch.bfloat16, monitor=False, trainer_class=Trainer):
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
    trainer = trainer_class(
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
    trainer.compiled_model = torch.nn.Linear(2, 2)
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
        assert not trainer.compiled_model.training
        assert torch.is_autocast_enabled("cpu")
        return {"bpb": 1.5, "loss": 2.0}

    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr("gpt_lab.train.trainer.compute_bpb", evaluator)

    trainer.evaluate_bpb()

    trainer.val_loader.assert_called_once_with()
    evaluator.assert_called_once_with(
        trainer.compiled_model, trainer.val_loader.return_value, 3,
        comp_info=trainer.config.comp_info,
        token_bytes=trainer.tokenizer.token_bytes,
    )
    improved = previous_best is None or previous_best > 1.5
    best = 1.5 if improved else previous_best
    assert trainer.ckpt_state.best_eval_value == best
    assert trainer.ckpt_state.best_eval_step == (step if improved else 1)
    assert trainer.save_checkpoint.call_count == int(should_save)
    assert trainer.compiled_model.training
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
    trainer.compiled_model = torch.nn.Linear(2, 2)

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


@pytest.mark.fast
@pytest.mark.parametrize("state", [None, DataLoaderState(epoch=3)])
def test_prepare_batch_preserves_payload_and_optional_state(tmp_path, state):
    trainer, _, _ = _make_trainer(tmp_path)
    x, y = torch.ones(2, 3), torch.zeros(2, 3)
    raw_batch = (x, y) if state is None else (x, y, state)

    payload, loader_state, tokens = trainer.prepare_batch(raw_batch)

    assert payload[0] is x
    assert payload[1] is y
    assert loader_state is state
    assert tokens == 6


@pytest.mark.fast
def test_prepare_batch_rejects_custom_formats_without_override(tmp_path):
    trainer, _, _ = _make_trainer(tmp_path)
    with pytest.raises(ValueError, match="override prepare_batch"):
        trainer.prepare_batch({"input_ids": torch.ones(2, 3)})
    with pytest.raises(TypeError, match="DataLoaderState or None"):
        trainer.prepare_batch((torch.ones(2, 3), torch.zeros(2, 3), {"offset": 1}))


@pytest.mark.fast
@pytest.mark.parametrize("model_returns_loss", [False, True])
def test_compute_loss_accepts_prepared_payload(tmp_path, model_returns_loss):
    trainer, _, _ = _make_trainer(tmp_path)
    x = torch.ones(1, 2, dtype=torch.long)
    y = torch.tensor([[0, 1]])
    logits = torch.tensor([[[2.0, 0.0], [1.0, 3.0]]], requires_grad=True)
    expected = torch.nn.functional.cross_entropy(logits.view(-1, 2), y.view(-1))
    output = SimpleNamespace(loss=expected) if model_returns_loss else logits
    trainer.compiled_model = Mock(return_value=output)

    loss = trainer.compute_loss((x, y))

    trainer.compiled_model.assert_called_once_with(x, y)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


class _PayloadTrainer(Trainer):
    def prepare_batch(self, raw_batch):
        return raw_batch, raw_batch.get("state"), raw_batch["tokens"]

    def compute_loss(self, payload):
        return (self.compiled_model(payload["features"]) - payload["targets"]).square().mean()


@pytest.mark.fast
def test_train_step_matches_accumulated_update_and_preserves_prefetch_state(tmp_path):
    trainer, model, _ = _make_trainer(
        tmp_path, compute_dtype=torch.float32, trainer_class=_PayloadTrainer,
    )
    trainer.compiled_model = model
    trainer.config = trainer.config.model_copy(update={"n_acc_steps": 2})
    trainer.state.step = 7
    trainer.lr_schedule = lambda step: 0.75
    trainer.muon_momentum_schedule = lambda step: 0.95
    trainer.weight_decay_schedule = lambda step: 0.25
    trainer.optimizer.update_hyperparams = Mock()
    trainer.optimizer_step = Mock(wraps=trainer.optimizer_step)
    batches = [dict(features=torch.full((2, 2), float(i + 1)),
                    targets=torch.zeros(2, 2), tokens=3 + i,
                    state=DataLoaderState(epoch=i + 1, offset_in_row_group=i))
               for i in range(4)]
    reference = torch.nn.Linear(2, 2, bias=False)
    reference.load_state_dict(model.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected_loss = sum(
        (reference(batch["features"]) - batch["targets"]).square().mean()
        for batch in batches[:2]
    ) / 2
    expected_loss.backward()
    reference_optimizer.step()
    train_iter = iter(batches)
    batch = trainer.prepare_batch(next(train_iter))

    next_batch, metrics = trainer.train_step(train_iter, batch, should_log_step=True)

    torch.testing.assert_close(model.weight, reference.weight)
    assert model.weight.grad is None
    torch.testing.assert_close(metrics["loss"], expected_loss.detach())
    assert not metrics["loss"].requires_grad
    assert metrics["tokens"] == 7
    assert metrics["step_time"] >= 0
    assert (metrics["lrm"], metrics["muon_momentum"], metrics["weight_decay"]) == (0.75, 0.95, 0.25)
    trainer.optimizer.update_hyperparams.assert_called_once_with(
        lrm=0.75, muon_momentum=0.95, weight_decay=0.25,
    )
    trainer.optimizer_step.assert_called_once_with(True)
    assert trainer.state.train_loader_state is batches[1]["state"]
    assert next_batch[0] is batches[2]
    assert next_batch[1] is batches[2]["state"]
    assert next_batch[2] == batches[2]["tokens"]
    assert next(train_iter) is batches[3]
    # Preserve the existing epoch update from the prefetched batch.
    assert trainer.state.n_epochs == batches[2]["state"].epoch


@pytest.mark.fast
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("log_every", [0, 1])
def test_training_custom_payload_counts_tokens_and_checkpoints_optional_state(
    tmp_path, monkeypatch, caplog, with_state, log_every,
):
    trainer, model, board = _make_trainer(
        tmp_path, compute_dtype=torch.float32, trainer_class=_PayloadTrainer,
    )
    trainer.config = trainer.config.model_copy(update={
        "n_steps": 2, "n_acc_steps": 2, "total_batch_size": 6,
        "n_flops_per_token": 100, "eval_bpb_every": 0,
        "eval_core_every": 0, "sample_every": 0, "log_every": log_every,
    })
    batches = [dict(features=torch.ones(2, 2), targets=torch.zeros(2, 2), tokens=3,
                    state=DataLoaderState(epoch=i + 1, offset_in_row_group=i) if with_state else None)
               for i in range(5)]
    trainer.train_loader = batches
    trainer.optimizer.update_hyperparams = Mock()
    trainer.ckpt_manager.save = lambda **kwargs: save_checkpoint(
        checkpoint_dir=tmp_path, comp_info=trainer.config.comp_info, **kwargs,
    )
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    # Keep the trainer's process-wide GC policy from affecting other tests.
    for name in ("collect", "freeze", "disable"):
        monkeypatch.setattr(f"gpt_lab.train.trainer.gc.{name}", lambda: None)
    initial_weight = model.weight.detach().clone()
    seen_payloads = []
    compute_loss = trainer.compute_loss

    def capture_loss(payload):
        seen_payloads.append(payload)
        return compute_loss(payload)

    trainer.compute_loss = capture_loss
    caplog.set_level("WARNING", logger="gpt_lab.train.trainer")

    trainer.train()

    assert all(seen is raw for seen, raw in zip(seen_payloads, batches))
    assert len(seen_payloads) == 4
    assert trainer.state.n_tokens == 12
    assert not torch.equal(model.weight, initial_weight)
    assert model.weight.grad is None
    warnings = [record for record in caplog.records if "No dataloader state provided" in record.message]
    assert len(warnings) == (0 if with_state else 1)
    loaded = load_checkpoint(tmp_path, step=1, comp_info=trainer.config.comp_info, load_scaler=False)
    assert loaded.trainer_state.train_loader_state == batches[3]["state"]
    assert loaded.trainer_state.n_tokens == 12
    assert len(board.records) == (2 if log_every else 0)


@pytest.mark.fast
@pytest.mark.parametrize("prior_duration_count", [0, 10])
def test_update_metrics_tracks_state_and_log_metrics_only_reports(
    tmp_path, capsys, prior_duration_count,
):
    trainer, _, board = _make_trainer(tmp_path)
    trainer.config = trainer.config.model_copy(update={"n_steps": 20, "n_flops_per_token": 100})
    trainer.state.step = 10
    trainer.state.n_tokens = 40
    trainer.state.n_epochs = 2
    trainer.state.smooth_train_loss = 1.8
    trainer.state.total_training_time = 5.0
    total_dt = [0.25] * prior_duration_count
    step_metrics = dict(loss=torch.tensor(2.0), tokens=4, step_time=0.5,
                        lrm=0.75, muon_momentum=0.95, weight_decay=0.25)

    metrics = trainer.update_metrics(step_metrics, total_dt)

    expected = {
        "epochs": 2, "train/loss": 1.82 / (1 - 0.9**11), "train/raw_loss": 2.0,
        "train/tokens_per_sec": 8.0, "train/step_time_ms": 500,
        "train/total_tokens": 44, "train/flops_per_sec": 800, "train/mfu": 0.0,
        "lrm": 0.75, "muon_momentum": 0.95, "weight_decay": 0.25,
        "train/total_training_flops": 4000, "train/total_training_time": 5.5,
        "train/eta_seconds": 2.75 if prior_duration_count else float("inf"),
    }
    assert metrics == pytest.approx(expected)
    assert trainer.state.n_tokens == 44
    assert trainer.state.smooth_train_loss == pytest.approx(1.82)
    assert trainer.state.total_training_time == 5.5
    assert total_dt == [0.25] * prior_duration_count + [0.5]
    assert board.records == []
    assert trainer.metrics.step == []
    state_before_logging = trainer.state.model_dump()

    trainer.log_metrics(metrics)

    assert trainer.state.model_dump() == state_before_logging
    assert board.records == [(metrics, 10)]
    assert trainer.metrics.step == [10]
    assert trainer.metrics.raw_loss == [2.0]
    output = capsys.readouterr().out
    assert "Step 00010/00020" in output
    assert "tok/s: 8" in output
    assert ("ETA:" in output) == bool(prior_duration_count)


@pytest.mark.fast
@pytest.mark.parametrize("serializable", [False, True])
def test_nonfinite_custom_payload_reports_loss_without_loader_state(tmp_path, monkeypatch, serializable):
    trainer, _, _ = _make_trainer(tmp_path, trainer_class=_PayloadTrainer)
    trainer.config = trainer.config.model_copy(update={
        "n_flops_per_token": 100, "eval_bpb_every": 0,
        "eval_core_every": 0, "sample_every": 0,
    })
    payload = dict(features=torch.ones(2, 2), targets=torch.full((2, 2), float("nan")), tokens=4)
    if not serializable:
        payload["custom_metadata"] = object()
    trainer.train_loader = [payload]
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)

    with pytest.raises(ValueError, match="Loss is NaN or Inf"):
        trainer.train()

    assert (tmp_path / "bad_batch.safetensors").exists() == serializable


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
