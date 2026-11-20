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
import itertools

from gpt_lab.utils.schemas import DataLoaderState, TrainerConfig, TrainerState
from gpt_lab.model.checkpoint import load_checkpoint, save_checkpoint
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh


class _CaptureBoard:
    def __init__(self):
        self.records = []

    def log(self, values, step=None):
        self.records.append((values, step))


def _distributed_loss_metrics_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        context = ComputeContext(
            device=torch.device("cpu"),
            mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("dp",)),
        )
        trainer = Trainer.__new__(Trainer)
        trainer.device = context.device
        trainer.config = TrainerConfig(
            comp_info=context, n_steps=3, total_batch_size=8, n_flops_per_token=100,
        )
        trainer.state = TrainerState()
        parameter = torch.tensor(float(rank + 1), requires_grad=True)
        expected_ema = 0.0
        durations = []
        for step in range(2):
            trainer.state.step = step
            local_loss = parameter * (2 * (step + 1))
            local_loss.backward()
            original_loss = local_loss.detach().clone()
            original_grad = parameter.grad.clone()
            metrics = trainer.update_metrics(dict(
                loss=local_loss, tokens=8, step_time=(rank + 1) * 0.5,
                lrm=1.0, muon_momentum=1.0, weight_decay=1.0,
            ), durations)

            global_loss = 3.0 * (step + 1)
            expected_ema = 0.9 * expected_ema + 0.1 * global_loss
            assert metrics["train/raw_loss"] == pytest.approx(global_loss)
            assert metrics["train/loss"] == pytest.approx(expected_ema / (1 - 0.9 ** (step + 1)))
            assert trainer.state.smooth_train_loss == pytest.approx(expected_ema)
            assert metrics["train/step_time_ms"] == 1000.0
            assert metrics["train/tokens_per_sec"] == 8.0
            torch.testing.assert_close(local_loss, original_loss)
            torch.testing.assert_close(parameter.grad, original_grad)
        assert trainer.state.n_tokens == 16
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_distributed_metrics_average_loss_before_smoothing(tmp_path):
    # File rendezvous avoids a free-port selection race; workers own the group.
    mp.spawn(
        _distributed_loss_metrics_worker,
        args=((tmp_path / "loss-metrics-rendezvous").as_uri(),), nprocs=2, join=True,
    )


def test_step_fields_are_canonical():
    config = TrainerConfig(stop_after_step=5)
    state = TrainerState(completed_steps=3)
    assert config.stop_after_step == 5
    assert state.completed_steps == 3
    assert config.model_dump()['stop_after_step'] == 5
    assert state.model_dump()['completed_steps'] == 3
    assert 'stop_after_updates' not in config.model_dump()
    assert 'completed_updates' not in state.model_dump()
    assert TrainerConfig(stop_after_updates=5).stop_after_step is None
    assert TrainerState(completed_updates=3).completed_steps is None


@pytest.mark.parametrize('horizon,cap,completed', [(2317, 945, 945), (3, None, 3), (3, 1, 1), (3, 0, 0), (2, 3, 2)])
def test_execution_cap_preserves_horizon_and_checkpoint_resume(tmp_path, monkeypatch, horizon, cap, completed):
    trainer, model, board = _make_trainer(tmp_path, compute_dtype=torch.float32, trainer_class=_PayloadTrainer)
    trainer.config = trainer.config.model_copy(update=dict(
        n_steps=horizon, stop_after_step=cap, n_acc_steps=1, total_batch_size=3,
        n_flops_per_token=100, eval_bpb_every=0, eval_core_every=0, sample_every=0, log_every=-1))
    trainer.train_loader = itertools.repeat(dict(features=torch.ones(2, 2), targets=torch.zeros(2, 2), tokens=3, state=None))
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer.optimizer.update_hyperparams = lambda **kwargs: None
    trainer.ckpt_manager.save = lambda **kwargs: save_checkpoint(
        checkpoint_dir=tmp_path, comp_info=trainer.config.comp_info, **kwargs)
    monkeypatch.setattr(torch, 'compile', lambda model, **kwargs: model)
    for name in ('collect', 'freeze', 'disable'):
        monkeypatch.setattr(f'gpt_lab.train.trainer.gc.{name}', lambda: None)
    trainer.train()
    assert trainer.config.n_steps == horizon
    assert trainer.state.completed_steps == completed
    assert trainer.state.step == max(0, completed - 1)
    assert trainer.state.n_tokens == completed * 3
    assert sum('train/raw_loss' in values for values, _ in board.records) == int(completed > 0)
    if cap is not None:
        assert board.records[-1][0] == {'train/completed_steps': completed}
    loaded = load_checkpoint(tmp_path, step=max(0, completed - 1), comp_info=trainer.config.comp_info, load_scaler=False)
    assert loaded.trainer_state.completed_steps == completed
    torch.testing.assert_close(loaded.model_state['weight'], model.weight)
    assert bool(loaded.optimizer_state['state']) == bool(completed)

    # Legacy checkpoints omitted the counter; include the step-zero boundary.
    if completed:
        loaded.trainer_state.completed_steps = None
    restored_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    restored_optimizer.load_state_dict(loaded.optimizer_state)
    restored_optimizer.update_hyperparams = lambda **kwargs: None
    resumed = _PayloadTrainer(model, None, None, None, config=trainer.config,
                              optimizer=restored_optimizer, board=board,
                              checkpoint_manager=trainer.ckpt_manager, resume_state=loaded.trainer_state)
    before = model.weight.detach().clone()
    logged_steps = sum('train/raw_loss' in values for values, _ in board.records)
    resumed.train() # None loader must never be accessed once the absolute limit is met.
    assert resumed.state.completed_steps == completed
    assert sum('train/raw_loss' in values for values, _ in board.records) == logged_steps
    torch.testing.assert_close(model.weight, before)
    if cap == 1:
        resumed.config = resumed.config.model_copy(update={'stop_after_step': 2})
        resumed.train_loader = trainer.train_loader
        resumed.train()
        assert resumed.state.completed_steps == 2 and resumed.state.step == 1
        assert resumed.state.n_tokens == 6 and resumed.config.n_steps == horizon


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
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, "bfloat16", torch.float32, "float32"])
def test_trainer_enables_amp_for_torch_and_legacy_dtypes(tmp_path, compute_dtype):
    trainer, model, _ = _make_trainer(tmp_path, compute_dtype=compute_dtype)
    inputs = torch.ones(2, 2)

    with trainer.train_context():
        train_output = model(inputs)
    with trainer.val_context(model):
        val_output = model(inputs)

    expected_dtype = torch.bfloat16 if compute_dtype in (torch.bfloat16, "bfloat16") else torch.float32
    assert trainer.use_amp == (expected_dtype == torch.bfloat16)
    assert trainer.dtype == expected_dtype
    assert train_output.dtype == expected_dtype
    assert val_output.dtype == expected_dtype


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
    (10, 2.0, True, True),  # Best selection is independent of periodic saving.
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
    trainer.state.completed_steps = step + 1
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
    assert logged_step == step + 1
    assert trainer.ckpt_state.best_eval_completed_steps == (step + 1 if improved else None)
    assert values["eval/best_bpb"] == best
    assert trainer.eval_metrics.step == [step + 1]
    assert trainer.eval_metrics.bpb == [1.5]
    assert trainer.eval_metrics.loss == [2.0]
    assert trainer.eval_metrics.step_time_ms == [values["eval/step_time_ms"]]


@pytest.mark.fast
@pytest.mark.parametrize("previous_best", [None, 1.0])
def test_evaluate_core_tracks_metrics_and_best_score(tmp_path, monkeypatch, previous_best):
    trainer, model, board = _make_trainer(tmp_path)
    trainer.state.step = 5
    trainer.state.completed_steps = 6
    trainer.config = trainer.config.model_copy(update={"save_on_best": True})
    trainer.save_checkpoint = Mock()
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
    assert step == 6
    assert trainer.ckpt_state.best_core_completed_steps == (6 if previous_best is None else None)
    assert trainer.save_checkpoint.call_count == int(previous_best is None)
    assert values["core/core"] == 0.6
    assert values["core/task/accuracy"] == 0.8
    assert trainer.core_metrics.step == [6]
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
def test_train_step_matches_accumulated_update_and_preserves_prefetch_state(tmp_path, monkeypatch):
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
    backward_microsteps = []
    trainer.optimizer.begin_backward_communication = backward_microsteps.append
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

    events = []
    prepare_batch = trainer.prepare_batch
    tensor_bool = torch.Tensor.__bool__
    isfinite = torch.isfinite

    def prepare(raw_batch):
        events.append("prepare")
        return prepare_batch(raw_batch)

    def inspect_loss(tensor):
        events.append("host_read")
        return tensor_bool(tensor)

    def check_finite(tensor):
        events.append("isfinite")
        return isfinite(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "prepare_batch", prepare)
        patch.setattr(torch.Tensor, "__bool__", inspect_loss)
        patch.setattr(torch, "isfinite", check_finite)
        next_batch, metrics = trainer.train_step(train_iter, batch, should_log_step=True)

    assert events == ["prepare", "prepare", "isfinite", "host_read"]
    assert backward_microsteps == [False, True]

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


@pytest.mark.parametrize("unsafe", ["scaler", "capture"])
def test_backward_communication_rejects_incompatible_training_modes(tmp_path, unsafe):
    trainer, _, _ = _make_trainer(tmp_path, compute_dtype=torch.float32)
    trainer.optimizer.communication = SimpleNamespace(mode="backward")
    if unsafe == "scaler":
        trainer.scaler = object()
    else:
        trainer.config = trainer.config.model_copy(update={"cuda_graph_mode": "forward_backward"})
    with pytest.raises(ValueError, match="unscaled gradients and disabled CUDA capture"):
        trainer.train_step(iter(()), None, should_log_step=False)


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
def test_loss_reporting_reduces_a_detached_copy(tmp_path, monkeypatch):
    trainer, model, _ = _make_trainer(tmp_path, compute_dtype=torch.float32)
    context = SimpleNamespace(is_distributed=True, world_size=2, gpu_peak_flops=float("inf"))
    trainer.config = trainer.config.model_copy(update={
        "comp_info": context, "n_steps": 2, "n_flops_per_token": 100,
    })
    local_loss = model(torch.ones(2, 2)).square().mean()
    original_loss = local_loss.detach().clone()
    local_loss.backward()
    original_grad = model.weight.grad.clone()

    def reduce(tensor, op):
        assert not tensor.requires_grad
        if op == dist.ReduceOp.MAX:
            tensor.fill_(1.0)
        else:
            assert op == dist.ReduceOp.SUM
            assert tensor.data_ptr() != local_loss.data_ptr()
            tensor.add_(4.0)  # The other rank's mean loss.

    monkeypatch.setattr("gpt_lab.train.trainer.dist.all_reduce", reduce)
    metrics = trainer.update_metrics(dict(
        loss=local_loss, tokens=4, step_time=0.5,
        lrm=1.0, muon_momentum=1.0, weight_decay=1.0,
    ), [])
    expected = (original_loss.item() + 4.0) / 2
    assert metrics["train/raw_loss"] == pytest.approx(expected)
    assert metrics["train/loss"] == pytest.approx(expected)
    assert trainer.state.smooth_train_loss == pytest.approx(0.1 * expected)
    torch.testing.assert_close(local_loss, original_loss)
    torch.testing.assert_close(model.weight.grad, original_grad)


@pytest.mark.fast
@pytest.mark.parametrize("bad_loss,microstep,with_state", [
    (float("nan"), 0, True), (float("inf"), 1, True), (float("nan"), 1, False),
])
def test_nonfinite_loss_completes_accumulation_reports_last_metadata_and_aborts(
    tmp_path, bad_loss, microstep, with_state,
):
    trainer, model, _ = _make_trainer(
        tmp_path, compute_dtype=torch.float32, trainer_class=_PayloadTrainer,
    )
    trainer.compiled_model = model
    trainer.config = trainer.config.model_copy(update={"n_acc_steps": 2})
    trainer.state.step = 7
    trainer.optimizer_step = Mock()
    initial_weight = model.weight.detach().clone()
    features, targets = torch.ones(2, 2), torch.zeros(2, 2)
    states = [DataLoaderState(epoch=i + 1, offset_in_row_group=10 + i) if with_state else None
              for i in range(3)]
    fetched = []

    def batches():
        for i, state in enumerate(states):
            targets.fill_(bad_loss if i == microstep else 0.0)
            fetched.append(i)
            yield dict(features=features, targets=targets, tokens=4, state=state)

    train_iter = batches()
    batch = trainer.prepare_batch(next(train_iter))

    with pytest.raises(ValueError, match="Accumulated loss is non-finite") as error:
        trainer.train_step(train_iter, batch, should_log_step=False)

    message = str(error.value)
    metadata = states[1].model_dump() if with_state else None
    assert "optimizer step=7" in message
    assert "microstep=" not in message
    assert f"Last consumed dataloader state: {metadata}." in message
    assert "Exact consumed-batch tensors were not captured" in message
    assert "Batch payload:" not in message
    assert fetched == [0, 1, 2]
    assert trainer.state.train_loader_state is states[1]
    trainer.optimizer_step.assert_not_called()
    torch.testing.assert_close(model.weight, initial_weight)
    assert not (tmp_path / "bad_batch.safetensors").exists()


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


@pytest.mark.parametrize("interval,expected", [(2, [2, 4, 5]), (-1, [5]), (0, [])])
def test_evaluation_runs_after_completed_updates_and_includes_final(tmp_path, monkeypatch, interval, expected):
    trainer, model, _ = _make_trainer(tmp_path, compute_dtype=torch.float32, trainer_class=_PayloadTrainer)
    trainer.config = trainer.config.model_copy(update=dict(
        n_steps=9, stop_after_step=5, n_acc_steps=1, total_batch_size=3,
        n_flops_per_token=100, eval_bpb_every=interval, eval_core_every=interval,
        sample_every=0, save_every=2, log_every=0))
    trainer.train_loader = itertools.repeat(dict(features=torch.ones(2, 2), targets=torch.zeros(2, 2), tokens=3, state=None))
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.optimizer.update_hyperparams = lambda **kwargs: None
    observations = {"bpb": [], "core": [], "save": []}
    def record(kind):
        observations[kind].append(trainer.state.completed_steps)
        assert trainer.state.step + 1 == trainer.state.completed_steps
        assert trainer.state.n_tokens == trainer.state.completed_steps * 3
    trainer.evaluate_bpb = lambda: record("bpb")
    trainer.evaluate_core = lambda: record("core")
    trainer.save_checkpoint = lambda: record("save")
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    for name in ("collect", "freeze", "disable"):
        monkeypatch.setattr(f"gpt_lab.train.trainer.gc.{name}", lambda: None)
    trainer.train()
    assert observations == {"bpb": expected, "core": expected, "save": [2, 4, 5]}
