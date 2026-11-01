from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.train import DistillationTrainer, Trainer
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.schemas import TrainerConfig, TransformerConfig


def _model(width):
    model = DenseTransformer(TransformerConfig(
        vocab_size=16, max_context=3, d_model=width, d_ffn=2 * width,
        n_heads=2, n_kv_heads=2, n_layers=1, d_head=width // 2,
        window_pattern="L", attn_impl="sdpa",
    )).to("cpu")
    model.init_weights()
    return model


def _trainer(tmp_path, **kwargs):
    student = _model(16)
    teacher = _model(32)
    trainer = DistillationTrainer(
        model=student, teacher=teacher, tokenizer=None, train_loader=None,
        val_loader=None, config=TrainerConfig(
            n_steps=1, comp_info=ComputeContext(device=torch.device("cpu")),
        ),
        optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
        checkpoint_manager=SimpleNamespace(source_dir=tmp_path), **kwargs,
    )
    trainer.compiled_model = student
    return trainer


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_distillation_matches_masked_token_mean_and_only_trains_student(tmp_path, alpha):
    trainer = _trainer(tmp_path, alpha=alpha, temperature=2.0)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[2, -1, 4], [-1, -1, 7]])
    payload, state, count = trainer.prepare_batch((x, y))
    assert isinstance(trainer, Trainer)
    assert state is None
    assert count == 6

    loss = trainer.compute_loss(payload)

    with torch.no_grad():
        student = trainer.model(x, y)
        teacher = trainer.teacher(x, y)
        temperature = trainer.temperature
        student_log_probs = F.log_softmax(student.logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher.logits / temperature, dim=-1)
        kl_per_token = (
            teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
        expected = (1 - alpha) * student.loss + alpha * temperature**2 * kl_per_token[y != -1].mean()
    torch.testing.assert_close(loss, expected)
    assert loss.ndim == 0
    assert trainer.model.training
    assert not trainer.teacher.training
    assert all(not param.requires_grad for param in trainer.teacher.parameters())

    teacher_weights = [param.detach().clone() for param in trainer.teacher.parameters()]
    loss.backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in trainer.model.parameters())
    assert all(param.grad is None for param in trainer.teacher.parameters())
    trainer.optimizer_step(should_log_step=False)
    for before, after in zip(teacher_weights, trainer.teacher.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_distillation_rejects_batches_without_valid_targets(tmp_path):
    trainer = _trainer(tmp_path)
    with pytest.raises(ValueError, match="non-ignored target"):
        trainer.compute_loss((torch.ones(1, 3, dtype=torch.long), torch.full((1, 3), -1)))


@pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"temperature": 0}, {"temperature": float("nan")}])
def test_distillation_rejects_invalid_objective_settings(tmp_path, kwargs):
    with pytest.raises(ValueError, match="alpha|temperature"):
        _trainer(tmp_path, **kwargs)


def test_distillation_rejects_shared_teacher_and_student(tmp_path):
    student = _model(16)
    with pytest.raises(ValueError, match="must not share parameters"):
        DistillationTrainer(
            model=student, teacher=student, tokenizer=None, train_loader=None,
            val_loader=None, optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
            checkpoint_manager=SimpleNamespace(source_dir=tmp_path),
        )
    assert all(param.requires_grad for param in student.parameters())


def test_distillation_rejects_mismatched_vocabularies(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.teacher.config = trainer.teacher.config.model_copy(update={"vocab_size": 15})
    x = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="matching .* shapes"):
        trainer.compute_loss((x, x))
