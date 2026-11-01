"""Logit distillation for pretraining with a fixed teacher."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer import Trainer


class DistillationTrainer(Trainer):
    """Train a student with (1 - alpha) * CE + alpha * temperature**2 * KL.

    Pass the student as ``model`` and a separate frozen teacher as ``teacher``.
    Both models must use the same token IDs/vocabulary, be on the same device,
    and return full-sequence ``.logits`` when called with (x, y). Targets of -1
    are ignored. The inherited prepare_batch and student evaluation are reused.

    The caller owns teacher placement. Checkpoints save the student and its
    training state; reconstruct this class with the same teacher, alpha, and
    temperature to resume. Inherited from_checkpoint cannot supply the teacher.
    Training FLOP/MFU estimates omit teacher compute.
    """

    def __init__(
        self,
        *args,
        teacher: nn.Module,
        alpha: float = 0.5,
        temperature: float = 2.0,
        **kwargs,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        super().__init__(*args, **kwargs)
        student_parameters = {id(param) for param in self.model.parameters()}
        if any(id(param) in student_parameters for param in teacher.parameters()):
            raise ValueError("Teacher and student must not share parameters.")
        self.teacher = teacher.eval().requires_grad_(False)
        # shortcut: teacher/settings are not checkpointed; add metadata before automatic resume.
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, payload: Any) -> torch.Tensor:
        """Compute forward KL(teacher || student), averaged over valid tokens."""
        x, y = payload
        valid = y != -1
        if not valid.any():
            raise ValueError("Distillation requires at least one non-ignored target.")

        # Labels request full-sequence logits from the current GPT implementation.
        student_logits = self.compiled_model(x, y).logits
        with torch.no_grad():
            teacher_logits = self.teacher(x, y).logits
        if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != y.shape:
            raise ValueError("Student and teacher logits must have matching (batch, sequence, vocabulary) shapes.")
        if student_logits.device != teacher_logits.device:
            raise ValueError("Student and teacher logits must be on the same device.")

        student_logits = student_logits[valid].float()
        teacher_logits = teacher_logits[valid].float()
        ce_loss = F.cross_entropy(student_logits, y[valid])
        temperature = self.temperature
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        )
        return (1 - self.alpha) * ce_loss + self.alpha * temperature**2 * kl_loss
