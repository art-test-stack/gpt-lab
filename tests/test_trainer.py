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
        dist_info={
            "compute_dtype": compute_dtype,
            "DEVICE_TYPE": "cpu",
            "DEVICE": torch.device("cpu"),
            "IS_DDP_INITIALIZED": False,
            "WORLD_SIZE": 1,
            "RANK": 0,
            "gpu_peak_flops": float("inf"),
        },
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
