from gpt_lab.utils.distributed import ComputeContext
import pytest
import torch

from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.optim.kernels.muon import muon_step_fused
from torch.utils._python_dispatch import TorchDispatchMode
from gpt_lab.utils.schemas import TrainerConfig


@pytest.mark.parametrize("momentum", [0.9, 0.95])
def test_default_momentum_is_independent_of_lr_warmup_and_warmdown(momentum):
    context = ComputeContext(device=torch.device("cpu"))
    config = TrainerConfig(comp_info=context, n_steps=2317, lr_warmup_steps=300)
    param = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = OptimizerFactory(
        [dict(params=[param], opt="muon", lr=0.02, momentum=momentum,
              beta=0.9, ns_steps=5, weight_decay=0.28)],
        comp_info=context,
    )

    # Revisit an early step to cover resume/reapplication without compounding.
    for step in (0, 149, 299, 300, 944, 2316, 149):
        optimizer.update_hyperparams(
            lrm=config.lr_multiplier_schedule(step),
            muon_momentum=config.muon_momentum_schedule(step),
        )
        group = optimizer.param_groups[0]
        assert group["momentum"] == pytest.approx(momentum)
        assert group["lr"] == pytest.approx(0.02 * config.lr_multiplier_schedule(step))


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float32])
def test_muon_restores_parameter_dtype_after_orthogonalization(compute_dtype):
    generator = torch.Generator().manual_seed(123)
    params = torch.randn(2, 3, 4, generator=generator)
    grads = torch.randn(params.shape, generator=generator)
    momentum = torch.zeros_like(params)
    variance = torch.zeros(2, 3, 1)
    initial_params = params.clone()
    matmul_dtypes, scaling_dtypes = [], []

    class ObserveDtypes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.bmm.default:
                matmul_dtypes.append((args[0].dtype, args[1].dtype))
            if (func == torch.ops.aten.mul.Tensor
                    and isinstance(args[1], torch.Tensor)
                    and args[0].shape == params.shape and args[1].shape == variance.shape):
                scaling_dtypes.append((args[0].dtype, args[1].dtype))
            return func(*args, **(kwargs or {}))

    with ObserveDtypes():
        muon_step_fused.__wrapped__(
            grads, params, momentum, variance,
            *(torch.tensor(value) for value in (0.95, 0.02, 0.28, 0.9)),
            5, -1, compute_dtype,
        )

    assert matmul_dtypes == [(compute_dtype, compute_dtype)] * 15
    assert scaling_dtypes == [(torch.float32, torch.float32)]
    for tensor in (params, grads, momentum, variance):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    assert not torch.equal(params, initial_params)


def test_hyperparameter_schedules_preserve_group_specific_values():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = OptimizerFactory(
        [
            dict(
                params=[param],
                opt="muon",
                lr=0.2,
                momentum=0.95,
                beta=0.9,
                ns_steps=5,
                weight_decay=0.28,
            )
        ],
        comp_info=ComputeContext(device=torch.device("cpu")),
    )

    optimizer.update_hyperparams(lrm=0.5, muon_momentum=0.8, weight_decay=0.25)

    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.1)
    assert group["momentum"] == pytest.approx(0.76)
    assert group["weight_decay"] == pytest.approx(0.07)


def test_legacy_checkpoint_gets_schedule_baselines():
    param = torch.nn.Parameter(torch.zeros(1))
    config = dict(
        params=[param],
        opt="muon",
        lr=0.2,
        momentum=0.95,
        beta=0.9,
        ns_steps=5,
        weight_decay=0.28,
    )
    optimizer = OptimizerFactory([config], comp_info=ComputeContext(device=torch.device("cpu")))
    state = optimizer.state_dict()
    for key in ("initial_lr", "initial_momentum", "initial_weight_decay"):
        state["param_groups"][0].pop(key)

    optimizer.load_state_dict(state)
    optimizer.update_hyperparams(lrm=0.5, muon_momentum=0.8, weight_decay=0.25)

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.07)
