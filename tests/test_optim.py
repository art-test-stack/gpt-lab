from gpt_lab.utils.distributed import ComputeContext
import pytest
import torch

from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.optim.kernels.muon import muon_step_fused
from gpt_lab.optim.kernels.normuon import ANVIL2_MAPS, _orthogonalize
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state
from torch.utils._python_dispatch import TorchDispatchMode


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


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
def test_hyperparameter_schedules_preserve_group_specific_values(opt_name):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = OptimizerFactory(
        [
            dict(
                params=[param],
                opt=opt_name,
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


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
def test_legacy_checkpoint_gets_schedule_baselines(opt_name):
    param = torch.nn.Parameter(torch.zeros(1))
    config = dict(
        params=[param],
        opt=opt_name,
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


def _reference_normuon_update(grad, momentum, second_moment, beta, beta2, steps):
    # Unbatched official reference; MIT (c) 2025 zichongli5, LICENSES/NorMuon-MIT.txt.
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta)
    x = update.bfloat16()
    if grad.size(0) > grad.size(1):
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    if grad.size(0) > grad.size(1):
        x = x.T
    update = x.to(grad.dtype)
    norm = update.norm()
    second_moment.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - beta2)
    update.mul_(1 / (second_moment.sqrt() + 1e-10))
    update.mul_(norm / (update.norm() + 1e-10))
    return update * max(1, grad.size(0) / grad.size(1)) ** 0.5


@pytest.mark.parametrize("shape", [(4, 2), (2, 4)])
def test_normuon_matches_reference_updates_and_checkpoint(shape):
    parameters = [torch.nn.Parameter(torch.linspace(-0.3, 0.7, 8).reshape(shape) + i)
                  for i in range(2)]
    expected = [p.detach().clone() for p in parameters]
    moments = [torch.zeros_like(p) for p in parameters]
    variances = [torch.zeros(shape[0], 1) for _ in parameters]
    config = dict(opt="normuon", lr=0.02, momentum=0.95, beta=0.95,
                  ns_steps=5, weight_decay=0.1)
    context = ComputeContext(device=torch.device("cpu"))
    optimizer = OptimizerFactory([dict(params=parameters, **config)], comp_info=context)
    with torch.compiler.set_stance("force_eager"):
        for step in range(3):
            for i, parameter in enumerate(parameters):
                gradient = torch.sin(torch.arange(8).reshape(shape) + i + step)
                parameter.grad = gradient
                update = _reference_normuon_update(gradient, moments[i], variances[i], 0.95, 0.95, 5)
                expected[i].mul_(1 - config["lr"] * config["weight_decay"])
                expected[i].add_(update, alpha=-config["lr"])
            optimizer.step()
            for parameter, reference in zip(parameters, expected):
                torch.testing.assert_close(parameter, reference)
            state = optimizer.state[parameters[0]]
            torch.testing.assert_close(state["momentum_buffer"], torch.stack(moments))
            torch.testing.assert_close(state["second_momentum_buffer"], torch.stack(variances))
            if step == 0:
                resumed = OptimizerFactory([dict(params=parameters, **config)], comp_info=context)
                resumed.load_state_dict(optimizer.state_dict())
                optimizer = resumed


@pytest.mark.parametrize("key,value", [("beta", 1.0), ("momentum", -0.1), ("lr", float("nan")),
                                       ("ns_steps", 0), ("ns_steps", 1.5)])
def test_normuon_rejects_invalid_hyperparameters(key, value):
    group = dict(params=[torch.nn.Parameter(torch.zeros(2, 3))], opt="normuon",
                 lr=0.02, momentum=0.95, beta=0.95, ns_steps=5, weight_decay=0.1)
    group[key] = value
    with pytest.raises(ValueError, match="NorMuon"):
        OptimizerFactory([group], comp_info=ComputeContext(device=torch.device("cpu")))


def test_anvil_spectral_envelope():
    sigma = torch.logspace(torch.log10(torch.tensor(0.003)), torch.log10(torch.tensor(1 / 1.05)),
                           10001, dtype=torch.float64)
    for a, b, c in ANVIL2_MAPS:
        sigma = a * sigma + b * sigma**3 + c * sigma**5
    assert sigma.min() > 0.995
    assert sigma.max() < 1.0005


@pytest.mark.parametrize("shape", [(4, 8), (8, 4)])
def test_anvil_spectral_bf16_rank_one_and_full_rank(shape):
    for rank in (0, 1, 4):
        matrix = torch.zeros(shape)
        matrix[:rank, :rank] = torch.eye(rank)
        result = _orthogonalize(matrix, 5, True)
        assert result.dtype == torch.bfloat16
        singular_values = torch.linalg.svdvals(result.float())
        assert torch.isfinite(result).all()
        torch.testing.assert_close(singular_values[:rank], torch.ones(rank), rtol=0, atol=0.06)
        torch.testing.assert_close(singular_values[rank:], torch.zeros(4 - rank))


@pytest.mark.parametrize("shape", [(4, 2), (2, 4)])
def test_anvil_spectral_update_norm_state_memory_and_resume(shape, tmp_path):
    parameters = [torch.nn.Parameter(torch.linspace(-0.3, 0.7, 8).reshape(shape) + i)
                  for i in range(2)]
    config = dict(opt="normuon", lr=0.02, momentum=0.95, beta=0.95,
                  ns_steps=5, weight_decay=0.1, anvil_spectral=True)
    context = ComputeContext(device=torch.device("cpu"))
    optimizer = OptimizerFactory([dict(params=parameters, **config)], comp_info=context)
    with torch.compiler.set_stance("force_eager"):
        for step in range(3):
            before = [p.detach().clone() for p in parameters]
            for i, parameter in enumerate(parameters):
                parameter.grad = torch.sin(torch.arange(8).reshape(shape) + i + step)
            if step == 0:
                # With zero initial EMA, the first Nesterov velocity is (1 - mu^2) g.
                velocity = torch.stack([p.grad for p in parameters]) * (1 - config["momentum"]**2)
                expected_norm = _orthogonalize(velocity, 5, True).float().norm(dim=(-2, -1))
                expected_norm *= max(1, shape[0] / shape[1])**0.5
            else:
                for parameter, resumed_parameter in zip(parameters, resumed_parameters):
                    resumed_parameter.grad = parameter.grad.clone()
                resumed.step()
            optimizer.step()
            state = optimizer.state[parameters[0]]
            assert set(state) == {"momentum_buffer", "second_momentum_buffer"}
            assert all(value.dtype == torch.float32 for value in state.values())
            assert sum(value.numel() * value.element_size() for value in state.values()) == 4 * 2 * (8 + shape[0])
            if step == 0:
                update = torch.stack([
                    (old * (1 - config["lr"] * config["weight_decay"]) - new.detach()) / config["lr"]
                    for old, new in zip(before, parameters)
                ])
                torch.testing.assert_close(update.norm(dim=(-2, -1)), expected_norm, rtol=1e-5, atol=1e-5)
                path = tmp_path / "optimizer.safetensors"
                save_tensor_state(optimizer.state_dict(), path)
                resumed_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
                resumed = OptimizerFactory([dict(params=resumed_parameters, **config)], comp_info=context)
                resumed.load_state_dict(load_tensor_state(path))
            else:
                for parameter, resumed_parameter in zip(parameters, resumed_parameters):
                    torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed.state[resumed_parameters[0]][key], rtol=0, atol=0)
