import pytest
import torch

from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.optim.kernels.normuon import ANVIL2_MAPS, _orthogonalize
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state


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

