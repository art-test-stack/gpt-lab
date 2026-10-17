from gpt_lab.utils.distributed import ComputeContext
import pytest
import torch

from gpt_lab.optim.factory import OptimizerFactory


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
