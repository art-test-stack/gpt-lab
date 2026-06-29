import pytest
import torch

from gpt_lab.optim.factory import OptimizerFactory


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
        dist_info={"IS_DDP_INITIALIZED": False},
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
    optimizer = OptimizerFactory([config], dist_info={"IS_DDP_INITIALIZED": False})
    state = optimizer.state_dict()
    for key in ("initial_lr", "initial_momentum", "initial_weight_decay"):
        state["param_groups"][0].pop(key)

    optimizer.load_state_dict(state)
    optimizer.update_hyperparams(lrm=0.5, muon_momentum=0.8, weight_decay=0.25)

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.07)
