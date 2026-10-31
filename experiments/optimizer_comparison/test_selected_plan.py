"""Offline recipe/ordering checks; never import Modal or run CUDA work."""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from gpt_lab.cli.parser import build_parser
from experiments.optimizer_comparison import launch, recipe
from experiments.optimizer_comparison.check_h100 import assert_tree, fingerprint


def test_selected_commands_match_native_parser_and_selection():
    commands = launch.job_commands(Path("/runs/check"), "train")
    assert len(commands) == 4
    assert "check_h100.py" in commands[0][5]
    for command, variant in zip(commands[1:], recipe.SELECTION["variants"]):
        assert "--nproc_per_node=8" in command
        settings = vars(build_parser().parse_args(["train", "auto", *command[command.index("--")+1:]]))
        for key, value in recipe.COMMON.items():
            assert settings[key] == value
        assert settings["model_name"] == variant["model_name"]
        assert settings["optim_config_path"] == variant["optimizer_config"]
        assert settings["stop_after_step"] is None
        assert settings["num_steps"] == settings["target_flops"] == settings["target_time"] == -1
    assert len(launch.job_commands(Path("/runs/check"), "validate")) == 1


def test_gate_and_training_failure_stop_sequence():
    commands = launch.job_commands(Path("/runs/check"), "train")
    for failure in (0, 1, 2):
        run = Mock(side_effect=[None]*failure+[subprocess.CalledProcessError(1, "probe")])
        with pytest.raises(subprocess.CalledProcessError):
            launch.run_commands(commands, run)
        assert run.call_count == failure+1
    run = Mock()
    launch.run_commands(commands, run)
    assert [call.args[0] for call in run.call_args_list] == commands


def test_dry_run_is_offline(monkeypatch, capsys):
    monkeypatch.setattr(launch, "resolve_source", lambda *args: pytest.fail("dry-run contacted Git"))
    launch.main(["--sha", "e"*40, "--run-name", "offline", "--mode", "train", "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["gpu"] == "H100:8" and len(report["commands"]) == 4


def test_resolved_parameter_horizon_and_fingerprint():
    model, report = recipe.model_and_horizon()
    from torch._subclasses.fake_tensor import FakeTensor
    assert all(isinstance(p, FakeTensor) for p in model.parameters())
    assert (report["d_model"], report["d_ffn"]) == (1280, 5120)
    assert report["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert report["scaling_parameters"] == model.n_scaling_params()
    for key in ("parameter_count", "scaling_parameters", "target_tokens", "horizon", "rounded_tokens"):
        assert report[key] == recipe.SELECTION["resolved_plan"][key]
    assert report["target_tokens"] == 10*report["scaling_parameters"]
    model._enforce_precision(torch.bfloat16)
    assert set(recipe.storage_summary(model)) == {"torch.bfloat16", "torch.float32"}
    assert report["horizon"]*524288 == report["rounded_tokens"]
    assert 0 <= report["target_tokens"]-report["rounded_tokens"] < 524288
    import numpy as np
    rng_array = np.arange(10000)
    changed_rng = rng_array.copy()
    changed_rng[5000] += 1
    assert fingerprint(rng_array) != fingerprint(changed_rng)
    a = dict(weight=torch.ones(2), step=torch.tensor(2.))
    b = {key:value.clone() for key,value in a.items()}
    assert fingerprint(a) == fingerprint(b)
    assert_tree(a, b)
    b["weight"][0] += 1
    assert fingerprint(a) != fingerprint(b)
    with pytest.raises(AssertionError):
        assert_tree(a, b)
