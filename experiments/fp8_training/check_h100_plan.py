"""Offline plan/lifecycle checks. No Git, network or Modal calls."""

import subprocess
from unittest.mock import patch

import pytest

from experiments.fp8_training import run_h100_compare as comparison


def test_matched_one_gpu_plans():
    baseline, fp8 = comparison.commands("local-check")
    for command in (baseline, fp8):
        for flag, value in {"--gpus": "1", "--gpu": "H100", "--dtype": "bfloat16",
                            "--act-func": "swiglu", "--n-acc-steps": "16",
                            "--attn-impl": "fa3",
                            "--num-steps": "2317", "--stop-after-step": "10",
                            "--total-batch-size": "524288"}.items():
            assert command[command.index(flag)+1] == value
        assert "--detach" not in command
    assert "--fp8-ffn-forward" not in baseline
    assert fp8[-1] == "--fp8-ffn-forward"
    a, b = list(baseline), list(fp8[:-1])
    a[a.index("--run-name")+1] = b[b.index("--run-name")+1]
    assert a == b


def test_dry_run_and_sequential_execution():
    with patch.object(comparison.subprocess, "run") as run:
        comparison.main(["--dry-run", "--run-prefix", "local-check"])
        run.assert_not_called()
        comparison.main(["--run-prefix", "local-check"])
        assert [call.args[0] for call in run.call_args_list] == comparison.commands("local-check")
        assert all(call.kwargs == {"cwd": comparison.ROOT, "check": True} for call in run.call_args_list)


def test_baseline_failure_stops_pair():
    with patch.object(comparison.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "fake")) as run:
        with pytest.raises(subprocess.CalledProcessError):
            comparison.main(["--run-prefix", "local-check"])
        assert run.call_count == 1


def test_invalid_prefix_rejected():
    with pytest.raises(ValueError):
        comparison.commands("../escape")
