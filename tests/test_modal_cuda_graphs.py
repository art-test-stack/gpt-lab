import importlib.util
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from gpt_lab.cli.parser import build_parser


spec = importlib.util.spec_from_file_location(
    "modal_cuda_graphs", Path(__file__).parents[1] / "scripts/modal_cuda_graphs.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("gpus", [1, 4])
@pytest.mark.parametrize("mode", ["disabled", "forward_backward"])
def test_dry_run_builds_matched_native_commands(monkeypatch, capsys, gpus, mode):
    monkeypatch.setattr(launcher, "resolve_source", lambda _: ("https://example.com/repo", "exp/cuda-graphs", "a" * 40))
    monkeypatch.setitem(sys.modules, "modal", None)  # Dry run must not load Modal.
    launcher.main(["--dry-run", "--gpus", str(gpus), "--mode", mode, "--require-sxm", "--run-name", "test-capture",
                   "--", "--save-every", "1", "--eval-bpb-every", "1"])
    output = capsys.readouterr().out
    assert "Required device: NVIDIA A100-SXM4-80GB" in output
    command = output.splitlines()[1]
    settings = build_parser().parse_args(shlex.split(command)[1:])
    assert settings.cuda_graph_mode == mode
    assert settings.optim_config_path == "configs/normuon.yaml"
    assert settings.stop_after_step == 150
    assert settings.num_steps == 2317 and settings.warmup_steps == 300
    assert settings.random_seed == 42 and settings.device == "cuda"
    assert settings.total_batch_size == 524288
    assert settings.total_batch_size // (gpus * settings.device_batch_size * settings.max_seq_len) == 16 // gpus
    assert (settings.save_every, settings.eval_bpb_every, settings.eval_core_every, settings.sample_every) == (0, 0, 0, 0)
    compile(launcher.SHORT_RUNNER, "train_short.py", "exec")


@pytest.mark.parametrize("names,require_sxm,accepted", [
    (["NVIDIA A100-SXM4-80GB"], True, True),
    (["NVIDIA A100 80GB PCIe"], True, False),
    (["NVIDIA A100-SXM4-80GB"] * 3 + ["NVIDIA A100 80GB PCIe"], True, False),
    (["NVIDIA A100 80GB PCIe"], False, True),
])
def test_gpu_probe_enforces_opt_in_sxm_guard(monkeypatch, names, require_sxm, accepted):
    cuda = SimpleNamespace(device_count=lambda: len(names), get_device_name=lambda index: names[index],
                           is_bf16_supported=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    probe = launcher.gpu_probe(len(names), "bfloat16", "sdpa", require_sxm)
    if accepted:
        exec(probe, {})
    else:
        with pytest.raises(AssertionError, match="SXM4 required"):
            exec(probe, {})


def test_summary_excludes_startup_and_uses_global_tokens_once():
    ranks = [dict(
        n_steps=2317, completed_steps=150, mode="forward_backward", dtype="torch.bfloat16",
        device_name="A100", n_acc_steps=4, device_batch_size=16,
        total_batch_size=524288, parameter_count=100,
        setup_before_first_update_seconds=10 + rank, time_to_first_update_seconds=20 + rank,
        capture_startup_seconds=5 + rank,
        steps=[dict(step=step, seconds=1000 if step < 10 else 2,
                    iteration_seconds=2000 if step < 10 else 3 + rank, tokens=524288) for step in range(150)],
    ) for rank in range(4)]
    result = launcher.summarize(ranks)
    assert result["world_size"] == 4
    assert result["n_steps"] == 2317 and result["completed_steps"] == 150
    assert result["steady_first_update"] == 11 and result["steady_last_update"] == 150
    assert result["steady_updates"] == 140 and result["steady_seconds"] == 840
    assert result["steady_global_tokens_per_second"] == 524288 / 6
    assert result["steady_mean_train_step_ms"] == 2000
    assert result["setup_before_first_update_seconds"] == 13
    assert result["time_to_first_update_seconds"] == 23
    assert result["capture_startup_seconds"] == 8
    ranks[1]["completed_steps"] = 149
    with pytest.raises(RuntimeError, match="Ranks disagree"):
        launcher.summarize(ranks)
