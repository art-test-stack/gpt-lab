import importlib.util
from pathlib import Path
import shlex
import sys
import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from gpt_lab.cli.parser import build_parser


spec = importlib.util.spec_from_file_location(
    "modal_cuda_graphs", Path(__file__).parents[1] / "scripts/modal_cuda_graphs.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("gpus", [1, 4, 8])
@pytest.mark.parametrize("mode", ["disabled", "forward_backward", "paired", "verify"])
def test_dry_run_builds_matched_native_commands(monkeypatch, capsys, gpus, mode):
    monkeypatch.setattr(launcher, "resolve_source", lambda _: ("https://example.com/repo", "exp/cuda-graphs", "a" * 40))
    monkeypatch.setitem(sys.modules, "modal", None)  # Dry run must not load Modal.
    launcher.main(["--dry-run", "--gpus", str(gpus), "--mode", mode, "--require-sxm", "--run-name", "test-capture",
                   "--", "--save-every", "1", "--eval-bpb-every", "1"])
    output = capsys.readouterr().out
    assert "Required device: NVIDIA A100-SXM4-80GB" in output
    command = output.splitlines()[1]
    settings = build_parser().parse_args(shlex.split(command)[1:])
    assert settings.cuda_graph_mode == (mode if mode in ("disabled", "forward_backward") else "disabled")
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


def test_profile_is_a_separate_short_job(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "resolve_source", lambda _: ("repo", "branch", "a" * 40))
    monkeypatch.setitem(sys.modules, "modal", None)
    launcher.main(["--dry-run", "--gpus", "8", "--mode", "paired", "--profile"])
    assert "14 updates per mode; record three updates" in capsys.readouterr().out


def test_summary_excludes_startup_and_uses_global_tokens_once():
    ranks = [dict(
        rank=rank,
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


@pytest.mark.parametrize("microbatch,accumulation", [(16, -1), (16, 2), (32, -1), (32, 1)])
def test_eight_gpu_preflight_preserves_matched_settings(monkeypatch, tmp_path, microbatch, accumulation):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda command, **kwargs:
                        json.dumps(vars(build_parser().parse_args(command[3:]))))
    monkeypatch.setattr(launcher, "RUNS", tmp_path)
    arguments = launcher.cli_defaults(launcher.DEFAULT_TRAIN | dict(
        device_batch_size=microbatch, n_acc_steps=accumulation))
    settings = launcher.check_inputs(arguments, "new-run", 8, check_data=False)
    assert settings["total_batch_size"] // (8 * settings["device_batch_size"] * settings["max_seq_len"]) == 32 // microbatch
    with pytest.raises(ValueError, match="8-GPU comparison"):
        launcher.check_inputs([*arguments, "--num-steps", "150"], "new-run", 8, check_data=False)
    if microbatch == 32:
        with pytest.raises(ValueError, match="Explicit n_acc_steps"):
            launcher.check_inputs([*arguments, "--n-acc-steps", "2"], "new-run", 8, check_data=False)


def test_preflight_worker_needs_no_project_packages(tmp_path):
    # Modal's worker lacks gpt_lab; only the selected project interpreter has it.
    repository = Path(__file__).parents[1]
    code = """
import importlib.util, json, pathlib, runpy, sys
assert importlib.util.find_spec("gpt_lab") is None
namespace = runpy.run_path(sys.argv[1])
check = namespace["check_inputs"]
check.__globals__.update(PYTHON=sys.argv[2], REPO=pathlib.Path(sys.argv[3]),
                        RUNS=pathlib.Path(sys.argv[4]))
settings = check(namespace["cli_defaults"](namespace["DEFAULT_TRAIN"]), "new-run", 8, False)
print(json.dumps(settings))
"""
    result = subprocess.check_output(
        [sys.executable, "-I", "-S", "-c", code,
         str(repository / "scripts/modal_cuda_graphs.py"), sys.executable,
         str(repository), str(tmp_path)], cwd=tmp_path, text=True,
    )
    settings = json.loads(result)
    assert settings["optim_config_path"] == "configs/normuon.yaml"
    assert settings["num_steps"] == 2317 and settings["stop_after_step"] == 150


def initial_rank(rank=0):
    return dict(rank=rank, world_size=1, initial_model_sha256="model",
                initial_optimizer_sha256="optimizer", initial_rng_sha256="rng",
                first_batch_sha256="inputs", initial_data_position={"offset": 0},
                final_data_position={"offset": 150}, n_steps=2317, completed_steps=150,
                n_acc_steps=16, device_name="NVIDIA A100-SXM4-80GB", dtype="torch.bfloat16",
                total_batch_size=524288, device_batch_size=16, parameter_count=422314440,
                torch_version="2.9.1", cuda_version="12.8")


@pytest.mark.parametrize("field", ["initial_model_sha256", "initial_optimizer_sha256",
                                  "first_batch_sha256", "final_data_position", "device_name"])
def test_pair_requires_identical_reset_and_data(field):
    left = [initial_rank()]
    right = copy.deepcopy(left)
    launcher.verify_pair(left, right)
    right[0][field] = "different"
    with pytest.raises(RuntimeError, match=field):
        launcher.verify_pair(left, right)


@pytest.mark.parametrize("gate_passes", [False, True])
@pytest.mark.parametrize("world,microbatch,accumulation,gate_accumulation", [(1, 16, -1, 16), (8, 32, 1, 1)])
def test_same_allocation_pair_is_blocked_by_failed_equivalence(monkeypatch, tmp_path, gate_passes,
                                                              world, microbatch, accumulation, gate_accumulation):
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.setattr(launcher, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(launcher, "git", lambda directory, *args: "sha" if args[0] == "rev-parse" else "")
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "test topology")
    events = []
    def run(command, **kwargs):
        if "scripts/benchmark/cuda_graph_equivalence.py" not in " ".join(command):
            events.append("probe")
            return
        events.append("equivalence")
        assert command[command.index("--n-acc-steps") + 1] == str(gate_accumulation)
        if not gate_passes:
            raise subprocess.CalledProcessError(1, command)
        directory = Path(command[command.index("--output-dir") + 1])
        directory.mkdir(parents=True)
        for rank in range(world):
            (directory / f"rank_{rank}.json").write_text(json.dumps(dict(passed=True, world_size=world)))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    def train_one(arguments, name, sha, branch, env, settings, world, volume, directory):
        mode = settings["cuda_graph_mode"]
        events.append(mode)
        if mode == "forward_backward":
            assert env["GPTLAB_EXPECTED_INITIAL"] == str(directory.parent / "disabled")
        directory.mkdir()
        for rank in range(world):
            result = initial_rank(rank) | dict(world_size=world, device_batch_size=microbatch,
                                               n_acc_steps=16 if world == 1 else 1)
            (directory / f"rank_{rank}.json").write_text(json.dumps(result))
        (directory / "summary.json").write_text(json.dumps(dict(mode=mode)))
    monkeypatch.setattr(launcher, "train_one", train_one)
    env = dict(GPTLAB_DTYPE="bfloat16", GPTLAB_REQUIRE_SXM="1", GPTLAB_PROFILE="0",
               GPTLAB_DIAGNOSTIC_MODE="paired")
    settings = launcher.DEFAULT_TRAIN | dict(n_acc_steps=accumulation, device_batch_size=microbatch)
    volume = SimpleNamespace(commit=lambda: None)
    if gate_passes:
        result = launcher.train([], "pair", "sha", "branch", env, settings, world, volume)
        assert result["identical_initial_state_and_data_position"]
        assert events == ["probe", "equivalence", "disabled", "forward_backward"]
    else:
        with pytest.raises(subprocess.CalledProcessError):
            launcher.train([], "pair", "sha", "branch", env, settings, world, volume)
        assert events == ["probe", "equivalence"]
        result = json.loads((tmp_path / "runs/pair/comparison.json").read_text())
        assert result["status"] == "failed" and "results" not in result
