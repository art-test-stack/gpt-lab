"""Direct launcher checks; no Modal calls or paid compute."""
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from gpt_lab.cli.parser import build_parser
from runs import modal_trainbase as launch
from experiments.optimizer_comparison import recipe
from experiments.optimizer_comparison.check_h100 import assert_tree, fingerprint


def test_three_direct_plans_preserve_recipe_without_implicit_cap(monkeypatch, capsys):
    monkeypatch.setattr(launch, "resolve_source", lambda *args: ("https://example.invalid/repo", "exp/anvil-normuon", "e"*40))
    commands = re.findall(r"```bash\n(.*?)\n```", (recipe.ROOT/"experiments/optimizer_comparison/README.md").read_text(), re.S)
    assert len(commands) == 3
    for command, variant in zip(commands, recipe.SELECTION["variants"]):
        name = variant["model_name"]+"-v2"
        parts = shlex.split(command.replace("\\\n", ""))
        arguments = parts[parts.index("runs/modal_trainbase.py")+1:]
        arguments[arguments.index("--sha")+1] = "e"*40
        arguments.insert(arguments.index("--"), "--dry-run")
        launch.main(arguments)
        output = capsys.readouterr().out
        settings = vars(build_parser().parse_args(shlex.split(output.splitlines()[1])[1:]))
        for key, value in recipe.COMMON.items():
            assert settings[key] == value
        assert settings["model_name"] == variant["model_name"]
        assert settings["optim_config_path"] == variant["optimizer_config"]
        assert settings["run_name"] == name and settings["stop_after_step"] is None
        assert recipe.selected_run(settings, name, 8)["reference_run"] == "ic1-d16-d20-muon-v2"
    generic = vars(build_parser().parse_args(["train", "auto", *launch.cli_defaults(launch.DEFAULT_TRAIN)]))
    assert generic["stop_after_step"] is None
    capped = vars(build_parser().parse_args(["train", "auto", *launch.cli_defaults(launch.DEFAULT_TRAIN), "--stop-after-step", "15"]))
    assert capped["stop_after_step"] == 15
    compile(launch.SHORT_RUNNER, "native-training-entry", "exec")


def test_failed_hardware_gate_prevents_native_training(monkeypatch, tmp_path):
    sha = "e"*40
    monkeypatch.setattr(launch, "RUNS", tmp_path)
    monkeypatch.setattr(launch, "REPO", tmp_path)
    monkeypatch.setattr(launch, "git", lambda directory, *args: "" if args[0] == "status" else sha)
    monkeypatch.setattr(launch.subprocess, "run", Mock())
    process = Mock(side_effect=subprocess.CalledProcessError(1, "hardware gate"))
    monkeypatch.setattr(launch, "run_process", process)
    with pytest.raises(subprocess.CalledProcessError):
        launch.train([], "probe", sha, "exp/anvil-normuon", {"GPTLAB_DTYPE": "bfloat16"},
                     True, {"attn_impl": "fa3", "_selected_run": {"reference_run": "probe"}}, 8, Mock())
    process.assert_called_once()
    assert "check_h100.py" in process.call_args.args[0][-3]
    assert json.loads((tmp_path/"probe/launch.json").read_text())["status"] == "failed"


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



def test_tracked_direct_launcher_preflight_without_personal_files(tmp_path):
    import shutil
    checkout = tmp_path/"checkout"
    paths = subprocess.check_output(["git", "ls-files", "--", "experiments/optimizer_comparison", "runs/modal_trainbase.py"],
                                    cwd=recipe.ROOT, text=True).splitlines()
    assert "runs/modal_trainbase.py" in paths
    assert "experiments/optimizer_comparison/training_checks.py" in paths
    for relative in set(paths):
        source = recipe.ROOT/relative
        if not source.is_file() or source.suffix not in (".py", ".json", ".yaml"):
            continue
        target = checkout/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    assert sorted(path.name for path in (checkout/"runs").iterdir()) == ["modal_trainbase.py"]
    cache = tmp_path/"cache"
    data = cache/"data/climbmix-base"
    data.mkdir(parents=True)
    for index in range(5):
        (data/f"shard_{index:05d}.parquet").write_bytes(b"availability fixture")
    tokenizer = cache/"tokenizers/gpt2"
    tokenizer.mkdir(parents=True)
    (tokenizer/"fixture").write_bytes(b"availability fixture")
    code = '''
import builtins, json, sys
from pathlib import Path
checkout, project_python, cache = sys.argv[1:]
sys.path.insert(0, checkout)
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "modal":
        raise AssertionError("Offline plan imported Modal")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from runs import modal_trainbase as launch
from experiments.optimizer_comparison import recipe, training_checks
launch.resolve_source = lambda *args: ("https://example.invalid/repo", "exp/anvil-normuon", "e"*40)
launch.REPO = Path(checkout)
launch.PYTHON = project_python
launch.CACHE = Path(cache)
launch.RUNS = Path(cache)/"outputs"
variant = recipe.SELECTION["variants"][0]
name = variant["model_name"]+"-v2"
arguments = recipe.arguments(variant, name, launch.RUNS/name)
launch.main(["--sha", "e"*40, "--run-name", name, "--gpu", "H100", "--gpus", "8", "--checkpoints", "--dry-run", "--", *arguments])
settings = launch.check_inputs(arguments, name, 8, True, "e"*40)
assert settings["num_steps"] == -1 and settings["stop_after_step"] is None
assert settings["_selected_run"]["reference_run"] == name
# Later independent commands must use a completed Muon reference at this SHA.
reference = launch.RUNS/name
reference.mkdir(parents=True)
(reference/"launch.json").write_text(json.dumps(dict(status="finished", git_sha="e"*40)))
for variant in recipe.SELECTION["variants"][1:]:
    later = variant["model_name"]+"-v2"
    settings = launch.check_inputs(recipe.arguments(variant, later, launch.RUNS/later), later, 8, True, "e"*40)
    assert settings["_selected_run"]["reference_run"] == name
    finished = launch.RUNS/later
    finished.mkdir()
    (finished/"launch.json").write_text(json.dumps(dict(status="finished", git_sha="e"*40)))
print("isolated direct preflight passed")
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(checkout),
                             sys.executable, str(cache)], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert result.stdout.rstrip().endswith("isolated direct preflight passed")
