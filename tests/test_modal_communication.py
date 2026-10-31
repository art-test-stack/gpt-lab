"""Local handoff checks; these never import Modal or launch GPU work."""
import os
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark/modal_communication.py"
    return runpy.run_path(str(path))


def result(positions, optimizer="muon"):
    return dict(status="completed", selected_optimizer=optimizer, initial_fingerprints_per_rank=[dict(
        rank=0, model_sha256="model", optimizer_state_sha256="optimizer", optimizer_groups=["muon"],
        first_batch_sha256="batch", first_batch_position={"offset": 0},
        last_consumed_data_positions=[{"offset": i} for i in positions],
    )])


def test_fresh_process_reset_and_data_prefix_are_checked():
    validate = launcher()["validate_matched"]
    validate(result(range(3)), result(range(5)))
    different = result(range(3))
    different["initial_fingerprints_per_rank"][0]["model_sha256"] = "different"
    with pytest.raises(ValueError, match="Reset mismatch"):
        validate(result(range(3)), different)
    different = result([0, 1, 4])
    with pytest.raises(ValueError, match="data positions diverged"):
        validate(result(range(3)), different)
    validate(result(range(3), "normuon"), result(range(5), "normuon"))
    with pytest.raises(ValueError, match="same selected optimizer"):
        validate(result(range(3)), result(range(3), "normuon"))


def test_required_files_are_root_relative_and_present():
    source = launcher()
    for name in source["REQUIRED"]:
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert (source["ROOT"] / name).is_file()


@pytest.mark.parametrize("optimizer", ["muon", "normuon"])
def test_dry_run_never_imports_modal(monkeypatch, capsys, optimizer):
    source = launcher()
    main = source["main"]
    monkeypatch.setitem(main.__globals__, "verify_source", lambda sha: "https://github.com/art-test-stack/gpt-lab.git")
    monkeypatch.setitem(sys.modules, "modal", None)
    main(["sweep", "--optimizer", optimizer, "--sha", "a" * 40, "--run-name", "handoff-test", "--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["gpu"] == "A100-80GB:4" and output["optimizer"] == optimizer
    assert output["screen_updates"] - output["warmup_updates"] >= 30
    assert output["confirmation_updates"] == 150 and output["min_throughput_gain"] == .01


def test_distributed_deadline_kills_process_and_persists(tmp_path, monkeypatch):
    run = launcher()["run_process"]
    monkeypatch.setitem(run.__globals__, "REPO", tmp_path)
    commits = []
    volume = SimpleNamespace(commit=lambda: commits.append(True))
    with pytest.raises(TimeoutError, match="process tree terminated"):
        run([sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "candidate", dict(os.environ), volume, .05)
    assert commits == [True]
    assert (tmp_path / "candidate/command.json").is_file()


def test_skipped_nccl_is_rejected(tmp_path, monkeypatch):
    check = launcher()["correctness"]
    monkeypatch.setitem(check.__globals__, "REPO", tmp_path)
    def skipped(command, directory, *args):
        basetemp = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--basetemp=")))
        assert basetemp.parent == tmp_path / ".tmp/codex/modal-correctness"
        assert basetemp.parent.is_dir()
        directory.mkdir(parents=True)
        (directory / "junit.xml").write_text('<testsuites><testsuite tests="1" skipped="1" /></testsuites>')
    monkeypatch.setitem(check.__globals__, "run_process", skipped)
    with pytest.raises(ValueError, match="must execute and pass, not skip"):
        check(tmp_path / "check", "muon", ["pipeline"], 2, {}, SimpleNamespace(commit=lambda: None), 1)


def test_source_verification_uses_this_branch_not_the_preserved_muon_tip(tmp_path, monkeypatch):
    verify = launcher()["verify_source"]
    monkeypatch.setitem(verify.__globals__, "ROOT", tmp_path)
    calls = []
    def git(*args):
        calls.append(args)
        return {
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("rev-parse", "HEAD"): "a" * 40,
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "exp/comm-optimizer-sweeps",
            ("ls-remote", "--exit-code", "origin", "refs/heads/exp/comm-optimizer-sweeps"): "a" * 40 + "\trefs/heads/exp/comm-optimizer-sweeps",
            ("remote", "get-url", "origin"): "git@github.com:example/repo.git",
        }.get(args, "")
    monkeypatch.setitem(verify.__globals__, "git", git)
    assert verify("a" * 40) == "https://github.com/example/repo.git"
    assert any(call[-1] == "refs/heads/exp/comm-optimizer-sweeps" for call in calls)
    assert not any(call[-1] == "refs/heads/exp/comm-buckets" for call in calls)


def test_profile_preflight_rejects_another_optimizer_or_source(tmp_path, monkeypatch):
    preflight = launcher()["preflight"]
    monkeypatch.setitem(preflight.__globals__, "RUNS", tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setitem(preflight.__globals__, "CACHE", cache)
    shards = cache / "data/climbmix-base"
    shards.mkdir(parents=True)
    for index in range(5):
        (shards / f"shard_{index:05d}.parquet").write_bytes(b"fixture")
    tokenizer = cache / "tokenizers/gpt2"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    directory = tmp_path / "previous"
    directory.mkdir()
    previous = dict(status="completed", stage="sweep", sha="a" * 40,
                    selected_optimizer="normuon", candidate_mode="pipeline")
    (directory / "manifest.json").write_text(json.dumps(previous))
    for optimizer, sha in (("muon", "a" * 40), ("normuon", "b" * 40)):
        with pytest.raises(ValueError, match="same SHA and optimizer"):
            preflight("profile", "new", "previous", sha, optimizer)
    assert preflight("profile", "new", "previous", "a" * 40, "normuon")


def test_nccl_request_uses_the_selected_optimizer(tmp_path, monkeypatch):
    check = launcher()["correctness"]
    monkeypatch.setitem(check.__globals__, "REPO", tmp_path)
    def completed(command, directory, env, *args):
        assert env["GPTLAB_COMM_TEST_OPTIMIZER"] == "normuon"
        assert env["GPTLAB_COMM_TEST_MODES"] == "buckets,pipeline,backward"
        directory.mkdir()
        (directory / "junit.xml").write_text('<testsuites><testsuite tests="1" /></testsuites>')
        for rank in range(4):
            (directory / f"correctness.rank{rank}.json").write_text(json.dumps(
                dict(ok=True, optimizer="normuon", backend="nccl")))
    monkeypatch.setitem(check.__globals__, "run_process", completed)
    rows = check(tmp_path / "nccl", "normuon", ["buckets", "pipeline", "backward"],
                 4, {}, SimpleNamespace(commit=lambda: None), 1)
    assert len(rows) == 4


@pytest.mark.parametrize("optimizer", ["muon", "normuon"])
@pytest.mark.parametrize("screen_ratio,full_ratio,selected", [
    (1.01, 1.01, "pipeline"), (1.005, 1.02, "baseline"),
    (1.02, 1.005, "baseline"), (.99, .99, "baseline"),
])
def test_sweep_and_profile_select_optimizer_and_retain_baseline(
        tmp_path, monkeypatch, optimizer, screen_ratio, full_ratio, selected):
    run = launcher()["run_suite"]
    globals_ = run.__globals__
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    repo.mkdir()
    runs.mkdir()
    monkeypatch.setitem(globals_, "REPO", repo)
    monkeypatch.setitem(globals_, "RUNS", runs)
    monkeypatch.setattr(globals_["subprocess"], "check_output", lambda *a, **kw: "a" * 40 + "\n")
    requests, commits = [], []
    def correctness(directory, selected_optimizer, modes, world, *args):
        assert selected_optimizer == optimizer and world == 4
        assert tuple(modes) == ("buckets", "pipeline", "backward")
        return [dict(ok=True, optimizer=selected_optimizer)]
    monkeypatch.setitem(globals_, "correctness", correctness)
    def process(command, directory, *args):
        if "-c" in command:
            return
        mode = command[command.index("--communication") + 1]
        updates = int(command[command.index("--updates") + 1])
        assert command[command.index("--optimizer") + 1] == optimizer
        assert command[command.index("--warmup-updates") + 1] == "25"
        assert float(command[command.index("--bucket-mib") + 1]) == 128
        requests.append((directory.name, updates, "--profile" in command))
        ratio = 1 if mode == "baseline" else (
            (screen_ratio if updates == 55 else full_ratio) if mode == "pipeline" else .98)
        row = result(range(updates), optimizer)
        row.update(communication=mode, updates_requested=updates, steady_updates=updates - 25,
                   steady_train_loop_tokens_per_second=100000 * ratio,
                   steady_wall_step_time_ms={}, steady_train_step_time_ms={},
                   startup_to_first_update_rank_max_s=1, process_startup_to_first_update_rank_max_s=1,
                   memory_per_rank=[], communication_rank_max_summary={},
                   world_size=4, git_sha="a" * 40, completed_steps_by_rank=[updates] * 4,
                   scheduler_horizon_by_rank=[2317] * 4, bucket_mib=128)
        directory.mkdir()
        (directory / "communication-results.json").write_text(json.dumps(row))
    monkeypatch.setitem(globals_, "run_process", process)
    volume = SimpleNamespace(commit=lambda: commits.append(True))
    manifest = run("sweep", "sweep", None, "a" * 40, 128, optimizer, volume, 1)
    assert manifest["recommended_mode"] == selected
    assert manifest["candidate_mode"] == "pipeline"
    assert manifest["selected_optimizer"] == optimizer
    assert [updates for _, updates, _ in requests] == [55, 55, 55, 55, 150, 150]
    assert not any(profile for _, _, profile in requests)
    assert len(commits) >= 2 * len(requests)
    profile = run("profile", "profile", "sweep", "a" * 40, 64, optimizer, volume, 1)
    assert profile["selected_optimizer"] == optimizer and profile["bucket_mib"] == 128
    assert requests[-2:] == [("profile-baseline", 55, True), ("profile-pipeline", 55, True)]
