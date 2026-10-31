"""Local handoff checks; these never import Modal or launch GPU work."""
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark/modal_communication.py"
    return runpy.run_path(str(path))


def result(positions):
    return dict(status="completed", selected_optimizer="muon", initial_fingerprints_per_rank=[dict(
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


def test_required_files_are_root_relative_and_present():
    source = launcher()
    for name in source["REQUIRED"]:
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert (source["ROOT"] / name).is_file()


def test_dry_run_never_imports_modal(monkeypatch, capsys):
    source = launcher()
    main = source["main"]
    monkeypatch.setitem(main.__globals__, "verify_source", lambda sha: "https://github.com/art-test-stack/gpt-lab.git")
    monkeypatch.setitem(sys.modules, "modal", None)
    main(["sweep", "--sha", "a" * 40, "--run-name", "handoff-test", "--dry-run"])
    assert '"gpu": "A100-80GB:4"' in capsys.readouterr().out


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
