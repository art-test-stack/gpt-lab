"""Include experiment-owned archive checks in the existing dashboard/CI suite."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("modal_results_checks", Path(__file__).resolve().parents[2] / "experiments/modal_results/check.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
ArchiveTests = checks.ArchiveTests
