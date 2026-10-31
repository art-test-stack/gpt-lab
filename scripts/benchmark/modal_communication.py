"""Compatibility entry point; implementation moved to experiments/."""
from pathlib import Path
import runpy

globals().update(runpy.run_path(str(Path(__file__).resolve().parents[2] / "experiments/communication_buckets/modal_communication.py"), run_name=__name__))
