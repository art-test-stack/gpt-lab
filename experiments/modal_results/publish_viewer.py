"""Update only the results Space interface, reusing all previously published evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

ASSETS = Path(__file__).resolve().parents[2] / "spaces/results"
VIEWER_FILES = ("index.html", "app.js", "data.mjs", "style.css", "evidence.js", "evidence_data.mjs", "logo.png")


def publish_viewer(repo_id, api=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="space")
    if getattr(info, "sdk", None) != "static":
        raise ValueError("The destination must be an existing static Hugging Face Space")
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        for name in VIEWER_FILES:
            shutil.copy2(ASSETS / name, stage / name)
        shutil.copytree(ASSETS / "vendor", stage / "vendor")
        return api.upload_folder(
            repo_id=repo_id, repo_type="space", folder_path=str(stage),
            commit_message="Show experiment families and readable run comparisons on the landing page",
            parent_commit=info.sha,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="ai-testack/gpt-lab")
    args = parser.parse_args()
    print(publish_viewer(args.space))


if __name__ == "__main__":
    main()
