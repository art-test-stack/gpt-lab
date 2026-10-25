"""Run with python -m unittest discover -s tests/dashboard (no training deps)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_results.py"
spec = importlib.util.spec_from_file_location("export_results", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportTests(unittest.TestCase):
    def test_sparse_eval_and_zero_values_are_preserved_without_filling(self):
        rows = export.normalize_history([
            {"_step": 10, "core/core": 0, "_runtime": 80},
            {"_step": 9, "train/raw_loss": 4.2, "train/total_tokens": 1000},
            {"_step": 10, "eval/bpb": "1.4", "eval/loss": float("nan")},
        ])
        self.assertEqual([r["step"] for r in rows], [9, 10, 10])
        self.assertEqual(rows[1]["core/core"], 0)
        self.assertNotIn("train/total_tokens", rows[1])
        self.assertNotIn("eval/loss", rows[2])
        json.dumps(rows, allow_nan=False)

    def test_export_omits_credentials_paths_and_machine_identity(self):
        clean = export.export_config({
            "system_info": {"hostname": "private-host"}, "model_dir": "/private/path",
            "optimizer": {"name": "muon", "api_key": "secret-value", "nested": {"hf_token": "hf_example"}},
            "n_core_tokens": 100, "meta_config": {"dirname": "/private/path", "tokenizer_cfg": {"vocab_size": 32000}},
        })
        text = json.dumps(clean)
        self.assertNotIn("secret-value", text)
        self.assertNotIn("hf_example", text)
        self.assertNotIn("private", text)
        self.assertEqual(clean["n_core_tokens"], 100)
        self.assertEqual(clean["meta_config"]["tokenizer_cfg"]["vocab_size"], 32000)

    def test_wandb_export_does_not_filter_sparse_rows_with_keys(self):
        class Run:
            name = "Muon baseline"
            id = "abc"
            group = "optimizer-comparison"
            state = "finished"
            url = "https://wandb.ai/e/p/runs/abc"
            config = {"random_seed": 42}
            def scan_history(self):
                yield {"_step": 1, "train/raw_loss": 3}
                yield {"_step": 2, "core/core": 0.1}
        result = export.export_wandb("e/p/abc", SimpleNamespace(run=lambda path: Run()))
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("core/core", result["history"][1])

    def test_append_replaces_same_run_and_preserves_other_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            export.write_snapshot(output, [{"id": "a", "history": []}, {"id": "b", "history": []}])
            export.write_snapshot(output, [{"id": "a", "history": [{"step": 1}]}], append=True)
            data = json.loads(output.read_text())
            self.assertEqual(len(data["runs"]), 2)
            self.assertEqual(data["runs"][0]["history"], [{"step": 1}])

    def test_cli_assembles_a_space_without_importing_torch_or_wandb(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history = base / "history.jsonl"
            history.write_text('{"_step":0,"train/raw_loss":4}\n{"_step":1,"core/core":0.2}\n')
            export.main(["--history", str(history), "--id", "test", "--output", str(base / "results.json"), "--space-dir", str(base / "space")])
            self.assertTrue((base / "space" / "index.html").exists())
            data = json.loads((base / "space" / "results.json").read_text())
            self.assertEqual(data["runs"][0]["history"][1]["core/core"], 0.2)

    def test_missing_step_is_rejected(self):
        with self.assertRaises(ValueError):
            export.normalize_history([{"eval/bpb": 1.2}])


if __name__ == "__main__":
    unittest.main()
