"""Offline accounting fixtures; no W&B, GPU, or model imports."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "repair_flops_reporting", Path(__file__).parents[1] / "scripts/repair_flops_reporting.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def d20():
    return dict(source_run_id="entity/project/d20", classification="causal-pair",
                source_sha="b2ab865", evidence=["recorded model_card and source estimator"],
                d_model=1280, context_length=2048, n_layers=20,
                left_windows=[258, 258, 258, -1] * 5,
                padded_vocab_size=50304, head_parameters=64389120,
                block_parameters=393217200, original_flops_per_token=2880234382.5)


class ReportingRepairTests(unittest.TestCase):
    def test_recorded_d20_and_swiglu_estimators(self):
        r = d20()
        self.assertEqual(repair.estimate(r, causal=True), 2880234382.5)
        self.assertEqual(repair.estimate(r), 2962367520)
        r.update(d_model=768, n_layers=12, left_windows=[512, 512, 512, -1] * 3,
                 block_parameters=113246640, head_parameters=38633472)
        self.assertEqual(repair.estimate(r, causal=True), 976837536)
        self.assertEqual(repair.estimate(r), 1010371104)

    def test_pr45_attention_cases_and_tied_head(self):
        # Same attention fixtures as PR #45, with an explicitly counted tied head.
        for context, windows, expected in [
            (4, [-1] * 4, 3072), (4, [1, 1, 1, -1], 1344),
            (4, [2, -1, 2, -1], 2304), (4, [4, 4, 4, -1], 3072),
            (4, [8, 8, 8, -1], 3072), (1, [1, 1, 1, -1], 768),
        ]:
            r = d20()
            r.update(d_model=16, context_length=context, n_layers=4,
                     left_windows=windows, block_parameters=100, head_parameters=1024,
                     padded_vocab_size=64, embedding_sharing="tied")
            self.assertEqual(repair.estimate(r), 6 * 1124 + expected)

    def test_sparse_steps_offset_counters_and_measurements(self):
        r = d20()
        doc = {"config": {"n_flops_per_token": r["original_flops_per_token"],
                          "target_flops": 123, "n_steps": 4364, "gpu_peak_flops": 989e12},
               "history": [{"_step": 0, "_timestamp": 1234, "_runtime": 80,
                            "train/total_tokens": 1048576, "train/total_training_flops": 0,
                            "train/mfu": 40, "train/loss": 3},
                           {"_step": 400, "eval/bpb": 0.9, "train/mfu": None},
                           {"_step": 410, "train/total_training_flops": 12345,
                            "train/tokens_per_sec": 100, "train/step_time_ms": 1000}],
               "summary": {"train/mfu": 40},
               "original_attrs": {"train/mfu": {"previousValue": 40}}}
        fixed, changes = repair.repair(doc, r, ["n_flops_per_token", "train/mfu", "train/total_training_flops"])
        factor = repair.estimate(r) / r["original_flops_per_token"]
        self.assertEqual(fixed["history"][0]["train/total_training_flops"], 0)
        self.assertEqual(fixed["history"][2]["train/total_training_flops"], 12345 * factor)
        self.assertEqual(fixed["history"][1], doc["history"][1])
        for before, after in zip(doc["history"], fixed["history"]):
            for key in before.keys() - {"train/mfu", "train/total_training_flops"}:
                self.assertEqual(after[key], before[key])
        self.assertEqual(fixed["config"]["target_flops"], 123)
        self.assertEqual(fixed["config"]["gpu_peak_flops"], 989e12)
        self.assertEqual(fixed["original_attrs"], doc["original_attrs"])
        self.assertEqual(repair.repair(fixed, r, ["train/mfu"]), (fixed, []))
        self.assertEqual(len(changes), 5)

    def test_compatible_unresolved_and_conflicting_evidence(self):
        r = d20()
        r.update(classification="nanochat-compatible", original_flops_per_token=repair.estimate(r))
        doc = {"train/mfu": 51.61060690246965}
        self.assertEqual(repair.repair(doc, r, ["train/mfu"]), (doc, []))
        for bad in [{**r, "classification": "unresolved"},
                    {**r, "original_flops_per_token": 1},
                    {**r, "left_windows": []}, {**r, "head_parameters": 1}]:
            with self.assertRaises(ValueError):
                repair.repair(doc, bad, ["train/mfu"])

    def test_plan_restores_and_rejects_changed_files(self):
        temporary_root = Path(__file__).parents[1] / ".tmp/codex/flops-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as tmp:
            root = Path(tmp)
            original = {"runs": [{"train/mfu": 40, "loss": 3}]}
            repair.write(root / "before/data.json", original)
            audit = {"targets": [{"file": "data.json", "path": ["runs", 0],
                                  "accounting": d20(), "metric_keys": ["train/mfu"]}]}
            manifest = repair.plan(audit, root / "before", root / "after")
            self.assertEqual(repair.plan(audit, root / "after", root / "second")["changes"], [])
            repair.restore(manifest, root / "after", root / "restored")
            self.assertEqual(repair.read(root / "restored/data.json"), original)
            repair.write(root / "after/data.json", {})
            with self.assertRaises(ValueError):
                repair.restore(manifest, root / "after", root / "restored")

    def test_replay_verifies_explicit_source_times(self):
        doc = {"history": [{"_step": 20, "_timestamp": 1234, "_runtime": 42, "train/loss": 3}],
               "summary": {"train/mfu": 45, "_runtime": 99}}
        uploaded = [{"_step": 20, "_timestamp": 9999, "_runtime": 1,
                     "source/step": 20, "source/timestamp": 1234,
                     "source/runtime": 42, "train/loss": 3}]
        summary = {"train/mfu": 45, "_runtime": 1, "source_summary/runtime": 99}
        repair.verify_replay(doc, uploaded, summary)
        uploaded[0]["source/runtime"] = 1
        with self.assertRaises(ValueError):
            repair.verify_replay(doc, uploaded, summary)


if __name__ == "__main__":
    unittest.main()
