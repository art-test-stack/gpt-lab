"""Offline checks: python -m unittest discover -s experiments/modal_results -p check.py."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("modal_results_collect", Path(__file__).with_name("collect.py"))
collect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collect
SPEC.loader.exec_module(collect)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.input, self.output = self.base / "volume", self.base / "space"
        self.input.mkdir()

    def write(self, path, value):
        target = self.input / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return target

    def build(self, **kwargs):
        return collect.collect(collect.LocalSource(self.input), self.output, **kwargs)

    def test_models_and_cache_files_are_excluded_before_read(self):
        paths = ["run/checkpoints/step1/config.json", "run/model.safetensors", "run/state.pt", "run/cache/summary.json", "run/weights.bin", "run/weights.pt.gz"]
        for path in paths:
            self.write(path, "NOT EVIDENCE")
        self.write("comm-a/manifest.json", {"status": "failed", "sha": "a" * 40})
        self.write("comm-a/console.log", "NCCL failed")
        class GuardedSource(collect.LocalSource):
            def chunks(self, path):
                if path in paths:
                    raise AssertionError("model bytes were requested")
                yield from super().chunks(path)
        catalog = collect.collect(GuardedSource(self.input), self.output)
        self.assertEqual(len(catalog["runs"]), 1)
        run = catalog["runs"][0]
        self.assertEqual((run["experiment"], run["status"], run["commits"]), ("communication_buckets", "failed", ["a" * 40]))
        self.assertEqual(len(run["artifacts"]), 2)
        self.assertTrue(catalog["excluded"])

    def test_metrics_keep_context_and_no_missing_values_are_invented(self):
        self.write("comm-test/manifest.json", {"status": "completed", "candidates": {"baseline": {"steady_train_loop_tokens_per_second": 618857}, "pipeline": {"steady_train_loop_tokens_per_second": 615619}}})
        catalog = self.build()
        measurements = catalog["runs"][0]["measurements"]
        self.assertEqual([m["context"] for m in measurements], ["candidates/baseline", "candidates/pipeline"])
        self.assertEqual(measurements[0]["values"], {"steady_train_loop_tokens_per_second": 618857})
        self.assertNotIn("history", catalog["runs"][0])

    def test_json_and_logs_redact_credentials_preserve_token_counts_and_hashes(self):
        token = "hf_" + "a" * 32
        self.write("cg-test/summary.json", {"status": "completed", "api_key": "private", "training": {"total_tokens": 1234, "HF_TOKEN": token}, "measurement": float("nan")})
        self.write("cg-test/console.log", f"HF_TOKEN={token}\nAPI_KEY=private\nAuthorization: Bearer abcdefghijklmnopqrst\n")
        run = self.build()["runs"][0]
        text = "\n".join((self.output / a["path"]).read_text() for a in run["artifacts"])
        self.assertNotIn(token, text)
        self.assertNotIn("private", text)
        self.assertIn('"total_tokens": 1234', text)
        self.assertTrue(all(a["transformed"] for a in run["artifacts"]))
        self.assertTrue(all(len(a["source_sha256"]) == 64 and len(a["sha256"]) == 64 for a in run["artifacts"]))

    def test_trace_streaming_gzip_and_invalid_reports_remain_downloadable(self):
        p = self.input / "fp8-test/trace.json.gz"
        p.parent.mkdir()
        with gzip.open(p, "wt") as handle:
            handle.write('{"traceEvents":[],"api_key":"private"}')
        self.write("fp8-test/partial.json", '{"status":')
        catalog = self.build()
        run = catalog["runs"][0]
        self.assertEqual(run["status"], "unknown")
        self.assertFalse(catalog["errors"])
        trace = next(a for a in run["artifacts"] if a["kind"] == "trace")
        with gzip.open(self.output / trace["path"], "rt") as handle:
            self.assertNotIn("private", handle.read())
        partial = next(a for a in run["artifacts"] if a.get("parse_error"))
        self.assertEqual((self.output / partial["path"]).read_text(), '{"status":')

    def test_repeat_and_partial_collection_merge_by_stable_run_id(self):
        self.write("cg-a/summary.json", {"steady_tokens_per_second": 100})
        self.write("fp8-b/report.json", {"median_ms": 20})
        first = self.build()
        second = self.build(prefixes=["cg-a"])
        self.assertEqual({r["id"] for r in first["runs"]}, {r["id"] for r in second["runs"]})
        self.assertEqual(len(second["runs"]), 2)
        self.assertTrue((self.output / "evidence.js").exists())

    def test_unknown_family_mapping_and_no_symlink_following(self):
        self.write("unusual-run/report.json", {"status": "failed"})
        (self.input / "outside.json").symlink_to(self.write("hidden-target/report.json", {"status": "completed"}))
        self.assertEqual(self.build()["runs"][-1]["experiment"], "other")
        result = self.build(mapping={"unusual-run": "optimizer_comparison"})
        self.assertIn("optimizer_comparison", {r["experiment"] for r in result["runs"]})
        self.assertFalse(any(a["name"] == "outside.json" for r in result["runs"] for a in r["artifacts"]))

    def test_path_traversal_and_overlapping_output_are_rejected(self):
        for path in ("../evil.json", "//evil.json", "x/../../evil.json", "x\\evil.json"):
            with self.assertRaises(ValueError):
                collect.safe_path(path)
        with self.assertRaises(ValueError):
            collect.collect(collect.LocalSource(self.input), self.input / "output")

    def test_publish_preserves_training_snapshot_metadata_and_previous_archive(self):
        self.write("comm-a/manifest.json", {"status": "completed"})
        self.build()
        remote = {"schema_version": 1, "experiments": [{"id": "cuda_graphs", "title": "CUDA"}], "runs": [{"id": "older", "name": "old", "experiment": "cuda_graphs", "artifacts": []}]}
        class Api:
            def repo_info(self, **kwargs):
                return SimpleNamespace(sdk="static", sha="HEAD")
            def upload_folder(self, **kwargs):
                root = Path(kwargs["folder_path"])
                assert not (root / "results.json").exists()
                assert not (root / "README.md").exists()
                assert "delete_patterns" not in kwargs
                assert kwargs["parent_commit"] == "HEAD"
                assert {r["id"] for r in json.loads((root / "experiments.json").read_text())["runs"]} == {"older", "local:local:comm-a"}
                return "published"
        self.assertEqual(collect.publish(self.output, "owner/space", api=Api(), read_remote=lambda *args: remote), "published")

    def test_nested_success_does_not_mark_failed_parent_completed(self):
        self.write("cg-test/nested/summary.json", {"status": "completed"})
        self.write("cg-test/console.log", "interrupted")
        self.assertEqual(self.build()["runs"][0]["status"], "unknown")

    def test_modal_adapter_only_lists_and_reads_existing_volume(self):
        calls = []
        class Volume:
            def iterdir(self, path, *, recursive):
                calls.append(("list", path, recursive))
                return [SimpleNamespace(path="/cg-a", size=0, type=SimpleNamespace(name="DIRECTORY")),
                        SimpleNamespace(path="/cg-a/summary.json", size=22, type=SimpleNamespace(name="FILE")),
                        SimpleNamespace(path="/cg-a/model.pt", size=10**9, type=SimpleNamespace(name="FILE"))]
            def read_file(self, path):
                calls.append(("read", path))
                self.assert_path = path
                if path.endswith(".pt"):
                    raise AssertionError("model downloaded")
                return iter([b'{"status":"completed"}'])
        def from_name(name, **kwargs):
            calls.append(("open", name, kwargs))
            return Volume()
        with patch.dict(sys.modules, {"modal": SimpleNamespace(Volume=SimpleNamespace(from_name=from_name))}):
            source = collect.ModalSource("gpt-lab-runs", "main")
            result = collect.collect(source, self.output, volume="gpt-lab-runs", environment="main")
        self.assertEqual(calls[0], ("open", "gpt-lab-runs", {"environment_name": "main", "create_if_missing": False}))
        self.assertEqual([call for call in calls if call[0] == "read"], [("read", "cg-a/summary.json")])
        self.assertEqual(result["runs"][0]["source"]["type"], "modal-volume")

    def test_publish_refuses_non_static_space_before_any_upload(self):
        api = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sdk="gradio", sha="HEAD"))
        with self.assertRaisesRegex(ValueError, "static"):
            collect.publish(self.output, "owner/space", api=api)


if __name__ == "__main__":
    unittest.main()
