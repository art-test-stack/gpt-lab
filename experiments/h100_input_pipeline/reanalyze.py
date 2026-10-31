"""Offline correction only: preserve the source report and unmodified raw trace."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

from recipe import distribution
from tracing import summarize


def reanalyze(trace_path, report_path):
    source_paths = (trace_path,report_path,report_path.with_name("harness_manifest.json"))
    hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    report = json.loads(report_path.read_text())
    manifest = json.loads(source_paths[2].read_text())
    # Verify the actual uploaded runner, not an assumption about the current checkout.
    runner = subprocess.check_output(["git","show",f"{report['harness_sha']}:experiments/h100_input_pipeline/runner.py"],
                                     cwd=Path(__file__).resolve().parents[2])
    assert hashlib.sha256(runner).hexdigest() == manifest["h100_input_pipeline/runner.py"]
    diagnostic = next(node for node in ast.parse(runner).body if isinstance(node,ast.FunctionDef) and node.name == "diagnostic")
    measured_loop = next(node for node in diagnostic.body if isinstance(node,ast.For)
                         and isinstance(node.iter,ast.Name) and node.iter.id == "ORDER")
    calls = [node for node in ast.walk(diagnostic) if isinstance(node,ast.Call)]
    profile_line = next(node.lineno for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr == "profile")
    analysis_line = next(node.lineno for node in calls if isinstance(node.func,ast.Name) and node.func.id == "summarize")
    assert measured_loop.end_lineno < profile_line < analysis_line
    for call in ast.walk(measured_loop):
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id == "run_updates":
            assert not any(kw.arg == "profile" and not (isinstance(kw.value,ast.Constant) and kw.value.value is False)
                           for kw in call.keywords)
    resolved = report["resolved_batch"]
    assert (resolved["device_batch_size"] * resolved["seq_len"] * resolved["world_size"]
            * resolved["n_acc_steps"] == resolved["total_batch_size"])
    verified = {}
    for name,leg in report["legs"].items():
        assert len(leg["timing"]["samples_ms"]) == report["measured_updates"]
        stats = distribution(leg["timing"]["samples_ms"])
        for key in ("mean_ms","median_ms","p05_ms","p95_ms"):
            assert math.isclose(stats[key],leg["timing"][key],rel_tol=1e-12)
        rate = report["measured_updates"] * resolved["total_batch_size"] / leg["measured_wall_seconds"]
        assert math.isclose(rate,leg["global_tokens_per_second"],rel_tol=1e-12)
        verified[name] = dict(tokens_per_second=rate,median_ms=stats["median_ms"],p95_ms=stats["p95_ms"])
    controls = [report["legs"][name] for name in ("production-before","production-after")]
    faster_rate = max(leg["global_tokens_per_second"] for leg in controls)
    faster_median = min(leg["timing"]["median_ms"] for leg in controls)
    drift = abs(controls[0]["timing"]["median_ms"]-controls[1]["timing"]["median_ms"])
    gains = {}
    for name in ("cpu-cache","gpu-cache"):
        leg = report["legs"][name]
        spread = max(item["timing"]["p95_ms"]-item["timing"]["p05_ms"] for item in (*controls,leg))
        saving = faster_median-leg["timing"]["median_ms"]
        exceeds = saving > max(drift,spread)
        assert exceeds == report["comparisons"][name]["saving_exceeds_control_drift_and_within_leg_width"]
        gains[name] = dict(throughput_gain_percent_vs_faster_control=100*(leg["global_tokens_per_second"]/faster_rate-1),
                           median_saving_ms=saving,variability_threshold_ms=max(drift,spread),
                           exceeds_variability_threshold=exceeds)
    profile = summarize(trace_path,expected_microsteps=resolved["n_acc_steps"],
                        expected_steps=range(report["warmup_updates"],report["warmup_updates"]+2))
    assert hashes == {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    here = Path(__file__).resolve().parent
    return report | dict(profile=profile,complete=True,offline_reanalysis=dict(
        source_sha256=hashes,analysis_code_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                                                for name in ("tracing.py","reanalyze.py")},
        original_measurements_unchanged=True,verified_unprofiled_measurements=verified,gains=gains,
        independence_evidence=dict(uploaded_runner_matches_manifest=True,measured_loop_end_line=measured_loop.end_lineno,
                                   profiler_start_line=profile_line,analysis_call_line=analysis_line,
                                   note="All uninstrumented measured legs completed before the separate profiler and failing analyzer; throughput recomputed from saved global-token counts and wall times, median/p95 from saved samples."),
        scope="Corrected batch-16 analysis only; source report and raw trace unchanged; batch 32 remains unmeasured."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new corrected-report path; never overwrite existing results")
    result = reanalyze(args.trace,args.report)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output),gains=result["offline_reanalysis"]["gains"],
                         updates=result["profile"]["updates"]),indent=2))
