"""Summarize a CORE sweep from xctrace's signpost and GPU interval XML exports."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def xml_rows(path):
    trace = ET.parse(path)
    for node in trace.iter("node"):
        by_id = {item.get("id"): item for item in node.iter() if item.get("id") is not None}
        fields = [item.text for item in node.findall("schema/col/mnemonic")]
        for row in node.findall("row"):
            yield dict(zip(fields, (by_id[item.get("ref")] if item.get("ref") else item for item in row)))


def bounds(row):
    start = int(row["start"].text)
    return start, start + int(row["duration"].text)


def merge(intervals, sweep):
    combined = []
    for start, end in sorted(intervals):
        start, end = max(start, sweep[0]), min(end, sweep[1])
        if start >= end:
            continue
        if combined and start <= combined[-1][1]:
            combined[-1] = (combined[-1][0], max(end, combined[-1][1]))
        else:
            combined.append((start, end))
    return combined


def summarize(signposts, metal):
    posts = [row for row in xml_rows(signposts) if row["subsystem"].text == "org.gpt_lab.core"]
    sweep, = [post for post in posts if post["name"].text == "CORE sweep"]
    window = bounds(sweep)
    process = sweep["process"].get("fmt")
    collations = sorted(bounds(row) for row in posts if row["name"].text == "CPU collation")
    submissions = sorted(bounds(row) for row in posts if row["name"].text == "Model submission")
    gpu = [bounds(row) for row in xml_rows(metal)
           if row["process"].get("fmt") == process and row["state"].text == "Active"]
    busy = merge(gpu, window)
    cpu = merge(collations, window)
    if not busy or not cpu:
        raise ValueError("Missing CORE GPU execution or CPU collation intervals")
    if len(collations) != len(submissions):
        raise ValueError("The trace has incomplete CPU collation or model submission intervals")
    adjacent = zip([(window[0], window[0]), *busy], [*busy, (window[1], window[1])])
    gaps = [right[0] - left[1] for left, right in adjacent]
    overlap = sum(max(0, min(a_end, b_end) - max(a_start, b_start))
                  for a_start, a_end in cpu for b_start, b_end in busy)
    return {
        "sweep_seconds": (window[1] - window[0]) / 1e9,
        "gpu_active_seconds": sum(end - start for start, end in busy) / 1e9,
        "gpu_idle_seconds": sum(gaps) / 1e9,
        "gpu_idle_gaps_over_one_ms": sum(gap > 1e6 for gap in gaps),
        "largest_gpu_idle_gap_seconds": max(gaps) / 1e9,
        "cpu_collation_seconds": sum(end - start for start, end in cpu) / 1e9,
        "collation_gpu_overlap_seconds": overlap / 1e9,
        "collation_count": len(collations),
        "collation_after_previous_submission_seconds": sum(
            max(0, end - max(start, submitted))
            for (start, end), (_, submitted) in zip(collations[1:], submissions)
        ) / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    args = parser.parse_args()
    report = {
        name: summarize(args.traces / f"{name}-OSSignpostIntervals.xml",
                        args.traces / f"{name}-metal-gpu-intervals.xml")
        for name in ("current", "prefetch", "deferred", "both")
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
