"""Fixed production recipe and offline reporting helpers for this diagnostic."""

import math
import statistics


SHA = "3fcd250669bb2f0bb5c25397c058f8cf7bd7d429"
REPOSITORY = "https://github.com/art-test-stack/gpt-lab.git"
TRAIN_ARGS = [
    "train", "auto", "--model-name", "ic1", "--ds-name", "climbmix-base",
    "--tokenizer-model", "gpt2", "--depth", "12", "--aspect-ratio", "64",
    "--d-head", "128", "--max-seq-len", "2048", "--act-func", "swiglu",
    "--tie-embeddings", "embs;value_embs;head", "--window-pattern", "SSSL",
    "--window-size", "512", "--softcap", "18.0", "--attn-impl", "fa3",
    "--device-batch-size", "16", "--total-batch-size", "524288",
    "--n-acc-steps", "16", "--num-steps", "2317", "--warmup-steps", "300",
    "--warmdown-ratio", "0.65", "--final-lr-frac", "0.05",
    "--save-every", "0", "--eval-bpb-every", "0", "--eval-core-every", "0",
    "--sample-every", "0", "--log-every", "0", "--board", "dummy",
]


def distribution(samples):
    ordered = sorted(samples)
    if not ordered or not all(math.isfinite(x) and x > 0 for x in ordered):
        raise ValueError("Expected positive finite timing samples")
    def percentile(p):
        index = (len(ordered)-1)*p
        low = int(index)
        high = min(low+1, len(ordered)-1)
        return ordered[low] + (ordered[high]-ordered[low])*(index-low)
    return dict(samples_ms=samples, mean_ms=statistics.mean(samples),
                median_ms=statistics.median(samples), stdev_ms=statistics.stdev(samples) if len(samples)>1 else 0,
                p05_ms=percentile(.05), p95_ms=percentile(.95),
                tokens_per_second=524288/(statistics.mean(samples)/1000))


def compare_aba(first, fp8, last):
    a, b, c = (distribution(x) for x in (first, fp8, last))
    gain = min(a["median_ms"], c["median_ms"]) - b["median_ms"]
    drift = abs(a["median_ms"]-c["median_ms"])
    spread = max(x["p95_ms"]-x["p05_ms"] for x in (a,b,c))
    return dict(bf16_before=a, fp8_forward=b, bf16_after=c,
                bf16_drift_ms=drift, conservative_median_saving_ms=gain,
                max_within_leg_p05_p95_width_ms=spread,
                gain_exceeds_observed_variation=gain>max(drift,spread),
                fp8_p95_below_both_bf16_p05=b["p95_ms"]<min(a["p05_ms"],c["p05_ms"]),
                criterion="Median saving versus the faster BF16 leg exceeds both BF16 median drift and every leg's p05–p95 width; descriptive, not a statistical significance test.")


def union_length(intervals):
    end = -math.inf
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop-max(start,end))
        end = max(end,stop)
    return total


def partition_intervals(items, start, stop):
    """Exclusive wall time: overlaps get their own bucket, never summed twice."""
    edges = [(start, 0, None), (stop, 0, None)]
    for left, right, category in items:
        left, right = max(start,left), min(stop,right)
        if right>left:
            edges.extend([(left,1,category),(right,-1,category)])
    active, result, previous = {}, {}, start
    for point, delta, category in sorted(edges, key=lambda item:item[0]):
        labels = sorted(k for k,v in active.items() if v)
        label = labels[0] if len(labels)==1 else "overlap" if labels else "gpu_idle"
        result[label] = result.get(label,0) + point-previous
        if category is not None:
            active[category] = active.get(category,0)+delta
        previous = point
    return result
