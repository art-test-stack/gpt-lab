"""Attribute profiler CUDA intervals; never add overlapping kernel durations."""

from collections import defaultdict
from bisect import bisect_right
import json
from pathlib import Path
import re

from common import partition_intervals, union_length


def node_category(node, backward=False):
    text = json.dumps(node).lower()
    direction = "backward" if backward else "forward"
    if "swiglufeedforward" in text or "ffn" in text:
        if any(word in text for word in ("float8_scaling", "float8_utils", "torchao_fp8", "amax", "quantize", "_to_copy", "convert_element_type")) and "scaled_mm" not in node.get("target", ""):
            return f"ffn/{direction}/quantization_casts_or_fused"
        return f"ffn/{direction}/other"
    if any(word in text for word in ("lm_head", "cross_entropy", "model/loss.py", "log_softmax", "nll_loss", "logits")):
        return f"head_loss/{direction}"
    if "causalselfattention" in text or "attention" in text:
        return f"attention/{direction}"
    return f"other_model/{direction}"


def generated_kernel_categories(graphs, debug_dir):
    """Use Inductor's post-grad provenance, leaving ambiguous kernels mixed."""
    result = defaultdict(set)
    for graph in graphs:
        scope={node["name"]:node_category(node,graph["backward"]) for node in graph["nodes"]}
        for kernel,names in graph.get("provenance",{}).get("cppCodeToPost",{}).items():
            result[kernel].update(scope[name] for name in names if name in scope)
    for path in Path(debug_dir).rglob("inductor_provenance_tracking_node_mappings.json"):
        code_path=path.parent/"output_code.py"
        code=code_path.read_text() if code_path.exists() else ""
        aot_id=re.search(r"AOT ID: ([^\n]*)",code)
        matching=[graph for graph in graphs if code==graph.get("source_code") or
                  (aot_id and aot_id.group(0) in graph.get("source_code",""))]
        nodes=defaultdict(set)
        for graph in matching:
            for node in graph["nodes"]:
                nodes[node["name"]].add(node_category(node,graph["backward"]))
        mapping = json.loads(path.read_text()).get("cppCodeToPost",{})
        for kernel, names in mapping.items():
            for name in names:
                result[kernel].update(nodes.get(name,()))
    # Source-node comments are also retained when a compiler build omits provenance.
    for graph in graphs:
        scope = {node["name"]:node_category(node,graph["backward"]) for node in graph["nodes"]}
        categories = set()
        for line in graph.get("source_code","").splitlines():
            if "Source Nodes:" in line:
                match = re.search(r"Source Nodes: \[([^]]*)\]",line)
                categories = {scope[name.strip()] for name in match.group(1).split(",") if name.strip() in scope} if match else set()
            match = re.search(r"\b(triton_\w+)\.run\(",line)
            if match and categories:
                result[match.group(1)].update(categories)
    def combined(categories):
        if len(categories)==1:
            return next(iter(categories))
        prefixes={category.rsplit("/",1)[0] for category in categories}
        if len(prefixes)==1:
            return next(iter(prefixes))+"/mixed_fused_work"
        return "mixed/"+"+".join(sorted(categories))
    return {kernel:combined(categories)
            for kernel,categories in result.items() if categories}


def trace_report(path, kernel_categories):
    events = json.loads(Path(path).read_text())["traceEvents"]
    spans = [event for event in events if event.get("ph")=="X" and event.get("dur",0)>0]
    updates = [event for event in spans if event.get("name","").startswith("diagnostic/update/")]
    annotations = [event for event in spans if event.get("name","").startswith(("diagnostic/gemm/","diagnostic/kernel/","diagnostic/optimizer"))]
    by_tid = defaultdict(list)
    for event in annotations:
        by_tid[event.get("tid")].append(event)
    indices={}
    for tid,ranges in by_tid.items():
        ranges.sort(key=lambda span:(span["ts"],-span["dur"]))
        parents=[]
        stack=[]
        for index,span in enumerate(ranges):
            while stack and ranges[stack[-1]]["ts"]+ranges[stack[-1]]["dur"]<=span["ts"]:
                stack.pop()
            parents.append(stack[-1] if stack else -1)
            stack.append(index)
        indices[tid]=([span["ts"] for span in ranges],parents)
    def category_for_cpu(event):
        tid=event.get("tid")
        if tid not in indices:
            return None
        starts,parents=indices[tid]
        index=bisect_right(starts,event["ts"])-1
        selected=None
        while index>=0:
            span=by_tid[tid][index]
            if span["ts"]+span["dur"]>=event["ts"]+event["dur"]:
                if span["name"]=="diagnostic/optimizer":
                    return "optimizer"
                selected=selected or span
            index=parents[index]
        return selected["name"].removeprefix("diagnostic/gemm/").removeprefix("diagnostic/kernel/").split("|",1)[0] if selected else None
    external = {}
    correlation = {}
    for event in spans:
        if event.get("cat") not in ("cpu_op","cuda_runtime","cuda_driver"):
            continue
        category = category_for_cpu(event)
        if category:
            args = event.get("args",{})
            if "External id" in args:
                external[args["External id"]] = category
            if "correlation" in args:
                correlation[args["correlation"]] = category
    kernels = [event for event in spans if event.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
    histogram = defaultdict(lambda:dict(calls=0,summed_duration_us=0.0,intervals=[]))
    attributed = []
    fp8_evidence = []
    name_categories={}
    for event in kernels:
        args, name = event.get("args",{}), event["name"]
        category = correlation.get(args.get("correlation")) or external.get(args.get("External id"))
        if category=="unattributed":
            category=None
        if category is None:
            if name not in name_categories:
                hits = {value for kernel,value in kernel_categories.items() if kernel==name or kernel in name}
                name_categories[name]=next(iter(hits)) if len(hits)==1 else "mixed/provenance" if hits else None
            category=name_categories[name]
        if category is None:
            lower = name.lower()
            category = ("attention/kernel" if any(word in lower for word in ("flash", "fmha")) else
                        "transfer" if event.get("cat")!="kernel" else "unattributed")
        interval = (event["ts"],event["ts"]+event["dur"],category)
        attributed.append(interval)
        row = histogram[(name,category)]
        row["calls"] += 1
        row["summed_duration_us"] += event["dur"]
        row["intervals"].append(interval[:2])
        if "ffn/forward/gemm_fp8"==category:
            fp8_evidence.append(dict(kernel=name,correlation=args.get("correlation"),external_id=args.get("External id"),
                                     gemm_named=bool(re.search(r"gemm|mma|matmul|nvjet|cutlass",name,re.I))))
    windows=[]
    for update in updates:
        start,stop = update["ts"],update["ts"]+update["dur"]
        exclusive = partition_intervals(attributed,start,stop)
        microsteps = sum(event.get("name","").startswith("diagnostic/microstep/") and start<=event["ts"]<stop for event in spans)
        if microsteps!=16:
            raise RuntimeError(f"Trace is incomplete: {microsteps} microsteps, expected 16")
        windows.append(dict(name=update["name"],wall_ms=(stop-start)/1000,microsteps=microsteps,
                            exclusive_ms={key:value/1000 for key,value in exclusive.items()},
                            component_exclusive_ms={prefix:sum(value for key,value in exclusive.items() if key.startswith(prefix))/1000
                                                    for prefix in ("ffn/forward/","ffn/backward/","attention/","head_loss/","optimizer")},
                            exposed_gpu_idle_ms=exclusive.get("gpu_idle",0)/1000))
    if not windows:
        raise RuntimeError("No complete update scopes in trace")
    return dict(updates=windows,fp8_cuda_evidence=fp8_evidence,
                kernels=[dict(name=name,category=category,calls=row["calls"],
                              summed_duration_ms_nonadditive=row["summed_duration_us"]/1000,
                              union_busy_ms=union_length(row["intervals"])/1000)
                         for (name,category),row in histogram.items()],
                accounting="exclusive_ms partitions each complete update across all CUDA streams. Overlap is separate. Kernel sums and component rollups are nonadditive. GPU-idle includes exposed host/launch/data/synchronization gaps; it is not a CPU-compute measurement. Mixed fused kernels stay mixed.")
