"""Native Hopper checks for A/B: BF16 versus FP8 FFN forward, BF16 backward.

Reports errors without inventing a numerical acceptance threshold. GPU dispatch
and finiteness are checked. This is an isolated FFN probe, not a quality result.
"""

import argparse
import copy
import json
import statistics
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from gpt_lab.model.fp8 import check_fp8_support, enable_fp8_ffn_forward
from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.utils.schemas import TransformerConfig


class GemmTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in ("aten._scaled_mm.default", "aten.mm.default"):
            self.operations.append({"operator": str(func), "inputs": [str(x.dtype) for x in args[:2]]})
        return func(*args, **(kwargs or {}))


def errors(actual, reference):
    delta = actual.detach().float()-reference.detach().float()
    return {"max_absolute_error": delta.abs().max().item(),
            "relative_l2_error": (delta.norm()/reference.detach().float().norm().clamp_min(1e-12)).item(),
            "reference_norm": reference.detach().float().norm().item(),
            "actual_norm": actual.detach().float().norm().item(),
            "finite": bool(torch.isfinite(actual).all())}


def milliseconds(fn):
    for _ in range(3):
        fn()
    times = []
    for _ in range(10):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times), "samples_ms": times}


def run(output, activation="swiglu"):
    check_fp8_support(torch.device("cuda"), torch.bfloat16)
    from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
    from torchao.float8.float8_training_tensor import GemmInputRole
    import torchao

    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    model = DenseTransformer(TransformerConfig(vocab_size=512, d_model=128, d_ffn=512,
        n_layers=1, n_heads=2, d_head=64, max_context=16, act_func=activation)).to("cuda")
    model._enforce_precision(torch.bfloat16)
    model.init_weights()
    with torch.no_grad():
        model.blocks[0].ffn.w_2.weight.normal_(std=.02)
    reference = copy.deepcopy(model.blocks[0].ffn)
    ids = {name: id(p) for name, p in model.named_parameters()}
    enable_fp8_ffn_forward(model, torch.bfloat16)
    assert ids == {name: id(p) for name, p in model.named_parameters()}
    candidate = model.blocks[0].ffn
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x)
    results = []
    traces = []
    for module in (reference, candidate):
        input_tensor = x.detach().clone().requires_grad_()
        forward, backward = GemmTrace(), GemmTrace()
        with forward:
            y = module(input_tensor)
        with backward:
            gradients = torch.autograd.grad(y, (input_tensor, *module.parameters()), upstream)
        assert gradients[0].dtype == torch.bfloat16
        assert all(gradient.dtype == parameter.dtype for gradient, parameter in zip(gradients[1:], module.parameters()))
        results.append((y, gradients))
        traces.append({"forward": forward.operations, "backward": backward.operations})
    fp8_ops = [op for op in traces[1]["forward"] if op["operator"] == "aten._scaled_mm.default"]
    assert len(fp8_ops) == 2 and all(op["inputs"] == ["torch.float8_e4m3fn"]*2 for op in fp8_ops)
    assert not any(op["operator"] == "aten._scaled_mm.default" for op in traces[1]["backward"])
    assert all(op["inputs"] == ["torch.bfloat16"]*2 for op in traces[1]["backward"])
    comparison = {"output": errors(results[1][0], results[0][0]),
                  "gradients": {name: errors(a, b) for name, a, b in zip(
                      ["input", *dict(candidate.named_parameters())], results[1][1], results[0][1])}}
    assert comparison["output"]["finite"] and all(v["finite"] for v in comparison["gradients"].values())

    operands = {}
    hooks = []
    for name, layer in candidate.named_children():
        def capture(module, args, name=name):
            operands[name] = (args[0].detach().reshape(-1, module.in_features), module.weight.detach().to(torch.bfloat16).t())
        hooks.append(layer.register_forward_pre_hook(capture))
    candidate(x)
    for hook in hooks:
        hook.remove()
    quantization = {}
    for name, (activation, weight_t) in operands.items():
        layer = getattr(candidate, name)
        tensors = []
        diagnostics = []
        for tensor, role, axis in ((activation, GemmInputRole.INPUT, -1), (weight_t, GemmInputRole.WEIGHT, 0)):
            def quantize():
                return hp_tensor_to_float8_dynamic(tensor, torch.float8_e4m3fn, layer.linear_mm_config,
                    gemm_input_role=role, scaling_granularity=layer.config.cast_config_input.scaling_granularity,
                    axiswise_dim=axis, round_scales_to_power_of_2=True)
            q = quantize()
            tensors.append(q)
            scaled = tensor.float()*q._scale
            bound = torch.finfo(torch.float8_e4m3fn).max
            diagnostics.append({"role": role.value, "scale_min": q._scale.min().item(),
                "scale_max": q._scale.max().item(), "clipped_fraction": (scaled.abs()>bound).float().mean().item(),
                "at_format_limit_fraction": (q._data.float().abs()==bound).float().mean().item(),
                "nonzero_to_zero_fraction": ((tensor!=0)&(q._data.float()==0)).float().mean().item(),
                "quantization": milliseconds(quantize)})
        quantization[name] = {"operands": diagnostics,
                              "temporary_bf16_weight_cast": milliseconds(lambda: layer.weight.detach().to(torch.bfloat16)),
                              "weight_transpose": "view; backend layout-copy costs remain inside quantization/GEMM timings and trace",
                              "gemm_with_prequantized_operands": milliseconds(lambda: torch.mm(*tensors))}

    def whole_ffn(module, parameters):
        z = x.detach().requires_grad_()
        y = module(z)
        torch.autograd.grad(y, (z, *parameters), upstream)

    compiled_reference = torch.compile(reference, dynamic=False)
    ref_z = x.detach().requires_grad_()
    ref_y = compiled_reference(ref_z)
    ref_gradients = torch.autograd.grad(ref_y, (ref_z, *reference.parameters()), upstream)
    compiled = torch.compile(candidate, dynamic=False)
    z = x.detach().requires_grad_()
    compiled_y = compiled(z)
    compiled_gradients = torch.autograd.grad(compiled_y, (z, *candidate.parameters()), upstream)
    compiled_comparison = {"output": errors(compiled_y, ref_y),
                          "gradients": [errors(a, b) for a, b in zip(compiled_gradients, ref_gradients)]}
    assert compiled_comparison["output"]["finite"] and all(v["finite"] for v in compiled_comparison["gradients"])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        candidate(x)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(output/"trace.json"))
    report = {"activation": activation, "torch": torch.__version__, "torchao": torchao.__version__, "gpu": torch.cuda.get_device_name(),
              "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "fp8_use_fast_accum": candidate.w_1.config.gemm_config_output.use_fast_accum,
              "native_fp8_operands_dispatched": fp8_ops, "GEMM_traces": traces,
              "eager_comparison": comparison, "compiled_comparison": compiled_comparison,
              "bf16_eager_to_compiled_error": errors(ref_y, results[0][0]),
              "quantization": quantization,
              "compiled_bf16_whole_ffn_forward_backward": milliseconds(lambda: whole_ffn(compiled_reference, reference.parameters())),
              "compiled_fp8_whole_ffn_forward_backward": milliseconds(lambda: whole_ffn(compiled, candidate.parameters())),
              "limits": "isolated small FFN; no quality or training-step speedup claim; inspect trace.json for CUDA kernel details"}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".tmp/codex/fp8-training/native-check"))
    parser.add_argument("--activation", choices=("swiglu", "relu_squared"), default="swiglu")
    args = parser.parse_args()
    run(args.output, args.activation)
