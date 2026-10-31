"""Small real-CUDA compiled-versus-captured check, including sharded optimizer state.

Run with torchrun for NCCL coverage. Failure on any rank prevents timing.
This is a correctness diagnostic, not a throughput benchmark.
"""

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path

os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"

import torch
import torch.distributed as dist

from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.train.cuda_graph import CapturedForwardBackward
from gpt_lab.utils.distributed import init_dist_groups, cleanup_dist_groups, broadcast_model
from gpt_lab.utils.schemas import OptimizerConfig, TrainerConfig, TransformerConfig


def compare_values(reference, actual, name, errors, maxima, *, rtol, atol):
    """Compare complete nested state, including exact keys and Python counters."""
    if isinstance(reference, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or reference.shape != actual.shape or reference.dtype != actual.dtype:
            errors.append(f"{name}: tensor type/shape/dtype mismatch")
            return
        if reference.numel():
            delta = (reference.detach().double() - actual.detach().double()).abs().max().item()
            maxima[name.split("/")[0]] = max(maxima.get(name.split("/")[0], 0), delta)
        try:
            torch.testing.assert_close(reference, actual, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as error:
            errors.append(f"{name}: {error}")
    elif isinstance(reference, dict):
        if not isinstance(actual, dict) or reference.keys() != actual.keys():
            errors.append(f"{name}: state keys differ")
            return
        for key in reference:
            compare_values(reference[key], actual[key], f"{name}/{key}", errors, maxima, rtol=rtol, atol=atol)
    elif isinstance(reference, (tuple, list)):
        if type(reference) is not type(actual) or len(reference) != len(actual):
            errors.append(f"{name}: sequence differs")
            return
        for index, (left, right) in enumerate(zip(reference, actual)):
            compare_values(left, right, f"{name}/{index}", errors, maxima, rtol=rtol, atol=atol)
    elif type(reference) is not type(actual) or reference != actual:
        errors.append(f"{name}: {reference!r} != {actual!r}")


def run_check(profile_path, context, output_dir, *, n_acc_steps=3, updates=6, attn_impl="fused"):
    if context.device_type != "cuda":
        raise RuntimeError("Real CUDA is required; a skipped check must not authorize timing")
    dtype = context.compute_dtype
    rtol, atol = (1e-3, 1e-6) if dtype == torch.bfloat16 else (1e-5, 1e-6)
    config = TrainerConfig(
        comp_info=context, optim_config=OptimizerConfig.from_file(profile_path),
        n_steps=2317, lr_warmup_steps=300, n_acc_steps=n_acc_steps,
    )
    reference = DenseTransformer(TransformerConfig(
        vocab_size=128, max_context=16, n_layers=2, d_model=64, d_ffn=128,
        n_heads=1, n_kv_heads=1, d_head=64, window_pattern="SL", window_size=8,
        attn_impl=attn_impl, dropout=0, attention_dropout=0,
    )).to(context.device)
    reference.init_weights()
    reference._enforce_precision(dtype)
    reference.train()
    broadcast_model(reference, context)
    captured = copy.deepcopy(reference)
    ref_opt = reference.build_optimizer(config)
    cap_opt = captured.build_optimizer(config)
    # Explicitly restore the same initial model and optimizer values.
    captured.load_state_dict(copy.deepcopy(reference.state_dict()))
    cap_opt.load_state_dict(copy.deepcopy(ref_opt.state_dict()))
    ref_forward = torch.compile(reference.training_loss, dynamic=False, options={"triton.cudagraphs": False})
    cap_forward = torch.compile(captured.training_loss, dynamic=False, options={"triton.cudagraphs": False})
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype == torch.bfloat16 else nullcontext
    generator = torch.Generator(device="cpu").manual_seed(1700 + context.rank)
    batches = [tuple(torch.randint(0, 128, (2, 16), generator=generator).to(context.device)
                     for _ in range(2)) for _ in range(updates * n_acc_steps)]
    errors, maxima, lr_values = [], {}, []
    before = [tensor.detach().clone() for tensor in list(captured.parameters()) + list(captured.buffers())]
    rng_before = (torch.get_rng_state(), torch.cuda.get_rng_state(context.device))
    opt_before = copy.deepcopy(cap_opt.state_dict())
    graph = CapturedForwardBackward(captured, lambda payload: cap_forward(*payload),
                                    batches[0], n_acc_steps, amp)
    compare_values(before, [t.detach() for t in list(captured.parameters()) + list(captured.buffers())],
                   "warmup_model", errors, maxima, rtol=0, atol=0)
    compare_values(opt_before, cap_opt.state_dict(), "warmup_optimizer", errors, maxima, rtol=0, atol=0)
    compare_values(rng_before, (torch.get_rng_state(), torch.cuda.get_rng_state(context.device)), "warmup_rng",
                   errors, maxima, rtol=0, atol=0)
    addresses = [p.grad.data_ptr() if p.grad is not None else None for p in captured.parameters()]
    for step in range(updates):
        ref_opt.zero_grad(set_to_none=True)
        cap_opt.zero_grad(set_to_none=False)
        for name, param in captured.named_parameters():
            if param.grad is not None and torch.count_nonzero(param.grad).item():
                errors.append(f"gradient_clear/{step}/{name}: gradient was not cleared")
        losses = [torch.zeros((), device=context.device) for _ in range(2)]
        for micro in range(n_acc_steps):
            payload = batches[step * n_acc_steps + micro]
            with amp():
                ref_loss = ref_forward(*payload) / n_acc_steps
            ref_loss.backward()
            cap_loss = graph.replay(payload)
            losses[0].add_(ref_loss.detach().float())
            losses[1].add_(cap_loss.float())
        compare_values(losses[0], losses[1], "loss", errors, maxima, rtol=rtol, atol=atol)
        for (name, left), (_, right) in zip(reference.named_parameters(), captured.named_parameters()):
            compare_values(left.grad, right.grad, f"gradients/{step}/{name}", errors, maxima, rtol=rtol, atol=atol)
        schedule = dict(lrm=config.lr_multiplier_schedule(step),
                        muon_momentum=config.muon_momentum_schedule(step),
                        weight_decay=config.weight_decay_schedule(step))
        lr_values.append(schedule["lrm"])
        # All communication and updates execute normally, outside the CUDA graph.
        for optimizer in (ref_opt, cap_opt):
            optimizer.update_hyperparams(**schedule)
            optimizer.step()
        for (name, left), (_, right) in zip(reference.named_parameters(), captured.named_parameters()):
            compare_values(left, right, f"parameters/{step}/{name}", errors, maxima, rtol=rtol, atol=atol)
        compare_values(ref_opt.state_dict(), cap_opt.state_dict(), f"optimizer/{step}",
                       errors, maxima, rtol=rtol, atol=atol)
        if context.is_distributed:
            for model in (reference, captured):
                flattened = torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])
                rank_zero = flattened.clone()
                dist.broadcast(rank_zero, src=0)
                compare_values(rank_zero, flattened, f"replicated_parameters/{step}",
                               errors, maxima, rtol=rtol, atol=atol)
        if addresses != [p.grad.data_ptr() if p.grad is not None else None for p in captured.parameters()]:
            errors.append(f"gradient_storage/{step}: captured gradient addresses changed")
    cap_opt.zero_grad(set_to_none=False)
    for param in captured.parameters():
        if param.grad is not None and torch.count_nonzero(param.grad).item():
            errors.append("gradient_clear/final: gradient was not cleared")
    if len(set(lr_values)) != updates:
        errors.append("scheduler: LR did not change on every logical optimizer update")
    result = dict(passed=not errors, rank=context.rank, world_size=context.world_size,
                  dtype=str(dtype), optimizer_profile=str(profile_path), updates=updates,
                  n_acc_steps=n_acc_steps, rtol=rtol, atol=atol,
                  max_absolute_errors=maxima, lr_multipliers=lr_values, errors=errors)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank_{context.rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    failed = torch.tensor(int(bool(errors)), device=context.device)
    if context.is_distributed:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise AssertionError(f"CUDA graph equivalence failed; inspect {output}/rank_*.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optim-config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-impl", default="fused", choices=("fused", "sdpa"))
    parser.add_argument("--n-acc-steps", type=int, default=2)
    args = parser.parse_args()
    if args.n_acc_steps < 2:
        parser.error("Equivalence requires at least two accumulation microsteps")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: equivalence was NOT verified")
    context = init_dist_groups("cuda", random_seed=17)
    try:
        result = run_check(args.optim_config_path, context, args.output_dir,
                           n_acc_steps=args.n_acc_steps, attn_impl=args.attn_impl)
        if context.rank == 0:
            print(json.dumps(result, indent=2), flush=True)
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
