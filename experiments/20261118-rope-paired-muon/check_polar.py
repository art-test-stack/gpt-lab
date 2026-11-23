"""Check five-step Polar Express, including joint Q/K training updates."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.optim.kernels.muon import paired_polar_express, polar_express_coeffs
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state
from optimizer import configure_experiment_muon, pairing_indices, pairing_order


JOINT_L2_TOL = {torch.float32: 2e-4, torch.bfloat16: 0.05}


def complex_rows(proposal):
    return torch.complex(proposal[..., ::2, :], proposal[..., 1::2, :])


def interleave(real, imag):
    return torch.stack((real, imag), dim=-2).flatten(-3, -2)


def complex_polar_express(proposal, dtype):
    def round_components(z):
        return torch.complex(z.real.to(dtype).float(), z.imag.to(dtype).float())

    def mm(a, b):
        return round_components(round_components(a) @ round_components(b))

    z = complex_rows(proposal.to(dtype).float())
    z = z / (torch.linalg.matrix_norm(z)[..., None, None] * 1.01 + 1e-6)
    for a, b, c in polar_express_coeffs:
        gram = mm(z, z.mH)
        z = a * z + mm(b * gram + c * mm(gram, gram), z)
    z = round_components(z)
    return interleave(z.real, z.imag)


def real_polar_express(proposal, dtype):
    x = proposal.to(dtype)
    x = x / (torch.linalg.matrix_norm(x)[..., None, None] * 1.01 + 1e-6)
    for a, b, c in polar_express_coeffs:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.float()


def exact_polar_comparison(proposal):
    z = complex_rows(proposal.double())
    tolerance = max(z.shape[-2:]) * torch.finfo(z.real.dtype).eps

    def svd_polar(matrix):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        supported = s > tolerance * s[..., :1]
        return (u * supported.unsqueeze(-2)) @ vh

    top = torch.cat((z.real, -z.imag), dim=-1)
    bottom = torch.cat((z.imag, z.real), dim=-1)
    realified = torch.cat((top, bottom), dim=-2)
    real, imag = svd_polar(realified)[..., :z.shape[-1]].chunk(2, dim=-2)
    complex_output = svd_polar(z)
    return interleave(real, imag), interleave(complex_output.real, complex_output.imag)


def report(check, actual, expected, *, max_relative_l2=None):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{check} produced non-finite output")
    for index, (a, b) in enumerate(zip(actual.flatten(1).double(), expected.flatten(1).double(), strict=True)):
        a_norm, b_norm = a.norm().item(), b.norm().item()
        difference = (a - b).norm().item()
        relative_l2 = difference / b_norm if b_norm else (0.0 if difference == 0 else float("inf"))
        print(json.dumps({
            "check": check,
            "proposal": index,
            "relative_l2_error": relative_l2 if b_norm or difference == 0 else None,
            "cosine_similarity": a.dot(b).item() / (a_norm * b_norm) if a_norm and b_norm else None,
            "output_frobenius_norm": a_norm,
            "reference_frobenius_norm": b_norm,
            "output_norm_ratio": a_norm / b_norm if b_norm else None,
        }))
        if max_relative_l2 is not None and relative_l2 > max_relative_l2:
            raise AssertionError(f"{check} head {index}: relative L2 {relative_l2:.6g} exceeds {max_relative_l2}")


def check_joint_dispatch(device, dtype, checkpoint_path):
    generator = torch.Generator().manual_seed(42)
    # [update, projection, head, output row, input column]
    grads = torch.randn(2, 3, 2, 16, 32, generator=generator).to(device)
    # The two Q/K head pairs emphasize different RoPE planes.
    grads[:, :2, 0, :8] *= 4
    grads[:, :2, 1, 8:] *= 4
    initial = torch.zeros_like(grads[0])
    lr, momentum, beta = 0.125, 0.9, 0.8
    context = ComputeContext(device=torch.device(device), compute_dtype=dtype)

    def build_optimizer(arm, weights):
        weight = torch.nn.Parameter(weights.flatten(0, 2).clone())
        attention = SimpleNamespace(n_heads=2, n_kv_heads=2, d_head=16, qkv_proj=SimpleNamespace(weight=weight))
        model = SimpleNamespace(config=SimpleNamespace(positional_encoding="rope"),
                                blocks=[SimpleNamespace(attention=attention)])
        group = dict(params=[weight], opt="muon", lr=lr, momentum=momentum, beta=beta, ns_steps=5,
                     weight_decay=0.0, qkv_split_sizes=(16,) * 6)
        optimizer = OptimizerFactory([group], comp_info=context)
        configure_experiment_muon(model, optimizer, pairing=arm)
        return weight, optimizer

    def step(weight, optimizer, grad):
        weight.grad = grad.flatten(0, 2).clone()
        optimizer.step()
        return weight.detach().reshape_as(initial).clone()

    # The references and all arms start with identical momentum.
    proposal = grads[0].lerp(torch.zeros_like(initial).lerp_(grads[0], 1 - momentum), momentum)
    q, k = proposal[:2]

    def joint_reference(polar, keys):
        pair = polar(torch.cat((q, keys), dim=-1), dtype)
        return torch.stack(pair.chunk(2, dim=-1), dim=1)

    def shuffled_reference(proposals, dtype):
        update = torch.empty_like(proposals)
        update.index_copy_(-2, order, complex_polar_express(proposals.index_select(-2, order), dtype))
        return update

    def normalized_update(raw):
        # First update, with a zero initial variance buffer and column-wise reduction.
        variance = (1 - beta) * raw.square().mean(dim=-2, keepdim=True)
        scaled = raw * variance.clamp_min(1e-10).rsqrt()
        norm = torch.linalg.matrix_norm(raw) / torch.linalg.matrix_norm(scaled).clamp_min(1e-10)
        return scaled * norm[..., None, None]

    weight, optimizer = build_optimizer("real", initial)
    real_update = -step(weight, optimizer, grads[0]) / lr
    real_next_v = step(weight, optimizer, grads[1])[2]
    weight, optimizer = build_optimizer("rope", initial)
    rope_update = -step(weight, optimizer, grads[0]) / lr
    weight, optimizer = build_optimizer("shuffled", initial)
    shuffled_update = -step(weight, optimizer, grads[0]) / lr
    shuffled_rows = optimizer.param_groups[0]["qkv_pairing"][2]
    assert shuffled_rows == pairing_order(16, "shuffled")
    order = torch.tensor(shuffled_rows, device=device)

    tolerance = JOINT_L2_TOL[dtype]
    for arm, baseline, baseline_update, polar in (
        ("joint-real", "real", real_update, real_polar_express),
        ("joint-rope", "rope", rope_update, complex_polar_express),
        ("joint-shuffled", "shuffled", shuffled_update, shuffled_reference),
    ):
        label = f"{arm}/{dtype}"
        expected = normalized_update(joint_reference(polar, k))
        wrong_q, wrong_k = joint_reference(polar, k.flip(0)).unbind(1)
        swapped = normalized_update(torch.stack((wrong_q, wrong_k.flip(0)), dim=1))
        swap_error = (swapped - expected).flatten(1).norm(dim=-1) / expected.flatten(1).norm(dim=-1)
        if (swap_error <= tolerance).any():
            raise AssertionError(f"{label}: swapped heads must exceed relative L2 tolerance: {swap_error.tolist()}")

        weight, optimizer = build_optimizer(arm, initial)
        if arm == "joint-shuffled":
            assert optimizer.param_groups[0]["qkv_pairing"][2] == shuffled_rows
        first = step(weight, optimizer, grads[0])
        update = -first / lr
        qk_update = update[:2].transpose(0, 1)
        report(f"{label}/update_reference", qk_update, expected, max_relative_l2=tolerance)
        # Without decay, variance reduction preserves each polar output's Frobenius norm.
        report(f"{label}/vs_{baseline}", qk_update, baseline_update[:2].transpose(0, 1))
        torch.testing.assert_close(update[2], real_update[2], rtol=0, atol=0)

        save_tensor_state({"weight": first, "optimizer": optimizer.state_dict()}, checkpoint_path)
        checkpoint = load_tensor_state(checkpoint_path, device=device)
        resumed_weight, resumed = build_optimizer(arm, checkpoint["weight"])
        resumed.load_state_dict(checkpoint["optimizer"])
        continued = step(weight, optimizer, grads[1])
        replayed = step(resumed_weight, resumed, grads[1])
        torch.testing.assert_close(continued[2], real_next_v, rtol=0, atol=0)
        torch.testing.assert_close(replayed, continued, rtol=0, atol=0)
        torch.testing.assert_close(resumed.state[resumed_weight], optimizer.state[weight], rtol=0, atol=0)
        print(json.dumps({"check": label, "swapped_relative_l2": swap_error.tolist(),
                          "v_matches_real": True, "resume_matches": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--proposals", help="Saved tensor with trailing dimensions [even head rows, input columns].")
    parser.add_argument("--pairing", choices=("both", "rope", "shuffled"), default="both")
    parser.add_argument("--exact", action="store_true", help="Run only the FP64 SVD polar comparison.")
    args = parser.parse_args()

    if args.proposals:
        proposals = torch.load(args.proposals, map_location="cpu", weights_only=True)
    else:
        proposals = torch.randn(2, 128, 1536, generator=torch.Generator().manual_seed(42))
    if (
        not isinstance(proposals, torch.Tensor) or proposals.ndim < 2
        or not proposals.is_floating_point() or not proposals.numel() or proposals.shape[-2] % 2
    ):
        parser.error("expected nonempty real floating-point proposals with an even row count")
    proposals = proposals.reshape(-1, *proposals.shape[-2:]).to(args.device)
    torch.set_float32_matmul_precision("highest")
    production = paired_polar_express
    if args.device == "cuda" and not args.exact:
        production = torch.compile(production, dynamic=False, fullgraph=True)
    assert len(polar_express_coeffs) == 5
    pairings = ("rope", "shuffled") if args.pairing == "both" else (args.pairing,)
    for pairing in pairings:
        try:
            rows = pairing_order(proposals.shape[-2], pairing)
        except ValueError as exc:
            parser.error(str(exc))
        if pairing == "shuffled":
            surviving = sum(a // 2 == b // 2 for a, b in zip(rows[::2], rows[1::2], strict=True))
            assert surviving == 0 and sorted(rows) == list(range(proposals.shape[-2]))
            print(json.dumps({"pairing": pairing, "surviving_genuine_pairs": surviving}))
        reordered = proposals[..., list(rows), :]
        if args.exact:
            report(f"{pairing}/exact_svd_fp64", *exact_polar_comparison(reordered))
            continue
        indices = pairing_indices(rows, proposals.device) if pairing == "shuffled" else None
        for dtype in (torch.float32, torch.bfloat16):
            actual = production(proposals, 5, dtype, indices)
            expected = torch.empty_like(actual, dtype=torch.float32)
            expected[..., list(rows), :] = complex_polar_express(reordered, dtype)
            report(f"{pairing}/five_step_{dtype}", actual, expected)

    if not args.exact:
        scratch = Path(__file__).resolve().parents[2] / ".tmp" / "codex"
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="polar-dispatch-", dir=scratch) as directory:
            checkpoint_path = Path(directory) / "qkv.safetensors"
            with torch.compiler.set_stance("force_eager" if args.device == "cpu" else "default"):
                for dtype in JOINT_L2_TOL:
                    check_joint_dispatch(args.device, dtype, checkpoint_path)


if __name__ == "__main__":
    main()
