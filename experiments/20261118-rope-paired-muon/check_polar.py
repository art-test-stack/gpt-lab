"""Compare paired Polar Express with a complex reference on shared head proposals."""

import argparse
import json

import torch

from gpt_lab.optim.kernels.muon import paired_polar_express, polar_express_coeffs
from optimizer import pairing_indices, pairing_order


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


def report(check, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{check} produced non-finite output")
    for index, (a, b) in enumerate(zip(actual.flatten(1).double(), expected.flatten(1).double(), strict=True)):
        a_norm, b_norm = a.norm().item(), b.norm().item()
        difference = (a - b).norm().item()
        print(json.dumps({
            "check": check,
            "proposal": index,
            "relative_l2_error": difference / b_norm if b_norm else (0.0 if difference == 0 else None),
            "cosine_similarity": a.dot(b).item() / (a_norm * b_norm) if a_norm and b_norm else None,
            "output_norm_ratio": a_norm / b_norm if b_norm else None,
        }))


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


if __name__ == "__main__":
    main()
