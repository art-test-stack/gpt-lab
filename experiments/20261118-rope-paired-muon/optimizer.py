"""Configure the experiment's polar maps for per-head Muon Q/K matrices."""

import random

import torch
import torch.distributed as dist

from gpt_lab.optim.strategies.muon import MuonStrategy


NORM_NAMES = (
    "weight_gradient_norm", "momentum_proposal_norm", "raw_polar_output_norm",
    "lr_scaled_update_norm", "decay_contribution_norm", "displacement_norm", "displacement_to_weight_norm",
)


def pairing_order(d_head, pairing):
    if d_head < 2 or d_head % 2:
        raise ValueError("Paired Muon needs an even head dimension")
    if pairing == "rope":
        return tuple(range(d_head))
    if pairing != "shuffled" or d_head < 4:
        raise ValueError("Shuffled Muon needs at least two genuine RoPE pairs")

    rng = random.Random(20261118)
    rows = list(range(d_head))
    while True:
        rng.shuffle(rows)
        # Both coordinates of a genuine pair have the same integer quotient.
        if all(a // 2 != b // 2 for a, b in zip(rows[::2], rows[1::2], strict=True)):
            return tuple(rows)


def pairing_indices(rows, device):
    inverse = sorted(range(len(rows)), key=rows.__getitem__)
    return torch.tensor(rows, device=device), torch.tensor(inverse, device=device)


class ExperimentMuonStrategy(MuonStrategy):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.indices = {}
        self.samples = {}
        self.sample_names = []
        self.recorded = []
        self.monitor = False

    def _kernel_kwargs(self, group):
        kwargs = {}
        pairing = group.get("qkv_pairing")
        if pairing is not None:
            arm, qk_heads, rows = pairing
            if arm.startswith("joint-"):
                kwargs["joint_qk_heads"] = qk_heads // 2
            if rows:
                kwargs.update(paired_heads=qk_heads, pair_indices=self.indices.get(rows))
        if self.monitor and (sample := self.samples.get(group["params"][0])) is not None:
            kwargs["monitor_indices"] = sample[:2]
        return kwargs

    def _update_matrices(self, group, *args):
        values = super()._update_matrices(group, *args)
        if values is not None:
            self.recorded.append((self.samples[group["params"][0]][2], values))

    def collect_norms(self):
        context = self.opt.comp_info
        norms = torch.zeros(len(self.sample_names), len(NORM_NAMES), device=context.device, dtype=torch.float32)
        for slots, values in self.recorded:
            norms.index_copy_(0, slots, values)
        self.recorded.clear()
        if context.is_distributed:
            dist.reduce(norms, dst=0, op=dist.ReduceOp.SUM)
        if context.rank != 0:
            return None
        return {
            f"{sample}/{name}": value
            for sample, row in zip(self.sample_names, norms.tolist(), strict=True)
            for name, value in zip(NORM_NAMES, row, strict=True)
        }


def sampled_indices(count):
    return sorted({0, (count - 1) // 2, count - 1})


def configure_experiment_muon(model, optimizer, *, pairing, monitor_qk_norms=False):
    if model.config.positional_encoding != "rope":
        raise ValueError("The paired Muon experiment requires rotary positional encoding")

    strategy = ExperimentMuonStrategy(optimizer)
    attentions = [block.attention for block in model.blocks]
    if pairing.startswith("joint-") and any(
        attention.n_heads != attention.n_kv_heads for attention in attentions
    ):
        raise ValueError(f"{pairing} requires equal query and key head counts")
    attention_by_weight = {attention.qkv_proj.weight: attention for attention in attentions}
    samples_by_weight = {}
    if monitor_qk_norms:
        for layer in sampled_indices(len(attentions)):
            attention = attentions[layer]
            samples = []
            for projection, count, offset in (("q", attention.n_heads, 0),
                                               ("k", attention.n_kv_heads, attention.n_heads)):
                for head in sampled_indices(count):
                    samples.append((offset + head, len(strategy.sample_names)))
                    strategy.sample_names.append(f"qk/{projection}/layer_{layer}/head_{head}")
            samples_by_weight[attention.qkv_proj.weight] = samples

    context = optimizer.comp_info
    configured = set()
    for group in optimizer.param_groups:
        weights = group["params"]
        if not weights:
            continue
        attention = attention_by_weight.get(weights[0])
        if attention is None:
            continue
        heads = attention.n_heads + 2 * attention.n_kv_heads
        d_head = attention.d_head
        if (
            group["opt"] != "muon" or group["ns_steps"] != 5
            or group.get("qkv_split_sizes") != (d_head,) * heads
        ):
            raise ValueError("The experiment requires per-head Muon with five Polar Express steps")
        configured.update(weights)
        basis = pairing.removeprefix("joint-")
        rows = () if basis == "real" else pairing_order(d_head, basis)
        group["qkv_pairing"] = (pairing, attention.n_heads + attention.n_kv_heads, rows)
        if basis == "shuffled" and rows not in strategy.indices:
            strategy.indices[rows] = pairing_indices(rows, weights[0].device)

        if not monitor_qk_norms:
            continue
        # Match PR #50's whole-matrix owner slots, including the final partial chunk.
        chunk = (len(weights) + context.world_size - 1) // context.world_size
        start = context.rank * chunk
        owned = [
            (position, head, slot)
            for position, p in enumerate(weights[start:start + chunk])
            for head, slot in samples_by_weight.get(p, ())
        ]
        if owned:
            strategy.samples[weights[0]] = tuple(torch.tensor(axis, device=weights[0].device) for axis in zip(*owned))

    if configured != attention_by_weight.keys():
        raise ValueError("Every fused QKV weight must be in a per-head Muon group")
    optimizer.backend.strategies["muon"] = strategy
    return {"pairing": pairing}
