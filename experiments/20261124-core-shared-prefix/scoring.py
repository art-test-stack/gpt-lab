from dataclasses import dataclass
from itertools import accumulate

import torch

from gpt_lab.evaluate.core import choice_nll, find_common_length, forward_model, forward_rows, stack_sequences


@dataclass
class PackedChoices:
    tokens: list[int]
    positions: list[int]
    branches: list[int]
    prefix: int
    offsets: list[int]
    lengths: list[int]


def shared_prefix_length(tokens, starts, ends):
    # Different crops can move the continuation to different logical positions.
    prefix = starts[0]
    if len(tokens) < 2 or prefix <= 0 or any(start != prefix or end <= start for start, end in zip(starts, ends)):
        return None
    return prefix if find_common_length(tokens) >= prefix else None


def pack_choices(tokens, starts, ends):
    prefix = shared_prefix_length(tokens, starts, ends)
    if prefix is None:
        return None
    suffixes = [row[prefix:end] for row, end in zip(tokens, ends)]
    lengths = [len(suffix) for suffix in suffixes]
    packed = list(tokens[0][:prefix])
    positions = list(range(prefix))
    branches = [0] * prefix
    for branch, suffix in enumerate(suffixes, 1):
        packed.extend(suffix)
        positions.extend(range(prefix, prefix + len(suffix)))
        branches.extend([branch] * len(suffix))
    offsets = list(accumulate(lengths[:-1], initial=prefix))
    return PackedChoices(packed, positions, branches, prefix, offsets, lengths)


def tree_masks(positions, branches, windows):
    distance = positions[:, None] - positions
    visible = (distance >= 0) & ((branches == 0) | (branches[:, None] == branches))
    masks = {
        left: visible if left < 0 else visible & (distance <= left)
        for left in dict.fromkeys(left for left, _ in windows)
    }
    return tuple(masks[left] for left, _ in windows)


def separate_choices(model, tokens, starts, ends, pad_id, device, *, capture_logits=False):
    rows = stack_sequences(tokens, pad_id).to(device)
    if capture_logits:
        outputs = forward_rows(model, rows)
        logits = [outputs.logits[i, start - 1:end - 1]
                  for i, (start, end) in enumerate(zip(starts, ends))]
        return choice_nll(outputs.loss.view_as(rows), starts, ends), logits
    losses, _ = forward_model(model, rows)
    return choice_nll(losses, starts, ends), None


def continuation_nll(first_nll, tails):
    return [torch.cat((first_nll[i:i + 1], tail)).mean().item() for i, tail in enumerate(tails)]


def separate_choices_with_prefix_scoring(model, tokens, starts, ends, pad_id, device):
    prefix = shared_prefix_length(tokens, starts, ends)
    if prefix is None:
        return separate_choices(model, tokens, starts, ends, pad_id, device)[0]
    rows = stack_sequences(tokens, pad_id).to(device)
    outputs = forward_rows(model, rows)
    first_nll = -outputs.logits[:, prefix - 1].float().log_softmax(-1).gather(1, rows[:, prefix:prefix + 1])
    losses = outputs.loss.view_as(rows)
    return continuation_nll(first_nll[:, 0], (losses[i, prefix:end - 1] for i, end in enumerate(ends)))


def packed_scores(outputs, row, packed, *, capture_logits=False):
    # The roll supplies every within-branch target. The prefix predicts all
    # first tokens, so get their losses from the same prefix distribution.
    prefix_logit = outputs.logits[0, packed.prefix - 1]
    first_nll = -prefix_logit.float().log_softmax(0)[row[0, packed.offsets]]
    losses = outputs.loss.view_as(row)[0]
    tails = (losses[offset:offset + length - 1] for offset, length in zip(packed.offsets, packed.lengths))
    means = continuation_nll(first_nll, tails)
    logits = None
    if capture_logits:
        logits = [
            torch.cat((prefix_logit[None], outputs.logits[0, offset:offset + length - 1]))
            for offset, length in zip(packed.offsets, packed.lengths)
        ]
    return means, logits


def packed_sdpa(model, packed, device, *, capture_logits=False):
    row = torch.tensor([packed.tokens], device=device)
    positions = torch.tensor(packed.positions, device=device)
    branches = torch.tensor(packed.branches, device=device)
    masks = tree_masks(positions, branches, model.window_sizes)
    outputs = forward_rows(model, row, position_ids=positions, attn_mask=masks)
    return packed_scores(outputs, row, packed, capture_logits=capture_logits)
