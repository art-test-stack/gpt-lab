from dataclasses import dataclass

import torch

from gpt_lab.evaluate.core import stack_sequences
from fixture_model import wide_fixture
from scoring import pack_choices, tree_masks


CONTEXT = 2048


@dataclass
class PackedBatch:
    tokens: torch.Tensor
    positions: torch.Tensor
    branches: torch.Tensor
    sources: torch.Tensor
    valid: torch.Tensor
    prefix_ends: torch.Tensor
    first_targets: torch.Tensor


def collate_packed(choices, pad_id):
    """Build the CPU tensors for a batch of already eligible questions."""
    if any(max(question.positions) >= CONTEXT or min(question.positions) < 0 for question in choices):
        raise ValueError("A packed path exceeds the 2,048-token context")
    tokens = stack_sequences([question.tokens for question in choices], pad_id)
    positions = stack_sequences([question.positions for question in choices], 0)
    branches = stack_sequences([question.branches for question in choices], -1)
    batch_size, width = tokens.shape
    max_options = max(len(question.lengths) for question in choices)
    max_answer = max(max(question.lengths) for question in choices)
    sources = torch.zeros((batch_size, max_options, max_answer), dtype=torch.long)
    valid = torch.zeros_like(sources, dtype=torch.bool)
    first_targets = torch.zeros((batch_size, max_options), dtype=torch.long)
    for i, question in enumerate(choices):
        for j, (offset, length) in enumerate(zip(question.offsets, question.lengths)):
            sources[i, j, 0] = i * width + question.prefix - 1
            sources[i, j, 1:length] = torch.arange(i * width + offset, i * width + offset + length - 1)
            first_targets[i, j] = question.tokens[offset]
            valid[i, j, :length] = True
    prefix_ends = torch.tensor([question.prefix - 1 for question in choices])
    return PackedBatch(tokens, positions, branches, sources, valid, prefix_ends, first_targets)


def score_packed_batch(model, batch, device, *, capture_logits=False):
    tokens, positions, branches = (x.to(device) for x in (batch.tokens, batch.positions, batch.branches))
    sources, valid = batch.sources.to(device), batch.valid.to(device)
    masks = tree_masks(positions.to(torch.int32), branches, model.window_sizes)
    # Score the first branch targets separately from the shared prefix distribution.
    pads = branches < 0
    labels = tokens.roll(-1, dims=1).masked_fill(pads | pads.roll(-1, dims=1), -1)
    outputs = model(tokens, labels, reduction="none", position_ids=positions, complete_attn_masks=masks)
    first_logits = outputs.logits[torch.arange(tokens.size(0), device=device), batch.prefix_ends.to(device)]
    first_nll = -first_logits.float().log_softmax(-1).gather(1, batch.first_targets.to(device))
    nll = outputs.loss.flatten().index_select(0, sources.flatten()).view_as(sources)
    nll[..., 0] = first_nll
    lengths = valid.sum(-1)
    means = (nll.masked_fill(~valid, 0).sum(-1) / lengths.clamp_min(1)).masked_fill(lengths == 0, torch.inf)
    logits = None
    if capture_logits:
        flat = outputs.logits.flatten(0, 1)
        logits = [
            [flat.index_select(0, offsets[keep].to(device)) for offsets, keep in zip(row_sources, row_valid)
             if keep[0]]
            for row_sources, row_valid in zip(batch.sources, batch.valid)
        ]
    return means, means.argmin(dim=-1), logits


@torch.no_grad()
def check_wide_fixture(model, pad_id, device):
    wide, short = wide_fixture(pad_id)
    questions = [pack_choices(*wide), pack_choices(*short)]
    batch = collate_packed(questions, pad_id)
    assert batch.tokens.shape == (2, 2160)
    assert batch.positions.max().item() == 2047
    assert (batch.branches[1] < 0).sum().item() == 2160 - len(questions[1].tokens)
    observed = []
    hook = model.register_forward_hook(lambda _, args, output: observed.append(output.logits.cpu()))
    try:
        score_packed_batch(model, batch, device)
        batch.tokens[0, questions[0].offsets[0] + 1] = 90
        score_packed_batch(model, batch, device)
        batch.tokens[1, len(questions[1].tokens):] = 91
        score_packed_batch(model, batch, device)
    finally:
        hook.remove()

    original, mutated, different_padding = observed
    changed_branch = batch.branches[0] == 1
    other_question = slice(0, len(questions[1].tokens))
    assert not torch.equal(original[0, changed_branch], mutated[0, changed_branch])
    torch.testing.assert_close(original[0, ~changed_branch], mutated[0, ~changed_branch], rtol=0, atol=0)
    torch.testing.assert_close(original[1, other_question], mutated[1, other_question], rtol=0, atol=0)
    torch.testing.assert_close(mutated[1, other_question], different_padding[1, other_question], rtol=0, atol=0)
    return {"physical_width": 2160, "max_logical_position": 2047,
            "sibling_and_other_question_max_logit_abs": 0.0, "padding_max_logit_abs": 0.0}
