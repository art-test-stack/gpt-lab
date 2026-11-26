from dataclasses import dataclass

import torch
import torch.nn.functional as F

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


@dataclass
class SeparateBatch:
    tokens: torch.Tensor
    keys: torch.Tensor
    sources: torch.Tensor
    valid: torch.Tensor


def stack_at_width(rows, pad_id, width):
    tokens = stack_sequences(rows, pad_id)
    if width is None or width == tokens.size(1):
        return tokens
    if width < tokens.size(1):
        raise ValueError("Padded width must fit every row")
    return F.pad(tokens, (0, width - tokens.size(1)), value=pad_id)


def collate_separate(examples, pad_id, *, padded_width=None):
    rows = [row for tokens, _, _ in examples for row in tokens]
    tokens = stack_at_width(rows, pad_id, padded_width)
    width = tokens.size(1)
    keys = torch.arange(width) < torch.tensor(list(map(len, rows)))[:, None]
    spans = []
    row_index = 0
    for _, starts, ends in examples:
        spans.append([])
        for start, end in zip(starts, ends):
            left, right, _ = slice(start - 1, end - 1).indices(width)
            spans[-1].append(range(row_index * width + left, row_index * width + right))
            row_index += 1
    max_answer = max(1, max(len(span) for question in spans for span in question))
    sources = torch.zeros((len(examples), max(map(len, spans)), max_answer), dtype=torch.long)
    valid = torch.zeros_like(sources, dtype=torch.bool)
    for i, question in enumerate(spans):
        for j, span in enumerate(question):
            if span:
                sources[i, j, :len(span)] = torch.arange(span.start, span.stop)
                valid[i, j, :len(span)] = True
    return SeparateBatch(tokens, keys, sources, valid)


def collate_packed(choices, pad_id, *, max_context=CONTEXT, padded_width=None):
    """Build the CPU tensors for a batch of already eligible questions."""
    if any(max(question.positions) >= max_context or min(question.positions) < 0 for question in choices):
        raise ValueError(f"A packed path exceeds the model context ({max_context} tokens)")
    tokens = stack_at_width([question.tokens for question in choices], pad_id, padded_width)
    positions = stack_at_width([question.positions for question in choices], 0, padded_width)
    branches = stack_at_width([question.branches for question in choices], -1, padded_width)
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


def shifted_labels(tokens, keys):
    return tokens.roll(-1, dims=1).masked_fill(~(keys & keys.roll(-1, dims=1)), -1)


def reduce_options(losses, sources, valid, first_nll=None):
    nll = losses.flatten().index_select(0, sources.flatten()).view_as(sources)
    if first_nll is not None:
        nll[..., 0] = first_nll
    lengths = valid.sum(-1)
    means = nll.masked_fill(~valid, 0).sum(-1) / lengths
    # Python min retains a NaN first option and ignores later NaNs.
    ranked = means.masked_fill(means.isnan(), torch.inf)
    winners = ranked.argmin(-1).masked_fill(means[:, 0].isnan(), 0)
    return means, winners


def score_separate_batch(model, batch, device):
    tokens, keys = batch.tokens.to(device), batch.keys.to(device)
    outputs = model(tokens, shifted_labels(tokens, keys), reduction="none")
    return reduce_options(outputs.loss, batch.sources.to(device), batch.valid.to(device))


def score_packed_batch(model, batch, device, *, capture_logits=False, fa4=None):
    tokens, positions, branches = (x.to(device) for x in (batch.tokens, batch.positions, batch.branches))
    sources, valid = batch.sources.to(device), batch.valid.to(device)
    # Score the first branch targets separately from the shared prefix distribution.
    labels = shifted_labels(tokens, branches >= 0)
    if fa4 is None:
        masks = tree_masks(positions.to(torch.int32), branches, model.window_sizes)
        outputs = model(tokens, labels, reduction="none", position_ids=positions, complete_attn_masks=masks)
    else:
        with fa4.batched_layout(positions.to(torch.int32), branches.to(torch.int32)):
            outputs = model(tokens, labels, reduction="none", position_ids=positions)
    first_logits = outputs.logits[torch.arange(tokens.size(0), device=device), batch.prefix_ends.to(device)]
    first_nll = -first_logits.float().log_softmax(-1).gather(1, batch.first_targets.to(device))
    means, winners = reduce_options(outputs.loss, sources, valid, first_nll)
    logits = None
    if capture_logits:
        flat = outputs.logits.flatten(0, 1)
        logits = [
            [flat.index_select(0, offsets[keep].to(device)) for offsets, keep in zip(row_sources, row_valid)
             if keep[0]]
            for row_sources, row_valid in zip(batch.sources, batch.valid)
        ]
    return means, winners, logits


@torch.no_grad()
def check_wide_fixture(model, pad_id, device, *, fa4=None):
    wide, short = wide_fixture(pad_id)
    questions = [pack_choices(*wide), pack_choices(*short)]
    batch = collate_packed(questions, pad_id)
    assert batch.tokens.shape == (2, 2160)
    assert batch.positions.max().item() == 2047
    assert (batch.branches[1] < 0).sum().item() == 2160 - len(questions[1].tokens)
    observed = []
    hook = model.register_forward_hook(lambda _, args, output: observed.append(output.logits.cpu()))
    try:
        score_packed_batch(model, batch, device, fa4=fa4)
        batch.tokens[0, questions[0].offsets[0] + 1] = 90
        score_packed_batch(model, batch, device, fa4=fa4)
        batch.tokens[1, len(questions[1].tokens):] = 91
        score_packed_batch(model, batch, device, fa4=fa4)
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
