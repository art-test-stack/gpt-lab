# https://medium.com/@dip.patel.ict/bits-per-byte-bpb-a-tokenizer-agnostic-way-to-measure-llms-25dfed3f41af
import torch
import torch.distributed as dist
import math
from itertools import count
from typing import Optional

from gpt_lab.utils.distributed import ComputeContext, is_ddp_initialized

@torch.no_grad()
def compute_bpb(model, batches, steps: int | None, token_bytes: torch.Tensor, comp_info: Optional[ComputeContext]) -> dict:
    """
    Compute bits-per-byte (bpb) over the given batches.
    
    ## Shapes
        - B: batch size
        - S: sequence length
        - V: vocabulary size
    
    Args:
        - model (nn.Module): The language model to evaluate. model(x, y, loss_reduction='none') -> loss per token.
            Expects:
                -  x (torch.int64): token ids tensor with shape (B, S)
                - y (torch.int64): token ids tensor with shape (B, S)
            Returns:
                - loss2d (torch.float32/torch.float16): Tensor of shape (B, S) representing the loss per token.
         where logits has shape (B, S, V); i.e loss per token.
        - batches: An iterable of yielding batches (x,y).
        - steps: Number of batches per rank, or None to exhaust each rank's iterator.
        - token_bytes (torch.int64): Tensor of shape (V,) containing the byte lengths of each token in the vocabulary; 0 for special tokens.
        - comp_info (Optional[ComputeContext]): Runtime device and data-parallel mesh.
    Returns:
        - bpb (float): The computed bits-per-byte metric.
        - loss (float): Mean next-token negative log-likelihood in nats over valid targets.
        - scored_targets (int): Number of valid targets summed across ranks.
    """
    dist_is_init = comp_info.is_distributed if comp_info is not None else is_ddp_initialized()
    world_size = comp_info.world_size if comp_info is not None else (dist.get_world_size() if dist_is_init else 1)
    
    device = model.get_device() if hasattr(model, "get_device") else next(model.parameters()).device
    sum_dtype = torch.float64 if steps is None and device.type in ("cpu", "cuda") else torch.float32

    # Accumulators across steps (and later across ranks)
    total_nats  = torch.tensor(0.0, dtype=sum_dtype, device=device)  # scalar
    total_bytes = torch.tensor(0,   dtype=torch.int64,   device=device)  # scalar
    total_loss = torch.tensor(0.0, dtype=sum_dtype, device=device)
    total_targets = torch.tensor(0, dtype=torch.int64, device=device)

    token_bytes = token_bytes.to(device=device, dtype=torch.int64)     # (V,)
    
    short_ranks = None
    if steps is not None and dist_is_init and world_size > 1:
        rank = comp_info.rank if comp_info is not None else dist.get_rank()
        short_ranks = torch.zeros(world_size, dtype=torch.int32, device=device)

    batch_iter = iter(batches)
    for completed in count() if steps is None else range(steps):
        batch = next(batch_iter, None)
        short = batch is None
        if short_ranks is not None:
            short_ranks[rank] = short
            dist.all_reduce(short_ranks)
            short = bool(short_ranks.any().item())
        if short:
            if steps is None:
                break
            ranks = short_ranks.nonzero().flatten().tolist() if short_ranks is not None else [0]
            raise RuntimeError(f"Validation exhausted after {completed} of {steps} full batches on ranks {ranks}")
        x, y, _ = batch

        output = model(x, y, reduction="none")
        losses = (output.loss if hasattr(output, "loss") else output).reshape(-1)
        del output
        y = y.reshape(-1)
        valid = y >= 0
        total_loss += losses.masked_fill(~valid, 0).sum(dtype=total_loss.dtype)
        total_targets += valid.sum()

        n_bytes = token_bytes[y.masked_fill(~valid, 0)].masked_fill(~valid, 0)
        total_nats += losses.masked_fill(n_bytes == 0, 0).sum(dtype=total_nats.dtype)
        total_bytes += n_bytes.sum()

    # Distributed sum over all ranks, if initialized
    if dist_is_init and world_size > 1:
        dist.all_reduce(total_nats,  op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_targets, op=dist.ReduceOp.SUM)

    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    target_count = total_targets.item()
    loss = total_loss.item() / target_count if target_count else float("nan")

    bpb = total_nats / (math.log(2.0) * total_bytes) if total_bytes else float("inf")
    return {"bpb": bpb, "loss": loss, "scored_targets": target_count}
