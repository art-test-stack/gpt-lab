# https://medium.com/@dip.patel.ict/bits-per-byte-bpb-a-tokenizer-agnostic-way-to-measure-llms-25dfed3f41af
import torch
import torch.distributed as dist
import math 
from typing import Optional

from gpt_lab.utils.distributed import ComputeContext, is_ddp_initialized

@torch.no_grad()
def compute_bpb(model, batches, steps: int, token_bytes: torch.Tensor, comp_info: Optional[ComputeContext]) -> dict[str, float]:
    """
    Compute bits-per-byte and token perplexity over the given batches.
    
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
        - step: Number of batches to evaluate.
        - token_bytes (torch.int64): Tensor of shape (V,) containing the byte lengths of each token in the vocabulary; 0 for special tokens.
        - comp_info (Optional[ComputeContext]): Runtime device and data-parallel mesh.
    Returns:
        - bpb (float): The computed bits-per-byte metric.
        - loss (float): The average loss per token, for debugging.
        - perplexity (float): Exponentiated mean loss over non-ignored targets, or NaN if none.
    """
    dist_is_init = comp_info.is_distributed if comp_info is not None else is_ddp_initialized()
    world_size = comp_info.world_size if comp_info is not None else (dist.get_world_size() if dist_is_init else 1)
    
    device = model.get_device() if hasattr(model, "get_device") else next(model.parameters()).device

    # Accumulators across steps (and later across ranks)
    total_nats  = torch.tensor(0.0, dtype=torch.float32, device=device)  # scalar
    total_bytes = torch.tensor(0,   dtype=torch.int64,   device=device)  # scalar
    total_loss  = torch.tensor(0.0, dtype=torch.float32, device=device)  # scalar, for debugging
    total_token_nats = torch.zeros((), dtype=torch.float32, device=device)
    total_tokens = torch.zeros((), dtype=torch.int64, device=device)

    token_bytes = token_bytes.to(device=device, dtype=torch.int64)     # (V,)
    
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)

        output = model(x, y, reduction="none") # (B, Seq) NATs
        loss2d = output.loss if hasattr(output, "loss") else output
        total_loss += loss2d.detach().mean() # For debugging
        loss2d = loss2d.reshape(-1) # (B*Seq,)
        y = y.reshape(-1) # (B*Seq,)

        valid = y >= 0
        total_token_nats += loss2d.masked_fill(~valid, 0).sum(dtype=torch.float32)
        total_tokens += valid.sum()

        target_bytes = token_bytes[y.masked_fill(~valid, 0)].masked_fill(~valid, 0)
        total_nats += loss2d.masked_fill(target_bytes == 0, 0).sum(dtype=torch.float32)
        total_bytes += target_bytes.sum()

    # Distributed sum over all ranks, if initialized
    if dist_is_init and world_size > 1:
        dist.all_reduce(total_nats,  op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_token_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    total_loss = total_loss.item() / (world_size * steps)  # Average loss per token, for debugging
    perplexity = (total_token_nats / total_tokens).exp().item()

    # Guard against division by zero (e.g., all tokens were special/ignored)
    bpb = total_nats / (math.log(2.0) * total_bytes) if total_bytes else float("inf")
    return {"bpb": bpb, "loss": total_loss, "perplexity": perplexity}
