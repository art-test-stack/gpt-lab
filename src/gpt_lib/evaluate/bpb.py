# https://medium.com/@dip.patel.ict/bits-per-byte-bpb-a-tokenizer-agnostic-way-to-measure-llms-25dfed3f41af
import torch
import torch.distributed as dist
import torch.nn.functional as F
import math 
from typing import Optional

from gpt_lib.utils.distributed import is_ddp_initialized

@torch.no_grad()
def compute_bpb(model, batches, steps: int, token_bytes: torch.Tensor, dist_info: Optional[dict]) -> float:
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
        - step: Number of batches to evaluate.
        - token_bytes (torch.int64): Tensor of shape (V,) containing the byte lengths of each token in the vocabulary; 0 for special tokens.
        - dist_info (Optional[dict]): Distributed training information. Should contain "WORLD_SIZE" if using distributed evaluation.
    Returns:
        - bpb (float): The computed bits-per-byte metric.
        - loss (float): The average loss per token, for debugging.
    """
    dist_is_init = dist_info["IS_DDP_INITIALIZED"] if dist_info is not None else is_ddp_initialized()
    world_size = dist_info["WORLD_SIZE"] if dist_info is not None else 1
    
    device = model.get_device() if hasattr(model, "get_device") else next(model.parameters()).device

    # Accumulators across steps (and later across ranks)
    total_nats  = torch.tensor(0.0, dtype=torch.float32, device=device)  # scalar
    total_bytes = torch.tensor(0,   dtype=torch.int64,   device=device)  # scalar
    total_loss  = torch.tensor(0.0, dtype=torch.float32, device=device)  # scalar, for debugging

    token_bytes = token_bytes.to(device=device, dtype=torch.int64)     # (V,)
    
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)

        output = model(x, y, reduction="none") # (B, Seq) NATs
        try:
            loss2d = output.loss.view(y.size()) # (B, Seq) NATs
            logits = output.logits
        except:
            loss2d = output # Assume output is directly the logits tensor if not wrapped in a ModelOutput-like object. (B, Seq)
        total_loss += loss2d.detach().mean() # For debugging
        loss2d = loss2d.reshape(-1) # (B*Seq,)
        y = y.reshape(-1) # (B*Seq,)

        if (y.int() < 0).any():
            # Mask out ignore_index (<0) before indexing into token_bytes
            valid = (y >= 0) # (B*Seq,)
            ysafe = torch.where(valid, y, torch.zeros_like(y)) # (B*Seq,)
            n_bytes2d = torch.where(
                valid, 
                token_bytes[ysafe], 
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )  # (B*Seq,) int64
            total_nats += (loss2d * (n_bytes2d > 0)).sum() # Only count nats for tokens with positive byte length
            total_bytes += n_bytes2d.sum() # Sum bytes for all tokens in the batch, but only count nats for tokens with positive byte length
        else:
            n_bytes2d = token_bytes[y]  # (B*Seq,) int64
            total_nats += (loss2d * (n_bytes2d > 0)).sum() # Only count nats for tokens with positive byte length
            total_bytes += n_bytes2d.sum() # Sum bytes for all tokens in the batch, but only count nats for tokens with positive byte length

    # Distributed sum over all ranks, if initialized
    if dist_is_init and world_size > 1:
        dist.all_reduce(total_nats,  op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    total_loss = total_loss.item() / world_size / (steps * x.size(0))  # Average loss per token, for debugging

    # Guard against division by zero (e.g., all tokens were special/ignored)
    if total_bytes == 0:
        return { "bpb": float("inf"), "loss": total_loss }

    bpb = total_nats / (math.log(2.0) * total_bytes)
    return { "bpb": bpb, "loss": total_loss }