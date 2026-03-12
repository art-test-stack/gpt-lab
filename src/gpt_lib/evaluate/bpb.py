# https://medium.com/@dip.patel.ict/bits-per-byte-bpb-a-tokenizer-agnostic-way-to-measure-llms-25dfed3f41af
import torch
import torch.distributed as dist
import math 

@torch.no_grad()
def compute_bpb(model, batches, steps: int, token_bytes: torch.Tensor) -> float:
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
    """

    device = model.get_device() if hasattr(model, "get_device") else next(model.parameters()).device

    # Accumulators across steps (and later across ranks)
    sum_nats  = torch.tensor(0.0, dtype=torch.float32, device=device)  # scalar
    sum_bytes = torch.tensor(0,   dtype=torch.int64,   device=device)  # scalar

    token_bytes = token_bytes.to(device=device, dtype=torch.int64)     # (V,)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)                  # x: (B, Seq), y: (B, Seq)
        x = x.to(device)
        y = y.to(device)

        output = model(x, y, loss_reduction='none')  # (B, Seq) NATs
        try:
            loss2d = output.loss2d
        except:
            loss2d = output # Assume output is directly the loss2d tensor if not wrapped in a ModelOutput-like object. (B, Seq)
        loss1d = loss2d.reshape(-1)                       # (B*Seq,)
        y1d    = y.reshape(-1)                            # (B*Seq,)

        if (y1d < 0).any():
            # Mask out ignore_index (<0) before indexing into token_bytes
            valid  = (y1d >= 0)                                      # (B*Seq,)
            ysafe  = torch.where(valid, y1d, torch.zeros_like(y1d))  # (B*Seq,)
            nb     = torch.where(valid, token_bytes[ysafe], torch.zeros_like(y1d))  # (B*Seq,) int64
        else:
            nb = token_bytes[y1d]  # (B*Seq,) int64

        # Count only tokens with positive byte length
        counted = (nb > 0)                             # (B*Seq,) bool
        sum_nats  += (loss1d[counted]).sum()           # scalar
        sum_bytes += nb[counted].sum()                 # scalar int64

    # Distributed sum over all ranks, if initialized
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(sum_nats,  op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_bytes, op=dist.ReduceOp.SUM)

    total_nats  = float(sum_nats.item())
    total_bytes = int(sum_bytes.item())

    # Guard against division by zero (e.g., all tokens were special/ignored)
    if total_bytes == 0:
        return float("inf")

    bpb = total_nats / (math.log(2.0) * total_bytes)
    return bpb