"""Explicit capture of one fixed-shape compiled training microbatch."""

from contextlib import nullcontext

import torch


class CapturedForwardBackward:
    """Replay one microbatch into persistent parameter gradients.

    Caller copies/consumes each batch and loss before reusing loader/output
    storage, and clears gradients in place only after the accumulated update.
    Optimizer, scheduler, finite-loss checks and collective waits stay outside.
    Capture does not step the optimizer or advance the dataloader. Restore model
    tensors and RNG after warmup; retain allocated gradients for graph lifetime.
    Python forward side effects and changing parameter participation are unsupported.
    """
    def __init__(self, model, compute_loss, payload, n_acc_steps, context=nullcontext):
        if (
            not isinstance(payload, (tuple, list)) or len(payload) != 2
            or any(not isinstance(t, torch.Tensor) or not t.is_cuda for t in payload)
            or payload[0].device != payload[1].device
        ):
            raise ValueError("CUDA capture requires a pair of CUDA input/label tensors on one device")
        if n_acc_steps < 1:
            raise ValueError("CUDA capture requires positive n_acc_steps")
        self.inputs = tuple(t.detach().clone() for t in payload)
        self.graph = torch.cuda.CUDAGraph()
        device = self.inputs[0].device
        params = list(model.parameters())
        tensors = params + list(model.buffers())
        saved = [t.detach().clone() for t in tensors]
        grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

        def forward_backward():
            with context():
                loss = compute_loss(self.inputs) / n_acc_steps
            loss.backward()
            return loss.detach()

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            # Compilation/warmup must finish before recording CUDA operations.
            with torch.random.fork_rng(devices=[device]), torch.cuda.device(device):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        model.zero_grad(set_to_none=False)
                        forward_backward()
                    model.zero_grad(set_to_none=False)
                stream.synchronize()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.loss = forward_backward()
                stream.synchronize()
        finally:
            stream.synchronize()
            with torch.no_grad():
                for tensor, value in zip(tensors, saved):
                    tensor.copy_(value)
                for param, grad in zip(params, grads):
                    if param.grad is not None:
                        if grad is None:
                            param.grad.zero_()
                        else:
                            param.grad.copy_(grad)

    def replay(self, payload):
        if not isinstance(payload, (tuple, list)) or len(payload) != len(self.inputs):
            raise ValueError("CUDA replay requires the captured input/label pair")
        for source, target in zip(payload, self.inputs):
            if (
                not isinstance(source, torch.Tensor) or source.shape != target.shape
                or source.dtype != target.dtype or source.device != target.device
            ):
                raise ValueError("CUDA replay input shape, dtype, and device must match capture")
        for source, target in zip(payload, self.inputs):
            target.copy_(source)
        self.graph.replay()
        return self.loss
