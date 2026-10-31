"""Owned real batches and a literal replay of DistDataLoader's transfer tail."""
import copy
import hashlib

import torch


def owned(tensor, device):
    result = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=device)
    result.copy_(tensor, non_blocking=False)
    return result


def verify(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(actual.cpu(), expected.cpu())


def digest_batch(batch):
    digest = hashlib.sha256()
    for tensor in batch[:2]:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype), tensor.stride())).encode())
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture(loader, count):
    """Clone before advancing the reusable loader, with copies complete at return."""
    packed, batches, hashes = [], [], []
    for _ in range(count):
        raw = next(loader)
        x, y = (owned(tensor, loader.device) for tensor in raw[:2])
        # Explicit boundary outside timing: finish output copies before loader reuse.
        if loader.device.type == "cuda":
            torch.cuda.synchronize(loader.device)
        stage = owned(loader.cpu, "cpu")
        data = stage.view(loader.B, loader.T + 1)
        verify(x, data[:, :-1].contiguous())
        verify(y, data[:, 1:].contiguous())
        packed.append(stage)
        batch = (x, y, copy.deepcopy(raw[2]))
        batches.append(batch)
        hashes.append(digest_batch(batch))
    if loader.device.type == "cuda":
        torch.cuda.synchronize(loader.device)
    return packed, batches, hashes


class Replay:
    def __init__(self, packed, batches, device, gpu=False):
        self.packed, self.batches, self.gpu_resident = packed, batches, gpu
        self.index = 0
        x = batches[0][0]
        self.B, self.T = x.shape
        self.device = torch.device(device)
        self.cpu = torch.empty(self.B * (self.T + 1), dtype=x.dtype,
                               pin_memory=self.device.type == "cuda")
        self.gpu = torch.empty_like(self.cpu, device=self.device)
        out = torch.empty(2 * self.B * self.T, dtype=x.dtype, device=self.device)
        self.inputs = out[:self.B * self.T].view(self.B, self.T)
        self.targets = out[self.B * self.T:].view(self.B, self.T)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.batches):
            raise StopIteration
        index = self.index
        self.index += 1
        if self.gpu_resident:
            return self.batches[index]
        self.cpu.copy_(self.packed[index])
        # Keep this identical to the pinned production transfer tail.
        self.gpu.copy_(self.cpu, non_blocking=False)
        data = self.gpu.view(self.B, self.T + 1)
        self.inputs.copy_(data[:, :-1])
        self.targets.copy_(data[:, 1:])
        return self.inputs, self.targets, self.batches[index][2]


def cache_bytes(packed, batches):
    return dict(cpu_packed_bytes=sum(t.numel() * t.element_size() for t in packed),
                resident_output_bytes=sum(t.numel() * t.element_size()
                                          for batch in batches for t in batch[:2]))
