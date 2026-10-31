"""Experimental fixed-layout communication; optimizer equations stay in strategies."""
from dataclasses import dataclass, field
import time
from threading import Lock

import torch
import torch.distributed as dist


class _CompletedWork:
    def wait(self):
        return True


@dataclass
class _Piece:
    parameter: torch.Tensor
    start: int
    length: int
    destination: torch.Tensor


@dataclass
class _Bucket:
    kind: str
    device: torch.device
    dtype: torch.dtype
    owners: list[list[_Piece]]
    lengths: list[int]
    parameters: set = field(default_factory=set)
    capacity: int = 0
    packed: torch.Tensor | None = None
    received: torch.Tensor | None = None
    work: object = None


class BucketCommunication:
    """B: immediate waits; C: packing pipeline; D: final-backward leaf hooks.

    Fixed active gradients and matching layouts are required on all ranks.
    D learns the active set on its first complete optimizer step. Python hooks
    and c10d Work objects remain outside CUDA capture.
    """

    def __init__(self, optimizer, mode: str, bucket_bytes: int, timing: bool = False):
        if mode not in ("buckets", "pipeline", "backward"):
            raise ValueError(f"Unknown communication mode: {mode!r}")
        if isinstance(bucket_bytes, bool) or not isinstance(bucket_bytes, int) or bucket_bytes < 1:
            raise ValueError("bucket_bytes must be a positive integer (packed bytes, including padding)")
        if not optimizer.comp_info.is_distributed or not dist.is_initialized():
            raise ValueError("Bucket communication requires an initialized distributed optimizer")
        if mode == "backward" and optimizer.comp_info.compute_dtype == torch.float16:
            raise ValueError("Backward communication does not support FP16 gradient scaling")
        self.optimizer = optimizer
        self.mode, self.bucket_bytes, self.timing = mode, bucket_bytes, timing
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.average = dist.get_backend() == "gloo"
        self.reduce_op = dist.ReduceOp.SUM if self.average else dist.ReduceOp.AVG
        self.parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.parameter_indices = {id(p): index for index, p in enumerate(self.parameters)}
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("Communication plans require unique parameters; tied aliases must share one entry")
        if any(group["opt"] not in ("adamw", "muon", "normuon") for group in optimizer.param_groups):
            raise ValueError("Bucket communication supports AdamW, Muon and NorMuon only")
        self.handles = []
        self.prepared = False
        self.buckets = []
        self.infos = []
        self.active = []
        self.ready = set()
        self.next_bucket = 0
        self.final_microstep = False
        self.measures = []
        self.step_measures = []
        self.step_count = 0
        self.setup_count = 0
        self.validation_count = 0
        self.reduced = False
        self.status = torch.zeros((), dtype=torch.int32, device=optimizer.comp_info.device)
        self.invalid_pack = False
        self.launch_lock = Lock()
        if mode == "backward":
            for parameter in self.parameters:
                if parameter.requires_grad:
                    self.handles.append(parameter.register_post_accumulate_grad_hook(self._ready))

    def begin_backward(self, final_microstep: bool):
        if self.mode == "backward":
            if self.optimizer.comp_info.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Backward communication hooks cannot run inside CUDA capture")
            self.final_microstep = final_microstep

    def _ready(self, parameter):
        if not self.prepared or not self.final_microstep:
            return
        if self.optimizer.comp_info.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Backward communication hooks cannot run inside CUDA capture")
        with self.launch_lock, torch.no_grad():
            self.ready.add(parameter)
            self._launch_ready()

    def _measure(self, name, action):
        if not self.timing:
            return action()
        device = self.optimizer.comp_info.device
        with torch.profiler.record_function(f"gradient_communication::{name}"):
            host_start = time.perf_counter()
            if device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = action()
                end.record()
                self.measures.append((name, start, end))
            else:
                start = time.perf_counter()
                result = action()
                self.measures.append((name, (time.perf_counter() - start) * 1000, None))
            self.measures.append(("host_" + name, (time.perf_counter() - host_start) * 1000, None))
        return result

    def _add(self, kind, pieces):
        parameter = pieces[0][1].parameter
        owners = self.world if kind == "reduce_scatter" else 1
        previous = self.buckets[-1] if self.buckets else None
        compatible = previous is not None and (
            previous.kind, previous.device, previous.dtype
        ) == (kind, parameter.device, parameter.dtype)
        lengths = previous.lengths.copy() if compatible else [0] * owners
        for owner, piece in pieces:
            lengths[owner] += piece.length
        size = max(lengths) * owners * parameter.element_size()
        if not compatible or size > self.bucket_bytes:
            previous = _Bucket(kind, parameter.device, parameter.dtype,
                               [[] for _ in range(owners)], [0] * owners)
            self.buckets.append(previous)
        for owner, piece in pieces:
            previous.owners[owner].append(piece)
            previous.lengths[owner] += piece.length
            previous.parameters.add(piece.parameter)

    def _prepare(self):
        active = [p.grad is not None for p in self.parameters]
        layout = [
            (tuple(p.shape), str(p.dtype), enabled)
            for p, enabled in zip(self.parameters, active)
        ]
        signature = (self.mode, self.bucket_bytes,
                     [(group["opt"], len(group["params"])) for group in self.optimizer.param_groups], layout)
        signatures = [None] * self.world
        # A single setup exchange prevents rank-dependent None/layout mismatches.
        dist.all_gather_object(signatures, signature)
        self.setup_count += 1
        if any(other != signature for other in signatures):
            raise ValueError("Gradient presence, shapes and dtypes must match across ranks")
        self.active = active
        self.layout = [(p.shape, p.dtype, p.device) for p in self.parameters]
        self.infos = []
        for group in self.optimizer.param_groups:
            parameters = group["params"]
            if group["opt"] == "adamw":
                self.infos.append({"average": False, "param_infos": {}})
            elif parameters:
                if any(p.grad is None for p in parameters):
                    raise ValueError("Muon/NorMuon requires every grouped gradient; missing gradients are unsupported")
                p = parameters[0]
                if any(q.shape != p.shape or q.dtype != p.dtype or q.device != p.device for q in parameters):
                    raise ValueError("Muon/NorMuon groups must have one shape, dtype and device")
                chunk = (len(parameters) + self.world - 1) // self.world
                self.infos.append({
                    "work": _CompletedWork(), "average": False,
                    "grad_chunk": torch.zeros(chunk, *p.shape, dtype=p.dtype, device=p.device),
                    "stacked_grads": torch.empty(chunk * self.world, *p.shape, dtype=p.dtype, device=p.device),
                    "updated_params": torch.empty(chunk, *p.shape, dtype=p.dtype, device=p.device),
                    "chunk_size": chunk, "param_indices": list(range(len(parameters))),
                    "owner_positions": list(range(min(chunk, max(0, len(parameters) - self.rank * chunk)))),
                })
            else:
                self.infos.append({"work": None})

        # Fixed reverse group order favors late-layer gradients without changing owners.
        for group, info in reversed(list(zip(self.optimizer.param_groups, self.infos))):
            parameters = group["params"]
            if group["opt"] == "adamw":
                for p in reversed(parameters):
                    if p.grad is None:
                        continue
                    small = p.numel() < 1024
                    if not small and p.shape[0] % self.world:
                        raise ValueError("AdamW reduce_scatter requires shape[0] divisible by world_size")
                    shape = p.shape if small else (p.shape[0] // self.world, *p.shape[1:])
                    target = torch.empty(shape, dtype=p.dtype, device=p.device)
                    info["param_infos"][p] = {"work": _CompletedWork(), "grad_slice": target, "is_small": small}
                    length = target.numel()
                    if small:
                        self._add("all_reduce", [(0, _Piece(p, 0, length, target.view(-1)))])
                    else:
                        fragment = max(1, self.bucket_bytes // (self.world * p.element_size()))
                        for start in range(0, length, fragment):
                            count = min(fragment, length - start)
                            self._add("reduce_scatter", [
                                (owner, _Piece(p, owner * length + start, count, target.view(-1)[start:start + count]))
                                for owner in range(self.world)
                            ])
            elif parameters:
                chunk = info["chunk_size"]
                # Interleave owners to avoid unnecessary padding while preserving indices.
                for position in reversed(range(chunk)):
                    for owner in reversed(range(self.world)):
                        index = owner * chunk + position
                        if index < len(parameters):
                            p = parameters[index]
                            target = info["grad_chunk"][position].view(-1)
                            self._add("reduce_scatter", [(owner, _Piece(p, 0, p.numel(), target))])
        for bucket in self.buckets:
            bucket.capacity = max(bucket.lengths)
            bucket.packed = torch.empty(bucket.capacity * len(bucket.owners), device=bucket.device, dtype=bucket.dtype)
            bucket.received = (bucket.packed if bucket.kind == "all_reduce" else
                               torch.empty(bucket.capacity, device=bucket.device, dtype=bucket.dtype))
        self.prepared = True

    def _launch(self, bucket):
        def pack():
            bucket.packed.zero_()
            for owner, pieces in enumerate(bucket.owners):
                offset = owner * bucket.capacity
                for piece in pieces:
                    gradient = piece.parameter.grad
                    if (gradient is None or gradient.dtype != bucket.dtype
                            or gradient.device != bucket.device or not gradient.is_contiguous()
                            or gradient.numel() < piece.start + piece.length):
                        # Drain the canonical schedule with zeros, then reject on every rank.
                        self.invalid_pack = True
                    else:
                        bucket.packed[offset:offset + piece.length].copy_(gradient.view(-1)[piece.start:piece.start + piece.length])
                    offset += piece.length
        self._measure("pack_ms", pack)
        launch_start = time.perf_counter() if self.timing else 0
        if bucket.kind == "all_reduce":
            bucket.work = dist.all_reduce(bucket.packed, op=self.reduce_op, async_op=True)
        else:
            bucket.work = dist.reduce_scatter_tensor(bucket.received, bucket.packed, op=self.reduce_op, async_op=True)
        if self.timing:
            self.measures.append(("host_launch_ms", (time.perf_counter() - launch_start) * 1000, None))

    def _finish(self, bucket):
        self._measure("wait_ms", bucket.work.wait)
        bucket.work = None
        if self.average:
            bucket.received.div_(self.world)
        def unpack():
            owner = self.rank if bucket.kind == "reduce_scatter" else 0
            offset = 0
            for piece in bucket.owners[owner]:
                piece.destination.copy_(bucket.received[offset:offset + piece.length])
                offset += piece.length
        self._measure("unpack_ms", unpack)

    def _launch_ready(self):
        while self.next_bucket < len(self.buckets):
            bucket = self.buckets[self.next_bucket]
            if not bucket.parameters.issubset(self.ready):
                break
            self._launch(bucket)
            self.next_bucket += 1

    def _check_gradients(self):
        invalid = self.invalid_pack
        for parameter, enabled, layout in zip(self.parameters, self.active, self.layout):
            gradient = parameter.grad
            invalid |= (gradient is not None) != enabled or (parameter.shape, parameter.dtype, parameter.device) != layout
            if gradient is not None:
                invalid |= (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                            or gradient.device != parameter.device or not gradient.is_contiguous())
        self.status.fill_(int(invalid))
        work = dist.all_reduce(self.status, op=dist.ReduceOp.MAX, async_op=True)
        self.validation_count += 1
        work.wait()
        return bool(self.status.item())

    def finish_reductions(self):
        """Complete the fixed reduction schedule before another collective/update."""
        if self.reduced:
            return
        if not self.prepared:
            self._prepare()
        if self.mode == "backward":
            # Drain the canonical suffix even when a final hook/gradient is absent.
            for bucket in self.buckets[self.next_bucket:]:
                self._launch(bucket)
                self.next_bucket += 1
            invalid = self._measure("validation_ms", self._check_gradients)
        else:
            invalid = self._measure("validation_ms", self._check_gradients)
            if invalid:
                self.measures = []
                raise ValueError("Gradient presence/layout changed; a fixed active set is required on all ranks")
            for bucket in self.buckets:
                self._launch(bucket)
                if self.mode == "buckets":
                    self._finish(bucket)
        if self.mode != "buckets":
            for bucket in self.buckets:
                self._finish(bucket)
        if invalid:
            self.ready.clear()
            self.next_bucket = 0
            self.final_microstep = False
            self.invalid_pack = False
            self.measures = []
            raise ValueError("Gradient presence/layout changed; a fixed active set is required on all ranks")
        self.reduced = True

    def step(self, strategies, finish_gathers):
        self.finish_reductions()
        gathers = []
        for group, info in zip(self.optimizer.param_groups, self.infos):
            if group["opt"] == "adamw":
                for p, pinfo in info["param_infos"].items():
                    if pinfo["is_small"]:
                        p.grad.copy_(pinfo["grad_slice"])
            strategies[group["opt"]].dist_compute(group, info, gathers, self.rank, self.world)
        self._measure("gather_wait_and_copy_ms", lambda: finish_gathers(gathers))
        if self.timing:
            self.step_measures.append(self.measures)
        self.measures = []
        self.step_count += 1
        self.ready.clear()
        self.next_bucket = 0
        self.final_microstep = False
        self.invalid_pack = False
        self.reduced = False

    def reset(self):
        if any(bucket.work is not None for bucket in self.buckets):
            raise RuntimeError("Cannot reset communication while reductions are in flight")
        self.prepared = False
        self.buckets, self.infos, self.active = [], [], []
        self.ready.clear()
        self.next_bucket = 0
        self.final_microstep = False
        self.invalid_pack = False
        self.measures = []
        self.reduced = False

    def close(self):
        self.reset()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def stats(self):
        buckets = []
        persistent = [self.status]
        for bucket in self.buckets:
            packed = bucket.packed.numel() * bucket.packed.element_size()
            useful = sum(bucket.lengths) * bucket.packed.element_size()
            buckets.append(dict(kind=bucket.kind, dtype=str(bucket.dtype),
                                packed_bytes=packed, useful_bytes=useful,
                                padding_bytes=packed - useful,
                                owner_elements=bucket.lengths.copy(),
                                parameter_indices=sorted(self.parameter_indices[id(p)] for p in bucket.parameters),
                                oversized=packed > self.bucket_bytes))
            persistent.extend((bucket.packed, bucket.received))
        gather_sizes = []
        for group, info in zip(self.optimizer.param_groups, self.infos):
            if group["opt"] == "adamw":
                for p, pinfo in info["param_infos"].items():
                    persistent.append(pinfo["grad_slice"])
                    if not pinfo["is_small"]:
                        gather_sizes.append(p.numel() * p.element_size())
            elif group["params"]:
                persistent.extend(info[key] for key in ("grad_chunk", "stacked_grads", "updated_params"))
                gather_sizes.append(info["stacked_grads"].numel() * info["stacked_grads"].element_size())
        storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in persistent}
        timings = []
        for measures in self.step_measures if self.timing else []:
            values = {name: 0.0 for name in ("pack_ms", "unpack_ms", "wait_ms", "validation_ms", "gather_wait_and_copy_ms")}
            for name, start, end in measures:
                values[name] = values.get(name, 0.0) + (start.elapsed_time(end) if end is not None else start)
            timings.append(values)
        factor = (self.world - 1) / self.world
        ring_send = factor * (sum(b["packed_bytes"] * (2 if b["kind"] == "all_reduce" else 1)
                                 for b in buckets) + sum(gather_sizes) + 2 * self.status.element_size())
        return dict(mode=self.mode, bucket_target_bytes=self.bucket_bytes, buckets=buckets,
                    gradient_collective_count=len(buckets), gather_collective_count=len(gather_sizes),
                    collective_count=len(buckets) + len(gather_sizes) + 1,
                    control_collective_count=1, control_payload_bytes=self.status.element_size(),
                    reduction_input_bytes=sum(b["packed_bytes"] for b in buckets),
                    padding_bytes=sum(b["padding_bytes"] for b in buckets),
                    gather_output_bytes=sum(gather_sizes), persistent_bytes=sum(storages.values()),
                    ring_send_bytes_estimate_per_rank=ring_send,
                    traffic_estimate_note="one-way ring model including control; not measured NCCL link traffic",
                    setup_object_exchanges=self.setup_count, completed_updates=self.step_count,
                    validation_collectives_total=self.validation_count,
                    timing_enabled=self.timing, per_update=timings, cuda_capture_verified=False)
