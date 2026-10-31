import os
import socket
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpt_lab.utils.distributed import ComputeContext, init_dist_groups, cleanup_dist_groups


def _muon_bucket_worker(rank, port):
    from gpt_lab.optim.factory import OptimizerFactory

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        context = init_dist_groups("cpu")
        for opt_name, spectral in (("muon", False), ("normuon", False), ("normuon", True)):
            params = [
                torch.nn.Parameter(torch.linspace(-0.3, 0.7, 6).reshape(2, 3) + index / 10)
                for index in range(5)
            ]
            reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in params]

            def group(parameters):
                options = {"anvil_spectral": spectral} if opt_name == "normuon" else {}
                return dict(params=parameters, opt=opt_name, lr=0.01, momentum=0.95,
                            beta=0.95, ns_steps=5, weight_decay=0.1, **options)

            optimizer = OptimizerFactory([group(params)], comp_info=context)
            local = OptimizerFactory([group(reference)], comp_info=ComputeContext(device=torch.device("cpu")))
            strategy = optimizer.backend.strategies[opt_name]
            param_group = optimizer.param_groups[0]
            # Original owners stay [0, 0, 0, 1, 1], regardless of bucket order.
            owned_count = 3 if rank == 0 else 2
            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                for step in range(2):
                    for index, (parameter, expected) in enumerate(zip(params, reference)):
                        base = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                        gradients = [torch.sin(base + index + step + worker) * (worker + 1) for worker in range(2)]
                        parameter.grad = gradients[rank].clone()
                        expected.grad = (gradients[0] + gradients[1]) / 2
                    buckets = [strategy._reduce_bucket(param_group, indices, 2)
                               for indices in ([4, 1, 3], [2, 0])]
                    gathers = []
                    if step == 0:
                        for invalid in ([buckets[0]], [*buckets, buckets[0]]):
                            with pytest.raises(ValueError, match="exactly once"):
                                strategy.dist_compute(param_group, {"buckets": invalid}, gathers, rank, 2)
                        assert not optimizer.state and not gathers
                        for parameter, expected in zip(params, reference):
                            torch.testing.assert_close(parameter, expected)
                    if step == 1:
                        buckets.reverse()
                    strategy.dist_compute(param_group, {"buckets": buckets}, gathers, rank, 2)
                    assert len(gathers) == 1
                    optimizer.backend._finish_gathers(gathers)
                    local.step()
                    for parameter, expected in zip(params, reference):
                        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
                    for key in ("momentum_buffer", "second_momentum_buffer"):
                        actual = optimizer.state[params[0]][key]
                        expected = local.state[reference[0]][key][rank * 3:rank * 3 + owned_count]
                        assert actual.shape[0] == 3
                        torch.testing.assert_close(actual[:owned_count], expected)
                        torch.testing.assert_close(actual[owned_count:], torch.zeros_like(actual[owned_count:]))
    finally:
        cleanup_dist_groups()


def test_two_process_gloo_muon_buckets_preserve_owners():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_muon_bucket_worker, args=(port,), nprocs=2, join=True)
