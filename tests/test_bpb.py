from gpt_lab.utils.distributed import ComputeContext
from datetime import timedelta
import math
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from gpt_lab.data.loader import build_dataloader
from gpt_lab.evaluate.bpb import compute_bpb


class _FixedLossModel:
    def get_device(self):
        return torch.device("cpu")

    def __call__(self, x, y, reduction):
        assert reduction == "none"
        return SimpleNamespace(loss=x.float(), logits=None)


def test_compute_bpb_reports_mean_loss_per_token():
    batches = [
        (torch.tensor([[1, 2]]), torch.tensor([[0, 1]]), None),
        (torch.tensor([[3, 0]]), torch.tensor([[1, -1]]), None),
    ]
    token_bytes = torch.tensor([0, 1])

    result = compute_bpb(
        _FixedLossModel(),
        batches,
        len(batches),
        token_bytes,
        comp_info=ComputeContext(device=torch.device("cpu")),
    )

    assert result["loss"] == 2.0
    assert result["scored_targets"] == 3
    assert result["bpb"] == pytest.approx((2 + 3) / (2 * math.log(2.0)))


def _uneven_validation_worker(rank, rendezvous, cache):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        device = torch.device("cpu")
        context = ComputeContext(device=device, mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("dp",)))

        class Tokenizer:
            def get_bos_token_id(self):
                return 0

            def __call__(self, text, *, prepend_bos, threads):
                assert prepend_bos
                return [0, *map(int, text.split())]

        class RecordingModel(_FixedLossModel):
            def __call__(self, x, y, reduction):
                assert (x >= 0).all()
                targets.append(y[y >= 0].tolist())
                batch_shapes.append(tuple(y.shape))
                # Padding must contribute neither loss nor bytes.
                return y.float().masked_fill(y < 0, float("nan"))

        batches = build_dataloader("uneven", batch_size=2, seq_len=4, tokenizer=Tokenizer(),
                                   cachedir=cache, comp_info=context, split="val", max_shards=0)
        targets, batch_shapes = [], []
        token_bytes = torch.arange(12)
        result = compute_bpb(RecordingModel(), batches(include_partial=True), None, token_bytes, context)
        assert targets == ([[1, 2]] if rank == 0 else [list(range(3, 11)), [11]])
        assert batch_shapes == ([(1, 4)] if rank == 0 else [(2, 4), (1, 4)])
        assert result["scored_targets"] == 11
        assert result["loss"] == 6
        assert result["bpb"] == pytest.approx(1 / math.log(2))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_full_bpb_scores_each_target_across_uneven_parquet_ranks(tmp_path):
    dataset = tmp_path / "uneven"
    dataset.mkdir()
    with pq.ParquetWriter(dataset / "shard_00000.parquet", pa.schema([("text", pa.string())])) as writer:
        writer.write_table(pa.table({"text": ["1 2"]}))
        writer.write_table(pa.table({"text": ["3 4 5 6 7 8 9 10 11"]}))
    mp.spawn(_uneven_validation_worker, args=((tmp_path / "rendezvous").as_uri(), tmp_path), nprocs=2)
