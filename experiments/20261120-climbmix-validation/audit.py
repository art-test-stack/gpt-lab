"""Replay the real validation loader on eight logical CPU ranks."""

import argparse
from array import array
from bisect import bisect_right
from collections import defaultdict
import hashlib
from itertools import tee
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from gpt_lab.data.loader import DistDataLoader, build_dataloader
from gpt_lab.tokenizer import Tokenizer

WORLD_SIZE = 8
BATCH_SIZE = 16
SEQ_LEN = 2048
BATCHES = 160


def audit_rank(dataset, shard_id, group_sizes, tokenizer, rank, spans, span_file):
    context = SimpleNamespace(rank=rank, world_size=WORLD_SIZE, device=torch.device("cpu"))
    make_loader = build_dataloader(
        name=dataset.name, cachedir=dataset.parent, tokenizer=tokenizer, split="val",
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, comp_info=context,
    )
    real = make_loader()
    manager = real.dataset.sm
    assert manager.shard_idx == (shard_id,)
    assert real._lookahead_size == 16
    locations = [(group, doc) for group in range(rank, len(group_sizes), WORLD_SIZE)
                 for doc in range(group_sizes[group])]
    iterate = manager.iterate

    def check_source_cursor(**kwargs):
        consumed = 0
        for texts, cursor in iterate(**kwargs):
            group, offset = locations[consumed]
            consumed += len(texts)
            assert offset + len(texts) <= group_sizes[group]
            next_group, next_doc = locations[consumed] if consumed < len(locations) else (rank, 0)
            actual = (cursor.global_shard_idx, cursor.shard_idx, cursor.row_group_idx,
                      cursor.offset_in_row_group, cursor.epoch)
            expected = (shard_id, int(consumed == len(locations)), next_group, next_doc, 1)
            assert actual == expected, (rank, actual, expected)
            yield texts, cursor
        assert consumed == len(locations)

    manager.iterate = check_source_cursor
    real.iterator, tag_source = tee(real.iterator)
    bos = tokenizer.get_bos_token_id()
    token_lookup = array("I", [bos])
    starts = []

    class TaggedDataset:
        split = "val"
        tokenizer = SimpleNamespace(get_bos_token_id=lambda: 0)

        def __iter__(self):
            for tokens, cursor in tag_source:
                assert tokens[0] == bos
                start = len(token_lookup)
                starts.append(start)
                token_lookup.extend(tokens[1:])
                yield [0, *range(start, len(token_lookup))], cursor

    tagged = DistDataLoader(TaggedDataset(), BATCH_SIZE, SEQ_LEN, device="cpu")
    digest = hashlib.sha256()
    batch_count = content_count = span_count = 0
    for batch_idx in range(BATCHES):
        batch = next(real, None)
        tags = next(tagged, None)
        assert (batch is None) == (tags is None)
        if batch is None:
            break
        x, y, state = batch
        tag_x, tag_y, _ = tags
        lookup = np.frombuffer(token_lookup, dtype=np.uint32)
        for tensor, tag in ((x, tag_x), (y, tag_y)):
            values = tensor.numpy()
            assert np.array_equal(values, lookup[tag.numpy()])
            digest.update(values.astype("<i8", copy=False).tobytes())
        del lookup
        batch_content = 0
        for row_idx, row in enumerate(tag_y.numpy()):
            boundaries = np.flatnonzero((np.diff(row) != 1) | (row[:-1] == 0)) + 1
            for run in np.split(row, boundaries):
                if run[0] == 0:
                    continue
                doc_idx = bisect_right(starts, int(run[0])) - 1
                group, doc = locations[doc_idx]
                begin = int(run[0]) - starts[doc_idx]
                end = begin + len(run)
                doc_end = starts[doc_idx + 1] if doc_idx + 1 < len(starts) else len(token_lookup)
                assert starts[doc_idx] + end <= doc_end
                spans[shard_id, group, doc].append((begin, end, rank))
                print(rank, batch_idx, row_idx, shard_id, group, doc, begin, end, sep="\t", file=span_file)
                batch_content += len(run)
                span_count += 1
        assert batch_content == state.source_content_tokens
        batch_count += 1
        content_count += batch_content

    return dict(rank=rank, full_batches=batch_count, shortfall_batches=BATCHES - batch_count,
                content_tokens=content_count, spans=span_count, xy_sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="ClimbMix directory with meta.json and its holdout Parquet")
    parser.add_argument("spans", type=Path, help="Output TSV of half-open, zero-based content-token spans")
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = json.loads((args.dataset / "meta.json").read_text())
    shard_id = metadata["max_shards"]
    path = args.dataset / f"shard_{shard_id:05d}.parquet"
    with path.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    with pq.ParquetFile(path) as parquet:
        group_sizes = [parquet.metadata.row_group(group).num_rows for group in range(parquet.num_row_groups)]
    tokenizer = Tokenizer.from_pretrained("gpt2", source="tiktoken")
    spans = defaultdict(list)
    with args.spans.open("w") as span_file:
        print("rank\tbatch\trow\tshard\trow_group\tdocument\ttoken_begin\ttoken_end", file=span_file)
        ranks = [audit_rank(args.dataset, shard_id, group_sizes, tokenizer, rank, spans, span_file)
                 for rank in range(WORLD_SIZE)]

    for (shard, group, doc), intervals in spans.items():
        previous_end = 0
        for begin, end, rank in sorted(intervals):
            assert begin >= previous_end, (shard, group, doc, begin, previous_end)
            assert group % WORLD_SIZE == rank
            previous_end = end
    total_batches = sum(rank["full_batches"] for rank in ranks)
    print(json.dumps(dict(shard_id=shard_id, shard_sha256=checksum, row_groups=len(group_sizes),
                          duplicate_text_across_files_checked=False, repeated_scored_source_spans=0,
                          target_positions=total_batches * BATCH_SIZE * SEQ_LEN, ranks=ranks), indent=2))
    return int(total_batches != BATCHES * WORLD_SIZE)


if __name__ == "__main__":
    raise SystemExit(main())
