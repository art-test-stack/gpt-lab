"""Small synthetic CPU loader examples; no model, tokenizer, downloads or GPU."""

from dataclasses import asdict
from itertools import cycle
import json
import platform
import time

import torch

from gpt_lab.data.loader import DistDataLoader

BOS = 999
DOCUMENTS = [[BOS, *range(1, length)] for length in (3, 9, 23, 65)]


class Documents:
    def __iter__(self):
        for document in cycle(DOCUMENTS):
            yield torch.tensor(document, dtype=torch.long), None


def main():
    torch.set_num_threads(1)
    results = []
    for policy in ("stream", "bos_aligned", "bos_bestfit_crop"):
        loader = DistDataLoader(Documents(), batch_size=2, seq_len=16, device="cpu",
                               packing_strategy=policy, bos_token_id=BOS, best_fit_buffer_docs=8)
        for _ in range(10):
            x, y, _ = next(loader)
            assert torch.equal(x[:, 1:], y[:, :-1])
            if policy != "stream":
                assert torch.all(x[:, 0] == BOS)
        before = asdict(loader.packing_stats)
        started = time.perf_counter()
        for _ in range(200):
            next(loader)
        elapsed = time.perf_counter() - started
        after = asdict(loader.packing_stats)
        results.append(dict(policy=policy, seconds=elapsed, target_tokens=6400,
                            target_tokens_per_second=6400 / elapsed,
                            source_tokens_read=after["source_tokens_read"] - before["source_tokens_read"],
                            cropped_suffix_tokens=after["destructive_cropped_tokens"] - before["destructive_cropped_tokens"],
                            synthetic_bos_tokens=after["synthetic_bos_tokens_inserted"] - before["synthetic_bos_tokens_inserted"]))
    print(json.dumps(dict(device="cpu", machine=platform.machine(), torch=torch.__version__,
                         threads=1, world_size=1, batch_size=2, seq_len=16, token_dtype="int64",
                         model=None, best_fit_buffer_docs=8, warmup_batches=10,
                         measured_batches=200, document_lengths=[3, 9, 23, 65], results=results), indent=2))


if __name__ == "__main__":
    main()
