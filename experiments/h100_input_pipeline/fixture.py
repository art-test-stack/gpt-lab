"""Small local real-file fixture; its timings never represent climbmix or H100."""
import argparse
import json
from pathlib import Path
import tempfile

from cpu import measure


def factory(root, batch_size=2, seq_len=8):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from gpt_lab.data.loader import build_dataloader
    from gpt_lab.tokenizer.tokenizer import Tokenizer
    from gpt_lab.utils.schemas import TokenizerConfig
    from gpt_lab.utils.special_tokens import SpecialTokens
    from gpt_lab.utils.distributed import ComputeContext
    data = root/"fixture"
    data.mkdir(parents=True)
    texts = [f"Document {i}: a small local fixture for ownership checks. "*(1+i%8) for i in range(256)]
    for shard in range(2):
        pq.write_table(pa.table({"text":texts}),data/f"shard_{shard:05d}.parquet",row_group_size=64)
    (data/"meta.json").write_text(json.dumps(dict(name="fixture",base_url="",column_name="text",max_shards=1)))
    special = SpecialTokens()
    cfg = TokenizerConfig(name="fixture-byte",source="local",vocab_size=256+len(special),
                          pat_str=r"\S+|\s+",special_tokens=special)
    tokenizer = Tokenizer({bytes([i]):i for i in range(256)},special.list(),cfg)
    return lambda:build_dataloader(name="fixture",batch_size=batch_size,seq_len=seq_len,
                                   tokenizer=tokenizer,cachedir=root,
                                   comp_info=ComputeContext(torch.device("cpu"),compute_dtype=torch.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    scratch = Path(".tmp/codex/h100-input-pipeline")
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch,prefix="fixture-") as temporary:
        result = measure(factory(Path(temporary),16,2048),128)
    import platform
    import torch
    result.update(source="fixture-only: byte-level tokenizer, generated text in local Parquet",
                  host=platform.platform(),torch=torch.__version__,torch_threads=torch.get_num_threads(),
                  batch_size=16,sequence_length=2048,
                  limitations="Not production GPT2/climbmix throughput; no CUDA, H2D or concurrent training")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
