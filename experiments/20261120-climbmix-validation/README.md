# ClimbMix validation audit

Two fresh CPU processes replayed `gpt-lab-cache/data/climbmix-base/shard_01061.parquet`
from the Modal cache. The file has 84 row groups and SHA-256
`9025ed81d751886e7810238f2c71a4d68ee907860f3c506d1cd3e74bda135fd8`.
The full ClimbMix training manifest contains shard IDs `0` through `1060`.
Both paired PR #54 H100 arms used training source IDs `620`, `631`, `979`, and
`980`, with `1061` reserved for validation. Exact duplicate text across files
was unchecked.

The audit used eight logical ranks, GPT-2 tokenization, sequence length 2,048,
batch size 16, and the lossless BOS packer with a 16-document lookahead. Each
rank read row groups `r`, `r + 8`, `r + 16`, and so on. All eight reached
160 full batches, totaling **41,943,040 target positions** and **41,871,148
source content tokens**. The existing holdout met the budget on every rank.

| Rank | Full batches | Content tokens | Scored spans | SHA-256 of packed `(x, y)`, both runs |
|---:|---:|---:|---:|---|
| 0 | 160 | 5,234,204 | 11,232 | `88da24b1569d262c48f0983e81eeda94d3599af1b55884aa7adb4b6d51b73130` |
| 1 | 160 | 5,233,908 | 11,525 | `b9e6ff030e2b3d0d5435dfe9dbcddcd051f4d3878ac7db60fba4a30527cc9fab` |
| 2 | 160 | 5,233,899 | 11,534 | `0e7281e774bc016270308cdb5d19cb521c25c1500fe6d5c7bc29939ba8dc533f` |
| 3 | 160 | 5,233,648 | 11,789 | `6271e481c54330627ce61d1053c3aeb735ee01258acb27f0a700c021f7e42886` |
| 4 | 160 | 5,234,027 | 11,405 | `c06adabfca8ca77acdca60863fa5f56c316ae65ded918d78e35efb812199d06a` |
| 5 | 160 | 5,233,881 | 11,546 | `c47f7c5919aab9043767500292c8ba11d0a032b15200e405ce5066eae8c327dd` |
| 6 | 160 | 5,233,774 | 11,659 | `7e16639d3423a59d466b885eef7df01229bc1eb5ef2f48875a4c3e3a602a05b9` |
| 7 | 160 | 5,233,807 | 11,624 | `db514c5a8a2bbae1fb76811ef8cdf16f9f4e43986fd3872624c355c954d621d2` |

For each batch, the audit hashes `x` followed by `y`, serialized as row-major,
little-endian signed 64-bit integers. It verifies the Parquet cursor and
replays the same documents with unique token positions. Mapping these positions
back to actual token IDs reproduces both tensors exactly. The audit checked all
92,314 scored spans for interval overlap within and across ranks and found zero
overlaps. Each span is keyed by shard ID, row-group index, document index
within that row group, and a half-open GPT-2 content-token interval. Indices
are zero-based, and content offsets exclude BOS. The span TSVs matched
byte-for-byte with SHA-256
`f6ea316d1a495202bb9c3bd3159d276bf95a7e31e0c09f6fc515711710afeced`.

## Reproduce

Place `meta.json` and `shard_01061.parquet` from the project cache in the
local `climbmix-base` directory used below. Run from the repository root.

```bash
audit_dir=.tmp/codex/climbmix-validation
for run in 1 2; do
  uv run --locked --extra cpu python experiments/20261120-climbmix-validation/audit.py \
    "$audit_dir/climbmix-base" "$audit_dir/spans-$run.tsv" > "$audit_dir/audit-$run.json"
done
cmp "$audit_dir/audit-1.json" "$audit_dir/audit-2.json"
cmp "$audit_dir/spans-1.tsv" "$audit_dir/spans-2.tsv"
```
