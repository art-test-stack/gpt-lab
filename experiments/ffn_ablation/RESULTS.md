# FFN results and provenance

Baseline post-#37 SHA: `10ac35ff94e9d3ebec68f00280fc3c35913d8cdd`.
Native activation/checkpoint/FLOP integration SHA: `989acfbb96d3e3fc8be5a59729a704eaa825dbaf`.
Nanochat comparison: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.
Both residual additions occur in the early history; their combined intent remains
unknown. C preserves the inner residual and is not nanochat residual equivalence.

| Case/order | Architecture | Median update ms | Global tok/s | Peak allocated bytes |
|---|---|---:|---:|---:|
| 1-A | A | 834.178 | 157146 | 28653086720 |
| 2-B | B | 833.201 | 157324 | 28652496896 |
| 3-C | C | 718.403 | 182482 | 25840723968 |
| 4-C | C | 719.056 | 182295 | 25840723968 |
| 5-B | B | 832.858 | 157299 | 28652496896 |
| 6-A | A | 835.007 | 156967 | 28653742080 |

One A100-80GB, BF16, FA2, 12×768/3072, context 2048, batch 16,
4 accumulation microsteps, 131072 tokens/update, horizon 2317/warmup 300,
8 benchmark warmups then 10 measured updates, deterministic synthetic ring.
External run: `gpt-lab-runs/ffn/ffn-a100-01`. The full six-case samples/losses
remain in `a100-summary.json`, original uploaded-source SHA256s and commands in
`source-manifest.json`, resolved model/optimizer/software in `baseline.json`.

A/B have 422314440 parameters and 976837536 useful training FLOPs/token.
C has 394002888 parameters and 806968224 FLOPs/token. B changed the function
without establishing a throughput benefit. C improved throughput while reducing
matrix work; that need not improve useful-FLOP utilization. Ten synthetic updates
are not a model-quality result. A later one-GPU C result around 162197 tok/s is
not the 4-GPU pipeline and must not be treated as an apples-to-apples improvement.

Production C smoke: external `ffn-training/ffn-c-4xa100-01` and
`relu-squared-4xa100-10`, four A100s, real climbmix-base/GPT2, 524288 global
tokens/update, horizon 2317 capped at 10, no extra warmup. Reported later steps
were about 715k–721k tok/s; the historical SwiGLU reference was about 605k.
These are different architectures and short runs, with evolving useful-FLOP
accounting. No equal-quality or precision-acceleration conclusion is established.

Agreed defaults: fresh configs/AutoConfig use `relu_squared`; spelling alias
`relu_squarred` accepted; Python None and missing/null legacy checkpoint fields
select SwiGLU. Both production FFNs retain their internal residual. Matrix keys
and stored architecture metadata determine checkpoint interpretation.

The original temporary A/B/C Python sources were not tracked on the public
baseline; preserve their exact uploaded-source hashes rather than attributing
an invented source commit. Archived notes retain the historical setup/commands;
no new architecture variant or batch-32 experiment was introduced here.
