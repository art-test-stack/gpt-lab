# Communication buckets — closed investigation

Question: do bucket packing, pipelining or final-backward hooks beat baseline transport?
Tested all three opt-in modes with Muon and NorMuon on 4 A100-SXM4-80GB,
BF16/FA2, microbatch 16, sequence 2048, global batch 524288, horizon 2317/warmup 300.
Neither optimizer improved reliably; extra memory increased. **Retain baseline.**
See [RESULTS.md](RESULTS.md) for SHAs, run IDs, padding, memory and NCCL limits.
Entry point: `python experiments/communication_buckets/modal_communication.py --help`.
Original tested launchers remain on both preserved communication branches;
do not rerun a sweep. Profiling/longer quality comparisons were not performed.
Agent A owns subsequent optimizer work; see [HANDOFF.md](HANDOFF.md).
