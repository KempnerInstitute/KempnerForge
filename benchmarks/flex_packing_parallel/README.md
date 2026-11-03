# FlexAttention Packing Under Parallelism

**Date**: 2026-10-28
**Commit**: `cf3ea9f` (`feat/flex-attention-validation` with `main` 62df6ef merged in)
**Hardware**: 1 node, 4x NVIDIA H200 (141 GB), NVLink
**Software**: torch 2.11.0+cu128
**Dataset**: synthetic. Uniform random token ids; the packed arms split every sequence into 8 equal-length documents
**Steps**: 4 warmup + 8 timed per cell. Throughput is the mean over the 8 timed steps (CUDA events); peak memory is `torch.cuda.max_memory_allocated()` on rank 0 over the same steps

The single-GPU comparison in
[`docs/architecture/model.md`](../../docs/architecture/model.md#attention-backend)
(from `micro/bench_forward.py`) shows FlexAttention's `BlockMask` beating the
dense `(B, 1, S, S)` mask for packed sequences, by 1.07x at `seq_len` 512 up to
2.42x at 8192. This campaign asks whether that holds once the step also pays
for communication, one parallelism strategy at a time.

## Configuration

Same model as the single-GPU table: 134M parameters (`dim=768`, 12 layers,
12 heads, vocab 32000), bf16 compute, no activation checkpointing. The model is not
compiled; `flex_attention` and `create_block_mask` are compiled on their own,
as they are in every run.

Three arms per cell, at the same per-rank batch:

| Arm | Attention |
|-----|-----------|
| `unpacked` | No `doc_ids`, so SDPA takes its `is_causal` fast path. What packing has to beat. |
| `sdpa+pack` | `doc_ids` with the dense `(B, 1, S, S)` mask, rebuilt in every layer |
| `flex+pack` | `doc_ids` with one FlexAttention `BlockMask` per forward |

| Mode | Mesh | Sequences per step |
|------|------|-------------------:|
| `fsdp` | `dp_shard=4` | 8 |
| `tp` | `tp=4` | 2 |
| `tp_fsdp` | `dp_shard=2`, `tp=2` | 4 |
| `pp` | `pp=4`, GPipe, 4 microbatches | 8 |

The batch is 2 sequences per rank (per microbatch under PP) in every mode, so
`seq_len` is the only variable within a mode. Tokens/s counts each sequence
once: TP ranks all see the same batch, and counting it per rank would make TP
look free. Global batch differs between modes, so compare ratios within a
mode, not throughput across modes.

## Results

flex+pack throughput over sdpa+pack:

| seq_len | FSDP | TP | TP+FSDP | PP |
|--------:|-----:|---:|--------:|---:|
| 512 | 0.90x | 0.98x | 0.97x | 0.93x |
| 1024 | 0.93x | 0.99x | 0.98x | 1.10x |
| 2048 | 0.95x | 0.99x | 0.99x | 1.19x |
| 4096 | 1.16x | 1.20x | 1.20x | 1.47x |
| 8192 | 2.29x | 0.98x | 0.99x | 2.01x |
| 16384 | **3.69x** | 1.57x | 2.36x | **3.06x** |

Throughput in tok/s and rank-0 peak memory in GB, per mode:

| Mode | seq_len | unpacked | sdpa+pack | flex+pack | flex/sdpa | sdpa GB | flex GB |
|------|--------:|---------:|----------:|----------:|----------:|--------:|--------:|
| FSDP | 512 | 115,242 | 113,128 | 101,606 | 0.90x | 0.89 | 0.88 |
| FSDP | 1024 | 217,988 | 194,397 | 181,485 | 0.93x | 1.63 | 1.44 |
| FSDP | 2048 | 388,062 | 382,106 | 363,956 | 0.95x | 2.89 | 2.69 |
| FSDP | 4096 | 768,743 | 621,032 | 720,179 | 1.16x | 5.88 | 4.79 |
| FSDP | 8192 | 872,853 | 474,550 | 1,085,259 | 2.29x | 12.64 | 9.42 |
| FSDP | 16384 | 727,292 | 306,920 | 1,132,213 | 3.69x | 31.42 | 18.54 |
| TP | 512 | 8,250 | 8,191 | 8,042 | 0.98x | 0.72 | 0.60 |
| TP | 1024 | 13,882 | 16,237 | 16,127 | 0.99x | 1.14 | 1.20 |
| TP | 2048 | 33,097 | 32,582 | 32,272 | 0.99x | 2.35 | 2.24 |
| TP | 4096 | 66,059 | 53,244 | 64,036 | 1.20x | 4.70 | 3.24 |
| TP | 8192 | 128,948 | 128,969 | 127,012 | 0.98x | 9.51 | 6.40 |
| TP | 16384 | 260,985 | 159,487 | 251,174 | 1.57x | 25.23 | 11.69 |
| TP+FSDP | 512 | 15,936 | 15,754 | 15,228 | 0.97x | 0.82 | 0.73 |
| TP+FSDP | 1024 | 26,820 | 30,345 | 29,880 | 0.98x | 1.37 | 1.39 |
| TP+FSDP | 2048 | 62,127 | 61,240 | 60,574 | 0.99x | 2.75 | 2.61 |
| TP+FSDP | 4096 | 124,666 | 99,348 | 119,040 | 1.20x | 5.53 | 4.22 |
| TP+FSDP | 8192 | 245,135 | 241,163 | 237,601 | 0.99x | 11.52 | 8.38 |
| TP+FSDP | 16384 | 404,256 | 201,584 | 476,736 | 2.36x | 29.25 | 15.86 |
| PP | 512 | 189,573 | 174,442 | 161,591 | 0.93x | 0.71 | 0.70 |
| PP | 1024 | 299,646 | 273,889 | 301,935 | 1.10x | 1.25 | 1.20 |
| PP | 2048 | 384,072 | 333,717 | 398,541 | 1.19x | 2.40 | 2.20 |
| PP | 4096 | 423,926 | 315,562 | 463,208 | 1.47x | 5.04 | 4.23 |
| PP | 8192 | 411,409 | 242,926 | 489,286 | 2.01x | 11.51 | 8.29 |
| PP | 16384 | 360,805 | 162,452 | 496,615 | 3.06x | 29.63 | 16.42 |

## Analysis

**Where the step is GPU-bound, the single-GPU result holds and grows with
`seq_len`.** Under FSDP at 16384, flex is 3.69x faster than the dense mask and
1.56x faster than not packing at all (1.13M against 727k tok/s), and peak memory
falls from 31.4 to 18.5 GB. PP follows the same curve, to 3.06x at 16384. In
both modes the dense mask is slower than not packing at every length, and at
16384 it reaches only 42-45% of unpacked throughput.

**Short sequences reverse.** From 512 to 2048, an FSDP step takes 35-45 ms in
every arm, whatever the length, so fixed per-step costs dominate. There flex is
5-10% slower than the dense mask (0.90-0.95x), which fits its fixed per-call
cost showing through. PP shows the same at 512 (0.93x). On one GPU the 512 row
is 1.07x, so this reversal is new to the multi-GPU runs.

**The TP modes don't measure attention.** A flex step takes 127-131 ms under TP
and 134-138 ms under TP+FSDP at every `seq_len`, even 16384. The sweep runs
uncompiled, so the floor is most likely eager DTensor dispatch; it was not
profiled. The TP ratios only show where sdpa+pack rises above that floor: at
16384, and at 4096, where sdpa+pack is slower than at 8192 for a reason not
tracked down. The unpacked arm has a similar unexplained bump at 1024 in both
TP modes.

## Reproduction

On one node with 4 GPUs, from the repo root:

```bash
OUT=flex_packing_parallel_results   # a fresh directory: the .txt files are appended to
mkdir -p "$OUT"
for MODE in fsdp tp tp_fsdp pp; do
  MODE=$MODE BENCH_OUT_DIR=$OUT uv run torchrun --nproc_per_node=4 \
    benchmarks/flex_packing_parallel/bench_flex_parallel.py
done
```

`BENCH_SEQ_LENS=512,2048` restricts the lengths. The four modes took about
9 minutes in all on 4x H200.

## Notes

- `results/` holds the script's own output from this run: `flex_parallel_<mode>.txt`
  is the table it prints, and `flex_parallel_<mode>.json` adds the mesh, GPU and
  torch version to every cell.
- A cell that fails on every rank at the same point, such as an OOM that each FSDP
  rank hits at the same allocation, is recorded as NaN and the sweep moves on. A
  failure on one rank inside a step, such as an OOM on one PP stage, leaves the
  other ranks in a collective until the NCCL timeout ends the job. No cell failed
  in this run.
- The script's only changes since `cf3ea9f` are its docstring and that failure
  handling.
