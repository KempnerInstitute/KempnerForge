# Scaling guide

A walkthrough from 1 to 32 GPUs with Kempner's H200 cluster as the
reference hardware. Every number on this page is from the measured
benchmarks — if you want the raw data, go to
[MFU scaling (dense)](../reference/benchmarks.md#mfu-scaling-dense)
and [MoE Expert Parallelism](../reference/benchmarks.md#moe-expert-parallelism).
This page decides *when* to reach for each dimension and *what*
throughput you should expect.

## The order of parallelism

Before any scaling decision, keep the
[canonical order](../architecture/parallelism-order.md) in mind.
`build_parallel_model` applies the intra-stage parallelisms in this
exact sequence:

```
TP → EP → [FP8] → AC → FSDP (dp_shard)
```

Pipeline parallelism is applied separately in `scripts/train.py`
(split into stages *before* `build_parallel_model` runs on each
stage), and DP replicate / CP are mesh dimensions without a dedicated
apply step — they fall out of how the `DeviceMesh` is constructed.
Getting the order wrong is silent: gradients can be mathematically
wrong but the loss curve looks plausible. Let the validator enforce
the mesh shape, and pick your dimensions knowing what each costs.

## What each dimension costs

| Dimension | Memory win | Comm pattern | When it earns its keep |
|-----------|-----------|--------------|------------------------|
| **FSDP (`dp_shard`)** | params, grads, opt state sharded | once per step (all-gather + reduce-scatter) | Always, unless model doesn't fit even at `ep`+`tp`+`pp` |
| **TP** | shards attention + MLP activations | every matmul (all-gather + reduce-scatter) | When a model is too big for FSDP alone, or `n_layers` is the bottleneck |
| **PP** | shards layers across stages | once per micro-batch (P2P send/recv) | When TP+FSDP still can't fit (70B memory-tight) |
| **EP** (MoE only) | shards expert weights | all-to-all on dispatch + combine | When expert weights dominate param budget; required at 32+ experts |
| **CP** | shards along sequence | n/a (stub, not wired up) | Not yet |
| **DP replicate** | none | all-reduce on grads across replicas | Rarely — shrinks all-reduce domain at memory cost |

Matching principle: **cheap collectives first, expensive last**.
FSDP fires one all-gather per step; TP fires one per matmul. At
equal memory savings, FSDP wins until it can't fit.

## 1 GPU — smoke test

Your first job is to confirm the loop works end-to-end. Use either
`hf_wikitext.toml` (~40M params, HF streaming) or `debug.toml`
(pre-tokenized path). See
[End-to-end training run](end-to-end-training-run.md).

```bash
uv run python scripts/train.py configs/train/hf_wikitext.toml
```

MFU at this scale is *uninformative* — the loop is framework-bound,
not matmul-bound. Don't chase numbers here.

## 4 GPUs (single node) — FSDP is the answer

The benchmark record: **7B dense + FSDP=4 hits 53.8% MFU at 38,983
tok/s**, the highest intra-node efficiency measured. This is the
scale where FSDP shines:

```toml
[distributed]
dp_shard = -1     # auto-resolves to 4
tp       = 1
pp       = 1
```

Why pure FSDP beats TP here: at 4 GPUs on 7B, pure FSDP measures
53.8% MFU vs 34.7% for pure `tp=4` (same GPUs). TP fires
all-gather/reduce-scatter on every matmul; FSDP fires once per step.
See [the benchmark](../reference/benchmarks.md#mfu-scaling-dense).

Reference config:
[`configs/train/7b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b.toml).

## 8 GPUs — the scaling cliff

Going from 4 to 8 GPUs crosses a node boundary for the first time:
intra-node NVLink → inter-node InfiniBand. The 7B model drops from
**93% linear efficiency at 4 GPUs to 53% at 8 GPUs**. Two strategies:

1. **Stay with 7B + FSDP=8** and accept the hit. Simplest; still
   useful for short experiments.
2. **Move to 13B + FSDP=8** — 13B on 8 H200s measures 44.4% MFU.
   Better absolute throughput (35,405 tok/s) because compute density
   per GPU grows.

Past this cliff, the slope is gentle (8 → 32 GPUs degrades only
53% → 47% efficiency). The first inter-node hop is the expensive
one — design around it.

## 16 GPUs — TP enters the picture

At 16 GPUs, pure FSDP works but isn't the efficient pick for larger
models. Two measured options at 13B / 16 GPUs:

- **`TP=4 × FSDP=4`**: 33.7% MFU, 53,814 tok/s. Use when you need
  the memory headroom or plan to scale further to 32.
- **FSDP=16**: usable for 7B, not tested for 13B at this step count.

Why TP here and not at 8 GPUs: TP's per-matmul cost is only worth
paying when the memory savings unlock a larger model or a larger
batch. At 8 GPUs / 13B, FSDP alone fits.

## 32 GPUs — mix TP + PP + FSDP

The measured 32-GPU best for 13B is **`TP=4 × FSDP=8`** at 32.7% MFU
/ 104,309 tok/s. Three patterns are in the recipes:

| Model | Parallelism | MFU | tok/s |
|-------|-------------|:---:|------:|
| 13B | `tp=4 × dp_shard=8` | 32.7% | 104,309 |
| 70B | `tp=4 × dp_shard=8` | 25.4% | (memory fits) |
| 70B | `tp=4 × pp=4 × dp_shard=2` | not measured | alternative for memory-tight setups |

`tp=4` maps to one node (intra-node NVLink → cheap collective). The
remaining 8 ranks go into FSDP. PP only enters when memory forces it:
70B with `TP=4 × FSDP=8` *fits*, so the PP version is only the right
pick when activation memory pushes you over. The PP=4 variant adds a
pipeline bubble in exchange for halved FSDP sharding work.

Full recipe catalog: [Parallelism recipes](../reference/parallelism-recipes.md).

## When each dimension becomes load-bearing

```
1 GPU     → no parallelism (smoke test only)
4 GPUs    → FSDP-only (single node)
8-16 GPUs → FSDP, usually. TP only if model too big or batch tiny.
32 GPUs   → TP=4 (intra-node) + FSDP=N (across). 70B adds PP when memory-tight.
64+ GPUs  → Same pattern, larger FSDP. PP mandatory for 100B+ dense.
MoE       → Add EP when expert weights dominate memory, typically 32+ experts.
```

Quantitatively, from the benchmark:

- At 32 GPUs, **7B uses 27.6 GB / 140 GB** — room to spare, but
  compute-per-GPU is low (26.9% MFU).
- **13B uses 36.3 GB** — the goldilocks zone on H200 (32.7% MFU).
- **70B with TP=4 uses 93.2 GB** — near saturation, still holds 25.4%
  MFU.

Pick the model size to saturate ~60-70% of H200 memory at your target
GPU count. Smaller wastes compute; larger runs the OOM gauntlet.

## Batch-size scaling

The training config's `batch_size × grad_accum_steps × world_size ×
seq_len == global tokens per step`. Two knobs to raise global batch
when scaling:

1. **`batch_size`** (micro-batch per rank). Hard-capped by activation
   memory — raising it eventually OOMs.
2. **`grad_accum_steps`**. Cheap: each micro-batch is a standalone
   forward/backward, gradients accumulate, optimizer steps once every
   `grad_accum_steps`. `maybe_no_sync` skips the DP all-reduce on
   all but the last micro-batch, so communication cost scales with
   optimizer steps, not micro-batches.

Concrete example: `moe_24gpu.toml` uses `grad_accum=32` to saturate
throughput on a 24-GPU run where the per-rank batch size is bounded
by activation memory. See
[`configs/train/moe_24gpu.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_24gpu.toml).

For LR scaling with batch size: follow whatever recipe your reference
paper used (GPT-3 / Llama linear scaling rule up to a batch of ~1M
tokens, then sub-linear). KempnerForge doesn't auto-scale LR.

## Activation checkpointing

`activation_checkpointing = "full"` recomputes every block's forward
during backward — trades compute for memory. Every measured MFU
number on this page is with full AC because it was necessary to fit
the batch size. The cost is roughly 30% more compute per step;
usually you don't have a choice, because without AC you OOM.

`"selective"` checkpoints a subset of blocks — sometimes usable for
smaller models where full AC leaves GPUs underutilized. Measured
marginal on most configs in the benchmark.

## MoE scaling

A separate track. Even at 32 GPUs, the measured MoE run (8 experts,
4B total, 1.8B active, `TP=4 × EP=2 × FSDP=4`) hits only **1.5%
MFU**. This is correct, not broken:

- ~56M active params per GPU on H200 (designed for billions).
- Communication dominates (EP all-to-all + FSDP + TP).
- Reaching dense-MFU territory on MoE at this scale would require
  **~50B total parameters / ~10B active** — a different model size
  class.

Don't compare dense MFU to MoE MFU. Compare tok/s and loss/token for
the same target recipe. Full discussion:
[Benchmarks § Why MFU is 1.5%](../reference/benchmarks.md#why-mfu-is-15).

EP turn-on criteria live in [MoE experiments](moe-experiments.md#when-to-turn-on-ep):

- Expert weights > 50% of total params.
- FSDP alone OOMs.
- InfiniBand available (all-to-all on Ethernet is punishing).

## Common pitfalls

### Going TP too early

Using `tp=4 × dp_shard=1` on a 4-GPU single-node 7B job: **34.7%
MFU** vs **53.8% MFU** for pure FSDP. If memory fits, FSDP wins.
Don't reach for TP until you've proven FSDP can't.

### Forgetting `dp_shard = -1`

The `dp_shard = -1` sentinel auto-resolves to
`world_size / (dp_replicate × tp × pp × cp × ep)`. Setting it to a
fixed number forces a mismatch when you change GPU count. Use `-1`
unless you have a specific reason to pin it.

### Mis-sized activation memory

Symptoms: OOM on step 2 (allocator fragments), or peak memory climbs
slowly over hundreds of steps. Fix 1: enable
`activation_checkpointing = "full"`. Fix 2: set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — load-bearing
for the 32-GPU MoE config per the benchmark.

### Running MoE + PP

`JobConfig.__post_init__` raises this — MoE is data-dependent and
doesn't compose with static pipeline stages. No workaround in
KempnerForge today. See
[MoE experiments § Composition caveats](moe-experiments.md#composition-caveats).

### FP8 + TP

Also rejected in validation —
[`Float8Linear`'s DTensor path is incomplete](../distributed/fp8.md#tp-incompatibility).
You can compose FP8 with FSDP (that's the point) and FP8 with EP
(they're orthogonal), but FP8 + TP will raise at startup.

### Chasing MFU instead of tok/s

MFU is a fraction; tok/s is the rate you actually care about. A 7B
model at 4 GPUs hits 53.8% MFU / 38,983 tok/s. A 13B model at 8 GPUs
hits 44.4% MFU / 35,405 tok/s — *lower* MFU, but 13B is a better
model for the same tok/s. Optimize the loss-per-wall-clock-hour goal,
not the MFU scoreboard.

## See also

- [Benchmarks](../reference/benchmarks.md) — the raw measurements
  this page cites.
- [Parallelism recipes](../reference/parallelism-recipes.md) — every
  shipped TOML with the parallelism combo it demonstrates.
- [Parallelism order](../architecture/parallelism-order.md) — why
  the mesh dimensions must be applied in a specific sequence.
- [Validation rules](../configuration/validation-rules.md) — what
  the cross-section validator rejects and why.
- [End-to-end training run](end-to-end-training-run.md) — the
  single-job walkthrough this guide scales from.
- [MoE experiments](moe-experiments.md) — the MoE scaling workflow.
