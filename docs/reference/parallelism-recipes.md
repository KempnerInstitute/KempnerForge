# Parallelism recipes

Every recipe here corresponds to a TOML preset we've actually run, not
a theoretical configuration. Factors multiply to the given GPU count:
`dp_replicate × dp_shard × tp × pp × cp × ep == world_size` (see
[Validation rules § Parallelism arithmetic](../configuration/validation-rules.md#parallelism-arithmetic)).

Single-node = 4 H200 141GB per node. Intra-node uses NVLink; inter-node
uses InfiniBand.

## 7B (Llama-3)

| GPUs | Nodes | Parallelism | Config | When to pick this |
|-----:|------:|-------------|--------|-------------------|
| 1 | 1 | single GPU | [`debug.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/debug.toml) / [`7b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b.toml) | Smoke test; `7b.toml` fits in 80 GB with full AC |
| 4 | 1 | FSDP=4 | [`7b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b.toml) (`dp_shard=-1`) | Single-node baseline; highest MFU |
| 12 | 3 | TP=4 × FSDP=3 | [`7b_12gpu_tp4.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_12gpu_tp4.toml) | TP within node, FSDP across — uncommon GPU count |
| 16 | 4 | FSDP=16 | [`7b_16gpu_adamw.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_adamw.toml) | Long preemptible runs (`7b_requeue.sh`) |
| 16 | 4 | FSDP=16, FP8 | [`7b_16gpu_fp8.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_fp8.toml) | FP8 compute with bf16 masters; `tp=1` required |
| 16 | 4 | FSDP=16, Muon | [`7b_16gpu_muon.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_muon.toml) | Muon + z-loss + chunked CE integration test |
| 32 | 8 | FSDP=32 | [`7b_32gpu_fsdp.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_32gpu_fsdp.toml) | Simple multi-node baseline |

The 7B model underutilizes 32 H200s (27.6 GB / 140 GB in the benchmark)
— at that scale [13B delivers higher MFU](benchmarks.md#mfu-scaling-dense).

## 13B (Llama-3)

| GPUs | Nodes | Parallelism | Config | When to pick this |
|-----:|------:|-------------|--------|-------------------|
| 8 | 2 | FSDP=8 | (inline override) | Pure FSDP when 13B params fit ≤90 GB/GPU |
| 24 | 6 | TP=4 × FSDP=6 | [`13b_24gpu_validation.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/13b_24gpu_validation.toml) | Full validation: WandB + profiling + eval |
| 32 | 8 | TP=4 × PP=2 × FSDP=4 | [`13b_32gpu_tp4_pp2.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/13b_32gpu_tp4_pp2.toml) | Pipeline parallel recipe — 20 layers per stage |

## 29B (custom)

| GPUs | Nodes | Parallelism | Config | When to pick this |
|-----:|------:|-------------|--------|-------------------|
| 32 | 8 | TP=4 × PP=2 × FSDP=4 | [`29b_32gpu_tp4_pp2.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/29b_32gpu_tp4_pp2.toml) | `dim=6144`, 56 layers — sized to saturate 120 GB/GPU |

## 70B (Llama-3)

| GPUs | Nodes | Parallelism | Config | When to pick this |
|-----:|------:|-------------|--------|-------------------|
| 32 | 8 | TP=4 × FSDP=8 | [`70b_32gpu_tp4.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/70b_32gpu_tp4.toml) | No PP — avoids bubble when FSDP alone fits |
| 32 | 8 | TP=4 × PP=4 × FSDP=2 | [`70b_32gpu_tp4_pp4.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/70b_32gpu_tp4_pp4.toml) | Memory-tight alternative — 20 layers per PP stage |

`70b_32gpu_tp4.toml` is the preferred of the two when memory allows:
the PP version adds a pipeline bubble in exchange for less FSDP
sharding work. See
[Validation rules § Sequence length](../configuration/validation-rules.md#sequence-length)
and
[§ Tensor-parallel head divisibility](../configuration/validation-rules.md#tensor-parallel-head-divisibility)
— both 70B recipes rely on `n_heads=64 % tp=4 == 0` and
`n_kv_heads=8 % tp=4 == 0`.

## MoE (Mixtral-style, 8 experts top-2)

| GPUs | Nodes | Parallelism | Config | When to pick this |
|-----:|------:|-------------|--------|-------------------|
| 1+ | 1+ | FSDP | [`debug_moe.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/debug_moe.toml) | Smoke test; tiny MoE, `moe_frequency=2` |
| 8 | 2 | TP=4 × FSDP=2 | [`moe_8gpu_stress.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_8gpu_stress.toml) | Real dataset, full AC, saturates 2 nodes |
| 24 | 6 | TP=4 × FSDP=6 | [`moe_24gpu.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_24gpu.toml) | `grad_accum=32` for throughput saturation |
| 32 | 8 | TP=4 × EP=2 × FSDP=4 | [`moe_ep_32gpu.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_ep_32gpu.toml) | Expert Parallel — all-to-all across IB |

`PP=1` is mandatory for MoE (routing is data-dependent and resists
pipeline-stage splitting — see the cross-section rules in
[Validation rules](../configuration/validation-rules.md)).
`ep > 1` requires `num_experts > 0` and
`num_experts % ep == 0` — see
[§ Expert parallel](../configuration/validation-rules.md#expert-parallel).

## Choosing a parallelism combination

The order the cross-section validator enforces (see
[Parallelism order](../architecture/parallelism-order.md)) is:

1. **PP** — split layers across stages; set first because it rewrites
   the model into stage submodules.
2. **TP** — shard attention heads and MLP within each PP stage; needs
   `n_heads % tp == 0`, `n_kv_heads % tp == 0`.
3. **CP** — split along the sequence dimension (not wired up yet).
4. **EP** — shard experts across ranks (MoE only).
5. **FSDP (`dp_shard`)** — shard remaining params; the "fill the rest"
   dimension, usually `dp_shard=-1`.
6. **DP replicate (`dp_replicate`)** — multiple full copies, typically
   `1` until a recipe deliberately trades memory for smaller all-reduce
   domains.

Rules of thumb from [the benchmarks](benchmarks.md):

- **FSDP-only wins when it fits.** At 4 GPUs / 7B, pure FSDP beats
  `tp=4` by ~18 MFU points; TP all-gather/reduce-scatter on every
  matmul dwarfs FSDP's once-per-step gradient communication.
- **TP when memory forces it or `n_layers` blocks FSDP.** 70B at 32
  GPUs needs TP — pure FSDP=32 OOMs on activation memory. TP within a
  node (NVLink) is cheap; across nodes (IB) is slow.
- **PP when even TP+FSDP can't fit.** 70B with PP=4 fits with half the
  FSDP shards (`dp_shard=2` vs `dp_shard=8`) at the cost of the PP
  bubble; only pick it when memory requires.
- **EP when expert count outgrows a single rank.** At 8 experts and 32
  GPUs, EP=2 halves per-rank expert storage. At higher expert counts
  (64+, DeepSeekMoE-scale) EP becomes load-bearing.

## See also

- [Available configs](available-configs.md) — the same recipes, indexed
  by filename.
- [Benchmarks](benchmarks.md) — measured throughput and MFU for the
  dense 7B/13B/70B and MoE EP recipes above.
- [Architecture § Parallelism order](../architecture/parallelism-order.md)
  — the invariants the cross-section validator enforces.
- [Validation rules](../configuration/validation-rules.md) — the
  precise checks `JobConfig.validate(world_size)` runs against these
  combinations.
