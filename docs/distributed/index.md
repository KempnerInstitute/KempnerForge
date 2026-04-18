# Distributed

The parallelism families — FSDP2, tensor parallelism, expert
parallelism, pipeline parallelism, FP8 — and the `DeviceMesh` that
composes them.

```{toctree}
:maxdepth: 1

device-mesh
fsdp2
tensor-parallelism
expert-parallelism
pipeline-parallelism
fp8
```

## What lives where

| Page | Covers |
|------|--------|
| [Device mesh](device-mesh.md) | Mesh construction, dimension order, sub-mesh extraction (`get_dp_mesh` / `get_tp_mesh` / `get_pp_mesh`), example meshes from shipped configs |
| [FSDP2](fsdp2.md) | Composable `fully_shard()`, per-block + top-level wrap, mixed-precision policy, `reshard_after_forward`, EP-MoE per-sub-module wrapping, activation checkpointing modes |
| [Tensor parallelism](tensor-parallelism.md) | `ColwiseParallel` / `RowwiseParallel` / `SequenceParallel` plan, forward / post-hooks on embeddings + attention + MLP, meta-device init, head divisibility |
| [Expert parallelism](expert-parallelism.md) | All-to-all dispatch + combine, unused-expert grad kludge, EP + TP + FSDP composition, gradient scaling |
| [Pipeline parallelism](pipeline-parallelism.md) | Layer assignment, `PipelineStageModule`, 1F1B / GPipe / interleaved schedules, per-stage DCP checkpointing |
| [FP8](fp8.md) | `torchao.float8` conversion, filter rules (no experts / router), `enable_fsdp_float8_all_gather`, TP incompatibility |

## Cross-cutting references

- [Architecture § Parallelism order](../architecture/parallelism-order.md)
  — the five-step apply sequence and the reasoning behind it.
- [Configuration § DistributedConfig](../configuration/config-sections.md)
  — the dataclass with `dp_replicate`, `dp_shard`, `tp`, `pp`, `cp`,
  `ep`, `pp_schedule`.
- [Configuration § Validation rules](../configuration/validation-rules.md)
  — arithmetic checks, head divisibility, FP8 + TP, MoE + PP,
  tie-embeddings + PP.
- [Reference § Parallelism recipes](../reference/parallelism-recipes.md)
  — which combinations work at which scales.
- [Reference § Benchmarks](../reference/benchmarks.md) — measured MFU
  and the MoE per-sub-module FSDP fix.

## Not covered here

- **NCCL health / liveness** — the all-reduce heartbeat and NaN
  detection are in [Resilience](../resilience/index.md).
- **Checkpointing under PP / TP / FSDP** — mesh-scoped DCP groups
  and resharding are in [Checkpointing](../checkpointing/index.md).
- **Context parallelism** — `cp` is declared in config and checked in
  validation but has no `apply_context_parallel` yet; see
  [Device mesh § Context parallelism](device-mesh.md#context-parallelism).
