# Mixture of Experts

KempnerForge's MoE implementation covers Mixtral-style (softmax top-k,
Switch aux loss) and DeepSeek-V3-style (sigmoid top-k with bias-based
balancing) routing. Everything MoE-specific lives under
[`kempnerforge/model/moe.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/moe.py)
and
[`kempnerforge/model/router.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/router.py);
the distributed mechanics (all-to-all, expert parallelism) live under
{doc}`../distributed/index`.

## What's on this section

- [**Routers**](routers.md) — `softmax_topk` vs `sigmoid_topk`, registry
  selection, shared-expert composition.
- [**Aux loss and balancing**](aux-loss-and-balancing.md) — Switch-style
  aux loss, bias-based EMA balancing, sequence-level aux loss,
  per-expert gradient scaling.
- [**Capacity and dispatch**](capacity-and-dispatch.md) — capacity
  factor, token drop policy, grouped GEMM vs sequential path, packed
  experts.
- [**MoE + FP8**](moe-fp8.md) — which Linears get excluded from Float8
  conversion and why.

## Config at a glance

| Field | Default | What it controls |
|-------|---------|------------------|
| `num_experts` | `0` | `>0` enables MoE layers |
| `moe_top_k` | `2` | Number of experts each token routes to |
| `moe_router` | `"softmax_topk"` | Router type (see [Routers](routers.md)) |
| `moe_frequency` | `1` | Every Nth layer is MoE; others are dense |
| `moe_shared_experts` | `0` | Dense expert applied to every token on top of routed |
| `moe_capacity_factor` | `0.0` | `>0` caps tokens per expert (see [Capacity and dispatch](capacity-and-dispatch.md)) |
| `moe_aux_loss_weight` | `0.01` | Coefficient on `aux_loss` added to main loss |
| `moe_packed_experts` | `false` | Packed weight tensors instead of `ModuleList` |
| `moe_gradient_scale` | `false` | Per-expert output scaling by utilization |
| `moe_sequence_aux_loss_weight` | `0.0` | Sigmoid-only: sequence-level balance penalty |
| `moe_bias_schedule` | `"constant"` | Sigmoid-only: bias update rate schedule |

All fields live in
[`kempnerforge/config/model.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/model.py)
(`ModelConfig`).

## MoE layer placement

`moe_frequency = 1` makes every transformer block an MoE block.
`moe_frequency = 2` alternates dense / MoE layers, matching the
DeepSeek-V2/V3 recipe. The dense layers use a plain `SwiGLUMLP`; only
the MoE blocks hit the routing, dispatch, and aux-loss machinery on
this section.

## Cross-section constraints

- **MoE + TP on experts**: not supported. Expert weights stay replicated
  across the TP group.
- **MoE + PP**: not supported — the config validator raises in
  `JobConfig.__post_init__`.
- **MoE + EP**: supported. `num_experts` must be divisible by
  `ep_world_size`. See
  [Expert parallelism](../distributed/expert-parallelism.md).
- **MoE + FP8**: supported with exclusions. Experts, shared expert, and
  router gate stay bf16 — [MoE + FP8](moe-fp8.md).

## Pages

```{toctree}
:maxdepth: 1

routers
aux-loss-and-balancing
capacity-and-dispatch
moe-fp8
```

## See also

- [Expert parallelism](../distributed/expert-parallelism.md) — the
  all-to-all dispatch/combine used when `ep_world_size > 1`.
- [MoE experiments](../how-to/moe-experiments.md) — end-to-end workflow
  and diagnosis recipes.
- [Validation rules](../configuration/validation-rules.md) — cross-section
  config checks (MoE + PP unsupported, `num_experts % ep == 0`).
- [Benchmarks § MoE Expert Parallelism](../reference/benchmarks.md#moe-expert-parallelism)
  — measured throughput across EP sizes.
