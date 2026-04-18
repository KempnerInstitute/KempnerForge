# MoE + FP8

FP8 training (via `torchao.float8`) converts dense `nn.Linear` modules
to `Float8Linear`. Three classes of Linear in an MoE model get
**excluded** from that conversion:

1. Routed experts (`experts.*`)
2. Shared expert (`shared_expert.*`)
3. Router gate (`router.gate`)

Everything else — attention projections, output head, dense-layer
MLPs — converts normally.

## Filter rule

From
[`apply_float8`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py):

```python
# kempnerforge/distributed/parallel.py — apply_float8._filter_fn
def _filter_fn(module, fqn):
    if "experts" in fqn or "shared_expert" in fqn:
        return False
    return "router" not in fqn
```

`convert_to_float8_training` walks the model, asks the filter for
each `Linear`, and only replaces the module when the filter returns
True.

## Why experts are excluded

The grouped GEMM path uses `torch._grouped_mm`, which is a distinct
kernel from `torch._scaled_mm` (what `Float8Linear.forward` calls
into). Even if an expert's `nn.Linear` *were* wrapped in `Float8Linear`,
the wrap would be bypassed:

- The forward path stacks per-expert weights into a single
  `(E, dim, hidden)` tensor (or reads them from pre-packed
  parameters) and calls `torch._grouped_mm` directly.
- `Float8Linear.forward` is never called on the expert weights.

The result would be FP8 *parameter storage* but bf16 *compute* — the
worst of both worlds (lost precision at storage, no matmul speedup).

See [Capacity and dispatch § Path A](capacity-and-dispatch.md#path-a-grouped-gemm-bf16fp16)
for the grouped GEMM call site.

## Why the router is excluded

Two reasons:

1. **Tiny output dim.** Router gate is `Linear(dim, num_experts)`.
   `num_experts` is typically 8-64 — not divisible by 16, which
   `torch._scaled_mm` requires for its fast path. Fallback paths
   give no speedup.
2. **Not compute-bound.** Routing is essentially a decision — the
   gate matmul is a tiny fraction of total FLOPs. FP8 quantization
   error on the routing decision could measurably perturb which
   experts get picked, which is a stability risk.

## Why the shared expert is excluded

Same reason as routed experts. The shared expert is a `SwiGLUMLP` (or
`StandardMLP`) with identical weight shapes to a routed expert — it
runs through the same grouped-GEMM path when that path is active (on
the dense forward side it's a plain `Linear` call, but convention
keeps it FP8-excluded for consistency).

In practice the shared expert runs on every token (not dispatched),
so it *could* use `Float8Linear`. The decision to exclude it is
conservative: keep the expert codepaths uniform, avoid a surprise if
the shared expert is later routed through the grouped path.

## What stays FP8

Everything not caught by the three filter rules:

| Module | Converted | FQN pattern |
|--------|:---------:|-------------|
| Attention Q/K/V/O | yes | `*.attention.{q_proj,k_proj,v_proj,o_proj}` |
| Dense MLP gate/up/down | yes | `*.mlp.{gate_proj,up_proj,down_proj}` (dense layers only) |
| Output head | yes | `output_head.proj` |
| Router gate | no | `*.router.gate` |
| Routed experts | no | `*.experts.*.{gate,up,down}_proj` |
| Shared expert | no | `*.shared_expert.*` |
| Embeddings | no | not `Linear` |
| RMSNorm | no | not `Linear` |

With `moe_frequency = 2` (alternating dense / MoE layers), the dense
layers' MLPs get FP8 and the MoE layers' MLPs don't — which is fine,
because the MoE layers' compute is dominated by grouped GEMM on the
expert weights anyway.

## Memory and throughput

Excluding expert Linears means FP8 gives smaller gains for MoE than
for dense training. A 4B-active MoE where half the FLOPs are in
grouped-GEMM experts sees FP8 speedup only on the remaining half
(attention + shared expert + output head + dense-layer MLPs).

For the 7B dense Llama reference (see
[Benchmarks § MFU scaling](../reference/benchmarks.md#mfu-scaling-dense)),
FP8 provides a measurable throughput lift at 16 GPUs. For MoE runs,
the same config would see a smaller lift because the expert portion
of compute runs at bf16 regardless.

## Config

One switch:

```toml
[train]
mixed_precision = "fp8"   # flips the conversion on
```

No separate MoE knob. The expert/router exclusions are hardcoded in
`_filter_fn`. If you want to experiment with FP8 on a specific
expert path — for example, trying FP8 on the shared expert — you'd
edit `apply_float8` directly.

## FP8 + EP + TP: what actually composes

Three separate constraints:

- **FP8 + TP**: Not supported.
  [`JobConfig.__post_init__`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/job.py)
  raises when `train.mixed_precision = "fp8"` and `distributed.tp > 1`.
  Reason: `Float8Linear`'s weight wrapper calls `aten.is_pinned` on
  DTensor, which has no sharding strategy yet. See
  [FP8 § TP incompatibility](../distributed/fp8.md#tp-incompatibility).
- **FP8 + EP**: Fine. Experts are excluded from FP8, so EP — which
  operates on experts — is orthogonal to the conversion.
- **FP8 + MoE without EP**: Fine. Non-expert Linears convert, expert
  Linears stay bf16, grouped GEMM continues to work.

## See also

- [FP8](../distributed/fp8.md) — the canonical FP8 reference (full
  conversion details, `enable_fsdp_float8_all_gather`, master weights,
  hardware requirements).
- [Expert parallelism § EP + FP8](../distributed/expert-parallelism.md#composition-with-other-parallelisms)
  — EP compatibility notes with FP8.
- [Capacity and dispatch § Path A](capacity-and-dispatch.md#path-a-grouped-gemm-bf16fp16)
  — the grouped-GEMM path that bypasses `Float8Linear`.
- [Parallelism order § Float8 before AC and FSDP](../architecture/parallelism-order.md)
  — where FP8 conversion sits in the apply sequence.
- [Validation rules](../configuration/validation-rules.md) — the
  FP8 + TP config check.
