# Expert parallelism

Expert parallelism (EP) partitions MoE experts across an EP process
group — each rank holds `num_experts / ep` experts and tokens are
shuffled between ranks by all-to-all so every token reaches its
assigned expert. With `ep=1` (the default), experts are replicated on
every rank and EP is a no-op.

Entry points:

- [`apply_expert_parallel(model, device_mesh)`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/expert_parallel.py)
  — prunes each `MoEMLP` to the local expert slice and stores EP
  metadata (`ep_group`, `ep_world_size`, `local_expert_start`,
  `num_local_experts`).
- `ep_dispatch_and_compute(x, weights, indices, moe, ...)` — runs
  inside `MoEMLP.forward()` when `ep_world_size > 1` and implements the
  all-to-all dispatch / local compute / all-to-all combine.

## When EP kicks in

```python
# kempnerforge/model/moe.py, MoEMLP.forward
if self.ep_world_size > 1:
    output = ep_dispatch_and_compute(
        x, weights, indices, self,
        self.ep_group, self.local_expert_start,
        self.num_local_experts, self.ep_world_size,
        gradient_scale=self.gradient_scale,
    )
```

With `ep=1`, `ep_world_size` stays at 1 (the default set in
`MoEMLP.__init__`) and the forward path runs experts locally. With
`ep>1`, `apply_expert_parallel` bumps `ep_world_size` to the EP mesh
size and populates the other metadata.

## Dispatch / combine flow

`ep_dispatch_and_compute` is a seven-step sequence:

| # | Step | What it does |
|---|------|--------------|
| 1 | Expand | `(num_tokens, top_k)` → flat `(num_tokens · top_k,)` list of `(token_id, expert_id, weight)` entries |
| 2 | Sort | Stable-sort entries by target EP rank so same-destination tokens are contiguous |
| 3 | Exchange counts | `dist.all_to_all_single` on `send_counts` → every rank learns `recv_counts` |
| 4 | Dispatch | `_AllToAll` sends `x_sorted` to expert-owning ranks; a second all-to-all ships the expert IDs |
| 5 | Local compute | Grouped GEMM over received tokens (sorted by local expert) when `torch._grouped_mm` is available; fallback is per-expert masked forward |
| 6 | Combine | Reverse all-to-all sends processed tokens back to the originating ranks |
| 7 | Weighted sum | `scatter_add_` combines the `top_k` expert outputs per token with their router weights |

The dispatch all-to-all is wrapped in
[`_AllToAll`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/expert_parallel.py)
— a custom `torch.autograd.Function` whose backward is the same
all-to-all with send/recv counts swapped. That's what makes the
forward path differentiable.

## Unused-expert kludge

If a local expert receives **zero tokens** in a step, its parameters
never enter the autograd graph — and FSDP2's reduce-scatter, which
fires only after every param in a unit has accumulated a gradient,
hangs forever.

`ep_dispatch_and_compute` forces an
`AccumulateGrad` hook to fire on each unused expert by adding a
zero-valued sum of its parameters into the output:

```python
for i in range(num_local_experts):
    if tokens_per_expert[i] == 0:
        for p in moe.experts[i].parameters():
            local_output = local_output + p.sum() * 0
```

Similar zero-contributions handle the packed-expert path and the
case where the dispatch all-to-all would otherwise have no gradient
edge back from `local_output` to `received_tokens` (which would
cause the backward all-to-all to be skipped on one side —
positional mismatch in NCCL → deadlock).

## Apply step

[`apply_expert_parallel`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/expert_parallel.py):

```python
for module in model.modules():
    if not isinstance(module, MoEMLP):
        continue
    assert num_experts % ep_size == 0
    start = ep_rank * (num_experts // ep_size)
    end   = start + (num_experts // ep_size)
    if module.packed_experts:
        # replace Parameter with sliced view (can't resize in place)
        module.up_w   = Parameter(module.up_w.data[start:end].clone())
        module.down_w = Parameter(module.down_w.data[start:end].clone())
        ...
    else:
        module.experts = ModuleList([module.experts[i] for i in range(start, end)])
    module.ep_world_size = ep_size
    module.ep_group = ep_group
    module.local_expert_start = start
    module.num_local_experts = num_experts // ep_size
```

The router (`moe.router`) is **not** sharded — every rank keeps the
full router weights so it can make the routing decision locally
before dispatch. Shared experts (`moe.shared_expert`) are also kept
on every rank.

## Composition with other parallelisms

EP runs after TP and before FSDP2 — see
[Parallelism order](../architecture/parallelism-order.md).

- **EP + TP**: TP shards the non-MoE Linears (attention q/k/v/o and
  shared-expert gate/up/down) along the `tp` mesh dim. EP shards the
  routed experts along `ep`. Dense TP layers are untouched by
  `apply_expert_parallel`.
- **EP + FSDP2**: FSDP2 wraps the MoE layer's `attention` and `mlp`
  **separately** (per-sub-module wrapping) rather than the whole
  block — see [FSDP2 § EP-MoE](fsdp2.md#ep-moe-per-sub-module-wrapping).
  Per-block wrapping would cause FSDP2's reduce-scatter to fire
  between the two EP all-to-alls in backward, deadlocking.
- **EP + FP8**: expert Linears are excluded from the Float8 pass
  (`"experts" in fqn → False` in the filter). The grouped GEMM path
  (`torch._grouped_mm`) doesn't go through `Float8Linear.forward`, so
  FP8 applied there is ineffective and adds surprise failures.
  See [FP8 § Exclusion rules](fp8.md#exclusion-rules).

## Config

```toml
[distributed]
ep = 2                           # expert parallelism degree

[model]
num_experts = 8                  # global expert count
moe_top_k   = 2                  # experts per token
moe_shared_experts = 1           # optional always-on expert
moe_packed_experts = false       # grouped GEMM with packed weights (opt-in)
moe_gradient_scale = false       # per-expert gradient normalization (opt-in)
```

`num_experts % ep == 0` is checked at apply time. The parallelism
arithmetic (see [DistributedConfig](../configuration/config-sections.md))
requires `dp_replicate · dp_shard · tp · pp · cp · ep == world_size`.

## Example: `moe_ep_32gpu.toml`

```
dp_shard=4, tp=4, ep=2, pp=1
num_experts=8, moe_top_k=2

mesh: ("dp_shard", "ep", "tp") → (4, 2, 4)
```

- Each EP group (size 2) splits the 8 experts as `experts 0-3` on
  rank 0 and `experts 4-7` on rank 1.
- Within each EP group, TP shards the per-expert Linears along `tp=4`.
- FSDP shards the remaining params across the `dp_shard=4` axis with
  per-sub-module wrapping.

Benchmark and reproducer: [Benchmarks § MoE Expert
Parallelism](../reference/benchmarks.md#moe-expert-parallelism).

## Gradient scaling (optional)

When `moe_gradient_scale = true`, the output of each local expert is
multiplied by `avg_tokens / tokens_for_this_expert` so high-traffic
experts don't dominate the gradient. The scaling happens on
`local_output` before the combine all-to-all, so the adjusted
gradient flows back through the dispatch all-to-all to the router
and expert params correctly. Disabled by default — it changes
gradient magnitudes and should be validated against a baseline run
before flipping on.

## See also

- [MoE overview](../moe/index.md) — architecture, routers, auxiliary
  losses. This distributed/ page is the canonical EP reference; the
  MoE pages link here.
- [FSDP2 § EP-MoE](fsdp2.md#ep-moe-per-sub-module-wrapping) — the
  per-sub-module wrapping pattern.
- [FP8 § Exclusion rules](fp8.md#exclusion-rules) — why experts +
  router + shared_expert are skipped by `apply_float8`.
- [Parallelism order](../architecture/parallelism-order.md) —
  EP's place in the apply sequence.
- [Validation rules § Expert parallel](../configuration/validation-rules.md#expert-parallel) —
  the `num_experts % ep == 0` check.
- [Benchmarks § MoE Expert Parallelism](../reference/benchmarks.md#moe-expert-parallelism) —
  measured EP speedup.
