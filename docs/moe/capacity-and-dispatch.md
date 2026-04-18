# Capacity and dispatch

After routing, each token has a `top_k` list of expert indices and
weights. The dispatch layer has to:

1. Optionally cap how many tokens each expert can receive
   (`capacity_factor`).
2. Actually run each expert on its assigned tokens — either in a
   Python loop (sequential) or via one batched `torch._grouped_mm`
   call (grouped GEMM).
3. Weight-combine the expert outputs back into one tensor per token.

Capacity and dispatch both live in
[`kempnerforge/model/moe.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/moe.py).

## Capacity factor

```toml
[model]
moe_capacity_factor = 0.0   # 0 = no cap, >0 = cap per expert
```

With `capacity_factor > 0`, each expert can only accept a bounded
number of tokens per forward:

```python
# kempnerforge/model/moe.py — _apply_capacity
capacity = max(1, ceil(num_tokens * top_k / num_experts * capacity_factor))
```

At `capacity_factor = 1.0`, each expert gets exactly the average load.
At `1.25`, each expert gets 25% headroom above the average — the
canonical Switch Transformer setting. At `0.0` (default), there is
no cap: every token goes to its top-`k` choices regardless of load.

### Drop policy

Tokens routed to an expert past capacity are **dropped from that
expert** — their routing weight is set to 0:

```python
# kempnerforge/model/moe.py — _apply_capacity (trimmed)
for k in range(top_k):
    for e in range(num_experts):
        assigned = (indices[:, k] == e).nonzero(as_tuple=True)[0]
        if assigned.numel() <= capacity:
            continue
        drop = assigned[capacity:]
        weights[drop, k] = 0.0
```

"First `capacity` tokens wins" — tokens are kept in sequence order,
later tokens get dropped. If a token's *only* chosen expert is
overloaded, the token contributes nothing to the MoE output — but
the residual connection still carries it through unchanged:

```python
# kempnerforge/model/transformer.py — TransformerBlock.forward
x = x + self.mlp(self.mlp_norm(x))   # MoE output + residual
```

So a dropped token is not an error — it passes through the layer
as if it skipped the MLP entirely.

### When to use capacity

**With EP**: sometimes necessary. All-to-all buffers are sized for
the *worst-case* token distribution. Without a cap, one rank can
overflow its dispatch buffer when many tokens happen to prefer
experts on that rank. A capacity factor of 1.25 bounds the buffer
size predictably.

**Without EP**: usually leave at 0. All experts live on the same
rank and the sequential/grouped path handles any distribution.
Setting a cap just costs throughput.

**Caveats**: the inner loop is O(num_experts × top_k) Python and
non-vectorized. At `num_experts = 64 × top_k = 8 = 512` Python
iterations per forward, it starts to show on profile. Okay for the
scale MoE runs we've tested; watch for it at 128+ experts.

## Dispatch: two compute paths

The dispatch happens in `MoEMLP._local_forward`. There are two paths
guarded by a feature check:

```python
# kempnerforge/model/moe.py — module-level
_HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")
_GROUPED_MM_DTYPES = {torch.bfloat16, torch.float16}
```

`torch._grouped_mm` exists in recent PyTorch (2.5+) but requires
bf16/fp16 inputs — the meta registration rejects fp32.

### Path A: grouped GEMM (bf16/fp16)

The fast path when available:

```python
# kempnerforge/model/moe.py — grouped_expert_forward (trimmed)
x_padded = x_sorted.new_zeros(num_experts, max_tokens, dim)
# ... pack per-expert token groups into (E, max_tokens, dim) with padding ...
if is_swiglu:
    gate    = torch._grouped_mm(x_padded, gate_w)     # (E, M, H)
    up      = torch._grouped_mm(x_padded, up_w)       # (E, M, H)
    hidden  = F.silu(gate) * up
else:
    hidden  = torch._grouped_mm(x_padded, up_w)
    hidden  = activation(hidden)
out_padded  = torch._grouped_mm(hidden, down_w)       # (E, M, dim)
```

Three batched matmuls for SwiGLU (gate + up + down) or two for
standard MLP (up + down). The `(E, max_tokens, dim)` padding is
wasted compute — if one expert gets 100 tokens and another gets 2,
both tensors sit at `max_tokens=100` padded. Imbalanced routing
amplifies this.

Implementation detail: the padding is also why capacity factor
helps throughput under EP. Bounded max per expert ⇒ bounded
padding ⇒ predictable compute.

### Path B: sequential loop (fp32)

Fallback when grouped GEMM can't run:

```python
# kempnerforge/model/moe.py — MoEMLP._local_forward (trimmed)
for i in range(self.num_experts):
    mask = (indices == i).any(dim=-1)
    if not mask.any():
        continue
    expert_input   = x_flat[mask]
    expert_output  = self.experts[i](expert_input)
    weight_for_i   = (weights * (indices == i).float()).sum(dim=-1)
    output[mask]  += weight_for_i[mask].unsqueeze(-1) * expert_output
```

One kernel launch per expert (or more, per matmul). Fine for
`num_experts ≤ 8`, painful at 32+.

**Rule of thumb**: if you're in fp32, you're not doing real MoE
training. The fallback exists for unit-testing and debugging the
grouped path without a CUDA device. Production runs use bf16
(default) or fp16, both of which hit the grouped path.

## Packed experts

Opt-in via `moe_packed_experts = true`. Replaces the `nn.ModuleList`
of experts with three packed `nn.Parameter` tensors of shape
`(num_experts, dim, hidden)`:

```python
# kempnerforge/model/moe.py — MoEMLP.__init__ (trimmed)
self.up_w   = nn.Parameter(stack(e.up_proj.weight.t()   for e in experts))
self.down_w = nn.Parameter(stack(e.down_proj.weight.t() for e in experts))
self.gate_w = nn.Parameter(stack(e.gate_proj.weight.t() for e in experts))  # SwiGLU
```

Grouped GEMM consumes these directly via
`grouped_expert_forward_packed` — no per-step `torch.stack` over the
`ModuleList`, and FSDP2 wraps three tensors instead of `3 × E`.

Measured impact (see
[Benchmark § MoE Expert Packing](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/moe_packed/moe_packed_benchmark.md)):

| Experts | Unpacked tok/s | Packed tok/s | Delta |
|:-------:|---------------:|-------------:|:-----:|
| 8  | 48,521 | 50,972 | +5.1% |
| 16 | 26,994 | 36,860 | +36.5% |
| 64 | 1,796  | 2,204  | +22.7% |

The speedup grows with `num_experts` — the per-step `torch.stack`
cost (in unpacked mode) scales with E, so eliminating it matters more
as E grows. Peak memory is approximately the same.

**Default is off** because EP integration (slicing packed weights
along the expert axis) is newer than the unpacked path. Flip it on for
MoE runs with ≥ 16 experts.

## Output combine

Both paths end in a weighted scatter-add back to the token axis:

```python
# kempnerforge/model/moe.py — _local_forward (grouped path, trimmed)
expert_out = expert_out * sorted_weights.unsqueeze(-1)
output = torch.zeros(num_tokens, dim, dtype=x_flat.dtype, device=x_flat.device)
output.scatter_add_(0, sorted_token_ids.unsqueeze(-1).expand_as(expert_out), expert_out)
```

Dropped tokens (weight=0) contribute 0 here. Tokens with multiple
experts (top-k > 1) get a sum of weighted outputs — the same token
appears in `top_k` sorted-positions, and each contribution is added.

After `_local_forward` returns, the shared expert (if any) is added:

```python
# kempnerforge/model/moe.py — MoEMLP.forward
if self.shared_expert is not None:
    output = output + self.shared_expert(x_flat)
```

Shared expert runs on *every* token, dense-style. It's additive, not
part of the top-k selection.

## EP branching

When `ep_world_size > 1`, `MoEMLP.forward` bypasses `_local_forward`
and calls `ep_dispatch_and_compute` from
`kempnerforge/distributed/expert_parallel.py`:

```python
# kempnerforge/model/moe.py — MoEMLP.forward (trimmed)
if self.ep_world_size > 1:
    output = ep_dispatch_and_compute(
        x_flat, weights, indices, self,
        self.ep_group, self.local_expert_start,
        self.num_local_experts, self.ep_world_size,
        gradient_scale=self.gradient_scale,
    )
else:
    output = self._local_forward(x_flat, weights, indices)
```

The capacity factor still applies before this branch — the `weights`
tensor handed to `ep_dispatch_and_compute` has dropped-token weights
already zeroed. See
[Expert parallelism § Dispatch / combine flow](../distributed/expert-parallelism.md#dispatch--combine-flow)
for the all-to-all mechanics.

## See also

- [Routers](routers.md) — where `(weights, indices)` come from.
- [Aux loss and balancing](aux-loss-and-balancing.md) — `moe_gradient_scale`
  lives on the dispatch path.
- [Expert parallelism](../distributed/expert-parallelism.md) — the EP
  branch of `MoEMLP.forward`.
- [Benchmarks § MoE Expert Parallelism](../reference/benchmarks.md#moe-expert-parallelism)
  — measured EP throughput with these dispatch paths.
- [MoE experiments § Capacity factor](../how-to/moe-experiments.md)
  — when to set `moe_capacity_factor > 0`.
