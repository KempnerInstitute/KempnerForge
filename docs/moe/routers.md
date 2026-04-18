# Routers

KempnerForge ships two routers, registered under the `"router"` category
of the component registry:

| Registry key | Class | Style |
|--------------|-------|-------|
| `softmax_topk` | [`SoftmaxTopKRouter`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/router.py) | Mixtral — softmax probabilities + Switch-Transformer aux loss |
| `sigmoid_topk` | [`SigmoidTopKRouter`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/router.py) | DeepSeek-V3 — per-expert sigmoid scores + learnable bias balancing |

Both produce the same output shape — `(weights, indices)` of shape
`(num_tokens, top_k)` — so `MoEMLP` works with either. They differ in
*how* load balancing is maintained and *what* loss signal (if any) gets
added to the training loss.

## `SoftmaxTopKRouter` — Mixtral-style

```python
# kempnerforge/model/router.py — forward
logits = self.gate(x)                                # (T, E)
probs  = F.softmax(logits, dim=-1)                   # (T, E)
weights, indices = torch.topk(probs, k=self.top_k)   # (T, K)
weights = weights / weights.sum(dim=-1, keepdim=True)
```

Each token gets softmax probabilities over all experts; the top `K`
are kept and renormalized. The auxiliary load-balancing loss is
Switch-Transformer's:

```
L_aux = num_experts · Σ_i (f_i · P_i)
```

where `f_i` is the fraction of tokens assigned to expert *i* (hard
counts, detached) and `P_i` is the mean softmax probability for
expert *i* (differentiable through the gate). Gradient flows through
`P_i` and pushes the gate to lower scores for over-utilized experts.

`self.aux_loss` is updated on every forward; the training loop picks
it up via `Transformer.get_moe_aux_loss()` and adds
`moe_aux_loss_weight · aux_loss` to the main loss (see
[Aux loss and balancing](aux-loss-and-balancing.md)).

**Use when**: default, Mixtral-style runs, or when you want the
balancing signal explicit in the loss. The aux loss coefficient is
well-studied territory (Switch set 0.01, Mixtral uses similar).

## `SigmoidTopKRouter` — DeepSeek-V3 style

```python
# kempnerforge/model/router.py — forward
logits = self.gate(x)                                # (T, E)
scores = torch.sigmoid(logits + self.expert_bias)    # (T, E) — bias added
weights, indices = torch.topk(scores, k=self.top_k)
weights = weights / weights.sum(dim=-1, keepdim=True)
```

Three things differ from the softmax router:

1. **Sigmoid, not softmax.** Expert scores are independent per expert —
   no competition for normalization. Raising expert *i*'s score doesn't
   lower expert *j*'s.
2. **`expert_bias` is a learnable `nn.Parameter`** added to the logits
   before sigmoid. It's *not* updated by the optimizer — the training
   loop adjusts it manually via an EMA of per-expert utilization (see
   [Aux loss and balancing § Bias adjustment](aux-loss-and-balancing.md#bias-adjustment)).
3. **No auxiliary loss by default.** `self.aux_loss = 0.0` unless
   `sequence_aux_loss_weight > 0` (opt-in lightweight balance penalty,
   covered on the aux-loss page).

The claim from DeepSeek-V3 is that bias-based balancing reaches
uniform expert utilization without injecting a balance-loss gradient
into the main loss — so the routed experts learn a slightly cleaner
signal. Empirically both approaches train; the sigmoid router is the
one you pick when chasing DeepSeek-V3's recipe specifically.

**Use when**: DeepSeek-V3 reproduction, or long MoE runs where
auxiliary-loss coefficient tuning is annoying and you want the
balancer off the main loss path.

## Side effects

Both routers store two tensors as forward-time side effects:

```python
self.aux_loss:       torch.Tensor  # scalar, picked up by Transformer.get_moe_aux_loss()
self.expert_counts:  torch.Tensor  # (num_experts,), picked up by get_expert_counts()
```

`expert_counts` holds per-expert token counts (hard, detached) for the
most recent forward. It's how hot/cold expert diagnosis works — see
[MoE experiments § Hot/cold expert diagnosis](../how-to/moe-experiments.md).

## Builders

Builders live in the same file and are registered at import time:

```python
# kempnerforge/model/router.py
def _build_softmax_topk(dim, num_experts, top_k): ...
def _build_sigmoid_topk(dim, num_experts, top_k, **kwargs): ...

registry.register("router", "softmax_topk", _build_softmax_topk)
registry.register("router", "sigmoid_topk", _build_sigmoid_topk)
```

Selection is config-driven:

```toml
[model]
num_experts = 8
moe_top_k   = 2
moe_router  = "sigmoid_topk"   # or "softmax_topk" (default)
```

[`build_moe()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/moe.py)
forwards the right kwargs to whichever router is selected (only
`sigmoid_topk` consumes `sequence_aux_loss_weight` and
`bias_schedule`).

## Shared experts

Independent of the router. When `moe_shared_experts > 0`, a
`SwiGLUMLP`-style expert runs on **every token** and its output is
*added* to the routed experts' weighted sum:

```python
# kempnerforge/model/moe.py — MoEMLP.forward
output = routed_experts_forward(...)        # (B, L, D) from top-k routing
if self.shared_expert is not None:
    output = output + self.shared_expert(x_flat)
```

DeepSeek's original motivation: the shared expert absorbs the
"universal" capacity so routed experts can specialize. In practice,
`moe_shared_experts=1` with `sigmoid_topk` reproduces the DeepSeek-V3
pattern; `0` (the default) is Mixtral-style "all experts are routed."

## Picking a router

|  | `softmax_topk` | `sigmoid_topk` |
|---|---|---|
| Balance mechanism | Auxiliary loss adds a term to main loss | EMA-driven bias adjustment; no loss term (by default) |
| Coefficient to tune | `moe_aux_loss_weight` (typical 0.01) | `bias_update_rate` (0.001 default) + optional `sequence_aux_loss_weight` |
| Best for | Mixtral baseline, pre-tuned aux-loss recipes | DeepSeek-V3 reproduction, minimal main-loss interference |
| Aux-loss-free | No — aux loss always non-zero | Yes by default |
| Shared experts | Works, but less common in published recipes | Standard DeepSeek pattern (`moe_shared_experts = 1`) |

## See also

- [Aux loss and balancing](aux-loss-and-balancing.md) — how the two
  routers' balance signals flow into training.
- [Capacity and dispatch](capacity-and-dispatch.md) — what happens to
  the `(weights, indices)` tuple after routing.
- [MoE + FP8](moe-fp8.md) — why `router.gate` is excluded from
  Float8 conversion.
- [Registry](../configuration/registry.md) — how the `"router"`
  category fits with the other 6 registry categories.
- [MoE experiments](../how-to/moe-experiments.md) — end-to-end
  workflow using one or both routers.
