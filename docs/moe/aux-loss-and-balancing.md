# Aux loss and balancing

MoE training collapses when a handful of experts eat all the tokens
and the rest starve — those starved experts are dead parameters.
KempnerForge has three mechanisms to prevent that, composable depending
on the router:

| Mechanism | Router it applies to | On by default? | Config |
|-----------|-----------------------|:--------------:|--------|
| Switch-style auxiliary loss | `softmax_topk` | always | `moe_aux_loss_weight` |
| Bias-based EMA balancing | `sigmoid_topk` | always | `bias_update_rate` (hardcoded 0.001) + `moe_bias_schedule` |
| Sequence-level aux loss | `sigmoid_topk` | opt-in | `moe_sequence_aux_loss_weight` |
| Per-expert gradient scaling | both | opt-in | `moe_gradient_scale` |

## Training-loop integration

Regardless of which router is active, the aux loss flows the same way:

```python
# scripts/train.py
main_loss = cross_entropy(logits, labels)
aux_loss  = model.get_moe_aux_loss()
total     = main_loss + mc.moe_aux_loss_weight * aux_loss
```

`Transformer.get_moe_aux_loss()` walks every `MoEMLP` layer and sums
`layer.mlp.aux_loss`. The coefficient `moe_aux_loss_weight`
(default 0.01) is the knob that scales the balance pressure.

For `sigmoid_topk` with no sequence aux loss, each layer's
`aux_loss` is the scalar 0.0 — so `total = main_loss` and the
coefficient has no effect. For `softmax_topk`, every step contributes
a non-zero Switch-style penalty.

## Softmax router — Switch-Transformer aux loss

```python
# kempnerforge/model/router.py — SoftmaxTopKRouter.forward (trimmed)
one_hot         = F.one_hot(indices, num_classes=self.num_experts).float()
tokens_per_exp  = one_hot.sum(dim=(0, 1))                       # (E,)
f               = tokens_per_exp / (num_tokens * self.top_k)    # hard counts
p               = probs.mean(dim=0)                             # soft probs
self.aux_loss   = self.num_experts * (f * p).sum()
```

The quantity `num_experts · Σ_i f_i · p_i` is minimized when every
expert gets `1/num_experts` of the tokens. `f_i` is detached (hard
assignment) so no gradient flows through it; gradient flows only
through `p_i`, lowering the gate logits for over-utilized experts.

**Typical coefficient**: `moe_aux_loss_weight = 0.01`. Higher values
(0.1) over-regularize — router gets uniform but can't specialize.
Lower (0.001) under-regularizes — collapse risk on long runs.

## Sigmoid router — bias-based balancing

The `sigmoid_topk` router holds a learnable `expert_bias` parameter
that's *added* to logits before sigmoid, and a buffer `expert_ema`
holding running utilization:

```python
# kempnerforge/model/router.py — SigmoidTopKRouter.__init__
self.expert_bias = nn.Parameter(torch.zeros(num_experts))
self.register_buffer("expert_ema", torch.ones(num_experts) / num_experts)
```

Inside forward (training only):

```python
# kempnerforge/model/router.py — SigmoidTopKRouter.forward (trimmed)
utilization = tokens_per_expert / (num_tokens * self.top_k + 1e-8)
self.expert_ema.lerp_(utilization, 1.0 - self.ema_decay)   # ema_decay = 0.99

with torch.no_grad():
    target = 1.0 / self.num_experts
    self.expert_bias.add_(effective_rate * (target - self.expert_ema))
```

Two moving parts:

- **`expert_ema`** — exponential moving average of per-expert fraction
  of tokens. `ema_decay = 0.99` means each step contributes 1% of the
  current batch to the smoothed estimate.
- **`expert_bias`** — incremented by `effective_rate · (target − ema)`.
  Under-utilized experts (`ema < target`) get a positive nudge →
  higher logits → more tokens next step. Over-utilized experts get a
  negative nudge.

The bias update is inside `torch.no_grad()` — the optimizer **does not
touch it**, even though it's an `nn.Parameter`. It's a parameter only
for DCP checkpointing and to keep it on the right device.

### Bias adjustment

Default `bias_update_rate = 0.001`. Not exposed in config — hardcoded
in `SigmoidTopKRouter.__init__`. The effective rate used per step is
modulated by the schedule:

```python
# kempnerforge/model/router.py — _effective_bias_rate
if self.bias_schedule == "constant":
    return rate
progress = self._step / self._max_steps
if self.bias_schedule == "cosine_decay":
    return rate * 0.5 * (1 + cos(pi * progress))
if self.bias_schedule == "linear_warmup":
    return rate * min(1.0, progress * 10.0)   # ramp over first 10% of training
```

| `moe_bias_schedule` | Effective-rate curve | Typical use |
|---------------------|----------------------|-------------|
| `"constant"` (default) | flat at `rate` | standard DeepSeek-V3 |
| `"cosine_decay"` | decays `rate → 0` over the run | late-training stability after balance is achieved |
| `"linear_warmup"` | `0 → rate` over first 10% | cold-start stabilization when bias would otherwise overshoot |

The schedule needs the current training step, which is pushed in each
step via `model.set_moe_step(step, max_steps)` in
`scripts/train.py`. `Transformer.set_moe_step` walks only sigmoid
routers:

```python
# kempnerforge/model/transformer.py — set_moe_step
for layer in self.layers.values():
    if isinstance(layer.mlp, MoEMLP) and isinstance(layer.mlp.router, SigmoidTopKRouter):
        layer.mlp.router.set_step(step, max_steps)
```

## Sequence-level aux loss (sigmoid only)

Opt-in. When `moe_sequence_aux_loss_weight > 0`, the sigmoid router
adds a Switch-style penalty on top of the bias mechanism:

```python
# kempnerforge/model/router.py — SigmoidTopKRouter.forward (trimmed)
if self.sequence_aux_loss_weight > 0:
    f = tokens_per_expert.detach() / (num_tokens * self.top_k + 1e-8)
    p = scores.mean(dim=0)                    # differentiable through sigmoid
    balance_loss = self.num_experts * (f * p).sum()
    self.aux_loss = self.sequence_aux_loss_weight * balance_loss
```

Same formula as the softmax router's Switch-style loss, but with `p`
as mean sigmoid *scores* rather than softmax probabilities. Gradient
flows through `p` only.

**When to use**: long runs (100k+ steps) where bias balancing alone
sometimes drifts into degenerate modes. Recommended weight is much
smaller than the softmax router's `moe_aux_loss_weight` — typical
`0.001` (one tenth of 0.01) or lower, because the bias mechanism is
already doing most of the balancing work.

Leave `moe_aux_loss_weight` at its default 0.01 — it's the
multiplier on `aux_loss` downstream, so the effective weight on the
balance loss is `moe_aux_loss_weight × sequence_aux_loss_weight`.

## Per-expert gradient scaling

Opt-in via `moe_gradient_scale = true`. Normalizes each expert's
output by its utilization so high-traffic experts don't dominate
training:

```python
# kempnerforge/model/moe.py — MoEMLP._local_forward (grouped path, trimmed)
if self.gradient_scale and self.training:
    avg_tokens = total_assignments / max(self.num_experts, 1)
    offset = 0
    for count in tokens_per_expert:
        if count > 0:
            scale = avg_tokens / count
            expert_out[offset:offset + count] *= scale
        offset += count
```

An expert that receives `count` tokens has its output multiplied by
`avg_tokens / count`. Hot experts get `scale < 1` (gradients shrunk);
cold experts get `scale > 1` (gradients amplified). The scale passes
through the `expert_out * sorted_weights` scatter-add, so the
routing weights still apply on top of it.

The same logic runs on the EP path (`ep_dispatch_and_compute`) using
per-local-expert receive counts — see
[EP § Gradient scaling](../distributed/expert-parallelism.md#gradient-scaling-optional).

**Reference**: DeepSeek-V3 §3.2. Empirically small effect on short
runs, visible on long runs where the router has converged to a
specific usage distribution.

**When to use**: long MoE training where expert utilization is
known-imbalanced and you want gradient-magnitude parity across
experts. Disabled by default because it changes gradient magnitudes
and should be validated against a baseline.

## Diagnosis

`Transformer.get_expert_counts()` returns
`dict[layer_idx → tensor(num_experts,)]` with the per-expert token
count from the most recent forward. Interpretation rule-of-thumb:

- **Balanced**: counts within ~20% of `num_tokens · top_k / num_experts`.
- **Hot/cold**: one expert ≥ 2× the mean → router is specializing.
  Fine if intentional, bad if unintentional.
- **Dead expert**: count = 0 for many consecutive steps → that
  expert isn't learning. Root cause is usually a too-small aux-loss
  coefficient or a cold-start bias issue.

See [MoE experiments § Hot/cold expert diagnosis](../how-to/moe-experiments.md).

## See also

- [Routers](routers.md) — where `aux_loss` is computed per router.
- [Capacity and dispatch](capacity-and-dispatch.md) — what happens
  when `capacity_factor` drops a token (the router still sees it,
  but its weight becomes 0).
- [Expert parallelism § Gradient scaling](../distributed/expert-parallelism.md#gradient-scaling-optional)
  — per-expert gradient scaling on the EP path.
- [MoE experiments](../how-to/moe-experiments.md) — recommended aux
  loss weights per training scale.
- [Validation rules § MoE + PP](../configuration/validation-rules.md)
  — cross-section config checks (MoE + PP is unsupported).
