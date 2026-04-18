# Losses

Two loss functions are registered in
[`kempnerforge/training/loss.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/loss.py):
`cross_entropy` and `chunked_cross_entropy`. z-loss is a regularizer
that wraps *either* — not a separate registry entry.

## `cross_entropy`

```toml
[train]
loss_fn = "cross_entropy"
```

Direct call to `F.cross_entropy` over flattened
`(batch·seq_len, vocab_size)` logits and `(batch·seq_len,)` labels.
`ignore_index = -100` is hard-coded; labels with `-100` are skipped in
the reduction. This is how
[document-packed sequences](../data/index.md) mask out cross-document
attention — tokens at packing boundaries get label `-100` so the loss
doesn't cross document edges.

Returns `0.0` (as a device tensor) when every label is `-100` — avoids
a NaN from dividing by zero valid tokens.

## `chunked_cross_entropy`

```toml
[train]
loss_fn = "chunked_cross_entropy"
ce_chunk_size = 4096                    # 0 -> auto 4096
```

Identical output to `cross_entropy`, but computed in chunks along the
flattened token dimension. Use this when the full logit tensor would
be large enough to matter — a 7B Llama-3 with `batch=4`, `seq=4096`,
`vocab=128000` materializes an **8 GB** float32 cross-entropy
intermediate if computed end-to-end.

Mechanics:

- The loop iterates `for i in range(0, num_tokens, chunk_size)` and
  accumulates per-chunk loss with `reduction="sum"`.
- At the end, divides by the number of non-ignored tokens across all
  chunks to recover the mean.
- `ignore_index = -100` is threaded through per chunk.

Important: chunking saves memory **inside the loss kernel**, not in the
output projection. The `(batch, seq_len, vocab_size)` logit tensor
that the model produces is still materialized — `chunked_cross_entropy`
only avoids the second full-size tensor that `F.cross_entropy`
internally allocates for the softmax.

`ce_chunk_size = 0` in the config is a sentinel for "use the default
4096" — see
[`build_loss_fn`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/loss.py).

## z-loss (logit magnitude regularizer)

```toml
[train]
loss_fn = "cross_entropy"               # or chunked_cross_entropy
z_loss_weight = 1e-4                    # 0 -> disabled
```

z-loss penalizes large log-sum-exp values — equivalent to penalizing
the softmax normalizer from running off. It is a PaLM / Gemma trick for
keeping logits in a numerically sane range without clamping.

Formula:

```
z_loss = z_loss_weight * mean(logsumexp(logits)^2)
```

How it wires in: `build_loss_fn` wraps the base loss with a closure
that adds `z_loss(logits, z_weight)` to the returned loss. The wrap
happens once at setup; the per-step overhead is one `logsumexp` over
the logit tensor.

Recommended weights from the literature:

| Value | Source |
|-------|--------|
| `1e-4` | PaLM |
| `1e-4` – `2e-4` | Gemma |
| `0.0` | off; default |

`z_loss_weight = 0.0` makes the wrap a no-op — the closure still runs
but `z_loss()` returns `0.0` immediately without computing
`logsumexp`.

## MoE auxiliary loss is separate

The MoE load-balancing loss is **not** part of this pipeline. It
lives on the model (`model.get_moe_aux_loss()`) and is added to the
base loss in the training loop:

```python
loss = loss_fn(logits, labels)
if mc.is_moe:
    loss = loss + mc.moe_aux_loss_weight * model.get_moe_aux_loss()
```

See [MoE](../moe/index.md) for how `get_moe_aux_loss()` sums the
per-layer router aux losses.

## Picking one

| Situation | Loss | z-loss |
|-----------|------|--------|
| Small model, tight compile, small vocab | `cross_entropy` | off |
| 7B+ with large vocab (≥ 50K) | `chunked_cross_entropy` | optional |
| Muon / aggressive LRs / long runs | either | `1e-4` |
| PaLM / Gemma-style recipe | `chunked_cross_entropy` | `1e-4` |

The shipped
[`7b_16gpu_muon.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_muon.toml)
is the reference combination: Muon + chunked cross-entropy +
z-loss — exercises all three paths together.

## See also

- [Training loop § Non-PP step](training-loop.md#non-pp-step-pp_enabled-is-false)
  — where `loss_fn(logits, labels)` is called.
- [Configuration § TrainConfig](../configuration/config-sections.md) —
  `loss_fn`, `ce_chunk_size`, `z_loss_weight`.
- [MoE](../moe/index.md) — how the MoE aux loss composes with the
  cross-entropy / z-loss stack.
