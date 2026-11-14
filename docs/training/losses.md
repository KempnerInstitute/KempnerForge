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

When every label is `-100` it returns an exact fp32 zero that stays on
the autograd graph of the logits, so there is no NaN from dividing by
zero valid tokens and backward and FSDP's collectives still run on
that rank.

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

## fp32 cross-entropy

```toml
[train]
loss_fn = "cross_entropy"               # or chunked_cross_entropy
fp32_cross_entropy = true               # default false
```

Under `mixed_precision = "bf16"` the output head returns bf16 logits,
and both losses above compute the log-softmax, the loss and its
gradient in bf16. Stacks that run the forward pass under
`torch.autocast` compute cross-entropy in fp32 instead.
`fp32_cross_entropy = true` does the same for either loss, without the
fp32 copy of the logits (8.4 GB for `batch=4`, `seq=4096`,
`vocab=128256`) that upcasting them would allocate, and without the
fp32 log-probabilities autograd would keep for backward. It changes
training numerics slightly. With the default `false`, both losses are
unchanged.

Both losses then run one custom autograd function in fp32 (fp64 logits
stay fp64):

- **Forward**: takes the log-sum-exp and target logit in fp32 and keeps
  only the per-token log-sum-exp (one fp32 value per token).
- **Backward**: recomputes the softmax from that log-sum-exp, subtracts
  the one-hot target, and writes the gradient in the logits' dtype.
  Rows labeled `-100` get a zero gradient.

`cross_entropy` does this over the whole batch at once;
`chunked_cross_entropy` does it `ce_chunk_size` tokens at a time. On
CUDA both steps are `torch.compile`d (once per process, as
FlexAttention is) into fused kernels that read the bf16 logits
directly. On CPU the same functions run eagerly.

Measured on one H100 for `batch=4`, `seq=4096`, `vocab=128256` (bf16
logits of 3.9 GiB), loss forward and backward:

| | time | memory beyond the logits | loss error vs fp64 |
|---|---|---|---|
| `cross_entropy` (bf16) | 12.4 ms | 11.7 GiB | 2.1e-3 |
| `F.cross_entropy(logits.float())` | 35.4 ms | 23.5 GiB | 4.3e-8 |
| `chunked_cross_entropy`, `fp32_cross_entropy = true`, `ce_chunk_size = 4096` | 8.8 ms | 4.9 GiB | 7.5e-8 |

The first call on each new shape compiles for a few seconds.

Limits:

- If `torch.compile` is disabled or falls back to eager on CUDA,
  whole-batch `cross_entropy` holds several fp32 tensors the size of
  the logits at once in its backward; `chunked_cross_entropy` bounds
  them to one chunk. Use `chunked_cross_entropy` if compile is off.
- The backward cannot itself be differentiated: with `create_graph=True`
  the gradient comes back detached, so a second backward through it
  raises.
- The flag is rejected for any `loss_fn` other than these two.

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
that adds `z_loss(logits, z_weight, chunk_size)` to the returned loss.
The wrap happens once at setup. `chunk_size` is `ce_chunk_size` for
`chunked_cross_entropy`; for `cross_entropy` every token is in one
chunk. `z_loss` takes each token's log-sum-exp in float32 one chunk at
a time and keeps only that per-token value for backward, which
recomputes each chunk's softmax instead of keeping a float32 copy of
the logits. Its memory cost is a gradient the size of the logits plus
float32 temporaries for one chunk (the whole batch with
`cross_entropy`). As with `fp32_cross_entropy`, the backward cannot
itself be differentiated.

Recommended weights from the literature:

| Value | Source |
|-------|--------|
| `1e-4` | PaLM |
| `1e-4` – `2e-4` | Gemma |
| `0.0` | off; default |

With `z_loss_weight = 0.0`, `build_loss_fn` returns the base loss
without the wrap; called directly, `z_loss(logits, 0.0)` returns `0.0`
without computing `logsumexp`.

Logits with no rows (the head's output for a packed VLM micro-batch
without text) give an exact fp32 zero that stays on the autograd graph
of the logits at any `chunk_size`, so there is no NaN from averaging
over zero tokens.

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
router aux losses: one per MoE layer, and on a MoT model one per
modality on each MoE layer.

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

- [Training loop § Non-PP step](training-loop.md#non-pp-step-text_step)
  — where `loss_fn(logits, labels)` is called.
- [Configuration § TrainConfig](../configuration/config-sections.md) —
  `loss_fn`, `ce_chunk_size`, `fp32_cross_entropy`, `z_loss_weight`.
- [MoE](../moe/index.md) — how the MoE aux loss composes with the
  cross-entropy / z-loss stack.
