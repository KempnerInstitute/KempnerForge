# Losses

Two loss functions are registered in
[`kempnerforge/training/loss.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/loss.py):
`cross_entropy` and `chunked_cross_entropy`. z-loss is a regularizer
that wraps *either* — not a separate registry entry.

## `cross_entropy`

```toml
[train]
loss_fn = "cross_entropy"
ce_chunk_size = 4096                    # 0 -> auto 4096
```

Mean cross-entropy over flattened `(batch·seq_len, vocab_size)` logits
and `(batch·seq_len,)` labels. `ignore_index = -100` is hard-coded;
labels with `-100` are skipped in the reduction. This is how
[document-packed sequences](../data/index.md) mask out cross-document
attention — tokens at packing boundaries get label `-100` so the loss
doesn't cross document edges.

Returns `0.0` (as an fp32 device tensor) when every label is `-100` —
avoids a NaN from dividing by zero valid tokens.

### Computed in fp32, one chunk at a time

Under `mixed_precision = "bf16"` the output head returns bf16 logits.
`F.cross_entropy` on them would compute the log-softmax and its gradient
in bf16; stacks that run the forward pass under `torch.autocast`
compute cross-entropy in fp32 instead. Casting the whole logit tensor
up front would allocate a full fp32 copy — 8.4 GB for `batch=4`,
`seq=4096`, `vocab=128256` — and autograd would also keep fp32
log-probabilities of the same size for the backward pass.

The loss is instead a custom autograd function that works on
`ce_chunk_size` tokens at a time, in fp32 (fp64 logits stay fp64):

- **Forward**: takes each chunk's log-sum-exp and target logit in fp32
  and keeps only the per-token log-sum-exp (one fp32 value per token).
- **Backward**: recomputes each chunk's softmax from that log-sum-exp,
  subtracts the one-hot target, and writes the gradient in the logits'
  dtype. Rows labeled `-100` get a zero gradient.

On CUDA both per-chunk steps are `torch.compile`d (once per process, as
FlexAttention is) into fused kernels that read the bf16 logits
directly, so not even a chunk is upcast in memory: beyond the logits
and their gradient, the loss holds one value per token and one chunk of
the gradient at a time. On CPU the same functions run eagerly, holding
a few fp32 chunks (`ce_chunk_size × vocab_size × 4` bytes each). It
does not shrink the logits themselves: the model's output head still
materializes the `(batch, seq_len, vocab_size)` logit tensor.

Measured on one H100 for `batch=4`, `seq=4096`, `vocab=128256` (bf16
logits of 3.9 GiB), loss forward and backward:

| | time | memory beyond the logits | loss error vs fp64 |
|---|---|---|---|
| `F.cross_entropy` on bf16 logits (before) | 12.4 ms | 11.7 GiB | 2.1e-3 |
| `F.cross_entropy(logits.float())` | 35.4 ms | 23.5 GiB | 4.3e-8 |
| `cross_entropy`, `ce_chunk_size = 4096` | 8.8 ms | 4.9 GiB | 7.5e-8 |

The first call on each new chunk shape compiles for a few seconds.

`ce_chunk_size = 0` in the config is a sentinel for "use the default
4096" — see
[`build_loss_fn`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/loss.py).

## `chunked_cross_entropy`

```toml
[train]
loss_fn = "chunked_cross_entropy"
ce_chunk_size = 4096                    # 0 -> auto 4096
```

An alias of `cross_entropy`, kept so configs that name it still load.
Both compute the same fp32 chunked loss and read `ce_chunk_size`.

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

The two cross-entropy names compute the same loss, so the choices are
the chunk size and z-loss:

| Situation | `ce_chunk_size` | z-loss |
|-----------|-----------------|--------|
| Default | `0` (4096) | off |
| Large vocab (≥ 128K) and tight memory | smaller, e.g. `1024` | optional |
| Muon / aggressive LRs / long runs | `0` (4096) | `1e-4` |
| PaLM / Gemma-style recipe | `0` (4096) | `1e-4` |

The shipped
[`7b_16gpu_muon.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_muon.toml)
is the reference combination: Muon + chunked cross-entropy +
z-loss — exercises all three paths together.

## See also

- [Training loop § Non-PP step](training-loop.md#non-pp-step-text_step)
  — where `loss_fn(logits, labels)` is called.
- [Configuration § TrainConfig](../configuration/config-sections.md) —
  `loss_fn`, `ce_chunk_size`, `z_loss_weight`.
- [MoE](../moe/index.md) — how the MoE aux loss composes with the
  cross-entropy / z-loss stack.
