# Model

The forward pass block by block, with pointers to the code that implements
each piece. KempnerForge's `Transformer` is a pre-norm Llama-style
decoder: token embedding → N transformer blocks → final RMSNorm →
output head. All components live under
[`kempnerforge/model/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/kempnerforge/model).

## High-level flow

```
tokens (batch, seq_len)
   ↓ TokenEmbedding
hidden (batch, seq_len, dim)
   ↓ for each of n_layers TransformerBlocks:
   ↓   x = x + Attention(RMSNorm(x), cos, sin)
   ↓   x = x + MLP(RMSNorm(x))           (SwiGLUMLP or MoEMLP)
   ↓ final RMSNorm
   ↓ OutputHead (nn.Linear, bias=False)
logits (batch, seq_len, vocab_size)
```

Pre-norm means `LayerNorm` happens before attention and MLP, not after.
Implemented in
[`kempnerforge/model/transformer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/transformer.py).

## Token embedding

`TokenEmbedding` wraps `nn.Embedding(vocab_size, dim)`. It is optional —
`None` on pipeline-parallel middle stages that only receive hidden states.
[`kempnerforge/model/embedding.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/embedding.py).

## RoPE (rotary position embedding)

Positions are injected inside each attention block, not added to the
embedding. `precompute_rope_frequencies(head_dim, max_seq_len, theta)`
returns two tables of shape `(max_seq_len, head_dim // 2)`:

- `cos` — cosines of `position * freq`
- `sin` — sines of the same

`apply_rope(x, cos, sin)` splits `x` along `head_dim` into two halves and
rotates them:

```
[x1, x2] → [x1·cos − x2·sin, x2·cos + x1·sin]
```

The rotation uses real-valued sin/cos (not complex arithmetic) so
`DTensor` metadata survives under `SequenceParallel` — a comment in
[`kempnerforge/model/position.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/position.py)
records that `.float()` stripped the `DTensor` wrapper in an earlier
version.

During generation, `Transformer.forward` slices `cos`/`sin` starting at
`kv_caches[0].seq_len` so incremental tokens get the correct absolute
positions.

## Attention

`Attention` in
[`kempnerforge/model/attention.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/attention.py)
implements grouped-query attention (GQA).

### Projections

Four `nn.Linear(..., bias=False)` projections:

| Name | Shape |
|------|-------|
| `q_proj` | `dim → n_heads * head_dim` |
| `k_proj` | `dim → n_kv_heads * head_dim` |
| `v_proj` | `dim → n_kv_heads * head_dim` |
| `o_proj` | `n_heads * head_dim → dim` |

GQA is configured by the ratio `n_heads / n_kv_heads`:

- `n_kv_heads == n_heads` → multi-head attention (MHA)
- `n_kv_heads == 1` → multi-query attention (MQA)
- anything in between → GQA (`configs/model/llama_7b.toml` uses
  `n_heads=32, n_kv_heads=8`)

After Q/K/V are computed, KV heads are expanded to `n_heads` via
`repeat_interleave(n_rep, dim=1)` so SDPA sees aligned shapes.

### QK-Norm

When `qk_norm=True`, per-head `RMSNorm(head_dim)` is applied to Q and K
before RoPE. Stabilizes attention logits at scale (Gemma, DeepSeek-V3).

### Three attention paths

1. **Packed sequences** (`doc_ids` passed in): a block-diagonal causal
   mask isolates documents within one packed sequence. Built from
   `doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)` intersected with the
   causal triangle, then passed to
   `F.scaled_dot_product_attention(..., attn_mask=...)`.
2. **Standard causal** (training and prefill, no `doc_ids`): SDPA with
   `is_causal=True`.
3. **Single-token decode** (`seq_len == 1` with a KV cache): no mask —
   the query attends to all cached positions. `is_causal=True` here
   would incorrectly restrict attention to only the first key.

A fourth, slower path fires only when `capture_attention_weights=True`:
`_attention_with_weights` computes `softmax(Q·Kᵀ / √d)` manually so
attention weights can be extracted for interpretability. Don't enable it
for training runs.

### KV cache placement

KV cache update happens **after** RoPE, **before** GQA expansion. That
ordering is intentional: cached keys already carry their rotary positions,
and the cache stores `n_kv_heads` tensors, not the expanded `n_heads`
copies.

### SDPA backend

`sdpa_backend` (default `"auto"`) selects flash / mem-efficient / cudnn /
math via a context manager. `"auto"` lets PyTorch pick.

## MLP

Two dense variants, keyed by `activation`:

- **`SwiGLUMLP`** (`activation="silu"`, Llama-style) — three linears:
  `gate_proj + up_proj → SiLU(gate) * up → down_proj`.
- **`StandardMLP`** (`"gelu"` or `"relu"`) — two linears:
  `up_proj → activation → down_proj`.

[`kempnerforge/model/mlp.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/mlp.py).
The helper `build_mlp(dim, hidden_dim, activation)` looks the variant up
in the `"mlp"` registry.

### MoE variant

When `ModelConfig.is_moe` is True, layers where
`(layer_idx + 1) % moe_frequency == 0` swap `SwiGLUMLP` for `MoEMLP` (see
[`kempnerforge/model/moe.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/moe.py)).
`moe_frequency=1` makes every layer MoE; `moe_frequency=2` interleaves
dense and MoE layers, with layer 0 staying dense. See
[`docs/moe/`](../moe/index.md) for the full MoE stack.

## RMSNorm

`RMSNorm(dim, eps=1e-5)` in
[`kempnerforge/model/norm.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/norm.py).
A single learned scale vector (`self.weight`, init 1s), no bias, no mean
subtraction:

```
x.float() * rsqrt(mean(x² , dim=-1, keepdim=True) + eps) * weight → cast back
```

The float32 cast-and-cast-back is deliberate — it prevents the variance
statistic from overflowing in bf16 for long sequences.

`LayerNorm` is also registered under the `"norm"` registry key
`"layernorm"` if you want it; everything else in the codebase assumes
`"rmsnorm"`.

## Output head

`OutputHead` is `nn.Linear(dim, vocab_size, bias=False)`. Optional, like
`TokenEmbedding` — `None` on pipeline-parallel non-final stages.

When `ModelConfig.tie_embeddings=True` (and both layers exist), the head
shares its weight with the embedding via `OutputHead.tie_weights(emb)`
(`self.proj.weight = emb.embedding.weight`).

## TransformerBlock

One block's `forward` is five lines:

```python
x = x + self.attention(self.attention_norm(x), rope_cos, rope_sin,
                       kv_cache=kv_cache, doc_ids=doc_ids)
x = x + self.mlp(self.mlp_norm(x))
return x
```

Pre-norm + residual, both for attention and for MLP. No
`drop_path`/`stochastic_depth`; no learnable scale on the residual branch.

## Transformer assembly

`Transformer` owns:

- `token_embedding` (optional)
- `layers: nn.ModuleDict[str, TransformerBlock]` — keyed by `str(i)`
  instead of `nn.ModuleList` so checkpoint FQNs are stable under DCP
- `norm` — final `RMSNorm`
- `output_head` (optional)
- `_rope_cos`, `_rope_sin` — precomputed frequency tables stored as plain
  attributes (not buffers) so `model.to(bf16)` doesn't cast them to bf16

Extra helpers on `Transformer`:

- `init_weights_and_freqs()` — called after `model.to_empty(device=...)`
  to fill parameter values after meta-device materialization
- `set_moe_step(step, max_steps)` — forwards the current step to every
  `SigmoidTopKRouter` for the adaptive bias schedule
- `get_moe_aux_loss()` — sums the per-layer MoE aux loss; returns 0 for
  dense models
- `get_expert_counts()` — returns `{layer_idx: counts}` for MoE layers;
  empty for dense

The whole class is registered under the `"model"` registry key
`"transformer"`, which is what the training loop asks for.

## Weight initialization

`init_weights(model, config)` in
[`kempnerforge/model/init.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/init.py)
applies the GPT-2 / Llama convention:

- 2-D parameters (`Linear` weights, `Embedding`) get
  `normal(0, config.init_std)` — default `0.02`.
- Parameters whose name ends in `o_proj.weight` or `down_proj.weight` get
  scaled by `1 / sqrt(2 * n_layers)`. Without this, the residual stream's
  variance grows linearly with depth.
- Biases and norm weights are left at their defaults (zeros and ones).
- Meta-device parameters are skipped — `init_weights_and_freqs()` runs
  after materialization.

## Registry keys

Swap pieces without touching the transformer code by registering a new
builder under the right category:

| Category | Existing keys | Builder |
|----------|---------------|---------|
| `"model"` | `"transformer"` | `@registry.register_model(...)` |
| `"mlp"` | `"swiglu"`, `"standard_gelu"`, `"standard_relu"` | `registry.register("mlp", ...)` |
| `"norm"` | `"rmsnorm"`, `"layernorm"` | `registry.register("norm", ...)` |
| `"router"` | `"softmax_topk"`, `"sigmoid_topk"` | `registry.register("router", ...)` |
| `"optimizer"` | `"adamw"`, `"lion"`, `"muon"`, `"schedule_free_adamw"` | `@registry.register_optimizer(...)` |
| `"scheduler"` | `"cosine"`, `"linear"`, `"wsd"`, `"constant"`, `"rex"`, `"none"` | `@registry.register_scheduler(...)` |
| `"loss"` | `"cross_entropy"`, `"chunked_cross_entropy"` | `@registry.register_loss(...)` |

See [Configuration](../configuration/index.md) for the step-by-step
pattern for registering new components.

## Where to read next

- [Parallelism order](parallelism-order.md) — how TP, EP, FP8, AC, and
  FSDP2 are layered on top of this module tree.
- [Data flow](data-flow.md) — the training-loop path that feeds this
  forward pass.
