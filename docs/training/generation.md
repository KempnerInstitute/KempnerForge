# Generation

Autoregressive decoding for research and debug, implemented in
[`kempnerforge/model/generate.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/generate.py).
Single-GPU only — not a production serving path.

## `sample`

```python
from kempnerforge.model.generate import sample

next_token = sample(logits, temperature=1.0, top_k=0, top_p=1.0)
# (batch, vocab_size) -> (batch,)
```

Applies, in order:

1. **`temperature == 0`** → `logits.argmax(dim=-1)` (greedy; short-circuit,
   none of the filters below run).
2. **Temperature scaling** → `logits / temperature`.
3. **Top-k filtering** → keep the `top_k` largest values per batch row,
   mask the rest with `-inf`. `top_k=0` disables.
4. **Top-p (nucleus) filtering** → sort descending, keep the smallest
   prefix whose probabilities sum to < `top_p`. `top_p=1.0` disables.
5. **Sample** → `torch.multinomial(probs, num_samples=1).squeeze(-1)`.

The order matters: top-k and top-p both operate on temperature-scaled
logits, not raw ones.

You can call `sample()` standalone for custom decode loops — it's the
one-shot primitive behind `generate()`.

## `generate`

```python
from kempnerforge.model.generate import generate

@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: torch.Tensor,        # (batch, prompt_len)
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:                      # (batch, total_len)
```

What it does:

1. Saves `was_training`, switches `model.eval()`.
2. Validates `prompt_len + max_new_tokens ≤ model.config.max_seq_len`
   — raises `ValueError` otherwise.
3. Allocates one `KVCache` per transformer layer, sized for the full
   sequence (`batch_size, total_len, n_kv_heads, head_dim`), matching
   the model's parameter dtype.
4. **Prefill**: forwards the whole prompt through the model with the
   KV caches; grabs the last-position logits.
5. **Decode loop**: for `max_new_tokens` iterations:
   - `sample()` to pick the next token.
   - If `eos_token_id` is set, OR it into a per-row `done` mask; break
     early if `done.all()`.
   - Forward the single sampled token (`next_token.unsqueeze(1)`) with
     the same KV caches to get the next logits.
6. Restores `model.train()` if `was_training`.
7. Returns `torch.cat([prompt_tokens, generated], dim=1)`.

## KV cache

Imported from
[`kempnerforge.model.attention`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/attention.py).
`generate()` allocates one cache per layer and passes them through the
model via `model(..., kv_caches=kv_caches)`.

Size: `batch_size × total_len × n_kv_heads × head_dim × 2 (K+V) × dtype_bytes`
per layer, times `n_layers`. For a 7B Llama-3 at `batch=1, total_len=8192,
bf16` that's ~2 GB of KV cache — easily fits on a single H200 but
watch the budget for larger batches or longer contexts.

## Stop criteria

Two:

- **`max_new_tokens`** — hard limit, always enforced.
- **`eos_token_id`** — optional. When provided, generation stops early
  only when **every** row in the batch has emitted EOS. A partial-batch
  early-stop would require padding, which `generate()` doesn't
  implement. If that matters, decode batches of size 1 or post-process
  the output yourself.

## Standalone CLI: `scripts/generate.py`

```bash
uv run python scripts/generate.py configs/train/7b.toml \
    --checkpoint.load_path /path/to/checkpoint \
    --prompt "The capital of France is" \
    --max_tokens 64 \
    --temperature 0.7 \
    --top_p 0.9
```

Interactive REPL:

```bash
uv run python scripts/generate.py configs/train/7b.toml \
    --checkpoint.load_path /path/to/checkpoint \
    --interactive
```

The CLI loads the model via DCP (not through the full distributed
stack — no FSDP2, no `init_distributed`), tokenizes the prompt with
the HF tokenizer specified in `data.tokenizer_path`, calls
`generate()`, decodes with `skip_special_tokens=True`, and prints the
generated suffix.

Args (beyond config / overrides):

| Flag | Default | Meaning |
|------|---------|---------|
| `--prompt` | (required unless `--interactive`) | Prompt string |
| `--max_tokens` | `128` | Max new tokens |
| `--temperature` | `1.0` | Sampling temperature; `0` = greedy |
| `--top_k` | `0` | Top-k cutoff; `0` = disabled |
| `--top_p` | `1.0` | Nucleus threshold; `1.0` = disabled |
| `--interactive` | `False` | REPL mode |
| `--device` | auto | `cuda` / `cpu` |
| `--dtype` | `bfloat16` | `float32` / `bfloat16` / `float16` |

## Limitations

- **Single-GPU only** — no FSDP / TP / PP model support in
  `generate()`. For distributed inference, export to HF and use
  vLLM or similar.
- **No KV-cache reuse across calls** — each `generate()` call
  allocates fresh caches. Good for research, wasteful for serving.
- **No speculative decoding, no beam search, no repetition penalty** —
  this is a minimal sampler for evaluating training progress, not a
  generation stack. If you need any of those, fork `generate.py`.

## See also

- [Training loop § Periodic work](training-loop.md#periodic-work) —
  training-time eval doesn't use `generate()`, only loss. Use
  `generate()` for qualitative spot-checks between runs.
- [Checkpointing](../checkpointing/index.md) — how `scripts/generate.py`
  loads the DCP checkpoint it decodes from.
- [Model § Attention paths](../architecture/model.md) — the KV-cache
  path vs the training path through attention.
