# Generate from a checkpoint

A DCP checkpoint + a tokenizer + a prompt →
[`scripts/generate.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/generate.py)
produces text. This page covers the CLI, the underlying
[`generate()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/generate.py)
call, and the KV-cache mechanics so you can build your own generation
loops when the script isn't enough.

## Quick start

```bash
uv run python scripts/generate.py configs/train/7b.toml \
    --checkpoint.load_path=checkpoints/7b/step_50000 \
    --data.tokenizer_path=meta-llama/Llama-2-7b-hf \
    --prompt "Once upon a time" \
    --max_tokens 256 \
    --temperature 0.8 \
    --top_p 0.9
```

```
Model: 6738M params on cuda:0 (torch.bfloat16)
Loaded checkpoint: checkpoints/7b/step_50000

--- Prompt ---
Once upon a time

--- Generated (256 tokens) ---
, in a small village nestled between two mountains, there lived …
```

## CLI reference

| Flag | Default | Purpose |
|------|---------|---------|
| `config` (positional) | — | TOML config (required — used for model architecture) |
| `--checkpoint.load_path` | from config | DCP directory to load (step_N or `latest` symlink) |
| `--data.tokenizer_path` | from config | HuggingFace hub ID or local dir |
| `--prompt` | `""` | Input text |
| `--max_tokens` | `128` | Max new tokens to generate |
| `--temperature` | `1.0` | Sampling temperature (0 = greedy) |
| `--top_k` | `0` | Top-k filtering (0 = disabled) |
| `--top_p` | `1.0` | Nucleus sampling threshold (1.0 = disabled) |
| `--interactive` | `false` | REPL mode — enter prompts, see output |
| `--device` | `cuda` if available | Target device |
| `--dtype` | `bfloat16` | Parameter dtype (also: `float32`, `float16`) |

The script uses `argparse.parse_known_args()` — flags it recognizes
(the table above) are consumed as script arguments, and everything
else (e.g. `--checkpoint.load_path=...`, `--model.dim=...`) is fed to
`load_config` as a config override, just like `scripts/train.py`.
Positional `config` must come first; the script flags and config
overrides can otherwise appear in any order.

### Interactive mode

```bash
uv run python scripts/generate.py configs/train/7b.toml \
    --checkpoint.load_path=checkpoints/7b/step_50000 \
    --data.tokenizer_path=meta-llama/Llama-2-7b-hf \
    --interactive
```

A small REPL loop: enter a prompt, see generation, loop. Each call
re-runs prefill from scratch — there's no conversation state between
prompts. Handy for qualitative sanity checks while training is
ongoing.

## How it loads a DCP checkpoint without `dist.init_process_group`

DCP files are multi-rank by default, but
`torch.distributed.checkpoint.load` supports single-process loading
without initializing a process group:

```python
# scripts/generate.py
state_dict = {"model": model.state_dict()}
dcp.load(state_dict, checkpoint_id=str(ckpt_path))
model.load_state_dict(state_dict["model"])
```

This is why `generate.py` is single-GPU even for models that were
trained with FSDP=N and TP=M — DCP handles the resharding on the read
side, loading the full unsharded model onto one device. For a 70B
model that means ~140 GB of bf16 parameters — run on a node with
enough memory, or use FSDP via a multi-GPU loader (not covered by
`scripts/generate.py`).

## The `generate()` function

```python
from kempnerforge.model.generate import generate

output_ids = generate(
    model,
    prompt_tokens,      # (batch, prompt_len)
    max_new_tokens,
    *,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    eos_token_id=None,  # stop when all batch entries emit this token
)                       # returns (batch, prompt_len + generated_len)
```

Called with `@torch.no_grad()`, flips the model to `.eval()`, restores
training mode on exit. The function does **not** call
`dist.init_process_group` — it works on raw tensors, so you can use it
both single-GPU (as `scripts/generate.py` does) and from within a
training script after FSDP summons parameters (more complex).

### Sampling details

Temperature → top-k → top-p → multinomial. All in
[`sample()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/generate.py):

```python
if temperature == 0:
    return logits.argmax(dim=-1)           # greedy
logits = logits / temperature

if top_k > 0:
    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
    logits = logits.where(logits >= threshold, -inf)

if top_p < 1.0:
    sorted_logits, idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mask = (probs.cumsum(dim=-1) - probs) >= top_p
    sorted_logits[mask] = -inf
    logits = scatter back to original order

probs = logits.softmax(dim=-1)
return torch.multinomial(probs, 1).squeeze(-1)
```

`sample()` is exported — useful if you want to plug your own decoding
loop with a custom sampler (contrastive, typical, etc.) while keeping
the model + KV-cache wiring.

## KV cache

Without a cache, generating `N` tokens re-runs attention over the
growing sequence `1, 2, …, N` times — O(N²) work for a problem that
should be O(N). The
[`KVCache`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/attention.py)
stores per-layer keys and values so each new token only needs one
attention pass over the cached history.

### Layout

```python
KVCache(
    batch_size,
    max_seq_len,   # pre-allocated up to this length
    n_kv_heads,    # GQA key heads (not query heads)
    head_dim,
    dtype,
    device,
)
```

Two pre-allocated tensors of shape
`(batch, n_kv_heads, max_seq_len, head_dim)`. Keys are stored **after**
RoPE but **before** GQA expansion — this saves memory by
`n_heads / n_kv_heads` (4× for the default Llama-style config).

### Update

```python
def update(k_new, v_new) -> (k_all, v_all):
    end = self.seq_len + k_new.shape[2]
    self.k[:, :, self.seq_len:end] = k_new
    self.v[:, :, self.seq_len:end] = v_new
    self.seq_len = end
    return self.k[:, :, :end], self.v[:, :, :end]
```

`update` returns slices, not copies — the returned tensors alias
into the pre-allocated buffer. Safe inside `@torch.no_grad()`.

### Prefill vs decode

`generate()` allocates one `KVCache` per transformer layer, then:

1. **Prefill**: one forward pass with the full prompt. Each layer's
   attention calls `kv_cache.update(k, v)` with the prompt's K/V
   tensors, filling positions `[0, prompt_len)`.
2. **Decode loop**: `max_new_tokens` steps, each a forward pass on
   a single-token input. Attention's `k_new`, `v_new` are shape
   `(batch, n_kv_heads, 1, head_dim)`; `update` appends at position
   `seq_len` and advances it.

If `prompt_len + max_new_tokens > model.config.max_seq_len` the call
raises — the cache is pre-allocated up to that bound.

## Batch generation from Python

`scripts/generate.py` tokenizes one prompt. For many prompts at
once, call `generate()` directly:

```python
import torch
from transformers import AutoTokenizer
from kempnerforge.model.generate import generate

tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompts = ["Hello, my name is", "The best way to learn Python is"]

# Pad to the same length (left-pad if you want all prompts to end
# together; right-pad is fine if you just want them to start together)
tokenizer.pad_token = tokenizer.eos_token
batch = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
batch = batch.to(device)

output = generate(model, batch, max_new_tokens=50, temperature=0.8, top_p=0.9)
# output: (batch=2, prompt_len + 50)

for row in output:
    print(tokenizer.decode(row, skip_special_tokens=True))
```

Two caveats:

- **Left-pad or right-pad consistently**. The KV cache assumes
  position 0 is the start of every sequence in the batch, so if you
  right-pad, the generation will treat pad tokens as a real prefix.
  For Llama-style tokenizers this is usually fine (pad is excluded
  from attention via the mask), but for greedy comparison runs,
  left-pad to line up end positions.
- **`eos_token_id` stops the whole batch when *all* sequences
  emit EOS.** Individual sequences past EOS continue generating garbage
  until every batch row hits it; filter post-hoc.

## What `scripts/generate.py` doesn't do

- **No beam search** — only sampling or argmax-greedy.
- **No distributed inference** — single-GPU only. For 70B+ you need
  to write your own FSDP-wrapped inference loop.
- **No cache reuse across calls in interactive mode** — each prompt
  re-prefills from scratch.
- **No streaming output** — it generates all `max_new_tokens` before
  printing. For streaming, loop `sample()` yourself and print after
  each token.

## See also

- [Training § Generation](../training/generation.md) — the
  `generate()` internals reference.
- [Training § Generation § KV cache](../training/generation.md) —
  `KVCache` class docs.
- [End-to-end training run § Generate from the checkpoint](end-to-end-training-run.md#6-generate-from-the-checkpoint)
  — the quickstart version of this page.
- [Checkpointing § DCP model format](../checkpointing/dcp-model.md) —
  why single-process DCP load works without `init_process_group`.
- [`scripts/generate.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/generate.py)
  — the script this page documents.
