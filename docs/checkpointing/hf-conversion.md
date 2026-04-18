# HuggingFace conversion

[`scripts/convert_checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/convert_checkpoint.py)
converts between KempnerForge's DCP format and HuggingFace's
safetensors (or `.bin`) format. Two directions:

- **`dcp-to-hf`** — export a DCP checkpoint for HF inference,
  model cards, or pushing to the Hub.
- **`hf-to-dcp`** — import HF pretrained weights (e.g.
  `meta-llama/Llama-3.1-8B`) into DCP for continued pretraining or
  fine-tuning.

Both directions are offline — no distributed training required,
runs on a single CPU or GPU.

## DCP → HuggingFace

```bash
uv run python scripts/convert_checkpoint.py dcp-to-hf \
    --dcp-dir checkpoints/step_10000 \
    --hf-dir exports/my_model \
    --config configs/model/llama_7b.toml
```

What happens:

1. Build a bare `Transformer(model_config)` on CPU — the default
   random-initialized parameters serve only as target shapes that
   DCP overwrites on load.
2. Detect whether `dcp_dir` contains PP stage subdirectories
   (`pp0/`, `pp1/`, …). If yes, load each stage's shards one at a
   time with `strict=False` — each stage holds a disjoint slice of
   the full model.
3. Convert state dict keys from KempnerForge → HF naming.
4. Cast to the requested dtype (default `bfloat16`).
5. Write `model.safetensors` (or fall back to `pytorch_model.bin`
   if the `safetensors` package isn't installed) plus a
   `config.json` compatible with `AutoModelForCausalLM`.

```
exports/my_model/
├── config.json              ← HF model config
└── model.safetensors        ← weights
```

### Dtypes

```bash
--dtype bfloat16      # default
--dtype float16       # e.g. for vLLM
--dtype float32       # rarely useful; largest files
```

No optimizer / scheduler / RNG — only weights. If you need those,
keep the DCP checkpoint; HF format doesn't carry them.

### With PP

A PP checkpoint has per-stage shards. The conversion script
handles this automatically:

```python
# scripts/convert_checkpoint.py _detect_pp_stages
pp_dirs = sorted((d for d in dcp_path.iterdir()
                  if d.is_dir() and d.name.startswith("pp")),
                 key=lambda d: int(d.name[2:]))
```

If any `pp0/`, `pp1/`, … exist, the script loads each one sequentially
into the same full-model state dict via `load_state_dict(..., strict=False)`.
Each stage fills in its owned slice of keys; at the end, the full
model state is assembled.

This is the path to use when you want to **reshard a PP checkpoint to
different PP** — no direct DCP resharding across PP boundaries
exists, but `dcp-to-hf` → `hf-to-dcp` round-trips through a
non-sharded format and lets you re-emit at a different `pp` degree.

## HuggingFace → DCP

```bash
# From a local HF directory
uv run python scripts/convert_checkpoint.py hf-to-dcp \
    --hf-dir /shared/models/llama-3.1-8b \
    --dcp-dir checkpoints/pretrained \
    --config configs/model/llama_8b.toml

# Or from the HF Hub directly
uv run python scripts/convert_checkpoint.py hf-to-dcp \
    --hf-dir meta-llama/Llama-3.1-8B \
    --dcp-dir checkpoints/pretrained \
    --config configs/model/llama_8b.toml
```

What happens:

1. Load the HF state dict:
   - Local dir: prefer `*.safetensors`, fall back to
     `pytorch_model*.bin`.
   - HF ID: download via
     `transformers.AutoModelForCausalLM.from_pretrained`.
2. Convert keys HF → KempnerForge naming.
3. `model.load_state_dict(converted, strict=False)` — missing KF
   keys (e.g. learned RoPE frequencies) stay at their default init
   and are logged as warnings.
4. Write as DCP via `dcp.save({"model": model.state_dict()}, ...)`.

```
checkpoints/pretrained/
├── .metadata               ← DCP index
└── __*_0.distcp            ← weight shards
```

Point a training config at this directory via
`[checkpoint] load_path = "checkpoints/pretrained"` and
training picks it up like any other DCP checkpoint.

## Key mapping

KempnerForge uses Llama-style naming with a few differences. The
main translations:

| KempnerForge | HuggingFace |
|--------------|-------------|
| `token_embedding.embedding.weight` | `model.embed_tokens.weight` |
| `output_head.proj.weight` | `lm_head.weight` |
| `norm.weight` | `model.norm.weight` |
| `layers.{i}.attention.{q,k,v,o}_proj.weight` | `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` |
| `layers.{i}.attention_norm.weight` | `model.layers.{i}.input_layernorm.weight` |
| `layers.{i}.mlp_norm.weight` | `model.layers.{i}.post_attention_layernorm.weight` |
| `layers.{i}.mlp.{gate,up,down}_proj.weight` | `model.layers.{i}.mlp.{gate,up,down}_proj.weight` |

MoE-specific:

| KempnerForge | HuggingFace |
|--------------|-------------|
| `layers.{i}.mlp.router.gate.weight` | `model.layers.{i}.mlp.gate.weight` |
| `layers.{i}.mlp.router.expert_bias` | `model.layers.{i}.mlp.expert_bias` |
| `layers.{i}.mlp.shared_expert.*` | `model.layers.{i}.mlp.shared_experts.*` (plural in HF) |
| `layers.{i}.mlp.experts.{j}.*` | `model.layers.{i}.mlp.experts.{j}.*` (identical) |

The mapping logic is in `_kf_to_hf_key` and `_hf_to_kf_key` in
`convert_checkpoint.py` — order-sensitive string replacements (e.g.
`.attention_norm.` handled before `.attention.` to avoid partial
matches).

## Generated `config.json`

For a dense model:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "max_position_embeddings": 4096,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}
```

For MoE the architecture is `MixtralForCausalLM` and three extra
fields (`num_local_experts`, `num_experts_per_tok`,
`router_aux_loss_coef`) are added.

Not included: tokenizer files. Copy `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, etc. from the source HF repo separately
(they're not part of the weights).

## Limitations

- **One architecture at a time** — the converter assumes the model
  is Llama or Mixtral style. Models with non-standard attention
  variants (MLA, linear attention) need additional mapping.
- **No optimizer state** — HF format doesn't carry optimizer state,
  so `hf-to-dcp` produces a checkpoint with **model weights only**.
  Training resumes with a fresh optimizer.
- **Missing keys on `hf-to-dcp`** — the HF model may not have some
  KempnerForge-specific buffers (e.g. `freqs_cis` for RoPE). These
  stay at default init; the warning is informational, not an error.
- **Key renames on `dcp-to-hf`** — if you've added custom modules to
  the model, they'll land in the HF output under the KempnerForge
  names (no translation rule). Downstream HF code won't recognize
  them.

## See also

- [DCP model + optimizer](dcp-model.md) — the source format for
  `dcp-to-hf` and the target format for `hf-to-dcp`.
- [Resharding](resharding.md) — using `dcp-to-hf` → `hf-to-dcp` as
  an escape hatch when DCP's automatic resharding can't handle your
  change (e.g. changing PP degree).
- [Configuration § CheckpointConfig](../configuration/config-sections.md) —
  `load_path` to pick up a converted checkpoint.
