# Available configs

The
[`configs/train/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/configs/train)
directory ships 17 training presets. Most are named
`<model>_<gpu-count>_<parallelism>.toml` so the filename doubles as a
recipe. The
[`configs/model/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/configs/model)
directory ships 4 model-only presets used by
[`scripts/convert_checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/convert_checkpoint.py)
for DCP ↔ HuggingFace conversion.

## Dense training configs

| File | Purpose | Model | GPUs | Parallelism | Notes |
|------|---------|-------|-----:|-------------|-------|
| [`debug.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/debug.toml) | Tiny model, fast smoke test | 256-dim, 4-layer | 1+ | FSDP (`dp_shard=-1`) | 100 steps, `compile_model=false`, `act_ckpt="none"` |
| [`hf_wikitext.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/hf_wikitext.toml) | End-to-end HuggingFace streaming | 512-dim, 8-layer | 1+ | FSDP (`dp_shard=-1`) | GPT-2 tokenizer, `wikitext-103-raw-v1`, 500 steps |
| [`7b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b.toml) | General-purpose Llama-3 7B | 7B | any | FSDP (`dp_shard=-1`) | 100K steps, `compile_model=true`, full AC |
| [`7b_32gpu_fsdp.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_32gpu_fsdp.toml) | Baseline multi-node 7B | 7B | 32 | pure FSDP | 2M tokens/step, simplest multi-node recipe |
| [`7b_12gpu_tp4.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_12gpu_tp4.toml) | 7B with intra-node TP | 7B | 12 | TP=4 × FSDP=3 | TP within node (NVLink), FSDP across (IB) |
| [`7b_16gpu_adamw.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_adamw.toml) | Long preemptible 7B run | 7B | 16 | FSDP (`dp_shard=-1`) | 100K steps, 210B tokens, ckpt every 500 steps |
| [`7b_16gpu_fp8.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_fp8.toml) | FP8 compute + FSDP2 float8 AG | 7B | 16 | FSDP (`dp_shard=-1`) | `mixed_precision="fp8"`, E4M3 fwd / E5M2 bwd |
| [`7b_16gpu_muon.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_muon.toml) | Muon optimizer + z-loss + chunked CE | 7B | 16 | FSDP (`dp_shard=-1`) | Tests Muon, chunked cross-entropy, z-loss together |
| [`13b_24gpu_validation.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/13b_24gpu_validation.toml) | Full-stack validation run | 13B | 24 | TP=4 × FSDP=6 | WandB, profiling, eval, HF tokenizer, 1000 steps |
| [`13b_32gpu_tp4_pp2.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/13b_32gpu_tp4_pp2.toml) | 13B with pipeline parallel | 13B | 32 | TP=4 × PP=2 × FSDP=4 | 40 layers → 20 per PP stage, `1f1b` schedule |
| [`29b_32gpu_tp4_pp2.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/29b_32gpu_tp4_pp2.toml) | Custom 29B sized for H200 140GB | 29B | 32 | TP=4 × PP=2 × FSDP=4 | `dim=6144`, 56 layers, saturates 120 GB/GPU |
| [`70b_32gpu_tp4.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/70b_32gpu_tp4.toml) | 70B without PP bubble | 70B | 32 | TP=4 × FSDP=8 | No PP — fits via FSDP sharding alone |
| [`70b_32gpu_tp4_pp4.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/70b_32gpu_tp4_pp4.toml) | 70B when memory is tight | 70B | 32 | TP=4 × PP=4 × FSDP=2 | 80 layers → 20 per PP stage, less FSDP sharding |

## MoE training configs

| File | Purpose | Model | GPUs | Parallelism | MoE |
|------|---------|-------|-----:|-------------|-----|
| [`debug_moe.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/debug_moe.toml) | Tiny MoE smoke test | 256-dim, 4-layer | 1+ | FSDP (`dp_shard=-1`) | 4 experts, top-2, `moe_frequency=2` |
| [`moe_8gpu_stress.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_8gpu_stress.toml) | Saturate 2 nodes with MoE | ~4B total / 1.8B active | 8 | TP=4 × FSDP=2 | 8 experts, top-2, `moe_frequency=1` |
| [`moe_24gpu.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_24gpu.toml) | 24-GPU MoE stress test | ~7B total / 1.8B active | 24 | TP=4 × FSDP=6 | 8 experts, top-2, `grad_accum=32` for full saturation |
| [`moe_ep_32gpu.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_ep_32gpu.toml) | MoE + Expert Parallel | ~4B total / 1.8B active | 32 | TP=4 × EP=2 × FSDP=4 | 8 experts, top-2, all-to-all across IB |

See the [MoE Expert Parallel benchmark](benchmarks.md#moe-expert-parallelism)
for the numbers the last config produced.

## Model-only configs

These don't include training fields — they're loaded by
[`scripts/convert_checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/convert_checkpoint.py)
to describe the architecture when round-tripping checkpoints to
HuggingFace.

| File | Architecture |
|------|-------------|
| [`llama_7b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/model/llama_7b.toml) | `dim=4096`, 32 layers, 32 heads, 8 kv-heads, `ffn_dim_multiplier=1.3` |
| [`llama_13b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/model/llama_13b.toml) | `dim=5120`, 40 layers, 40 heads, 8 kv-heads, `ffn_dim_multiplier=1.3` |
| [`llama_70b.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/model/llama_70b.toml) | `dim=8192`, 80 layers, 64 heads, 8 kv-heads, `ffn_hidden_dim=28672` |
| [`moe_small.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/model/moe_small.toml) | `dim=2048`, 24 layers, 16 heads, 4 kv-heads, 8 experts top-2 |

## Conventions

- **`dp_shard = -1`** in a config means "fill the remaining mesh
  dimension with FSDP" — the loader resolves this to
  `world_size / (dp_replicate·tp·pp·cp·ep)`. Most single-dimension
  configs (7B, FP8, Muon, debug) use `-1` so the same config works on
  any GPU count.
- **`compile_model = false`** is set on every MoE config. Routing
  produces data-dependent shapes that break `torch.compile`'s graph —
  [`JobConfig.validate(world_size)`](../configuration/validation-rules.md)
  logs a warning (not an error) if you combine them.
- **Paths in these configs** (`dataset_path = "/path/to/..."`) are
  placeholders — replace them with a real tokenized shard directory
  before running. The `hf_wikitext.toml` config is the only one that
  runs end-to-end without path edits (it streams from the HF Hub).
- **Short `max_steps`** (20–100) in the multi-node configs is a
  benchmark-sizing default, not a training budget. Override with
  `--train.max_steps=…` for real runs.

## See also

- [Parallelism recipes](parallelism-recipes.md) — same data, indexed
  by (model, GPU count) rather than by filename.
- [Benchmarks](benchmarks.md) — measured throughput for the configs
  that were benchmarked end-to-end.
- [Config sections](../configuration/config-sections.md) — the fields
  every TOML key maps to.
