# VLM example

Vision-language training — images **or** video — on the core `Transformer`. A
frozen HF vision encoder produces visual tokens, a connector projects (and
optionally pools) them, and an arch-specific path feeds the backbone. Everything
here is configuration plus an entry point; `kempnerforge/` never imports it, so
this directory can be deleted without touching the core.

## Configs

`vlm_debug*` are 1-GPU smoke presets — tiny backbone, `random` encoder, so they
run on a fresh clone with no download. The `vlm_7b*` presets are 4-8 GPU
starting points.

| Config | Arch | Encoder | For |
| --- | --- | --- | --- |
| `vlm_debug.toml` | joint_decoder | random | 1-GPU smoke |
| `vlm_debug_mot.toml` | mot | random | 1-GPU smoke |
| `vlm_debug_moma.toml` | moma | random | 1-GPU smoke |
| `vlm_debug_moe.toml` | cross_attention | random | 1-GPU smoke, MoE FFN |
| `vlm_7b.toml` | joint_decoder | random | 7B, AC off (VRAM stress) |
| `vlm_7b_ac.toml` | joint_decoder | random | 7B, AC full + longer seq |
| `vlm_7b_mot.toml` | mot | random | 7B |
| `vlm_7b_moma.toml` | moma | random | 7B |
| `vlm_7b_cross_attn.toml` | cross_attention | random | 7B |
| `vlm_7b_freeze_schedule.toml` | cross_attention | random | multi-stage `FreezeStage` schedule |
| `vlm_7b_siglip2.toml` | joint_decoder | siglip2 | real-run starting point |
| `vlm_7b_siglip2_cross_attn.toml` | cross_attention | siglip2 | real-run starting point |
| `vlm_video_webvid.toml` | joint_decoder | siglip2 | video (WebVid-10M) |
| `vlm_qwen3_0.6b_joint_decoder_webvid.toml` | joint_decoder | siglip2 | video, from a pretrained backbone |

Paths in these configs are placeholders (`data_root = "path-to-webvid-10m"`) —
point them at your own data and output directories, or override on the CLI.

## Run it

```bash
# 1-GPU smoke
uv run python examples/vlm/train.py examples/vlm/configs/vlm_debug.toml

# 4 GPUs, single node
uv run torchrun --nproc_per_node=4 examples/vlm/train.py \
    examples/vlm/configs/vlm_7b_siglip2.toml

# Override anything on the CLI
uv run python examples/vlm/train.py examples/vlm/configs/vlm_debug.toml \
    --train.max_steps=20 --checkpoint.dir=/your/run/dir
```

Video needs PyAV: `uv sync --group video`.

Tests: `uv run pytest examples/vlm/tests/ -v` (they are outside the core
`testpaths`, so run them by path).

## Video from a pretrained backbone

`vlm_qwen3_0.6b_joint_decoder_webvid.toml` trains video captioning on a frozen
pretrained 0.6B backbone, with a pretrained vision encoder and a fresh `avgpool`
adapter that both train. The backbone's attention is wider than its model dim
(`head_dim_override = 128`).

Build its starting checkpoint once, with the converter described under
*Start from a pretrained backbone*:

```bash
uv run python examples/vlm/scripts/convert_hf_backbone.py \
    --hf-dir Qwen/Qwen3-0.6B \
    --config examples/vlm/configs/vlm_qwen3_0.6b_joint_decoder_webvid.toml \
    --out path-to-init-checkpoint
```

Then set `[video].data_root`, `[checkpoint].load_path` and `[checkpoint].dir`,
and train:

```bash
uv run torchrun --nproc_per_node=4 examples/vlm/train.py \
    examples/vlm/configs/vlm_qwen3_0.6b_joint_decoder_webvid.toml
```

The first run loads the converted weights only (`exclude_from_loading =
["optimizer"]`); a resume from `[checkpoint].dir` restores the full training
state. For a short shakedown, add `--train.max_steps=20 --video.max_samples=512`.

## Data prep

`data/prep_vlm_coco.py` writes a COCO-Karpathy `save_to_disk` directory for
`data.hf_dataset_name` to point at.

## Evaluation

Benchmark evaluation of the resulting checkpoints lives in `eval/`.
