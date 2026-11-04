# Image-understanding example

The multimodal project's image-understanding recipe on the core `Transformer`:
a **Joint-Decoder** VLM with a **SigLIP2** image encoder
(`google/siglip2-so400m-patch16-512`), an **avgpool** connector and a
**Qwen3-0.6B** backbone, trained in two stages. Everything here is
configuration, entry points and their tests; `kempnerforge/` never imports it,
so this directory can be deleted without touching the core.

[`examples/vlm/`](../vlm/) stays the general VLM example: the four arches,
video, and its evaluation CLI. This example is image-only and fixed to one
architecture, so the recipe's remaining features (its datasets and chat
rendering, packing, tiling, per-module learning rates, evaluation parity) land
here without widening that example.

## Layout

```
examples/image_understanding/
├── README.md
├── train.py              # training entry point
├── configs/              # TOML presets (below)
└── tests/
    ├── test_image_understanding_configs.py   # every preset loads and validates; the entry point
    └── test_image_understanding_backbone.py  # the stage presets build the Qwen3-0.6B backbone
```

## Configs

| Config | Stage | Encoder | Trains | For |
| --- | --- | --- | --- | --- |
| `debug.toml` | 1 | random | connector | 1-GPU smoke, no download |
| `qwen3_0.6b_siglip2_stage1.toml` | 1 | siglip2 | encoder + connector | captioning on a frozen backbone, 16 GPUs |
| `qwen3_0.6b_siglip2_stage2.toml` | 2 | siglip2 | everything | instruction mixture from a stage-1 checkpoint, 16 GPUs |

The geometry is the recipe's: a 512 px view is 32 × 32 = 1024 patch tokens,
pooled 2 × 2 to 256 visual tokens, plus `max_text_len = 256` text tokens, so
`max_seq_len = 512`. The stage presets train at a global batch of 256
(stage 1) and 512 (stage 2) on 16 GPUs; on fewer GPUs raise
`train.grad_accum_steps` to keep it.

Paths in these configs are placeholders (`dir = "path-to-output-dir"`) and the
dataset is a 30-image COCO stand-in, so a fresh clone validates and smokes.
Point them at your own data and output directories, or override on the CLI.

## Run it

```bash
# 1-GPU smoke
uv run python examples/image_understanding/train.py \
    examples/image_understanding/configs/debug.toml

# Stage 1 on 4 GPUs (grad accumulation keeps the global batch at 256)
uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py \
    examples/image_understanding/configs/qwen3_0.6b_siglip2_stage1.toml \
    --train.grad_accum_steps=4 \
    --checkpoint.load_path=/your/init/checkpoint --checkpoint.dir=/your/run/dir

# Stage 2 from the stage-1 run
uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py \
    examples/image_understanding/configs/qwen3_0.6b_siglip2_stage2.toml \
    --train.grad_accum_steps=8 \
    --checkpoint.load_path=/your/run/dir/step_50000 --checkpoint.dir=/your/stage2/dir
```

## Start from the pretrained backbone

Stage 1 warm-starts from a checkpoint that holds the pretrained Qwen3-0.6B
backbone and the pretrained SigLIP2 encoder, with a fresh connector. Build it
once with the converter the VLM example ships; it reads the stage-1 config, so
the written checkpoint matches this example's model exactly:

```bash
uv run python examples/vlm/scripts/convert_hf_backbone.py \
    --hf-dir Qwen/Qwen3-0.6B \
    --config examples/image_understanding/configs/qwen3_0.6b_siglip2_stage1.toml \
    --out /your/init/checkpoint
```

Then pass it as `--checkpoint.load_path`. The config's
`exclude_from_loading = ["optimizer"]` makes the load weights-only. Stage 2
loads a stage-1 checkpoint the same way.

## Roadmap

What the recipe still needs, and where it is tracked. Each lands as its own
change in this directory (or in core where the issue says so).

| Feature | Tracked in |
| --- | --- |
| Training sources (COCO, VQAv2, TextVQA, DocVQA, ChartQA, PixMo, FineVision, DataComp), chat-template rendering, answer-only loss, the mixture loader | [#428](https://github.com/KempnerInstitute/KempnerForge/issues/428) |
| Typed stage contract and pinned source manifest | [#262](https://github.com/KempnerInstitute/KempnerForge/issues/262) |
| Packing image-text documents into one sequence | [#432](https://github.com/KempnerInstitute/KempnerForge/issues/432) |
| Text-only documents in image-text batches | [#433](https://github.com/KempnerInstitute/KempnerForge/issues/433) |
| Documents with several images | [#434](https://github.com/KempnerInstitute/KempnerForge/issues/434) |
| Tiling into a global view and a 3 × 3 grid of crops (stage 2) | [#435](https://github.com/KempnerInstitute/KempnerForge/issues/435) |
| Per-module learning rates and schedules | [#439](https://github.com/KempnerInstitute/KempnerForge/issues/439) |
| Vision-encoder precision separate from the backbone's | [#438](https://github.com/KempnerInstitute/KempnerForge/issues/438) |
| Untied output head from the tied Qwen3 checkpoint | [#441](https://github.com/KempnerInstitute/KempnerForge/issues/441) |
| Evaluation parity: chat-template prompts, image attention, the recipe's scorers | [#443](https://github.com/KempnerInstitute/KempnerForge/issues/443) |
| Run-comparison metrics: documents seen, token counts, MFU with the encoder | [#444](https://github.com/KempnerInstitute/KempnerForge/issues/444) |

Until those land, the stage presets train on one 512 px view per image with a
single learning rate, the plain caption loss and the tied output head.

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU, network or download:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
