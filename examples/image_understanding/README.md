# Image-understanding example

The image-understanding recipe on the core `Transformer`: a **Joint-Decoder**
VLM with a **SigLIP2** image encoder (`google/siglip2-so400m-patch16-512`), an
**avgpool** connector and a **Qwen3-0.6B** backbone, trained in two stages
(captioning on a frozen backbone, then an instruction mixture with everything
training), as specified by #262 and #428. Everything here is
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
├── image_text/           # image-text dataset readers and chat renderer (below)
├── scripts/
│   └── convert_finevision_to_arrow.py  # FineVision parquet -> memory-mappable Arrow
└── tests/
    ├── test_image_understanding_configs.py   # every preset loads and validates; the entry point
    ├── test_image_understanding_backbone.py  # the stage presets build the Qwen3-0.6B backbone
    └── image_text/       # reader and renderer tests (CPU)
```

## Configs

| Config | Stage | Encoder | Trains | For |
| --- | --- | --- | --- | --- |
| `qwen3_0.6b_siglip2_stage1.toml` | 1 | siglip2 | encoder + connector | captioning on a frozen backbone, 16 GPUs |
| `qwen3_0.6b_siglip2_stage2.toml` | 2 | siglip2 | everything | instruction mixture from a stage-1 checkpoint, 16 GPUs |

The geometry is the recipe's: a 512 px view is 32 × 32 = 1024 patch tokens,
pooled 2 × 2 to 256 visual tokens, plus `max_text_len = 256` text tokens, so
`max_seq_len = 512`. The stage presets train at a global batch of 256
(stage 1) and 512 (stage 2) on 16 GPUs; on fewer GPUs raise
`train.grad_accum_steps` to keep it.

Paths in these configs are placeholders (`dir = "path-to-output-dir"`), and
`data.hf_dataset_name` names `sayakpaul/coco-30-val-2014`, 30,000 COCO
image-caption pairs (about 4.9 GB on first download), as a stand-in for the
recipe's sources until its readers land. A fresh clone validates the configs
without downloading anything. Point them at your own data and output
directories, or override on the CLI.

## Run it

```bash
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

## Image-text datasets

`image_text/` reads the recipe's image-text sources and renders them into the
samples `VLMCollator` batches.

- `testbed.py`: COCO Karpathy captions, VQAv2 (LAVIS layout), TextVQA, DocVQA and
  ChartQA (human and augmented).
- `render.py`: `ChatRenderer` applies the tokenizer's chat template (prompt then
  answer for single-turn rows, the full template for conversations) and trains only
  the assistant replies when `answer_only`. `RenderedDataset` draws one reference
  answer per turn from a seed and the row index, so a resume draws the same answers.
- `pixmo.py`: PixMo captions, CapQA, pointing, counting and point-explanations;
  `points.py` writes the point answers.
- `finevision.py`: FineVision subsets from their memory-mappable Arrow copy.

Every sample carries one image, resized square to `image_size`, with no tiling or
packing. PixMo-Count stores its points in pixels; they are converted to percent of
the image, the unit the other PixMo sources use. `FineVisionArrow(max_images=1)`
keeps only the rows with one image, using the per-row image counts `index.npz`
stores; an index written without them takes the counts from `num_images`.

Convert FineVision once:

```bash
uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py convert \
    --src path-to-FineVision --dst path-to-FineVision-arrow --procs 8
uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py index \
    --src path-to-FineVision --dst path-to-FineVision-arrow
```

## Start from the pretrained backbone

Stage 1 warm-starts from a checkpoint that holds the pretrained Qwen3-0.6B
backbone and the pretrained SigLIP2 encoder, with a fresh connector. The
presets untie the output head (`tie_embeddings = false`); the converter starts
it as a copy of the tied source's embedding, so it trains separately from
stage 1 on. Build the checkpoint once with the converter the VLM example
ships; it reads the stage-1 config, so the written checkpoint matches this
example's model exactly:

```bash
uv run python examples/vlm/scripts/convert_hf_backbone.py \
    --hf-dir Qwen/Qwen3-0.6B \
    --config examples/image_understanding/configs/qwen3_0.6b_siglip2_stage1.toml \
    --out /your/init/checkpoint
```

Then pass it as `--checkpoint.load_path`. The presets set
`exclude_from_loading = ["optimizer", "train_state"]`, so only the weights
load: the step, LR schedule, data position and RNG start fresh, and the stage's
own freeze specs apply. Stage 2 loads a stage-1 checkpoint directory the same
way, so its 6,000-step warmup starts from step 0 rather than resuming stage 1's
step and schedule.

## Roadmap

What the recipe still needs, and where it is tracked. Each lands as its own
change in this directory (or in core where the issue says so).

| Feature | Tracked in |
| --- | --- |
| Training sources (COCO, VQAv2, TextVQA, DocVQA, ChartQA, PixMo, FineVision, DataComp), chat-template rendering with the loss on the assistant turns only, the mixture loader | [#428](https://github.com/KempnerInstitute/KempnerForge/issues/428) |
| Typed stage contract and pinned source manifest | [#262](https://github.com/KempnerInstitute/KempnerForge/issues/262) |
| Packing image-text documents into one sequence | [#432](https://github.com/KempnerInstitute/KempnerForge/issues/432) |
| Text-only documents in image-text batches | [#433](https://github.com/KempnerInstitute/KempnerForge/issues/433) |
| Documents with several images | [#434](https://github.com/KempnerInstitute/KempnerForge/issues/434) |
| Tiling into a global view and a 3 × 3 grid of crops (stage 2) | [#435](https://github.com/KempnerInstitute/KempnerForge/issues/435) |
| Per-module learning rates and schedules | [#439](https://github.com/KempnerInstitute/KempnerForge/issues/439) |
| Vision-encoder precision separate from the backbone's | [#438](https://github.com/KempnerInstitute/KempnerForge/issues/438) |
| Evaluation parity: chat-template prompts, image attention, the recipe's scorers | [#443](https://github.com/KempnerInstitute/KempnerForge/issues/443) |
| Run-comparison metrics: documents seen, token counts, MFU with the encoder | [#444](https://github.com/KempnerInstitute/KempnerForge/issues/444) |

Until those land, the stage presets train on one 512 px view per image with a
single learning rate and the plain caption loss.

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU; the renderer tests download
the `Qwen/Qwen3-0.6B` tokenizer once:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
