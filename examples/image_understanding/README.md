# Image-understanding example

The image-understanding recipe on the core `Transformer`: a **Joint-Decoder**
VLM with a **SigLIP2** image encoder (`google/siglip2-so400m-patch16-512`), an
**avgpool** connector and a **Qwen3-0.6B** backbone, trained in two stages
(captioning on a frozen backbone, then an instruction mixture with everything
training), as specified by #262 and #428. Everything here is entry points,
readers and their tests; `kempnerforge/` never imports it, so this directory
can be deleted without touching the core.

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
├── image_text/           # image-text dataset readers and chat renderer (below)
└── tests/
    ├── test_image_understanding_entry.py   # the entry point loads a config and hands it off
    └── image_text/       # reader and renderer tests (CPU)
```

## Run it

The entry point takes a TOML config and the usual `--section.key=value`
overrides:

```bash
uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py \
    <config.toml> --train.grad_accum_steps=4 \
    --checkpoint.load_path=/your/init/checkpoint --checkpoint.dir=/your/run/dir
```

The recipe's stage presets are not here yet. They need the roadmap below first
(the typed stage contract of #262 fixes their fields, and the readers, packing,
tiling and per-module learning rates fix their values), so they arrive with
those changes rather than as placeholders that would be rewritten by each one.
Until then, write a config the way the [VLM example](../vlm/README.md) does,
with `vlm.arch = "joint_decoder"`, `vision_encoder.type = "siglip2"` and
`adapter.type = "avgpool"`. The geometry is the recipe's: a 512 px view is
32 × 32 = 1024 patch tokens, pooled 2 × 2 to 256 visual tokens, plus
`max_text_len = 256` text tokens, so `max_seq_len = 512`.

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

Every sample carries one image, resized square to `image_size`, with no tiling or
packing. PixMo-Count stores its points in pixels; they are converted to percent of
the image, the unit the other PixMo sources use.

## Start from the pretrained backbone

Stage 1 warm-starts from a checkpoint that holds the pretrained Qwen3-0.6B
backbone and the pretrained SigLIP2 encoder, with a fresh connector. The
converter the VLM example ships builds it from your config, so the written
checkpoint matches the model exactly; with `tie_embeddings = false` it starts
the untied output head as a copy of the tied source's embedding:

```bash
uv run python examples/vlm/scripts/convert_hf_backbone.py \
    --hf-dir Qwen/Qwen3-0.6B \
    --config <config.toml> \
    --out /your/init/checkpoint
```

Then pass it as `--checkpoint.load_path` with
`checkpoint.exclude_from_loading = ["optimizer", "train_state"]`, so only the
weights load: the step, LR schedule, data position and RNG start fresh, and the
stage's own freeze specs apply. Stage 2 loads a stage-1 checkpoint directory
the same way, so its warmup starts from step 0 rather than resuming stage 1's
step and schedule.

## Roadmap

What the recipe still needs, and where it is tracked. Each lands as its own
change in this directory (or in core where the issue says so).

| Feature | Tracked in |
| --- | --- |
| Training sources (COCO, VQAv2, TextVQA, DocVQA, ChartQA, PixMo, FineVision, DataComp), chat-template rendering with the loss on the assistant turns only, the mixture loader | [#428](https://github.com/KempnerInstitute/KempnerForge/issues/428) |
| Typed stage contract and pinned source manifest, which fix the stage presets | [#262](https://github.com/KempnerInstitute/KempnerForge/issues/262) |
| Packing image-text documents into one sequence | [#432](https://github.com/KempnerInstitute/KempnerForge/issues/432) |
| Text-only documents in image-text batches | [#433](https://github.com/KempnerInstitute/KempnerForge/issues/433) |
| Documents with several images | [#434](https://github.com/KempnerInstitute/KempnerForge/issues/434) |
| Tiling into a global view and a 3 × 3 grid of crops (stage 2) | [#435](https://github.com/KempnerInstitute/KempnerForge/issues/435) |
| Per-module learning rates and schedules | [#439](https://github.com/KempnerInstitute/KempnerForge/issues/439) |
| Vision-encoder precision separate from the backbone's | [#438](https://github.com/KempnerInstitute/KempnerForge/issues/438) |
| Evaluation parity: chat-template prompts, image attention, the recipe's scorers | [#443](https://github.com/KempnerInstitute/KempnerForge/issues/443) |
| Run-comparison metrics: documents seen, token counts, MFU with the encoder | [#444](https://github.com/KempnerInstitute/KempnerForge/issues/444) |

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU; the renderer tests download
the `Qwen/Qwen3-0.6B` tokenizer once:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
