# Image-understanding example

The image-understanding recipe on the core `Transformer`: a **Joint-Decoder**
VLM with a **SigLIP2** image encoder (`google/siglip2-so400m-patch16-512`), an
**avgpool** connector and a **Qwen3-0.6B** backbone, trained in two stages
(captioning on a frozen backbone, then an instruction mixture with everything
training). Everything here is entry points and their tests; `kempnerforge/`
never imports it, so this directory can be deleted without touching the core.

## Layout

```
examples/image_understanding/
├── README.md
├── train.py              # training entry point
├── image_text/           # image-text dataset readers and chat renderer (below)
├── scripts/
│   └── convert_finevision_to_arrow.py  # FineVision parquet -> memory-mappable Arrow
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
(the typed stage contract fixes their fields, and the readers, packing, tiling
and per-module learning rates fix their values), so they arrive with those
changes rather than as placeholders that would be rewritten by each one.
Until then, write a config the way the [VLM example](../vlm/README.md) does,
with `vlm.arch = "joint_decoder"`, `vision_encoder.type = "siglip2"` and
`adapter.type = "avgpool"`. The geometry is the recipe's: a 512 px view is
32 × 32 = 1024 patch tokens, pooled 2 × 2 to 256 visual tokens, plus
`max_text_len = 256` text tokens, so `max_seq_len = 512`.

## Image-text datasets

`image_text/` reads the recipe's image-text sources and renders them into the
samples `VLMCollator` batches.

- `testbed.py`: COCO Karpathy captions, VQAv2 (LAVIS layout), TextVQA, DocVQA and
  ChartQA (human and augmented); a row whose image does not decode is passed over.
- `render.py`: `ChatRenderer` applies the tokenizer's chat template (prompt then
  answer for single-turn rows, the full template for conversations) and trains only
  the assistant replies when `answer_only`. `RenderedDataset` draws one reference
  answer per turn from a seed and the row index, so a resume draws the same answers,
  and passes over a document the renderer refuses (an empty reply, or one the text
  budget cuts away), up to `max_attempts` in a row; `skipped` counts them per
  process, so a loader worker's skips are its own.
- `pixmo.py`: PixMo captions, CapQA, pointing, counting and point-explanations;
  `points.py` writes the point answers.
- `finevision.py`: FineVision subsets from their memory-mappable Arrow copy.

Every sample carries one image, resized square to `image_size`, with no tiling or
packing. PixMo-Count stores its points in pixels; they are converted to percent of
the image, the unit the other PixMo sources use. A counting answer lists its
points before the count, so a row with many points outruns `max_text_len` and
trains only the leading points; `max_points` bounds the answer so the count
fits. `FineVisionArrow` keeps only the rows with one image (`max_images`, which
`None` lifts), using the per-row image counts `index.npz` stores; an index
written without them takes the counts from `num_images`. Turns missing their
user message or reply are dropped, and a row left without a turn is passed over
like one whose image does not decode.

Convert FineVision once; any number of `convert` processes can share the work,
a file that fails to convert is listed at the end without stopping the others,
and `index` refuses a subset whose parquet files changed since their conversion:

```bash
uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py convert \
    --src path-to-FineVision --dst path-to-FineVision-arrow --procs 8
uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py index \
    --src path-to-FineVision --dst path-to-FineVision-arrow
```

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
`checkpoint.exclude_from_loading = ["optimizer"]`, so the converted weights load
and the optimizer starts fresh. Stage 2 should start from a stage-1 checkpoint
the same way, as weights alone with its own warmup from step 0; a warm start
today also restores the saved step and schedule, so that is on the roadmap.

## Roadmap

What the recipe still needs. Each lands as its own change in this directory,
or in core where it belongs:

- Training sources (COCO, VQAv2, TextVQA, DocVQA, ChartQA, PixMo, FineVision,
  DataComp), chat-template rendering with the loss on the assistant turns only,
  and the mixture loader
- Typed stage contract and pinned source manifest, which fix the stage presets
- Starting stage 2 from stage 1's weights alone, with a fresh step and schedule
- Packing image-text documents into one sequence
- Text-only documents in image-text batches
- Documents with several images
- Tiling into a global view and a 3 × 3 grid of crops (stage 2)
- Per-module learning rates and schedules
- Vision-encoder precision separate from the backbone's
- Evaluation parity: chat-template prompts, image attention, the recipe's scorers
- Run-comparison metrics: documents seen, token counts, MFU with the encoder

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU; the renderer tests download
the `Qwen/Qwen3-0.6B` tokenizer once:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
