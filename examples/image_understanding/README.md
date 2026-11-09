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
└── tests/
    └── test_image_understanding_entry.py   # the entry point loads a config and hands it off
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

## Pack documents

By default every row holds one document: its image tokens, then its text
right-padded to `vlm.max_text_len`. With `data.pack_sequences = true` on a
`joint_decoder` run, `PackedVLMCollator` packs the `train.batch_size` documents
of each micro-batch first-fit into rows of `train.seq_len` instead. Each
document is its image's token slots followed by its real text, `doc_ids` keeps
attention inside a document, and the wrapper writes each image's tokens over
its slots (`ModalityContext.inline_embeds`). The number of rows follows the
documents' lengths, so it varies between micro-batches.

```toml
[vision_encoder]
type = "siglip2"
path = "google/siglip2-so400m-patch16-512"
num_tokens = 1024          # the packer needs the count before the encoder is built

[adapter]
type = "avgpool"
pool_window = 2            # 256 tokens per image in the row

[vlm]
arch = "joint_decoder"
max_text_len = 256         # per-document text cap

[data]
pack_sequences = true

[train]
seq_len = 2048             # row length; batch_size documents per micro-batch
```

`attention_backend = "flex"` makes the packed mask block-sparse. Positions run
through the row without restarting per document, and the views of an image
attend causally; per-document positions and bidirectional image attention are
follow-ups on the core's explicit position ids and attention groups. A
`[video]` run and the other arches do not pack.

## Mix in text-only documents

Image-text mixes often carry documents without an image, such as FineVision's
`text_*` subsets. With `data.hf_allow_text_only = true`, a row whose image
column is `None` or an empty list is a text-only document. Without packing it
takes a blank image and a `frame_mask` entry of `False`, so its image tokens
are hidden from attention; the document still occupies the image positions,
so its text sits after them in the row and in the position encoding, as a
masked image document rather than plain text. With packing it is just its
text, and a micro-batch of nothing but text carries no images. On every rank
a sharded vision encoder or adapter still runs, on a blank image when there is
nothing else, because its FSDP2 collectives need every rank to take part; a
frozen encoder is skipped. The flag needs an image-prefix arch or packing, not
`cross_attention`, whose image K/V cannot hide a blank image. Without the flag
a dataset with such a row is rejected when it is loaded.

## Documents with several images

An image column that holds a list of images is a multi-image document.
`data.hf_max_images = n` (`n >= 2`) accepts up to `n` of them per document, in
list order, and keeps the first `n` of a longer list; when the dataset is
loaded it counts the rows that trims in one warning. At the default of `1` a
dataset with a list of more than one is rejected when it is loaded. The
images' tokens go before the document's text, each image in its own block,
which is the layout packing uses. With `data.pack_sequences = true` such
documents pack like any other; without it `MultiImageVLMCollator` lays each
document out in its own row of `n` images' tokens plus `vlm.max_text_len`, so
rows keep one shape across micro-batches and ranks. Both need
`vlm.arch = "joint_decoder"` and `vision_encoder.num_tokens`, and `n` images'
tokens plus `vlm.max_text_len` have to fit `train.seq_len` and
`model.max_seq_len`, which the config checks, so every document fits its row.

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
- Tiling into a global view and a 3 × 3 grid of crops (stage 2)
- Per-module learning rates and schedules
- Vision-encoder precision separate from the backbone's
- Evaluation parity: chat-template prompts, image attention, the recipe's scorers
- Run-comparison metrics: documents seen, token counts, MFU with the encoder

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU, network or download:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
