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
├── configs/
│   └── eval_tasks.toml   # sample evaluation tasks, with placeholder paths
├── evaluation/           # evaluation tasks and answer scorers (below)
├── image_text/           # image-text dataset readers and chat renderer (below)
├── scripts/
│   └── convert_finevision_to_arrow.py  # FineVision parquet -> memory-mappable Arrow
└── tests/
    ├── test_image_understanding_entry.py   # the entry point loads a config and hands it off
    ├── evaluation/       # task-file and scorer tests (CPU)
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
  budget cuts away), up to `max_attempts` in a row; `skipped` counts them with
  the reader's skips, per rank, and loader workers forked from the training
  process add to its count.
- `pixmo.py`: PixMo captions, CapQA, pointing, counting and point-explanations;
  `points.py` writes the point answers.
- `finevision.py`: FineVision subsets from their memory-mappable Arrow copy.
- `datacomp.py`: DataComp-1B captions, streamed from its webdataset tar shards with
  no index or conversion step (`DataCompStream`); `RenderedStream` renders a
  stream the way `RenderedDataset` renders a reader, and `stream.py`'s
  `StreamLoader` loads it so that a resume continues where training stopped.

Every sample the renderer builds carries one image, resized square to `image_size`,
with no tiling or packing. PixMo-Count stores its points in pixels; they are
converted to percent of the image, the unit the other PixMo sources use. A
counting answer lists its points before the count, so a row with many points
outruns `max_text_len` and trains only the leading points. `max_points` retains
the count only when the cap is small enough for the label and the text budget:
with a 40-point `dot` row at 256 tokens, a cap of 30 still cuts the count, while
a cap of 3 keeps it.

`FineVisionArrow` keeps only the rows with one image (`max_images`, which `None`
lifts), using the per-row image counts `index.npz` stores; an index written
without them takes the counts from `num_images`, and `text_only` keeps the rows
without images. A conversation keeps its turns up to the first one missing its
user message or reply, so a row whose first turn is incomplete is not kept.

Convert FineVision once; any number of `convert` processes can share the work,
a file that fails to convert is listed at the end without stopping the others,
and `index` refuses a subset with a file missing or whose parquet file changed
size since its conversion:

```bash
uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py convert \
    --src path-to-FineVision --dst path-to-FineVision-arrow --procs 8
uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py index \
    --src path-to-FineVision --dst path-to-FineVision-arrow
```

DataComp has no index, so its reader streams: the shards are listed once,
reordered every epoch from the seed, and dealt out to data-parallel ranks and
then to loader workers, without end. Each worker keeps `open_shards` of its
shards open (4 by default) and takes every sample from one of them drawn at
random, so a batch mixes several shards; each shard is still read front to back,
with no buffer of samples to hold. A sample without a decodable image or a
caption is passed over, as is the rest of a shard that stops being a readable
tar, and a shard with no samples counts as one skip. `StreamLoader` records the
position of every batch it hands out, so the stream's `state_dict` holds where
training stopped, and a resume continues from the next sample with batches in
the order they would have come, whichever way the loader workers are started. A
resume must use the same seed, ranks, workers, `open_shards` and shard list.
Pass the stream's `num_workers` to the loader too:

```python
stream = DataCompStream(
    "/path/to/datacomp-1b/raw/webdataset",
    prompt="Describe the image briefly.",
    rank=dp_rank,
    world_size=dp_size,
    num_workers=4,
    seed=seed,
    open_shards=4,
)
rendered = RenderedStream(stream, renderer, seed=seed, rank=dp_rank)
loader = StreamLoader(rendered, batch_size=16, num_workers=4)
```

## Tile images

By default every image is one `hf_image_size` view. `vlm.tile_grid = n` encodes
it as `n × n` crops instead, each `hf_image_size` px, cut from the image resized
to `n × hf_image_size` square, preceded by the whole image at `hf_image_size`
unless `vlm.tile_global = false`:

```toml
[vlm]
arch = "joint_decoder"
max_text_len = 256
tile_grid = 3          # 9 crops in raster order
tile_global = true     # plus the whole image first: 10 views
```

The views travel like a clip's frames (`pixel_values` is `(B, views, 3, H, W)`),
so the sequence budget is `views × tokens per view + max_text_len`, and the
config and build checks enforce it. With this example's 256 tokens per view
and `max_text_len = 256`, the 10 views above need `model.max_seq_len` and
`train.seq_len` of at least 2816, so raise both from the presets' 512 before
turning tiling on. `tile_image` in `kempnerforge/data/vlm_dataset.py` is the
preprocessor at training and at evaluation, so an evaluation tiles exactly as
training did. The crops come from the image resized to a square canvas, so a
wide or tall image is stretched the way its single view already is; a grid
that follows the aspect ratio is not implemented. An image smaller than the
`n × hf_image_size` canvas is upsampled onto it, so its crops cost tokens
without adding detail; tiling pays off on sources larger than the canvas.
Tiling applies to the image path only; a `[video]` run rejects it.

## Pack documents

By default every row holds one document: its image tokens, then its text
right-padded to `vlm.max_text_len`. With `data.pack_sequences = true` on a
`joint_decoder` run, `PackedVLMCollator` packs the `train.batch_size` documents
of each micro-batch first-fit into rows of `train.seq_len` instead. Each
document is its image's token slots followed by its real text, `doc_ids` keeps
attention inside a document, and the wrapper writes each image's tokens over
its slots (`ModalityContext.inline_embeds`). Each document's positions
restart at 0, so a document sees the positions it would on the unpacked path,
and the LM head runs on the text positions only, so the loss and the head's
memory match the unpacked path too. The z-loss averages over those positions,
where the unpacked path's also averages over each row's text padding, so the
two differ slightly when `train.z_loss_weight > 0`. Packing does not run with
`train.mixed_precision = "fp8"`.
The number of rows follows the documents' lengths, so it varies between
micro-batches.

```toml
[vision_encoder]
type = "siglip2"
path = "google/siglip2-so400m-patch16-512"

[adapter]
type = "avgpool"
pool_window = 2            # 256 tokens per image in the row

[vlm]
arch = "joint_decoder"
max_text_len = 256         # per-document text cap

[data]
pack_sequences = true
hf_image_size = 512        # the checkpoint's resolution: 32 × 32 patches

[model]
max_seq_len = 2048         # the RoPE table must cover a full row

[train]
seq_len = 2048             # row length; batch_size documents per micro-batch
```

`attention_backend = "flex"` makes the packed mask block-sparse. Under
`compile_model`, a micro-batch that packs into a single row compiles its own
graph (a size of 1 cannot be made dynamic), so size `train.batch_size` for at
least two rows. The views of
an image attend causally; bidirectional image attention is a follow-up on the
core's attention groups. A `[video]` run and the other arches do not pack.

## Mix in text-only documents

Image-text mixes often carry documents without an image, such as FineVision's
`text_*` subsets. A row whose image column is `None` or an empty list is a
text-only document; no setting turns this on. Without packing it
takes a blank image and a `frame_mask` entry of `False`, so its image tokens
are hidden from attention; the document still occupies the image positions,
so its text sits after them in the row and in the position encoding, as a
masked image document rather than plain text. With packing it is just its
text, and a micro-batch of nothing but text carries no images; the vision
encoder and adapter then run on one blank image whose output carries no weight,
so FSDP2's collectives still see every rank, and under `train.compile_model`
such a micro-batch is a second compiled graph. When the loaded dataset holds
text-only rows, every unpacked batch carries a `frame_mask`, so the prefix
route's attention takes an explicit dense mask rather than the unmasked causal
fast path (or the flex backend's block-sparse mask) for that run, and never
flips between the two. On `cross_attention` a text-only row skips the
cross-attention blocks: their attention and MLP updates are zeroed for that
row, so its text runs through the backbone alone. On `mot` the hidden blank
image's tokens still pass through the image experts. On `moma` they are dropped
before routing, so they take no expert capacity; a micro-batch with no image
gives the image experts zero gradients. On `joint_decoder` and `moma` the frame
mask reaches attention under `model.sdpa_backend`, whose `"flash"` kernel takes
no mask on CUDA, so on CUDA an unpacked dataset with text-only rows and
`data.hf_max_images = 1` is refused with `model.sdpa_backend = "flash"` under
either `attention_backend`, `"sdpa"` or `"flex"`; on CPU it is not.

## Documents with several images

An image column that holds a list of images is a multi-image document.
`data.hf_max_images = n` (`n >= 2`) accepts up to `n` of them per document, in
list order. A row with more than `n` images, at any cap including the
default of `1`, is skipped when the dataset is loaded, and one warning counts
the skipped rows. The images' tokens go before the document's text, each image
in its own block, which is the layout packing uses; with `vlm.tile_grid` set, each
image contributes all of its views in order. With `data.pack_sequences = true`
such documents pack like any other; without it the same packer lays each
document out in its own row of `train.seq_len`, so the token tensors keep one
shape across micro-batches and ranks (the image tensor still holds the batch's
image count). Both need `vlm.arch = "joint_decoder"` and not
`train.mixed_precision = "fp8"`, nor `model.sdpa_backend = "flash"` under
`attention_backend = "sdpa"`; and `n` images' tokens (every view of each) plus `vlm.max_text_len` have to fit `train.seq_len` and
`model.max_seq_len`. The config checks that when `vision_encoder.num_tokens` is
set; the data pipeline checks it again against the built model's token count
before training starts, with packing or without, so a probed encoder is
covered too.

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
and the optimizer starts fresh. Stage 2 starts from a stage-1 checkpoint as
weights alone with `checkpoint.exclude_from_loading = ["optimizer", "train_state"]`,
so its step and schedule start again from 0.

## Evaluation tasks and scorers

`evaluation/` holds what an evaluation run is made of: the tasks to run and the
scorers of their answers. The evaluator that runs them on a checkpoint is on the
roadmap.

- `tasks.py`: `load_tasks` reads a TOML task file into frozen `EvalTask`s. A task
  has a unique `name`, a `reader` (one of the `testbed.py` loaders) with that
  loader's arguments in `reader_args`, a `scorer`, and `max_new_tokens`, the cap
  on a decoded answer. `reader_args` is a read-only copy. The file is checked as
  it loads, before a reader opens anything: an unknown key, reader or scorer, an
  unknown or missing reader argument, a required reader parameter that cannot be
  given by keyword, and a value of the wrong TOML type for a parameter annotated
  `str`, `Path`, `bool`, `int` or `float` (alone or with `None`) are refused with
  the valid ones listed. Nothing else is judged at load: other annotations,
  nested values and every limit on a value (a count of at least 1, say) are
  checked by the reader itself when `build_reader` constructs it. A relative
  path given to a reader parameter annotated `Path`, as the testbed loaders'
  `annotation_file` and `image_root` are, is taken from the task file's
  directory, so a task file reads the same data from any working directory.
- `scorers.py`: a scorer rates one prediction against a sample's reference
  answers with a value in [0, 1], and a task's score is the mean over its
  samples. The text scorers take the prediction as a `str` and the references as
  a non-empty sequence of `str`; anything else raises before any scoring, naming
  the argument. `get_scorer` returns one by name; its `score` rates a sample and its
  `aggregate` turns a task's sample scores into the task's metrics.
  - `vqa_accuracy` (VQAv2, TextVQA): each annotator is left out in turn, the
    prediction scores `min(1, m / 3)` against the `m` remaining annotators who
    gave the same answer, and the score is the mean over those rounds, after the
    official answer normalization (`vqa_normalization.py`) of the prediction and
    of every reference, whether or not the references agree. With ten references,
    three matching annotators score 0.9 and four score 1. Four or more references
    can reach 1, and a single reference scores 0, as in the official evaluation,
    so data with one answer per question needs another scorer.
  - `anls` (DocVQA): the similarity `1 - NL` to the closest reference, where `NL`
    is the Levenshtein distance over the longer length, and 0 once `NL` is above
    0.5. A similarity of exactly 0.5 scores 0.5, as in the docvqa.org evaluation
    script and lmms-eval; this departs from the recipe's `DOCVQAScorer`, which
    scores it 0 (`nl < tau`). Case is ignored, and whitespace is collapsed
    first: none at either end, and each run inside one space.
  - `relaxed_accuracy` (ChartQA): 1 for a number within 5 percent of a numeric
    reference, or for text equal to a reference but for case, surrounding
    whitespace and trailing `?`, `.` and `!`, applied in that order, so
    `"yes !"` becomes `"yes "`. A number is what `float` parses, and a trailing
    `%` makes hundredths, as in Pix2Struct's relaxed correctness: `45%` matches
    `0.45` and `45.1%` but not `45`. `1,000` is text, while `1_000` is 1000.

`configs/eval_tasks.toml` is a sample with five tasks and their caps (VQAv2 10,
TextVQA 75, DocVQA 50, ChartQA human and machine 50); its paths are
placeholders:

```toml
[[tasks]]
name = "docvqa"
reader = "docvqa"
scorer = "anls"
max_new_tokens = 50

[tasks.reader_args]
annotation_file = "path-to-docvqa-annotations"
image_root = "path-to-docvqa-images"
```

A reader is an entry of `READERS` and a scorer an entry of `SCORERS`, so a new
one is added there. A scorer may report several named components per sample and
aggregate them its own way; the default is the mean of each component.

## Roadmap

What the recipe still needs. Each lands as its own change in this directory,
or in core where it belongs:

- Training sources (COCO, VQAv2, TextVQA, DocVQA, ChartQA, PixMo, FineVision,
  DataComp), chat-template rendering with the loss on the assistant turns only,
  and the mixture loader
- Typed stage contract and pinned source manifest, which fix the stage presets
- Per-module learning rates and schedules
- Evaluation: the evaluator that runs the tasks on a saved checkpoint, periodic
  evaluation during training, PixMo point and count scorers, documents with
  several images or none, and image attention when decoding
- Run-comparison metrics: documents seen, token counts, MFU with the encoder

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU; the renderer tests download
the `Qwen/Qwen3-0.6B` tokenizer once:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
