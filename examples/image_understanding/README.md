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
├── evaluate_checkpoint.py   # evaluation entry point (below)
├── configs/
│   ├── eval_tasks.toml   # sample evaluation tasks, with placeholder paths
│   └── evaluate_checkpoint.sbatch   # sample job script for evaluating one checkpoint
├── evaluation/           # checkpoint evaluator, periodic evaluation, tasks and scorers (below)
├── image_text/           # image-text dataset readers and chat renderer (below)
└── tests/
    ├── test_image_understanding_entry.py   # the entry point loads a config and hands it off
    ├── evaluation/       # evaluator, periodic evaluation, task-file and scorer tests (CPU)
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

Every sample the renderer builds carries one image, resized square to `image_size`,
with no tiling or packing.

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

## Evaluate a checkpoint

`evaluate_checkpoint.py` decodes an answer for every row of each task's split
with a saved checkpoint and scores it. It takes the config the checkpoint was
trained with and the overrides that run used, a run directory (its latest
checkpoint is evaluated) or one step directory, a task file (next section) and
an output directory:

```bash
uv run torchrun --nproc_per_node=4 examples/image_understanding/evaluate_checkpoint.py \
    <config.toml> --checkpoint /your/run/dir --tasks <tasks.toml> \
    --output /your/results/dir --batch-size 32
```

The model is built from the config and loads the checkpoint's weights; every
rank holds all of it, whatever parallelism the run trained with. The language
model and the adapter compute in the dtype of `train.mixed_precision`, the image
encoder in `vision_encoder.mixed_precision` when that is set and otherwise in
the dtype it loads in. The decode takes a `joint_decoder` that sees one view of
one image, so another arch, a `[video]` config or `vlm.tile_grid` is refused
before anything is built. So is `model.moe_capacity_factor > 0`, which drops
tokens by what else is in the batch.

The inputs are training's. Each document goes through the training
`ChatRenderer`: the prompt is the text it puts before the assistant's reply, and
the image is preprocessed as in a training sample. A document needs one image
and one turn; any other stops the run with the limit named
(`evaluation/inputs.py` is where a document becomes model inputs).

Decoding is greedy. A batch is laid out like a training row (image tokens, then
text, padding last), each image is encoded once, and a row ends at the turn end
or EOS, or after the task's `max_new_tokens`. Attention is causal, so an answer
does not depend on the rows decoded with it: the scores are the same for any
`--batch-size` and any number of GPUs. A split is cut into consecutive batches,
the ranks take them in turn, and rank 0 merges the rows, each scored once.

A row is scored as it is or counted under a reason, never changed to fit:

- `unreadable_row`: the reader passed over the row because its image does not
  decode. The row it moved on to is not scored in its place.
- `prompt_too_long`: the prompt and `max_new_tokens` do not fit
  `vlm.max_text_len`, the text length training lays out.

A task's rows are the rows its reader yields: a reader that drops rows when it
loads a split (a row with no question or no answer, for one) does so before the
evaluation and does not count them. A reader that passes over rows while it is
read must say so, either through `read_with_index`, which returns the index of
the row it read with the document, or through a `skipped` count that moves
whenever it passes over a row. A reader that gives up after a run of unusable
rows raises `NoUsableRowError` (what `first_usable` raises); each such row is
counted as `unreadable_row`, and any other error fails the run.

The prediction that is scored is the generation decoded with special tokens
skipped and nothing else cleaned up; the scorers normalize their own input. The
predictions file keeps the raw `generation` beside it.

A task's metrics are over its scored rows, with the counts beside them. A
missing image file, a task whose cap leaves no room for any prompt, or a task
with no scored row fails the run.

```
/your/results/dir/
├── predictions/<task>.jsonl   # per row: question, references, generation, prediction, score
├── metrics.json               # <task>/<metric> to value
└── results.json               # per task: metrics, rows (its reader's), samples, skips by reason
```

`results.json` is removed when an evaluation starts, before the config and the
task file are loaded, and written last, so it is there only for an evaluation
that finished; it names the step directory it
evaluated. When that checkpoint's `metadata.json` records an MLflow run, rank 0
then appends the metrics to it as `eval/<task>/<metric>` at the checkpoint's
step. `--no-publish` turns that off, and a failure to publish is a warning with
the command that publishes `metrics.json` later, never a failed evaluation.

## Evaluate during training

`train.py` can evaluate every checkpoint the run saves, as the run goes on. It is
off unless `--eval-tasks`, `--eval-command` and `--eval-output` are all given, and
training is then exactly what it is without them:

```bash
uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
    --eval-tasks <tasks.toml> --eval-output /your/results/root \
    --eval-command "sbatch --job-name=eval-{step} \
        examples/image_understanding/configs/evaluate_checkpoint.sbatch \
        {config} {checkpoint} {tasks} {output} {overrides}"
```

`--eval-command` is a template for the command that evaluates one checkpoint.
Its placeholders are `{config}` and `{tasks}` (the two files above), `{checkpoint}`
(the step directory), `{step}`, `{output}` (that step's directory under
`--eval-output`, `step_<N>`) and `{overrides}`, which stands for the run's
`--section.key=value` overrides as separate arguments, so the evaluation builds the
model the run trained. `{checkpoint}` and `{output}` are required. The template is
split into arguments first and the placeholders filled in afterwards, so a path with
a space is still one argument, and it is run without a shell. A SLURM user passes an
`sbatch` line, as above, so each evaluation is its own job that outlives the run;
`examples/image_understanding/configs/evaluate_checkpoint.sbatch` is a job script
to copy, with placeholders for the partition, account and environment. Anyone else
passes a local script, which is started detached and keeps running when the run
ends. A local evaluation shares the training GPUs and slows training while it runs;
a scheduler job runs on GPUs of its own and does not.

Everything is checked before the run builds anything: the task file, the config (the
evaluator decodes the same runs as above), the template and its program, and the
output root. A mistake stops the run with the valid options named.

The hook is rank 0's alone and training never waits on it: apart from one look at
the checkpoint directory when training begins, the training thread only puts a
message on a queue, and a worker thread does everything else, so no filesystem call
or scheduler call is on the step path. A saved step is launched
once its directory is complete on disk, which is when the DCP `.metadata`,
`train_state.pt` and `metadata.json` are all there, and not before: with
asynchronous checkpointing the directory is still being written when the loop moves
on, so the worker checks for it again every few seconds, with three `stat` calls,
and once more when the loop ends, after it has drained the last save. A launch is
started and left, detached and without the training run's rendezvous variables
(`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `SLURM_PROCID`, `SLURM_NTASKS`
and the other per-process and `TORCHELASTIC_*` ones), so an
evaluator or an `sbatch` job starts its own process group. Every other variable is
inherited, the rest of the run's `SLURM_*` variables and `CUDA_VISIBLE_DEVICES`
included: an `sbatch` job gets its own from the scheduler, but a local script
shares the run's, so it should set the device it evaluates on. The command's exit status is only read later, without waiting,
to warn about a command that failed. A checkpoint the shutdown handler saves
outside the loop is found by the last look and by the next run's scan.

A step is launched at most once. Before launching, the worker writes
`launched.json` into the step's output directory, exclusively, and skips a step
that has it or a `results.json`; when a run resumes, it also takes up the complete
steps already on disk, so what an earlier incarnation saved and never started is
evaluated and nothing is evaluated twice. `launch.log` holds the command's output
(the job id, for `sbatch`). A step that was pruned before it was complete, or never
completed, is skipped with a log line, and a launch that cannot start is a warning
that releases its `launched.json`. A command that starts and then exits non-zero is
a warning too, and its step is not started again: look for a step directory without
`results.json`, and delete its `launched.json` to have the next run start it.
Once a command has started its step stays launched whatever fails afterwards. A step
whose directory existed before the run and that the run may save again (step 0 of a
dynamic checkpoint window) is evaluated once a later step has been saved, or at the
end of training, never while it may be half rewritten.

The evaluator reads a step directory while the run goes on, and the run removes old
ones (`checkpoint.keep_last_n`). Keep as many as the evaluations in flight need,
or the evaluator stops on a directory that is gone.

## Evaluation tasks and scorers

`evaluation/` also holds what an evaluation run is made of: the tasks to run and
the scorers of their answers.

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
  checked by the reader itself when `build_reader` constructs it.
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
    of every reference. With ten references, three matching annotators score 0.9
    and four score 1. A sample needs at least two references: with one the score
    could only be 0, so `vqa_accuracy` raises `ValueError` instead, and data with a
    single answer per question needs another scorer. Four or more can reach 1.
  - `anls` (DocVQA): the similarity `1 - NL` to the closest reference, where `NL`
    is the Levenshtein distance over the longer length, and 0 once `NL` reaches
    0.5. Case and surrounding whitespace are ignored.
  - `relaxed_accuracy` (ChartQA): 1 for a number within 5 percent of a numeric
    reference, or for text equal to a reference but for case, surrounding
    whitespace and trailing `?`, `.` and `!`, applied in that order, so
    `"yes !"` becomes `"yes "`. A number is what `float` parses, so `45%` and
    `1,000` are text while `1_000` is 1000.

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
- Starting stage 2 from stage 1's weights alone, with a fresh step and schedule
- Text-only documents in image-text batches
- Documents with several images
- Per-module learning rates and schedules
- Vision-encoder precision separate from the backbone's
- Evaluation: PixMo point and count scorers, documents with several images or
  none, and image attention when decoding
- Run-comparison metrics: documents seen, token counts, MFU with the encoder

## Tests

The tests live outside the repo's `tests/` tree (the main suite's
`testpaths`), so run them by path. They need no GPU; the renderer
tests download the `Qwen/Qwen3-0.6B` tokenizer once:

```bash
uv run pytest examples/image_understanding/tests
```

CI runs them on pushes and PRs that touch `examples/image_understanding/`.
