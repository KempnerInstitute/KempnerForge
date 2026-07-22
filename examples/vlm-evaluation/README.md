# VLM evaluation via lmms-eval

Evaluate a KempnerForge vision-language model (VLM) checkpoint on any standard
multimodal benchmark (MMMU, MMBench, ScienceQA, SEED, AI2D, …) by integrating
the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness.

A custom lmms-eval *chat model* loads `VLMWrapper` directly from the DCP
checkpoint. The pieces are [`adapter.py`](adapter.py) (the `KempnerForgeVLM`
adapter) and [`vlm_eval_harness.py`](vlm_eval_harness.py) (the CLI). The
harness constructs `KempnerForgeVLM` itself and passes the instance to
`simple_evaluate(model=...)` — there is no lmms-eval entry-point registration,
and the core `kempnerforge` package carries no lmms-eval-facing code. For
text-model evaluation (loss/perplexity and the `lm-eval` harness this example
parallels), see [Run evaluation](../../docs/how-to/run-evaluation.md).

## Contents

```
examples/vlm-evaluation/
├── README.md
├── adapter.py              # KempnerForgeVLM: lmms-eval chat model over a DCP checkpoint
├── vlm_eval_harness.py     # CLI: constructs the adapter, runs simple_evaluate
└── tests/
    ├── conftest.py         # path bootstrap + tiny VLM fixtures
    ├── integration/        # real-lmms-eval tests (skip when it is absent)
    │   ├── test_vlm_eval.py             # DCP-roundtrip generate_until (image + video)
    │   └── test_lmms_eval_contract.py   # pins the real API the unit fake imitates
    └── unit/
        ├── __init__.py     # load-bearing package marker (see its docstring)
        ├── conftest.py     # injects the hermetic fake lmms_eval
        ├── _fake_lmms_eval.py
        └── test_adapter.py # CPU unit tests for the adapter (fake lmms_eval)
```

## Install lmms-eval

`lmms-eval` is an **optional dependency** and is intentionally NOT declared in
`pyproject.toml`. Install it into your environment before running:

```bash
uv pip install lmms-eval
```

lmms-eval stays out of the core package entirely: the adapter lives here in the
example and is imported only by the harness (or these tests), so
`import kempnerforge` works without lmms-eval installed.

**Video evaluation** additionally needs the `av` (PyAV) video-decoding
dependency, which ships in the optional `video` group:

```bash
uv sync --group video
```

PyAV's manylinux wheel bundles FFmpeg, so no system FFmpeg or CUDA libraries are
required. (Image-only evaluation does not need this group.)

## Usage

```bash
# One task, write results JSON
uv run python examples/vlm-evaluation/vlm_eval_harness.py \
    --config     configs/train/vlm_jd.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      mmmu_val \
    --output     results/vlm_step_10000.json

# Several tasks, quick partial run (4 examples per task)
uv run python examples/vlm-evaluation/vlm_eval_harness.py \
    --config     configs/train/vlm_jd.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      mmmu_val,mmbench_en_dev,scienceqa_img \
    --limit      4
```

`--config` is the same KempnerForge TOML the checkpoint was trained with (it
carries the vision encoder, adapter, `vlm.arch`, and tokenizer settings).
`--checkpoint` accepts either a run directory (the latest `step_N` is resolved
automatically) or a specific `step_N` directory.

There is **no default task suite** — `--tasks` is required. A representative
default benchmark set is still being decided.

## Multi-GPU (data parallel)

A single benchmark can be **data-parallelized across GPUs**: launch the harness with
`accelerate launch --num_processes N` and lmms-eval shards the benchmark's documents
`[rank::world_size]` across the N processes, gathering results onto rank 0. Each rank loads a
**full replica** of the model on its own GPU, so throughput scales ~N× with identical scores.
`accelerate` ships with lmms-eval, so there is nothing extra to install.

```bash
accelerate launch --num_processes 4 examples/vlm-evaluation/vlm_eval_harness.py \
    --config     configs/train/vlm_jd.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      mmmu_val \
    --output     results/vlm_step_10000.json
```

- **No new flags.** DP is entirely launcher-driven — the adapter auto-detects the run from the
  `WORLD_SIZE` / `LOCAL_RANK` environment variables the launcher sets and binds each rank to
  `cuda:LOCAL_RANK`. Plain `uv run python vlm_eval_harness.py …` (no launcher) is unchanged
  single-GPU.
- **Pass `--device cuda` without an index.** Per-rank GPU binding triggers only on the bare
  `cuda`; an explicit `--device cuda:0` would pin *every* rank to GPU 0.
- **Replication, not model-parallel.** Every rank holds a full copy, so aggregate GPU memory is
  N× — the model must fit on one GPU. Sharded inference for larger models is separate future
  work (see [Limitations](#limitations)).
- **`--limit N` stays a global cap** (the per-rank shards union to `N` docs total); `--batch-size`
  is per rank.
- **Only rank 0 writes** the `--output` JSON; the other ranks score their shard and exit.
- On the Kempner cluster, request the GPUs in your allocation (`--gres=gpu:N` on one node) and keep
  the same `LD_LIBRARY_PATH` / `HF_HOME` environment as a single-GPU run (see
  [Cluster environment notes](#cluster-environment-notes)).

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--config` | — (required) | KempnerForge TOML the checkpoint was trained with |
| `--checkpoint` | — (required) | DCP checkpoint dir (run dir or `step_N` dir) |
| `--tasks` | — (required) | comma-separated lmms-eval task names |
| `--limit` | `None` | cap examples per task (int count, or `<1.0` fraction) |
| `--output` | `None` | save full JSON results |
| `--device` | `cuda` | inference device |
| `--dtype` | `None`(maps to model config setting) | model dtype |
| `--batch-size` | `1` | requests decoded together (grouped by `gen_kwargs`) |
| `--max-new-tokens` | `128` | fallback only; task `gen_kwargs` override it |

## Experiment tracking

Eval results can be logged through the framework's metrics backends — the same
`MetricsTracker` training uses (`kempnerforge/metrics/tracker.py`), so eval
inherits every backend the framework has (WandB, TensorBoard; MLflow once PR
#159 lands). Tracking is off by default and enabled with the same config flags
as training, forwarded as dotted overrides:

```bash
uv run python examples/vlm-evaluation/vlm_eval_harness.py \
    --config     configs/train/vlm_jd.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      mmmu_val \
    --metrics.enable_wandb=true --metrics.wandb_project=vlm-eval
```

Any unrecognized `--section.key=value` argument is layered over `--config` by
the KempnerForge config loader (unknown keys fail fast). TensorBoard works the
same way: `--metrics.enable_tensorboard=true --metrics.tensorboard_dir=...`.

- **Results land in the checkpoint's training run.** The harness reads the
  `wandb_run_id` training saved into the checkpoint's `train_state.pt` and
  resumes that run, so eval metrics sit next to the training curves and the run
  id can never be wrong. When the checkpoint has none (training ran with
  tracking off), a fresh run named `<run>-<step_N>` starts with a warning;
  attach to a specific run with `--metrics.wandb_run_id=<id>`.
- **Logged at the checkpoint's training step**, as one flat dict:
  `eval/benchmarks/agg/<benchmark>` (the aggregate, normalized into [0, 1] via
  the manifest), `eval/benchmarks/raw/<task>/<metric>` (every numeric metric of
  every task and subtask, unnormalized — nothing thrown away), and
  `eval/benchmarks/throughput/<task>/...`.
- **Per-benchmark knowledge lives in
  [`benchmark_manifest.py`](benchmark_manifest.py).** Each benchmark registers
  its authoritative aggregate metric and score range. An unregistered benchmark
  falls back to result metadata with a loud warning that includes the exact
  registry line to paste — add it when you add a benchmark. `egoschema` is
  submission-only and logs no aggregate by design.
- A tracking failure (missing wandb, no network, bad credentials) never fails a
  completed eval — it warns and moves on.

## Video evaluation

When `--config` is a **video checkpoint** (its TOML has a `[video]` section), the
harness evaluates lmms-eval *video* `generate_until` tasks: each request's video
is decoded into frames and fed to the model as a single clip. This needs the `av`
video group (see [Install lmms-eval](#install-lmms-eval)).

```bash
uv run python examples/vlm-evaluation/vlm_eval_harness.py \
    --config     configs/train/vlm_video_webvid.toml \
    --checkpoint checkpoints/vlm_video/step_10000 \
    --tasks      <a video generate_until task> \
    --limit      4
```

- **The frame budget is a property of the checkpoint, not a flag.** Frames are
  sampled by the model's own `[video]` policy (`fps` / `min_frames` /
  `max_frames`, the Molmo2 uniform `sample_timestamps`) and fixed to exactly
  `max_frames` (zero-padded when a clip yields fewer). You cannot change it at
  eval time — the transformer was built around `frames_per_clip = max_frames`.
  Comparability to externally published video-benchmark numbers therefore depends
  on the checkpoint's frame budget matching the reference's, which is a training
  choice rather than a knob here.
- **Scope.** One video per request, single-turn, zero-shot, generative arches
  (`joint_decoder` / `cross_attention` / `mot`). **Image** tasks also run on a
  video checkpoint — one image is a 1-frame clip, and multiple images are packed
  as an ordered clip (zero-padded, and truncated with a warning past
  `frames_per_clip`). Multiple videos, mixed image+video, audio, and multi-turn /
  few-shot raise a clear error; MoMa still fails fast. An **image** checkpoint
  cannot evaluate video and raises a clear error if handed a video task.

## Text-only evaluation

Text-only `generate_until` benchmarks (e.g. GSM8K, IFEval) run on **both image and
video checkpoints**, for the generative arches (`joint_decoder` / `cross_attention` /
`mot`). A request with no image or video renders as an empty-frame prompt and runs
the arch's **pure-text forward** — no vision encoder, no image prefix (JD/MoT), and
cross-attention blocks skipped (CA) — so the number reflects the text backbone. This
is how you measure how much VLM training drifted the base LM, in the same harness as
the multimodal tasks.

```bash
uv run python examples/vlm-evaluation/vlm_eval_harness.py \
    --config     configs/train/vlm_jd.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      gsm8k \
    --limit      8
```

- **Scope.** `generate_until` tasks only (generation / answer-extraction).
  `loglikelihood`-scored multiple-choice suites (ARC, HellaSwag, MMLU-style) are not
  supported — the adapter is generation-only. MoMa is excluded (non-generative).
  Text-only, image, and video requests may be freely mixed across a task suite; each
  request is decoded by its own modality path (text-only and visual requests within a
  batch are decoded as separate sub-batches).

## Limitations

Several are tracked follow-ups.

- **Data parallel, replicated.** Multi-GPU runs shard the benchmark's documents across GPUs
  via `accelerate launch --num_processes N` (see
  [Multi-GPU (data parallel)](#multi-gpu-data-parallel)); each rank holds a **full replica** of
  the model. Sharded / model-parallel inference for models too large for a single GPU is a
  larger, separate effort.
- **MoMa is not supported.** The `moma` arch uses non-causal expert-choice
  routing and cannot autoregressively generate, but eval tasks are
  generation-only. A MoMa checkpoint fails fast with a clear error. Joint-Decoder
  (`joint_decoder`), Cross-Attention (`cross_attention`), and MoT (`mot`) are
  supported.
- **One visual per request on image checkpoints; no multi-turn / few-shot.** An
  image checkpoint carries exactly one image per request (multiple images raise); a
  video checkpoint carries one video, or one or more images packed as an ordered
  clip (see [Video evaluation](#video-evaluation)). Audio, multiple videos, mixed
  image+video, and multi-turn / few-shot requests raise a clear error. Multi-turn /
  few-shot is a tracked follow-up (for chat tasks lmms-eval delivers few-shot as
  extra content blocks/turns, so it reduces to multi-turn support).
- **Prompt flattening discards structure.** Flattening drops role/turn structure
  and any model-specific chat template. KempnerForge pre-training uses no chat
  template; once a post-training format exists, repo-wide chat-template support
  should be added and the rendering step made configurable.
- **No KV cache.** Decoding re-runs the full transformer over the growing sequence
  each step (KempnerForge has no image-conditioned KV-cache decode path); this is
  correct but costs extra compute, and a KV-cache decode is future work. Raising
  `--batch-size` decodes multiple requests together
  (right-padded, grouped by `gen_kwargs`) to amortize the per-step transformer cost.

## Cluster environment notes

Installing lmms-eval pulls in extra packages that can clash with a CUDA-pinned
PyTorch. Two gotchas seen on the Kempner cluster:

- **torchvision must match the CUDA build of torch.** The default-index
  `torchvision` is ABI-incompatible with `torch …+cu128` (it fails
  `register_fake("torchvision::nms")`, which breaks `import lmms_eval`). Install
  the matching build from the same index:

  ```bash
  uv pip install --reinstall-package torchvision \
      --index https://download.pytorch.org/whl/cu128 "torchvision==0.26.0"
  ```

- **`GLIBCXX_… not found` when importing the evaluator.** lmms-eval's
  `simple_evaluate` pulls in a library that needs a newer `libstdc++` than the
  system one. Put a newer `libstdc++` first on the library path, e.g.
  `LD_LIBRARY_PATH=<conda-env>/lib uv run python examples/vlm-evaluation/vlm_eval_harness.py …`.

## Run the tests

Standalone (not part of the main `tests/` suite), as two separate invocations:

```bash
# Hermetic unit tests — always run (a fake lmms_eval is injected; no GPU/network)
uv run pytest examples/vlm-evaluation/tests/unit

# Integration tests — need real lmms-eval installed; skip otherwise
uv run pytest examples/vlm-evaluation/tests/integration
```

Keep the two directories in separate pytest sessions: in a combined run the unit
conftest's injected fake replaces the `adapter` module in `sys.modules` after the
integration modules have bound the real one, so the monkeypatch-based integration
tests patch the wrong module object and fail (the in-tree layout this example was
extracted from behaved the same way for a combined `pytest tests/` run).

The opt-in end-to-end test runs a small slice of a real task against a real
checkpoint (GPU node):

```bash
KF_VLM_EVAL_CONFIG=/path/to/train_config.toml \
KF_VLM_EVAL_CHECKPOINT=/path/to/checkpoints/step_N \
KF_VLM_EVAL_TASK=mmmu_val \
uv run pytest examples/vlm-evaluation/tests/integration/test_vlm_eval.py -k real_task
```

## See also

- [Run evaluation](../../docs/how-to/run-evaluation.md) — text-model
  loss/perplexity and the `lm-eval` harness this example parallels.
- [End-to-end training run](../../docs/how-to/end-to-end-training-run.md) —
  produces the checkpoints this harness consumes.
