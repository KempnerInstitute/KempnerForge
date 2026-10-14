# VLM example

Vision-language training — images **or** video — on the core `Transformer`. A
frozen HF vision encoder produces visual tokens, a connector projects (and
optionally pools) them, and an arch-specific path feeds the backbone. Everything
here is configuration, entry points and their tests; `kempnerforge/` never
imports it, so this directory can be deleted without touching the core.

## Layout

```
examples/vlm/
├── README.md
├── train.py              # training entry point
├── eval.py               # evaluation CLI: constructs the adapter, runs simple_evaluate
├── lmms_adapter.py       # KempnerForgeVLM: lmms-eval chat model over a DCP checkpoint
├── configs/              # TOML presets (below)
├── scripts/              # run-only utilities
│   └── prep_vlm_coco.py  # COCO-Karpathy dataset prep
└── tests/
    ├── test_configs.py   # every preset loads and validates; the training entry point
    └── eval/
        ├── conftest.py   # path bootstrap + tiny VLM fixtures
        ├── integration/  # real-lmms-eval tests (skip when it is absent)
        │   ├── test_vlm_eval.py            # DCP-roundtrip generate_until (image + video)
        │   └── test_lmms_eval_contract.py  # pins the real API the unit fake imitates
        └── unit/
            ├── __init__.py               # load-bearing package marker (see its docstring)
            ├── conftest.py               # injects the hermetic fake lmms_eval
            ├── _fake_lmms_eval.py
            ├── test_adapter.py           # CPU unit tests for the adapter (fake lmms_eval)
            └── test_import_isolation.py  # `import kempnerforge` must not need lmms-eval
```

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

Tests: see [Tests](#tests) (they are outside the core
`testpaths`, so run them by path).

## Data prep

`scripts/prep_vlm_coco.py` writes a COCO-Karpathy `save_to_disk` directory for
`data.hf_dataset_name` to point at.

## Evaluation

`eval.py` evaluates a checkpoint on any standard multimodal benchmark (MMMU,
MMBench, ScienceQA, SEED, AI2D, …) through the
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness.

A custom lmms-eval *chat model*, `KempnerForgeVLM` in
[`lmms_adapter.py`](lmms_adapter.py), loads `VLMWrapper` directly from the DCP
checkpoint. [`eval.py`](eval.py) constructs `KempnerForgeVLM` itself and passes
the instance to `simple_evaluate(model=...)` — there is no lmms-eval
entry-point registration, and the core `kempnerforge` package carries no
lmms-eval-facing code. For text-model evaluation (loss/perplexity and the
`lm-eval` harness this example parallels), see
[Run evaluation](../../docs/how-to/run-evaluation.md).

### Install lmms-eval

`lmms-eval` is an **optional dependency** and is intentionally NOT declared in
`pyproject.toml`. Install it into your environment before running:

```bash
uv pip install lmms-eval
```

lmms-eval stays out of the core package entirely: the adapter lives here in the
example and is imported only by `eval.py` (or the tests), so
`import kempnerforge` works without lmms-eval installed.

**Video evaluation** additionally needs the `av` (PyAV) video-decoding
dependency, which ships in the optional `video` group:

```bash
uv sync --group video
```

PyAV's manylinux wheel bundles FFmpeg, so no system FFmpeg or CUDA libraries are
required. (Image-only evaluation does not need this group.)

### Usage

```bash
# One task, write results JSON
uv run python examples/vlm/eval.py \
    --config     examples/vlm/configs/vlm_7b.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      mmmu_val \
    --output     results/vlm_step_10000.json

# Several tasks, quick partial run (4 examples per task)
uv run python examples/vlm/eval.py \
    --config     examples/vlm/configs/vlm_7b.toml \
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

### Flags

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

### Video evaluation

When `--config` is a **video checkpoint** (its TOML has a `[video]` section), the
harness evaluates lmms-eval *video* `generate_until` tasks: each request's video
is decoded into frames and fed to the model as a single clip. This needs the `av`
video group (see [Install lmms-eval](#install-lmms-eval)).

```bash
uv run python examples/vlm/eval.py \
    --config     examples/vlm/configs/vlm_video_webvid.toml \
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
  (`joint_decoder` / `cross_attention` / `mot`). A single **image** task also runs
  on a video checkpoint — the image is treated as a 1-frame clip, zero-padded to
  `frames_per_clip`. Multiple videos, mixed image+video, multiple images, audio,
  and multi-turn / few-shot raise a clear error; MoMa still fails fast. An
  **image** checkpoint cannot evaluate video and raises a clear error if handed a
  video task.

### Limitations

Several are tracked follow-ups.

- **Single GPU.** v1 runs on one GPU. Data-parallel
  multi-GPU is a localized
  future addition; sharded/model-parallel inference for models too large for one
  GPU is a larger, separate effort.
- **MoMa is not supported.** The `moma` arch uses non-causal expert-choice
  routing and cannot autoregressively generate, but eval tasks are
  generation-only. A MoMa checkpoint fails fast with a clear error. Joint-Decoder
  (`joint_decoder`), Cross-Attention (`cross_attention`), and MoT (`mot`) are
  supported.
- **One visual per request; no multi-turn / few-shot / multi-image.** A request
  carries exactly one image (image checkpoint) or one video (video checkpoint —
  see [Video evaluation](#video-evaluation)). Audio, multiple images, multiple
  videos, mixed image+video, and multi-turn / few-shot requests raise a clear
  error. Multi-image and multi-turn/few-shot are tracked follow-ups (for chat
  tasks lmms-eval delivers few-shot as extra content blocks/turns, so it reduces
  to multi-image + multi-turn support).
- **Prompt flattening discards structure.** Flattening drops role/turn structure
  and any model-specific chat template. KempnerForge pre-training uses no chat
  template; once a post-training format exists, repo-wide chat-template support
  should be added and the rendering step made configurable.
- **No KV cache.** Decoding re-runs the full transformer over the growing sequence
  each step (KempnerForge has no image-conditioned KV-cache decode path); this is
  correct but costs extra compute, and a KV-cache decode is future work. Raising
  `--batch-size` decodes multiple requests together
  (right-padded, grouped by `gen_kwargs`) to amortize the per-step transformer cost.

### Cluster environment notes

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
  `LD_LIBRARY_PATH=<conda-env>/lib uv run python examples/vlm/eval.py …`.
  The integration tests import the evaluator too and need the same workaround.

### See also

- [Run evaluation](../../docs/how-to/run-evaluation.md) — text-model
  loss/perplexity and the `lm-eval` harness this example parallels.
- [End-to-end training run](../../docs/how-to/end-to-end-training-run.md) —
  produces the checkpoints `eval.py` consumes.

## Tests

The tests live outside the repo's `tests/` tree (the main suite's `testpaths`),
so run them by path, as **two separate sessions**:

```bash
# Hermetic: the config tests and the evaluation unit tests
# (a fake lmms_eval is injected; no GPU or network)
uv run pytest examples/vlm/tests --ignore=examples/vlm/tests/eval/integration

# Integration: needs real lmms-eval installed; skips otherwise
uv run pytest examples/vlm/tests/eval/integration
```

CI runs the hermetic session on pushes and PRs that touch `examples/vlm/`; the
integration tier stays manual, since real lmms-eval is not installed there.

Keep the integration tier out of the hermetic session: in a combined run the unit
conftest's injected fake replaces the `lmms_adapter` module in `sys.modules` after
the integration modules have bound the real one, so the monkeypatch-based
integration tests patch the wrong module object and fail.

The opt-in end-to-end test runs a small slice of a real task against a real
checkpoint (GPU node):

```bash
KF_VLM_EVAL_CONFIG=/path/to/train_config.toml \
KF_VLM_EVAL_CHECKPOINT=/path/to/checkpoints/step_N \
KF_VLM_EVAL_TASK=mmmu_val \
uv run pytest examples/vlm/tests/eval/integration/test_vlm_eval.py -k real_task
```
