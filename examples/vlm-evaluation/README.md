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

> [!WARNING]
> **Kempner cluster: `libstdc++` Version:**
> `lmms-eval` imports packages that require version `GLIBCXX_3.4.30` or newer.
> If you experience `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`,
> please point the system towards a newer version with e.g.,
>
> ```bash
> export LD_LIBRARY_PATH=/n/sw/Miniforge3-25.3.1-0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  
> ```
>
> See [Cluster environment notes](#cluster-environment-notes) for more details.

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

`--output` saves the task results at the specified path, including KempnerForge
model metadata and job config to make runs identifiable and reproducible.

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--config` | — (required) | KempnerForge TOML the checkpoint was trained with |
| `--checkpoint` | — (required) | DCP checkpoint dir (run dir or `step_N` dir) |
| `--tasks` | — (required) | comma-separated lmms-eval task names |
| `--limit` | `None` | cap examples per task (int count, or `<1.0` fraction) |
| `--output` | `None` | save full JSON results |
| `--device` | `cuda` | inference device |
| `--dtype` | `None`(defaults to `train.param_dtype`) | model dtype |
| `--batch-size` | `1` | requests decoded together (grouped by `gen_kwargs`) |
| `--max-new-tokens` | `128` | fallback only; task `gen_kwargs` override it |
| `--override` | `None` | KempnerForge `SECTION.KEY=VALUE` config override merged over the TOML (repeatable), e.g. `--override frame_selector.type=qframe` |

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
  `max_frames`, the uniform `sample_timestamps`) and fixed to exactly
  `max_frames` (zero-padded when a clip yields fewer). You cannot change it at
  eval time — the transformer was built around `frames_per_clip = max_frames`.
  Comparability to externally published video-benchmark numbers therefore depends
  on the checkpoint's frame budget matching the reference's, which is a training
  choice rather than a knob here.
- **Query-aware frame selection.** If the config has a `[frame_selector]` section
  (or you add one with `--override frame_selector.type=qframe`), the harness
  decodes a larger candidate pool and keeps the `max_frames` frames most relevant
  to the request's prompt — the *which* frames change, the budget does not. This
  is how a uniform-trained checkpoint is A/B'd across `topk` / `qframe` / `mdp3`
  without editing its TOML. Per-frame times of the selected clip are threaded to
  the model, so a time-aware checkpoint gets its temporal signal at eval.
- **Scope.** One video per request, single-turn, zero-shot, generative arches
  (`joint_decoder` / `cross_attention` / `mot`). A single **image** task also runs
  on a video checkpoint — the image is treated as a 1-frame clip, zero-padded to
  `frames_per_clip`. Multiple videos, mixed image+video, multiple images, audio,
  and multi-turn / few-shot raise a clear error; MoMa still fails fast. An
  **image** checkpoint cannot evaluate video and raises a clear error if handed a
  video task.

## Limitations

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
  The integration test suite imports the evaluator too and needs the same
  workaround.

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
