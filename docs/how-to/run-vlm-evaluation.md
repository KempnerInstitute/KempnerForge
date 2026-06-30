# Run VLM evaluation

Evaluate a vision-language model (VLM) checkpoint on any standard multimodal
benchmark (MMMU, MMBench, ScienceQA, SEED, AI2D, …) by integrating the
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness.

[Run evaluation](run-evaluation.md).

| Entry point | When to use |
|-------------|-------------|
| `scripts/vlm_eval_harness.py` | Downstream VLM benchmarks via lmms-eval, on any DCP checkpoint |

A custom lmms-eval *chat model* loads `VLMWrapper` directly from the DCP checkpoint. The pieces are:
[`kempnerforge/eval/vlm/adapter.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/eval/vlm/adapter.py)
(the `KempnerForgeVLM` adapter),
[`kempnerforge/eval/vlm/manifest.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/eval/vlm/manifest.py)
(the lmms-eval registration manifest), and
[`scripts/vlm_eval_harness.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/vlm_eval_harness.py)
(the CLI).

## What's on this page

- [Install lmms-eval](#install-lmms-eval) — the optional, separately-installed dependency
- [Usage](#usage) — running the harness on a checkpoint
- [Flags](#flags) — CLI options at a glance
- [Video evaluation](#video-evaluation) — evaluating a video checkpoint
- [Limitations](#limitations) — single-GPU, MoMa, multi-turn/few-shot/multi-image, flattening, no KV cache
- [Cluster environment notes](#cluster-environment-notes) — torchvision / libstdc++ gotchas

## Install lmms-eval

`lmms-eval` is an **optional dependency** and is intentionally NOT declared in
`pyproject.toml`. Install
it into your environment before running:

```bash
uv pip install lmms-eval
```

The `lmms_eval.models` entry point that exposes the `kempnerforge_vlm` model is
declared in `pyproject.toml` as metadata only — it does not pull lmms-eval in as
a dependency, and `import kempnerforge` works without lmms-eval installed.

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
uv run python scripts/vlm_eval_harness.py \
    --config     configs/train/vlm_jd.toml \
    --checkpoint checkpoints/vlm/step_10000 \
    --tasks      mmmu_val \
    --output     results/vlm_step_10000.json

# Several tasks, quick partial run (4 examples per task)
uv run python scripts/vlm_eval_harness.py \
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

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--config` | — (required) | KempnerForge TOML the checkpoint was trained with |
| `--checkpoint` | — (required) | DCP checkpoint dir (run dir or `step_N` dir) |
| `--tasks` | — (required) | comma-separated lmms-eval task names |
| `--limit` | `None` | cap examples per task (int count, or `<1.0` fraction) |
| `--output` | `None` | save full JSON results |
| `--device` | `cuda` | inference device |
| `--dtype` | `bfloat16` | model dtype |
| `--batch-size` | `1` | requests decoded together (grouped by `gen_kwargs`) |
| `--max-new-tokens` | `128` | fallback only; task `gen_kwargs` override it |


## Video evaluation

When `--config` is a **video checkpoint** (its TOML has a `[video]` section), the
harness evaluates lmms-eval *video* `generate_until` tasks: each request's video
is decoded into frames and fed to the model as a single clip. This needs the `av`
video group (see [Install lmms-eval](#install-lmms-eval)).

```bash
uv run python scripts/vlm_eval_harness.py \
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
  correct but costs extra compute, and a KV-cache decode is future work. The vision
  tower + adapter, by contrast, are cached: they are encoded
  once per request and the projected embeds are reused across all decode steps via
  the `VLMWrapper.encode_visual` / `precomputed_embeds` seam (arch-agnostic across
  the modality strategies). Raising `--batch-size` decodes multiple requests together
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
  `LD_LIBRARY_PATH=<conda-env>/lib uv run python scripts/vlm_eval_harness.py …`.

## See also

- [Run evaluation](run-evaluation.md) — text-model loss/perplexity and the
  `lm-eval` harness this page parallels.
- [End-to-end training run](end-to-end-training-run.md) — produces the
  checkpoints this harness consumes.
- [`kempnerforge/eval/vlm/adapter.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/eval/vlm/adapter.py)
  and
  [`scripts/vlm_eval_harness.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/vlm_eval_harness.py)
  — the adapter and CLI this page documents.
