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
[`kempnerforge/eval/vlm/registry.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/eval/vlm/registry.py)
(the lmms-eval registration manifest), and
[`scripts/vlm_eval_harness.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/vlm_eval_harness.py)
(the CLI).

## What's on this page

- [Install lmms-eval](#install-lmms-eval) — the optional, separately-installed dependency
- [Usage](#usage) — running the harness on a checkpoint
- [Flags](#flags) — CLI options at a glance
- [Limitations](#limitations) — single-GPU, MoMa, images-only, flattening, no KV cache
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
- **Images only.** A request must carry exactly one image. Video/audio content,
  multi-image, and multi-turn/few-shot requests raise a clear error. Visual input
  is modeled internally as an ordered list of frames (a single image is the
  length-1 case), so **video** is a localized future addition — it will also
  require a video-decoding dependency (lmms-eval uses `decord` / `qwen-vl-utils`).
- **Prompt flattening discards structure.** Flattening drops role/turn structure
  and any model-specific chat template. KempnerForge pre-training uses no chat
  template; once a post-training format exists, repo-wide chat-template support
  should be added and the rendering step made configurable.
- **No KV cache; vision re-encoded per step.** Decoding re-runs the full forward
  over the growing sequence each step (KempnerForge has no image-conditioned
  KV-cache decode path), and the vision tower is re-encoded each step (there is
  no arch-agnostic public seam to pass cached image features). Both are correct
  but cost extra compute. Raising `--batch-size` decodes multiple requests
  together (right-padded, grouped by `gen_kwargs`) to amortize per-step overhead;
  encode-once and a KV-cache decode are future work.

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
