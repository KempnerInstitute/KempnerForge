# Notebooks

Six interactive Jupyter notebooks under
[`examples/notebooks/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/examples/notebooks)
for single-GPU exploration. All use tiny 1–5M-param configs sized for
interactive use — each runs end-to-end in well under a minute, except
notebook 5 (optimizer comparison, ~2 min).

Every notebook opens with the same header:

- **Objectives** — what you'll learn
- **Requirements** — hardware, data, prerequisites
- **Runtime** — approximate wall time for *Run All*

## Running

From the repo root:

```bash
uv run jupyter lab examples/notebooks/
```

Or execute a single notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute \
  examples/notebooks/01_inspect_model.ipynb
```

## Catalogue

| # | Notebook | What it shows |
|---|----------|---------------|
| 1 | [`01_inspect_model.ipynb`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/notebooks/01_inspect_model.ipynb) | Build a model from `ModelConfig`, inspect layer shapes, run a forward pass |
| 2 | [`02_attention_visualization.ipynb`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/notebooks/02_attention_visualization.ipynb) | Capture attention weights per layer and head, plot heatmaps |
| 3 | [`03_activation_extraction.ipynb`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/notebooks/03_activation_extraction.ipynb) | Extract intermediate activations via `ActivationStore` and `extract_representations()`, save to `.npz` |
| 4 | [`04_checkpoint_analysis.ipynb`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/notebooks/04_checkpoint_analysis.ipynb) | Train a tiny model, save a checkpoint, load it back, generate text |
| 5 | [`05_optimizer_comparison.ipynb`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/notebooks/05_optimizer_comparison.ipynb) | Train the same model with AdamW / Muon / Lion / Schedule-Free AdamW, plot loss curves |
| 6 | [`06_moe_routing.ipynb`](https://github.com/KempnerInstitute/KempnerForge/blob/main/examples/notebooks/06_moe_routing.ipynb) | Build an MoE model, visualize per-layer expert utilization |

## When to open which

- **Debugging a config**: start with notebook 1 — it builds the model from
  your config and prints every layer's shape.
- **Interpretability setup**: notebooks 2 and 3 cover the attention-capture
  and activation-extraction APIs you'll use in a larger probing pipeline.
- **Checkpoint round-trips**: notebook 4 is the minimal reproduction of
  "train → save → load → generate" that you can adapt for evaluating any
  checkpoint.
- **Optimizer ablations**: notebook 5 is the reference pattern for a
  controlled comparison with per-optimizer LR sweeps.
- **MoE diagnostics**: notebook 6 shows how to read `get_expert_counts()`
  output and spot dead or hot experts.

## Requirements

- 1 GPU (falls back to CPU where possible, but attention and training are
  slow).
- Project dev dependencies installed via `uv sync` from the repo root.

Notebook outputs are stripped on commit (via the `nbstripout` pre-commit
hook) to keep diffs clean, so you'll always see empty outputs until you
run them.

```{note}
These notebooks are meant to be **run**, not just **read**. The Sphinx site
does not execute them or embed their outputs — open them in JupyterLab.
```
