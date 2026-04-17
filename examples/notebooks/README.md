# KempnerForge Notebooks

Interactive examples for single-GPU exploration. All notebooks use tiny 1–5M-param
configs sized for interactive use — each runs end-to-end in well under a minute,
except notebook 05 (optimizer comparison, ~2 min).

Every notebook opens with the same header:

- **Objectives** — what you'll learn
- **Requirements** — hardware, data, prerequisites
- **Runtime** — approximate wall time if you select *Run All*

## Running

From the repo root:

```bash
uv run jupyter lab examples/notebooks/
```

Or execute a single notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute examples/notebooks/01_inspect_model.ipynb
```

## Notebooks

| # | Notebook | What it shows |
|---|----------|---------------|
| 1 | [`01_inspect_model.ipynb`](01_inspect_model.ipynb) | Build a model from `ModelConfig`, inspect layer shapes, run a forward pass |
| 2 | [`02_attention_visualization.ipynb`](02_attention_visualization.ipynb) | Capture attention weights per layer/head, plot heatmaps |
| 3 | [`03_activation_extraction.ipynb`](03_activation_extraction.ipynb) | Extract intermediate activations via `ActivationStore` and `extract_representations()`, save to `.npz` |
| 4 | [`04_checkpoint_analysis.ipynb`](04_checkpoint_analysis.ipynb) | Train a tiny model, save a checkpoint, load it back, generate text |
| 5 | [`05_optimizer_comparison.ipynb`](05_optimizer_comparison.ipynb) | Train the same model with AdamW / Muon / Lion / Schedule-Free AdamW, plot loss curves |
| 6 | [`06_moe_routing.ipynb`](06_moe_routing.ipynb) | Build a MoE model, visualize per-layer expert utilization |

## Requirements

- 1 GPU (falls back to CPU where possible, but attention/training is slow)
- Dev dependencies installed via `uv sync` from the repo root

Notebook outputs are stripped on commit (via the `nbstripout` pre-commit hook) to keep diffs clean.
