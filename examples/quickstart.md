# Quickstart has moved

The 5-minute walkthrough now lives in the hosted Sphinx documentation:

- **Source**: [`docs/getting-started/quickstart.md`](../docs/getting-started/quickstart.md)
- **Rendered**: published on GitHub Pages on every push to `main` (see
  [Docs workflow](../.github/workflows/docs.yml) for deployment details).

The `examples/` directory still hosts runnable files that complement the
docs:

- [`custom_hook.py`](custom_hook.py) — four training-hook examples
  (gradient-norm histogram, learning-dynamics, early stopping, MoE
  expert load balance).
- [`notebooks/`](notebooks/) — six single-GPU Jupyter notebooks for
  model inspection, interpretability, and MoE diagnostics.

For the notebook catalogue with short descriptions, see
[`docs/getting-started/notebooks.md`](../docs/getting-started/notebooks.md).
