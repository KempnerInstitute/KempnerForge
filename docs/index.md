# KempnerForge

PyTorch-native framework for fault-tolerant distributed training of
foundation models on AI clusters.

This site is the canonical documentation for KempnerForge. It builds from
the `docs/` folder and the package source, and deploys to GitHub Pages on
every push to `main`.

## New here?

Start with the getting-started pages, then move to the flagship how-to
once you want a real run:

- **{doc}`getting-started/install`** — prerequisites, `uv sync`,
  environment verification, SLURM-specific notes.
- **{doc}`getting-started/quickstart`** — five-minute walkthrough: debug
  run → multi-GPU → custom data → optimizer swap → MoE → hooks.
- **{doc}`getting-started/first-training-run`** — what the debug run
  actually did, log line by log line.
- **{doc}`how-to/end-to-end-training-run`** — flagship walkthrough:
  tokenize → write a config → launch 1 GPU → launch 4 GPUs → resume →
  generate. The integration test of the docs.
- **{doc}`getting-started/notebooks`** — six interactive notebooks for
  model inspection, attention visualization, activation extraction,
  checkpoint analysis, optimizer comparison, MoE routing.

## Looking for something specific?

- **{doc}`how-to/index`** — end-to-end researcher workflows: prepare
  data, scale 1→32 GPUs, compare optimizers, set up MoE experiments,
  extract activations, debug regressions.
- **{doc}`architecture/index`** — model forward pass, parallelism
  application order, data flow through the training loop.
- **{doc}`configuration/index`** — the typed dataclass system, CLI
  overrides, registry for swappable components.

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
