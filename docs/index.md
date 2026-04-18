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


### Subsystem reference

- **{doc}`training/index`** — training loop, optimizers, schedulers,
  loss functions, gradient utilities, hooks, evaluation, generation.
- **{doc}`distributed/index`** — DeviceMesh, FSDP2, tensor /
  expert / pipeline parallelism, FP8.
- **{doc}`moe/index`** — routers, capacity and dispatch, auxiliary
  losses and balancing, FP8 interaction.
- **{doc}`data/index`** — memory-mapped datasets, HuggingFace
  streaming, sampler, stateful dataloader, mixing and annealing.
- **{doc}`checkpointing/index`** — DCP model and train-state,
  auto-resume, resharding, HuggingFace conversion.
- **{doc}`metrics-and-profiling/index`** — metrics tracker, MFU,
  memory monitor, profiler, WandB / TensorBoard backends.
- **{doc}`resilience/index`** — SLURM preemption, NaN detection,
  NCCL liveness, GPU health, elastic training.

### Reference Tables and API Documentation

- **{doc}`reference/index`** — available configs, parallelism
  recipes, benchmarks, environment variables.
- **{doc}`api/index`** — API reference, auto-generated from
  docstrings.

## Contributing

Documentation PRs follow the same flow as code PRs. The editor loop, build
commands, and style conventions live in
[Contributing § Writing Docs](contributing.md#writing-docs).

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Architecture and How-to

architecture/index
how-to/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Subsystems

training/index
distributed/index
moe/index
data/index
checkpointing/index
metrics-and-profiling/index
resilience/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Configuration and Reference

configuration/index
reference/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API

api/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

contributing
```

