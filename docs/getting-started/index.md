# Getting Started

Four pages that take you from a fresh clone to a running model. Read in order if
you're new; skip around if you know what you're looking for.

```{toctree}
:maxdepth: 1

install
quickstart
first-training-run
notebooks
```

## What each page covers

**{doc}`install`** — prerequisites, `uv sync`, environment verification, and
SLURM-specific module setup for Kempner clusters.

**{doc}`quickstart`** — a 5-minute walkthrough: debug run on one GPU, then
multi-GPU, then point at your own data, then swap optimizer, then enable MoE,
then extend via hooks. Every step is a single command.

**{doc}`first-training-run`** — slows down and explains what the debug run
actually does: what the log line means, what's in the checkpoint directory,
how auto-resume finds the latest step, and what to change next.

**{doc}`notebooks`** — summaries of the six interactive notebooks under
`examples/notebooks/` and when to open each.

## Prerequisites before you start

- A Linux host with Python 3.12+ and at least one NVIDIA GPU (H100/H200/A100).
  CPU-only also works for the inspection notebooks; training steps will be
  slow.
- [uv](https://docs.astral.sh/uv/) installed. All commands in these pages use
  `uv run`, which activates the project venv automatically — you do not need
  to `source .venv/bin/activate` yourself.
- For multi-node runs on Kempner clusters: SLURM account and partition names
  from your PI or cluster admin.

If any of those are missing, start at {doc}`install`.
