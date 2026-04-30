# Install

KempnerForge uses [uv](https://docs.astral.sh/uv/) for dependency management.
There is no `pip install kempnerforge` — you work from a clone.

## Prerequisites

- **Python 3.12+** — pinned via `.python-version` (`>=3.12`) and
  `pyproject.toml`'s `requires-python`. uv reads `.python-version` and will
  use any Python ≥ 3.12 already on your machine, or fetch a managed CPython
  for you. No manual Python install needed.
- **CUDA 12.8 toolkit** — PyTorch wheels are pulled from the CUDA 12.8 index.
- **NVIDIA GPU** for any training or distributed test (H100, H200, or A100).
  Unit tests and the inspection notebooks run on CPU.
- **uv** — install once per machine:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Clone and sync

```bash
git clone git@github.com:KempnerInstitute/KempnerForge.git
cd KempnerForge

# Creates .venv, installs runtime + dev dependencies, resolves torch from the
# CUDA 12.8 index pinned in pyproject.toml.
uv sync
```

The `.venv` is project-local. Every command in the docs runs through
`uv run` so the venv is activated automatically — you do not need to
`source .venv/bin/activate`.

## Verify

Run the same checks CI runs on every PR:

```bash
uv run ruff check kempnerforge/ tests/ scripts/
uv run ruff format --check kempnerforge/ tests/ scripts/
uv run pyright kempnerforge/
uv run pytest tests/unit/ -v --timeout=60
```

If all four pass, your environment is ready.

## Optional: docs dependencies

If you plan to preview docs locally, add the `docs` dependency group:

```bash
uv sync --group docs
uv run make -C docs live  # live-reload server on http://127.0.0.1:8000
```

See the "Writing Docs" section of
[`CONTRIBUTING.md`](https://github.com/KempnerInstitute/KempnerForge/blob/main/CONTRIBUTING.md#writing-docs)
for the full editor loop.

## SLURM clusters (Kempner and similar)

Unit tests run on CPU login nodes. Anything that needs a GPU (integration,
distributed, end-to-end tests, and actual training) has to run inside a SLURM
allocation.

### Interactive allocation for development

```bash
# 1 node, 4 GPUs — enough for integration + distributed tests
salloc --partition=<partition-name> --account=<account-name> \
  --nodes=1 --gpus-per-node=4 --cpus-per-task=16 --mem=256G --time=2:00:00
```

Replace `<partition-name>` and `<account-name>` with values your PI gave
you. Once the prompt returns, you are on a compute node and can run
`uv run torchrun ...` directly.

### Modules and environment

On Kempner clusters, the provided `scripts/slurm/*.sh` wrappers detect the
InfiniBand interface and export `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
before invoking `torchrun` or `srun`. They do not `module load` anything —
`uv sync` brings in the CUDA-enabled PyTorch wheel, and the CUDA driver is
already present on compute nodes.

If you launch bare `torchrun` from an interactive shell, the venv from
`uv sync` is all you need.

## Next steps

- {doc}`quickstart` — run a tiny model end-to-end in under a minute.
- {doc}`first-training-run` — understand what that run actually did.
- [`CONTRIBUTING.md`](https://github.com/KempnerInstitute/KempnerForge/blob/main/CONTRIBUTING.md#environment-setup) for the contributor workflow (branch naming, PR template, pre-push checklist).
