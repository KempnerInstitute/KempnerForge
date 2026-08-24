#!/usr/bin/env python3
"""KempnerForge training entry point.

A thin CLI wrapper: parse argv, load the config, and hand off to
``kempnerforge.training.entry.run_training``.

Usage:
    # Single GPU
    uv run python scripts/train.py configs/train/debug.toml

    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml

    # Multi-node (via SLURM srun — see scripts/slurm/multinode.sh)
    # srun launches one process per GPU; MASTER_ADDR/MASTER_PORT are resolved
    # automatically from SLURM env vars by init_distributed().
    srun uv run python scripts/train.py configs/train/7b.toml

    # With overrides
    uv run python scripts/train.py configs/train/7b.toml \
        --train.max_steps=1000 --optimizer.lr=1e-4
"""

from __future__ import annotations

import sys

from kempnerforge.config.loader import load_config
from kempnerforge.training.entry import run_training


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: train.py <config.toml> [--section.key=value ...]")
        sys.exit(1)

    config = load_config(sys.argv[1], cli_args=sys.argv[2:])
    run_training(config)


if __name__ == "__main__":
    main()
