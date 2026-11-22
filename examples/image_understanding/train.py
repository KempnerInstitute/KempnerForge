#!/usr/bin/env python3
"""Training entry point for the image-understanding example.

Parses argv, loads the config, and hands off to the core training scaffold.
The VLM step body still lives in core, so ``run_training`` selects it from
``config.is_vlm``; when it moves out here it becomes a ``step_fn=`` argument.
``--mixture=<file>`` trains on the packed mixture of image-text sources the file
describes (``mixture.py``) instead of the ``[data]`` source.

Usage:
    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml>

    # With overrides
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --train.grad_accum_steps=4

    # On a mixture of image-text sources
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --mixture=<mixture.toml>
"""

from __future__ import annotations

import sys

from mixture import mixture_data_fn

from kempnerforge.config.loader import load_config
from kempnerforge.training import run_training

MIXTURE_FLAG = "--mixture="


def main() -> None:
    mixtures = [arg[len(MIXTURE_FLAG) :] for arg in sys.argv[1:] if arg.startswith(MIXTURE_FLAG)]
    args = [arg for arg in sys.argv[1:] if not arg.startswith(MIXTURE_FLAG)]
    if not args or len(mixtures) > 1:
        print("Usage: train.py <config.toml> [--mixture=<mixture.toml>] [--section.key=value ...]")
        sys.exit(1)

    config = load_config(args[0], cli_args=args[1:])
    run_training(config, data_fn=mixture_data_fn(mixtures[0]) if mixtures else None)


if __name__ == "__main__":
    main()
