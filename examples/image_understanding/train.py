#!/usr/bin/env python3
"""Training entry point for the image-understanding example.

Parses argv, loads the config, and hands off to the core training scaffold.
The VLM step body still lives in core, so ``run_training`` selects it from
``config.is_vlm``; when it moves out here it becomes a ``step_fn=`` argument.

Usage:
    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py \
        examples/image_understanding/configs/qwen3_0.6b_siglip2_stage1.toml

    # With overrides
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py \
        examples/image_understanding/configs/qwen3_0.6b_siglip2_stage1.toml \
        --train.grad_accum_steps=4
"""

from __future__ import annotations

import sys

from kempnerforge.config.loader import load_config
from kempnerforge.training import run_training


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: train.py <config.toml> [--section.key=value ...]")
        sys.exit(1)

    config = load_config(sys.argv[1], cli_args=sys.argv[2:])
    run_training(config)


if __name__ == "__main__":
    main()
