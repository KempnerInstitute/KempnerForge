#!/usr/bin/env python3
"""Evaluation entry point for the image-understanding example.

Parses argv, loads the training config and the task file, and hands off to
``evaluation.evaluator.evaluate_checkpoint``, which decodes each task's split
with the checkpoint's model, scores the answers and writes the results.

Usage:
    # One GPU
    uv run python examples/image_understanding/evaluate_checkpoint.py <config.toml> \
        --checkpoint <run or step dir> --tasks <tasks.toml> --output <results dir>

    # Several GPUs (single node, via torchrun), with the overrides training ran with
    uv run torchrun --nproc_per_node=4 examples/image_understanding/evaluate_checkpoint.py \
        <config.toml> --checkpoint <run or step dir> --tasks <tasks.toml> \
        --output <results dir> --batch-size 64 --section.key=value
"""

from __future__ import annotations

import argparse

from evaluation.evaluator import evaluate_checkpoint
from evaluation.tasks import load_tasks

from kempnerforge.config.loader import load_config


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """The entry point's own arguments and the ``--section.key=value`` config overrides."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint of the image-understanding example on the "
        "tasks of a task file. Config overrides follow as --section.key=value.",
        allow_abbrev=False,
    )
    parser.add_argument("config", help="TOML config the checkpoint was trained with")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="run directory (its latest checkpoint is evaluated) or step directory",
    )
    parser.add_argument("--tasks", required=True, help="TOML file of the tasks to evaluate")
    parser.add_argument("--output", required=True, help="directory the results are written to")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="rows decoded together (default: %(default)s)"
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="do not publish the metrics to the MLflow run recorded in the checkpoint",
    )
    args, overrides = parser.parse_known_args()
    stray = [arg for arg in overrides if not arg.startswith("--")]
    if stray:
        parser.error(f"unrecognized arguments: {' '.join(stray)}")
    return args, overrides


def main() -> None:
    args, overrides = _parse_args()
    config = load_config(args.config, cli_args=overrides)
    evaluate_checkpoint(
        config,
        load_tasks(args.tasks),
        checkpoint=args.checkpoint,
        output=args.output,
        batch_size=args.batch_size,
        publish=not args.no_publish,
    )


if __name__ == "__main__":
    main()
