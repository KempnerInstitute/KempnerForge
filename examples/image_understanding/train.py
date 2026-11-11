#!/usr/bin/env python3
"""Training entry point for the image-understanding example.

Parses argv, loads the config, and hands off to the core training scaffold.
The VLM step body still lives in core, so ``run_training`` selects it from
``config.is_vlm``; when it moves out here it becomes a ``step_fn=`` argument.

Periodic evaluation is off by default. With ``--eval-tasks``, ``--eval-command`` and
``--eval-output`` every checkpoint the run saves is evaluated: the command is a template
started once the checkpoint is complete and never waited for, and the output is the
directory that gets one results directory per step.

Usage:
    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml>

    # With overrides
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --train.grad_accum_steps=4

    # Evaluating each saved checkpoint as its own scheduler job
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --eval-tasks <tasks.toml> --eval-output <results root> \
        --eval-command "sbatch <job script> {config} {checkpoint} {tasks} {output} {overrides}"
"""

from __future__ import annotations

import argparse
import sys

from kempnerforge.config.loader import load_config
from kempnerforge.training import run_training


def _parse_args(argv: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace, list[str]]:
    """The periodic-evaluation flags among ``argv`` and the config overrides that remain."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        usage="train.py <config.toml> [--section.key=value ...] "
        "[--eval-tasks FILE --eval-command TEMPLATE --eval-output DIR]",
        description="Periodic evaluation is off unless an --eval-* flag is given; it then "
        "needs all three.",
        allow_abbrev=False,
    )
    parser.add_argument("--eval-tasks", help="TOML task file; turns periodic evaluation on")
    parser.add_argument(
        "--eval-command",
        help="command template that evaluates one checkpoint; its placeholders are listed when "
        "it names an unknown one",
    )
    parser.add_argument("--eval-output", help="directory that gets one results directory per step")
    args, overrides = parser.parse_known_args(argv)
    return parser, args, overrides


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: train.py <config.toml> [--section.key=value ...]")
        sys.exit(1)

    parser, args, overrides = _parse_args(sys.argv[2:])
    config = load_config(sys.argv[1], cli_args=overrides)
    flags = {
        "--eval-tasks": args.eval_tasks,
        "--eval-command": args.eval_command,
        "--eval-output": args.eval_output,
    }
    if all(value is None for value in flags.values()):
        run_training(config)
        return
    absent = [flag for flag, value in flags.items() if value is None]
    if absent:
        parser.error(f"periodic evaluation also needs {' and '.join(absent)}")

    from evaluation.checkpoint import check_supported
    from evaluation.periodic import PeriodicEvaluation, PeriodicEvaluationHook

    try:
        check_supported(config)
        settings = PeriodicEvaluation(
            config=sys.argv[1],
            tasks=args.eval_tasks,
            command=args.eval_command,
            output_root=args.eval_output,
            overrides=tuple(overrides),
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    run_training(config, hooks=PeriodicEvaluationHook(settings).runner())


if __name__ == "__main__":
    main()
