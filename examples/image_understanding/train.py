#!/usr/bin/env python3
"""Training entry point for the image-understanding example.

Parses argv, loads the config, and hands off to the core training scaffold.
The VLM step body still lives in core, so ``run_training`` selects it from
``config.is_vlm``; when it moves out here it becomes a ``step_fn=`` argument.
``--mixture=<file>`` trains on the packed mixture of image-text sources the file
describes (``mixture.py``) instead of the ``[data]`` source.

Periodic evaluation is off by default. With ``--eval-tasks``, ``--eval-command`` and
``--eval-output`` every checkpoint the run saves is evaluated: the command is a template
started once the checkpoint is complete and never waited for while the run trains, and
the output is the directory that gets one results directory per step. Every other
argument must then be a ``--section.key=value`` override: the evaluations receive the
same overrides, and ``--mixture=<file>`` with them, and refuse any other argument.

Usage:
    # Multi-GPU (single node, via torchrun)
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml>

    # With overrides
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --train.grad_accum_steps=4

    # On a mixture of image-text sources
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --mixture=<mixture.toml>

    # Evaluating each saved checkpoint as its own scheduler job
    uv run torchrun --nproc_per_node=4 examples/image_understanding/train.py <config.toml> \
        --eval-tasks <tasks.toml> --eval-output <results root> \
        --eval-command "sbatch <job script> {config} {checkpoint} {tasks} {output} {overrides}"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mixture import mixture_data_fn

from kempnerforge.config.loader import load_config
from kempnerforge.training import run_training

EVAL_FLAG_PREFIX = "--eval-"
MIXTURE_FLAG = "--mixture="


def _parse_args(argv: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace, list[str]]:
    """The periodic-evaluation flags among ``argv`` and the config overrides that remain.

    An argument that is neither, which the config loader would drop, stops the run through
    ``parser.error``.
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        usage="train.py <config.toml> [--mixture=<mixture.toml>] [--section.key=value ...] "
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
    stray = [argument for argument in overrides if not argument.startswith("--")]
    if stray:
        parser.error(
            f"unrecognized arguments: {' '.join(stray)}; config overrides take the form "
            "--section.key=value"
        )
    return parser, args, overrides


def main() -> None:
    mixtures = [arg[len(MIXTURE_FLAG) :] for arg in sys.argv[1:] if arg.startswith(MIXTURE_FLAG)]
    args = [arg for arg in sys.argv[1:] if not arg.startswith(MIXTURE_FLAG)]
    if not args or len(mixtures) > 1:
        print("Usage: train.py <config.toml> [--mixture=<mixture.toml>] [--section.key=value ...]")
        sys.exit(1)
    options = {"data_fn": mixture_data_fn(mixtures[0])} if mixtures else {}

    arguments = args[1:]
    if not any(argument.startswith(EVAL_FLAG_PREFIX) for argument in arguments):
        run_training(load_config(args[0], cli_args=arguments), **options)
        return

    parser, parsed, overrides = _parse_args(arguments)
    config = load_config(args[0], cli_args=overrides)
    flags = {
        "--eval-tasks": parsed.eval_tasks,
        "--eval-command": parsed.eval_command,
        "--eval-output": parsed.eval_output,
    }
    absent = [flag for flag, value in flags.items() if value is None]
    if absent:
        parser.error(f"periodic evaluation also needs {' and '.join(absent)}")

    from evaluation.checkpoint import check_supported
    from evaluation.periodic import PeriodicEvaluation, PeriodicEvaluationHook

    passed_on = [f"{MIXTURE_FLAG}{Path(mixtures[0]).resolve()}"] if mixtures else []
    try:
        check_supported(config)
        settings = PeriodicEvaluation(
            config=args[0],
            tasks=parsed.eval_tasks,
            command=parsed.eval_command,
            output_root=parsed.eval_output,
            overrides=(*passed_on, *overrides),
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    run_training(config, hooks=PeriodicEvaluationHook(settings).runner(), **options)


if __name__ == "__main__":
    main()
