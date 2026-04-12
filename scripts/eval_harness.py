#!/usr/bin/env python3
"""Run lm-eval-harness benchmarks on a KempnerForge checkpoint.

Converts a DCP checkpoint to HuggingFace format, then runs lm-eval-harness
on the converted model. Requires: `uv add lm-eval` (optional dependency).

Usage:
    # Default task suite (HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, BoolQ)
    uv run python scripts/eval_harness.py \
        --checkpoint checkpoints/7b/step_10000 \
        --config configs/train/7b.toml

    # Specific tasks
    uv run python scripts/eval_harness.py \
        --checkpoint checkpoints/7b/step_10000 \
        --config configs/train/7b.toml \
        --tasks hellaswag,arc_easy,mmlu

    # Pre-converted HuggingFace model (skip conversion)
    uv run python scripts/eval_harness.py \
        --hf-model ./exports/my_model \
        --tasks hellaswag,arc_easy

    # Output to file
    uv run python scripts/eval_harness.py \
        --checkpoint checkpoints/7b/step_10000 \
        --config configs/train/7b.toml \
        --output results/eval_step_10000.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TASKS = "hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval-harness on a KempnerForge checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="DCP checkpoint directory (will be converted to HF format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="TOML config file (required with --checkpoint for model architecture)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Pre-converted HuggingFace model directory (skip DCP conversion)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help=f"Comma-separated tasks (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task-specific)",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.hf_model is None:
        parser.error("Provide either --checkpoint (DCP) or --hf-model (HuggingFace)")

    # --- Resolve HF model path ---
    hf_model_path = args.hf_model
    tmp_dir = None

    if hf_model_path is None:
        if args.config is None:
            parser.error("--config is required when using --checkpoint")

        # Convert DCP → HuggingFace
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from convert_checkpoint import dcp_to_hf

        from kempnerforge.config.loader import load_config

        config = load_config(args.config, cli_args=[])

        tmp_dir = tempfile.mkdtemp(prefix="kf_eval_hf_")
        hf_model_path = tmp_dir
        logger.info(f"Converting DCP checkpoint to HuggingFace format: {tmp_dir}")
        dcp_to_hf(
            dcp_dir=args.checkpoint,
            hf_dir=hf_model_path,
            model_config=config.model,
        )

    # --- Run lm-eval-harness ---
    try:
        import lm_eval
    except ImportError:
        logger.error(
            "lm-eval-harness not installed. Install with: uv add lm-eval\n"
            "Or: uv pip install lm-eval"
        )
        sys.exit(1)

    logger.info(f"Running lm-eval-harness: tasks={args.tasks}, model={hf_model_path}")

    task_list = args.tasks.split(",")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_model_path},dtype=bfloat16",
        tasks=task_list,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
    )

    # --- Print results ---
    print(f"\n{'=' * 60}")
    print("lm-eval-harness Results")
    print(f"{'=' * 60}")
    if "results" in results:
        for task_name, task_results in sorted(results["results"].items()):
            print(f"\n  {task_name}:")
            for metric, value in sorted(task_results.items()):
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                elif metric != "alias":
                    print(f"    {metric}: {value}")
    print(f"{'=' * 60}\n")

    # --- Save results ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    # Cleanup temp dir
    if tmp_dir is not None:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
