#!/usr/bin/env python3
"""Run lmms-eval benchmarks on a KempnerForge VLM checkpoint.

Evaluates a VLM checkpoint directly via the
``kempnerforge_vlm`` lmms-eval chat-model adapter, on the standard image
benchmarks lmms-eval implements as ``generate_until`` tasks (MMMU, MMBench,
ScienceQA, SEED, AI2D, ...).

Requirements (lmms-eval is an OPTIONAL, separately-installed dependency, exactly
like lm-eval for text evaluation):

    uv pip install lmms-eval

v1 is single-GPU and image-only; MoMa checkpoints are not supported (see
docs/how-to/run-vlm-evaluation.md). On clusters where importing lmms-eval's
evaluator fails with ``GLIBCXX_... not found``, put a newer libstdc++ on the
library path (e.g. ``LD_LIBRARY_PATH=<conda>/lib``).

Usage:
    uv run python scripts/vlm_eval_harness.py \
        --config configs/train/vlm_jd.toml \
        --checkpoint checkpoints/vlm/step_10000 \
        --tasks mmmu_val \
        --output results/vlm_step_10000.json

    # Quick partial run (4 examples per task)
    uv run python scripts/vlm_eval_harness.py \
        --config configs/train/vlm_jd.toml \
        --checkpoint checkpoints/vlm/step_10000 \
        --tasks mmmu_val,mmbench_en_dev \
        --limit 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _limit_type(value: str) -> int | float:
    """Per-task example cap: an integer count, or a fraction < 1.0."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be > 0")
    if parsed < 1.0:
        return parsed
    if parsed.is_integer():
        return int(parsed)
    raise argparse.ArgumentTypeError("--limit must be an integer count, or a fraction < 1.0")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lmms-eval on a KempnerForge VLM checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="KempnerForge TOML the checkpoint was trained with",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="DCP checkpoint dir (run dir or step_N dir)"
    )
    # No default task suite: the representative default benchmark set is an open
    # decision; --tasks is required until one is provided.
    parser.add_argument(
        "--tasks", type=str, required=True, help="Comma-separated lmms-eval task names"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument(
        "--limit",
        type=_limit_type,
        default=None,
        help="Cap examples per task (int count, or <1.0 fraction); for quick partial runs",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype; default: the checkpoint config's train.param_dtype",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Requests decoded together (grouped by gen_kwargs)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Fallback max new tokens; task gen_kwargs override it (default: 128)",
    )
    args = parser.parse_args()

    # lmms-eval is optional and undeclared; import lazily with a helpful error.
    try:
        from lmms_eval.evaluator import simple_evaluate
    except ImportError as exc:
        logger.error(
            "Could not import lmms-eval's simple_evaluate (%s).\n"
            "lmms-eval is an optional dependency; install it with: uv pip install lmms-eval\n"
            "If this is a 'GLIBCXX_... not found' error, put a newer libstdc++ on the library "
            "path (e.g. LD_LIBRARY_PATH=<conda>/lib); see docs/how-to/run-vlm-evaluation.md.",
            exc,
        )
        sys.exit(1)

    logger.info(f"Running lmms-eval: tasks={args.tasks}, checkpoint={args.checkpoint}")

    model_args = (
        f"config={args.config},checkpoint={args.checkpoint},max_new_tokens={args.max_new_tokens}"
    )
    # Only pass dtype when explicitly set; otherwise the adapter defaults it from
    # the checkpoint config (train.param_dtype).
    if args.dtype is not None:
        model_args += f",dtype={args.dtype}"
    results = simple_evaluate(
        model="kempnerforge_vlm",
        model_args=model_args,
        tasks=args.tasks.split(","),
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # --- Print results ---
    print(f"\n{'=' * 60}")
    print("lmms-eval Results")
    print(f"{'=' * 60}")
    if results is not None and "results" in results:
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


if __name__ == "__main__":
    main()
