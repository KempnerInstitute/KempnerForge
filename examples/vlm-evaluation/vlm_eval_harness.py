#!/usr/bin/env python3
"""Run lmms-eval benchmarks on a KempnerForge VLM checkpoint.

Evaluates a VLM checkpoint via the ``KempnerForgeVLM`` lmms-eval chat-model
adapter (the sibling ``adapter.py``), on the standard benchmarks lmms-eval
implements as ``generate_until`` tasks (MMMU, MMBench, ScienceQA, SEED, AI2D,
...). The harness constructs the adapter directly and passes the instance to
``simple_evaluate`` — there is no lmms-eval entry-point registration.

Requirements (lmms-eval is an OPTIONAL, separately-installed dependency, exactly
like lm-eval for text evaluation):

    uv pip install lmms-eval

v1 is single-GPU; MoMa checkpoints are not supported (see README.md in this
directory). On clusters where importing lmms-eval's evaluator fails with
``GLIBCXX_... not found``, put a newer libstdc++ on the library path
(e.g. ``LD_LIBRARY_PATH=<conda>/lib``).

Usage:
    uv run python examples/vlm-evaluation/vlm_eval_harness.py \
        --config configs/train/vlm_jd.toml \
        --checkpoint checkpoints/vlm/step_10000 \
        --tasks mmmu_val \
        --output results/vlm_step_10000.json

    # Quick partial run (4 examples per task)
    uv run python examples/vlm-evaluation/vlm_eval_harness.py \
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
            "path (e.g. LD_LIBRARY_PATH=<conda>/lib); see README.md in examples/vlm-evaluation/.",
            exc,
        )
        sys.exit(1)

    # The adapter imports lmms-eval at module top; the guard above already proved
    # it importable. The script's own directory is sys.path[0], so the sibling
    # adapter.py resolves as a top-level module.
    from adapter import KempnerForgeVLM

    logger.info(f"Running lmms-eval: tasks={args.tasks}, checkpoint={args.checkpoint}")

    model = KempnerForgeVLM(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,  # None -> adapter defaults to the checkpoint's train.param_dtype
    )

    metadata = model.run_metadata()
    results = simple_evaluate(
        model=model,
        # Record-only on the instance path: simple_evaluate never constructs or
        # configures a prebuilt model with these; it stores them in results["config"].
        model_args=metadata["model_args"],
        batch_size=args.batch_size,
        device=args.device,
        tasks=args.tasks.split(","),
        limit=args.limit,
    )

    if results is not None:
        # The two identity records simple_evaluate has no parameter for.
        config_block = results.setdefault("config", {})
        config_block["checkpoint"] = metadata["checkpoint"]
        config_block["job_config"] = metadata["job_config"]

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
