#!/usr/bin/env python3
"""Run lmms-eval benchmarks on a KempnerForge VLM checkpoint.

Evaluates a VLM checkpoint via the ``KempnerForgeVLM`` lmms-eval chat-model
adapter, on the standard benchmarks lmms-eval implements as ``generate_until``
tasks.

Requirements (lmms-eval is an OPTIONAL, separately-installed dependency, exactly
like lm-eval for text evaluation):

    uv pip install lmms-eval

MoMa checkpoints are not supported (see README.md in this directory).
On clusters where importing lmms-eval's evaluator fails with
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

    # Log results to the checkpoint's training run (see README: Experiment tracking)
    uv run python examples/vlm-evaluation/vlm_eval_harness.py \
        --config configs/train/vlm_jd.toml \
        --checkpoint checkpoints/vlm/step_10000 \
        --tasks mmmu_val \
        --metrics.enable_wandb=true --metrics.wandb_project=vlm-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kempnerforge.config.schema import JobConfig

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


def _resolve_checkpoint(checkpoint: str) -> Path:
    """Resolve a checkpoint arg (run dir or step_N dir) to a concrete step directory.

    Mirrors how the adapter loads weights: a run directory resolves to its
    latest ``step_N`` via ``resolve_resume_path``.
    """
    from kempnerforge.resilience.elastic import resolve_resume_path

    return (resolve_resume_path(checkpoint) or Path(checkpoint)).resolve()


def _checkpoint_step(ckpt_dir: Path) -> int:
    """The checkpoint's training step: metadata.json, else the step_N dir name, else 0."""
    meta_file = ckpt_dir / "metadata.json"
    if meta_file.exists():
        try:
            return int(json.loads(meta_file.read_text())["step"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(f"Could not read step from {meta_file}; falling back to the dir name")
    step_suffix = ckpt_dir.name.removeprefix("step_")
    if ckpt_dir.name.startswith("step_") and step_suffix.isdigit():
        return int(step_suffix)
    logger.warning(f"Could not determine {ckpt_dir}'s training step; logging eval at step 0")
    return 0


def _resolve_run_id(config: JobConfig, ckpt_dir: Path) -> None:
    """Point ``config.metrics`` at the run these results belong to: an explicit
    ``--metrics.wandb_run_id`` override wins, else the id training saved into
    the checkpoint, else a fresh run named after the checkpoint.
    """
    mc = config.metrics
    if mc.wandb_run_id:
        return  # explicit override (or TOML) wins
    from kempnerforge.checkpoint import load_train_state_extras

    run_id = None
    try:
        run_id = load_train_state_extras(ckpt_dir).get("wandb_run_id")
    except Exception as exc:  # foreign-owned or corrupt train_state.pt — never fatal here
        logger.warning(f"Could not read {ckpt_dir / 'train_state.pt'} ({exc})")
    if run_id:
        mc.wandb_run_id = run_id
        logger.info(f"Attaching eval metrics to the checkpoint's training run ({run_id})")
        return
    logger.warning(
        f"{ckpt_dir} has no saved wandb_run_id — starting a fresh run "
        f"(attach to an existing one with --metrics.wandb_run_id=<id>)"
    )
    if mc.wandb_run_name is None:
        mc.wandb_run_name = f"{ckpt_dir.parent.name}-{ckpt_dir.name}"


def _track_eval(config: JobConfig, results: dict, tasks: list[str], ckpt_dir: Path) -> None:
    """Log eval metrics through the framework's MetricsTracker backends.

    ``gpu_peak_tflops`` is a nonzero sentinel: eval never computes MFU, and
    ``None``/``0.0`` would trigger the GPU probe. A tracking failure never
    fails a completed eval.
    """
    try:
        from benchmark_manifest import build_eval_metrics

        from kempnerforge.metrics.tracker import MetricsTracker

        _resolve_run_id(config, ckpt_dir)
        tracker = MetricsTracker(config, num_gpus=1, gpu_peak_tflops=1.0)
        tracker.init_backends(config)
        tracker.log_eval(build_eval_metrics(results, tasks), step=_checkpoint_step(ckpt_dir))
        tracker.close()
    except Exception as exc:  # tracking must never fail a completed eval
        logger.warning(f"Experiment tracking failed (eval results are unaffected): {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lmms-eval on a KempnerForge VLM checkpoint",
        epilog=(
            "Unrecognized --section.key=value arguments are forwarded to the KempnerForge "
            "config loader as dotted overrides on --config and apply to both the evaluated "
            "model (e.g. --video.max_frames=8) and experiment tracking, which is enabled "
            "that way, e.g. --metrics.enable_wandb=true --metrics.wandb_project=vlm-eval "
            "(see 'Experiment tracking' in README.md)."
        ),
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
    args, extra_overrides = parser.parse_known_args()

    # Forwarded --section.key=value overrides layer over the checkpoint TOML;
    # unknown keys raise here, before the expensive model build. The merged
    # config object is passed to the adapter below, so overrides reach the
    # evaluated model, not just experiment tracking.
    from kempnerforge.config.loader import load_config

    config = load_config(args.config, cli_args=extra_overrides)
    ckpt_dir = _resolve_checkpoint(args.checkpoint)

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
    # it importable. The script's own directory is normally sys.path[0], so the sibling
    # adapter.py resolves as a top-level module — but under `accelerate launch` the launcher
    # may run this file such that sys.path[0] is not its directory, so insert it explicitly
    # so `from adapter import ...` resolves on every rank.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from adapter import KempnerForgeVLM

    logger.info(f"Running lmms-eval: tasks={args.tasks}, checkpoint={args.checkpoint}")

    # Only pass dtype when explicitly set; otherwise the adapter defaults it from
    # the checkpoint config (train.param_dtype).
    dtype_kwargs = {"dtype": args.dtype} if args.dtype is not None else {}
    model = KempnerForgeVLM(
        config=config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        **dtype_kwargs,
    )
    # lmms-eval's evaluate() reads a few attributes off a `cli_args` namespace,
    # and some tasks dereference them directly rather than via getattr: judge-
    # scored tasks cache their GPT responses under `cli_args.output_path`, and
    # hallusion_bench's aggregation crashes on `args.output_path` when
    # `task.args` is None (which is what a missing cli_args leaves it as).
    judge_output_dir = Path(args.output).parent if args.output else Path(tempfile.mkdtemp())
    judge_output_dir.mkdir(parents=True, exist_ok=True)
    cli_args = argparse.Namespace(
        output_path=str(judge_output_dir),
        process_with_media=True,
    )
    results = simple_evaluate(
        model=model,
        tasks=args.tasks.split(","),
        limit=args.limit,
        cli_args=cli_args,
    )

    # Only rank 0 holds the aggregated results (simple_evaluate returns None on non-zero
    # ranks); every other DP rank must skip reporting so it does not print an empty banner,
    # dump `None` to --output, or race on the same file. Single-process: rank 0.
    if model.rank == 0:
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

        # --- Experiment tracking (opt-in) ---
        mc = config.metrics
        track = mc.enable_wandb or mc.enable_tensorboard
        if track and results is not None and "results" in results:
            _track_eval(config, results, args.tasks.split(","), ckpt_dir)


if __name__ == "__main__":
    main()
