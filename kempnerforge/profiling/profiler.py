"""torch.profiler integration for KempnerForge.

Provides a step-aware profiler wrapper that activates only within a
configured step range, exports Chrome traces, and integrates with
the training loop via a simple .step() interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from kempnerforge.config.schema import ProfilingConfig

logger = logging.getLogger(__name__)


def build_profiler(
    config: ProfilingConfig,
    rank: int = 0,
) -> torch.profiler.profile | None:
    """Build a torch.profiler instance from config.

    Returns None if profiling is disabled.

    Args:
        config: Profiling configuration.
        rank: Current rank (for output directory naming).

    Returns:
        A torch.profiler.profile context manager, or None.
    """
    if not config.enable:
        return None

    trace_dir = Path(config.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Profile schedule: wait → warmup → active → repeat
    # wait: skip steps before start_step
    # warmup: 1 step to stabilize profiler
    # active: profile for (end_step - start_step) steps
    wait_steps = max(0, config.start_step - 1)
    active_steps = config.end_step - config.start_step

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=wait_steps,
            warmup=1,
            active=active_steps,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    )

    logger.info(
        f"Profiler configured: steps {config.start_step}–{config.end_step}, traces → {trace_dir}"
    )

    return prof


def _analyze_profiler(prof: torch.profiler.profile) -> dict:
    """Analyze profiler events and return aggregate stats.

    Returns a dict with keys: total_cuda_time_us, matmul_time_us, comm_time_us,
    memory_time_us, other_time_us, total_flops, achieved_tflops, peak_tflops,
    and percentage breakdowns.
    """
    from kempnerforge.metrics.mfu import get_gpu_peak_tflops

    total_cuda_time = 0
    matmul_time = 0
    comm_time = 0
    memory_time = 0
    other_time = 0
    total_flops = 0

    for evt in prof.key_averages():
        cuda_us = evt.self_device_time_total
        name = evt.key.lower()

        if "profilerstep" in name:
            continue

        total_cuda_time += cuda_us
        total_flops += evt.flops if evt.flops else 0

        if any(
            k in name
            for k in ["gemm", "mm", "matmul", "dot", "bmm", "cublas", "nvjet", "cutlass"]
        ):
            matmul_time += cuda_us
        elif any(k in name for k in ["nccl", "allreduce", "allgather", "reduce_scatter"]):
            comm_time += cuda_us
        elif any(k in name for k in ["memcpy", "memset"]):
            memory_time += cuda_us
        else:
            other_time += cuda_us

    denom = max(total_cuda_time, 1)
    peak_tflops = get_gpu_peak_tflops()
    achieved_tflops = (
        total_flops / (total_cuda_time / 1e6) / 1e12 if total_cuda_time > 0 else 0.0
    )

    return {
        "total_cuda_time_us": total_cuda_time,
        "matmul_time_us": matmul_time,
        "comm_time_us": comm_time,
        "memory_time_us": memory_time,
        "other_time_us": other_time,
        "matmul_pct": 100 * matmul_time / denom,
        "comm_pct": 100 * comm_time / denom,
        "memory_pct": 100 * memory_time / denom,
        "other_pct": 100 * other_time / denom,
        "total_flops": total_flops,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak_tflops,
        "kernel_efficiency_pct": 100 * achieved_tflops / peak_tflops if peak_tflops > 0 else 0.0,
    }


def print_profiler_summary(prof: torch.profiler.profile, trace_dir: str | None = None) -> None:
    """Print kernel-level GPU profiling summary and optionally save to file.

    Prints top CUDA kernels by time and FLOPS, an aggregate GPU time
    breakdown (matmul, communication, memory, other), and achieved
    TFLOPS vs hardware peak.

    If trace_dir is provided, writes a summary.md file alongside the traces.

    Args:
        prof: A completed torch.profiler.profile instance.
        trace_dir: Optional directory to save summary.md report.
    """
    stats = _analyze_profiler(prof)

    print("\n" + "=" * 100)
    print("TOP CUDA KERNELS (by total CUDA time)")
    print("=" * 100)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30, top_level_events_only=False
        )
    )

    print("\n" + "=" * 100)
    print("TOP CUDA KERNELS (by FLOPS)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by="flops", row_limit=20, top_level_events_only=False))

    print("\n" + "=" * 100)
    print("AGGREGATE GPU TIME BREAKDOWN")
    print("=" * 100)
    print(f"  Total CUDA time:    {stats['total_cuda_time_us'] / 1e6:.3f} s")
    mm_s = stats["matmul_time_us"] / 1e6
    comm_s = stats["comm_time_us"] / 1e6
    mem_s = stats["memory_time_us"] / 1e6
    other_s = stats["other_time_us"] / 1e6
    print(f"  MatMul/GEMM:        {mm_s:.3f} s ({stats['matmul_pct']:.1f}%)")
    print(f"  Communication:      {comm_s:.3f} s ({stats['comm_pct']:.1f}%)")
    print(f"  Memory ops:         {mem_s:.3f} s ({stats['memory_pct']:.1f}%)")
    print(f"  Other kernels:      {other_s:.3f} s ({stats['other_pct']:.1f}%)")
    print(f"  Total FLOPS:        {stats['total_flops'] / 1e12:.2f} TFLOP")
    if stats["total_cuda_time_us"] > 0:
        print(f"  Achieved TFLOPS:    {stats['achieved_tflops']:.1f}")
        print(f"  GPU peak (bf16):    {stats['peak_tflops']:.0f} TFLOPS")
        print(f"  Kernel efficiency:  {stats['kernel_efficiency_pct']:.1f}%")

    if trace_dir is not None:
        _save_profiler_summary(stats, prof, trace_dir)


def _save_profiler_summary(
    stats: dict, prof: torch.profiler.profile, trace_dir: str
) -> None:
    """Save a markdown summary report alongside the trace files."""
    from datetime import datetime

    out_path = Path(trace_dir) / "summary.md"

    # Build top kernels table (clean markdown)
    kernel_rows = []
    events = sorted(
        prof.key_averages(),
        key=lambda e: e.self_device_time_total,
        reverse=True,
    )
    for evt in events[:20]:
        name = evt.key
        if "profilerstep" in name.lower():
            continue
        cuda_us = evt.self_device_time_total
        pct = 100 * cuda_us / max(stats["total_cuda_time_us"], 1)
        calls = evt.count
        flops_str = f"{evt.flops / 1e9:.1f}" if evt.flops else "—"
        # Truncate long kernel names
        if len(name) > 60:
            name = name[:57] + "..."
        kernel_rows.append(
            f"| {name} | {cuda_us / 1e3:.1f} | {pct:.1f} | {calls} | {flops_str} |"
        )

    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    total_s = stats["total_cuda_time_us"] / 1e6

    lines = [
        "# Profiling Summary",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**GPU**: {gpu_name}",
        f"**Traces**: `{trace_dir}/`",
        "",
        "## GPU Time Breakdown",
        "",
        "| Category | Time (s) | % |",
        "|----------|--------:|---:|",
        f"| MatMul/GEMM | {stats['matmul_time_us'] / 1e6:.3f} | {stats['matmul_pct']:.1f} |",
        f"| Communication (NCCL) | {stats['comm_time_us'] / 1e6:.3f} | {stats['comm_pct']:.1f} |",
        f"| Memory ops | {stats['memory_time_us'] / 1e6:.3f} | {stats['memory_pct']:.1f} |",
        f"| Other kernels | {stats['other_time_us'] / 1e6:.3f} | {stats['other_pct']:.1f} |",
        f"| **Total** | **{total_s:.3f}** | **100.0** |",
        "",
        "## Efficiency",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total FLOPS | {stats['total_flops'] / 1e12:.2f} TFLOP |",
        f"| Achieved TFLOPS | {stats['achieved_tflops']:.1f} |",
        f"| GPU peak (bf16) | {stats['peak_tflops']:.0f} TFLOPS |",
        f"| Kernel efficiency | {stats['kernel_efficiency_pct']:.1f}% |",
        "",
        "## Top CUDA Kernels",
        "",
        "| Kernel | CUDA (ms) | % | Calls | GFLOPS |",
        "|--------|----------:|---:|------:|-------:|",
        *kernel_rows,
        "",
        "## Viewing Traces",
        "",
        "Load the `.json` trace files in [Perfetto UI](https://ui.perfetto.dev/) or TensorBoard:",
        "",
        "```bash",
        f"tensorboard --logdir {trace_dir}",
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines))
    logger.info(f"Profiling summary saved to {out_path}")
