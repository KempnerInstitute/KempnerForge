"""Benchmark runner for KempnerForge.

Provides timing utilities and a CLI to run all benchmarks and produce a results
table. Uses CUDA events for accurate GPU timing.

Usage:
    # Run all benchmarks
    uv run python benchmarks/runner.py

    # Run a specific benchmark file
    uv run python benchmarks/bench_forward.py

    # Save results to JSON
    uv run python benchmarks/runner.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    name: str
    tokens_per_sec: float = 0.0
    peak_memory_gb: float = 0.0
    step_time_ms: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"{self.name:<40}"]
        if self.step_time_ms > 0:
            parts.append(f"step: {self.step_time_ms:>8.2f} ms")
        if self.tokens_per_sec > 0:
            parts.append(f"tok/s: {self.tokens_per_sec:>12,.0f}")
        if self.peak_memory_gb > 0:
            parts.append(f"mem: {self.peak_memory_gb:>6.2f} GB")
        for k, v in self.extra.items():
            parts.append(f"{k}: {v:.4f}")
        return "  |  ".join(parts)


def run_benchmark(
    fn: callable,
    warmup: int = 3,
    iterations: int = 10,
    name: str = "benchmark",
    tokens_per_iter: int = 0,
) -> BenchmarkResult:
    """Run a benchmark function with CUDA event timing.

    Args:
        fn: Function to benchmark. Called with no arguments.
        warmup: Number of warmup iterations (not timed).
        iterations: Number of timed iterations.
        name: Name for the result.
        tokens_per_iter: Tokens processed per iteration (for tok/s calculation).

    Returns:
        BenchmarkResult with timing and memory statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # Warmup
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    # Reset peak memory
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Timed iterations
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        total_ms = (time.perf_counter() - start) * 1000

    avg_ms = total_ms / iterations
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
    tok_per_sec = (tokens_per_iter / (avg_ms / 1000)) if tokens_per_iter > 0 else 0.0

    return BenchmarkResult(
        name=name,
        step_time_ms=avg_ms,
        tokens_per_sec=tok_per_sec,
        peak_memory_gb=peak_mem,
    )


def print_results(results: list[BenchmarkResult], title: str = "Benchmark Results") -> None:
    """Print a formatted table of benchmark results."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    for r in results:
        print(f"  {r}")
    print(f"{'=' * 80}\n")


def save_results(results: list[BenchmarkResult], path: str | Path) -> None:
    """Save benchmark results to a JSON file."""
    path = Path(path)
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {path}")


def main() -> None:
    """Run all benchmark suites."""
    parser = argparse.ArgumentParser(description="KempnerForge benchmarks")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: Benchmarks require a CUDA GPU.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    all_results: list[BenchmarkResult] = []

    # Import and run each benchmark suite
    from benchmarks.bench_data import run_data_benchmarks
    from benchmarks.bench_forward import run_forward_benchmarks
    from benchmarks.bench_moe import run_moe_benchmarks
    from benchmarks.bench_optimizer import run_optimizer_benchmarks

    all_results.extend(run_forward_benchmarks())
    all_results.extend(run_moe_benchmarks())
    all_results.extend(run_data_benchmarks())
    all_results.extend(run_optimizer_benchmarks())

    print_results(all_results, "All Benchmarks")

    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
