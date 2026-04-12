"""Data pipeline benchmarks: mmap iteration, packing, mixture sampling."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.runner import BenchmarkResult, print_results


def run_data_benchmarks() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic data files
        seq_len = 513  # +1 for input/label split
        n_tokens = 100_000
        for i in range(4):
            tokens = np.random.randint(0, 32000, size=n_tokens, dtype=np.uint16)
            np.save(tmpdir / f"shard_{i}.npy", tokens)

        # --- MemoryMappedDataset iteration ---
        from kempnerforge.data.dataset import MemoryMappedDataset

        dataset = MemoryMappedDataset(data_dir=str(tmpdir), seq_len=seq_len)
        n_samples = min(len(dataset), 1000)

        def mmap_iter(ds=dataset, n=n_samples):
            for i in range(n):
                _ = ds[i]

        # Warmup
        for _ in range(2):
            mmap_iter()

        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            mmap_iter()
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        samples_per_sec = n_samples / (elapsed / iterations)

        results.append(
            BenchmarkResult(
                name=f"MemoryMappedDataset ({n_samples} samples)",
                step_time_ms=avg_ms,
                tokens_per_sec=samples_per_sec * seq_len,
            )
        )

        # --- Packed dataset ---
        dataset_packed = MemoryMappedDataset(
            data_dir=str(tmpdir), seq_len=seq_len, pack_sequences=True, eos_token_id=2
        )
        n_packed = min(len(dataset_packed), 1000)

        def packed_iter(ds=dataset_packed, n=n_packed):
            for i in range(n):
                _ = ds[i]

        for _ in range(2):
            packed_iter()

        start = time.perf_counter()
        for _ in range(iterations):
            packed_iter()
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        results.append(
            BenchmarkResult(
                name=f"MemoryMappedDataset packed ({n_packed} samples)",
                step_time_ms=avg_ms,
                tokens_per_sec=n_packed * seq_len / (elapsed / iterations),
            )
        )

        # --- MixtureSampler ---
        from kempnerforge.data.sampler import MixtureSampler

        base_size = n_tokens // seq_len
        for n_datasets in [2, 4, 8]:
            cumulative = [base_size * (i + 1) for i in range(n_datasets)]
            weights = [1.0] * n_datasets
            sampler = MixtureSampler(
                cumulative_sizes=cumulative,
                weights=weights,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=42,
            )

            def sampler_iter(s=sampler):
                return len(list(s))

            for _ in range(2):
                sampler_iter()

            start = time.perf_counter()
            for _ in range(iterations):
                sampler_iter()
            elapsed = time.perf_counter() - start
            avg_ms = (elapsed / iterations) * 1000

            results.append(
                BenchmarkResult(
                    name=f"MixtureSampler ({n_datasets} datasets)",
                    step_time_ms=avg_ms,
                )
            )

    return results


if __name__ == "__main__":
    print_results(run_data_benchmarks(), "Data Pipeline Benchmarks")
