"""Optimizer comparison benchmarks: step time and memory for all optimizers."""

from __future__ import annotations

import torch

from benchmarks.runner import BenchmarkResult, print_results, run_benchmark
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer

BENCH_CONFIG = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=32000,
    max_seq_len=2048,
)

BATCH = 4
SEQ = 512
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

OPTIMIZERS = [
    ("AdamW", OptimizerConfig(name="adamw", lr=3e-4, fused=True)),
    ("AdamW (unfused)", OptimizerConfig(name="adamw", lr=3e-4, fused=False)),
    ("Lion", OptimizerConfig(name="lion", lr=1e-4)),
    ("Schedule-Free AdamW", OptimizerConfig(name="schedule_free_adamw", lr=3e-4)),
    ("Muon", OptimizerConfig(name="muon", lr=0.02)),
]


def run_optimizer_benchmarks() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    toks_per_iter = BATCH * SEQ

    for opt_name, opt_config in OPTIMIZERS:
        model = Transformer(BENCH_CONFIG).to(device=DEVICE, dtype=DTYPE)
        optimizer = build_optimizer(model, opt_config)
        input_ids = torch.randint(0, 32000, (BATCH, SEQ), device=DEVICE)

        def step_fn(m=model, o=optimizer, ids=input_ids):
            logits = m(ids)
            loss = logits.sum()
            loss.backward()
            o.step()
            o.zero_grad()

        result = run_benchmark(
            step_fn,
            name=f"optimizer: {opt_name}",
            tokens_per_iter=toks_per_iter,
        )
        results.append(result)

        del model, optimizer
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmarks require a CUDA GPU")
    print_results(run_optimizer_benchmarks(), "Optimizer Comparison")
