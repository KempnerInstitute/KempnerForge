"""Unit tests for KempnerForge performance modules (MFU, memory, profiling)."""

from __future__ import annotations

import torch

from kempnerforge.config.schema import ModelConfig, ProfilingConfig
from kempnerforge.metrics.memory import (
    format_memory_stats,
    get_memory_stats,
    get_memory_utilization,
)
from kempnerforge.metrics.mfu import (
    compute_mfu,
    estimate_model_flops_per_token,
    get_gpu_peak_tflops,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMALL_CONFIG = ModelConfig(
    dim=256, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=128
)


# ---------------------------------------------------------------------------
# MFU
# ---------------------------------------------------------------------------


class TestMFU:
    def test_flops_per_token_positive(self):
        flops = estimate_model_flops_per_token(SMALL_CONFIG)
        assert flops > 0

    def test_flops_scales_with_params(self):
        small = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=1000)
        large = ModelConfig(dim=512, n_layers=8, n_heads=8, vocab_size=1000)
        assert estimate_model_flops_per_token(large) > estimate_model_flops_per_token(small)

    def test_compute_mfu_basic(self):
        # 1M tokens/sec on a single GPU with 312 TFLOPS peak
        mfu = compute_mfu(SMALL_CONFIG, tokens_per_sec=1e6, num_gpus=1, gpu_peak_tflops=312.0)
        assert 0.0 < mfu < 1.0

    def test_compute_mfu_zero_throughput(self):
        mfu = compute_mfu(SMALL_CONFIG, tokens_per_sec=0.0, num_gpus=1, gpu_peak_tflops=312.0)
        assert mfu == 0.0

    def test_compute_mfu_scales_with_gpus(self):
        mfu_1 = compute_mfu(SMALL_CONFIG, tokens_per_sec=1e6, num_gpus=1, gpu_peak_tflops=312.0)
        mfu_4 = compute_mfu(SMALL_CONFIG, tokens_per_sec=1e6, num_gpus=4, gpu_peak_tflops=312.0)
        # Same throughput on 4 GPUs → lower MFU (more peak capacity)
        assert mfu_4 < mfu_1

    def test_gpu_peak_tflops_detected(self):
        tflops = get_gpu_peak_tflops()
        assert tflops > 0

    def test_llama_7b_flops_reasonable(self):
        config = ModelConfig(
            dim=4096, n_layers=32, n_heads=32, vocab_size=32000,
            ffn_hidden_dim=11008, max_seq_len=2048,
        )
        flops = estimate_model_flops_per_token(config)
        # Llama 7B: ~41e9 FLOPS/token (6*7B + attention term)
        assert 30e9 < flops < 60e9


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------


class TestMemory:
    def test_get_memory_stats_keys(self):
        stats = get_memory_stats()
        assert "allocated_gb" in stats
        assert "peak_gb" in stats
        assert "reserved_gb" in stats
        assert "total_gb" in stats

    def test_memory_utilization_range(self):
        util = get_memory_utilization()
        assert 0.0 <= util <= 1.0

    def test_format_memory_stats_string(self):
        s = format_memory_stats()
        assert "GPU mem:" in s

    def test_memory_after_allocation(self):
        if not torch.cuda.is_available():
            return
        # Allocate a tensor and check memory increased
        before = get_memory_stats()
        x = torch.randn(1024, 1024, device="cuda")
        after = get_memory_stats()
        assert after["allocated_gb"] >= before["allocated_gb"]
        del x


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class TestProfiler:
    def test_build_profiler_disabled(self):
        from kempnerforge.profiling.profiler import build_profiler

        config = ProfilingConfig(enable=False)
        prof = build_profiler(config)
        assert prof is None

    def test_build_profiler_enabled(self, tmp_path):
        from kempnerforge.profiling.profiler import build_profiler

        config = ProfilingConfig(enable=True, start_step=2, end_step=5, trace_dir=str(tmp_path))
        prof = build_profiler(config)
        assert prof is not None

    def test_cuda_timer(self):
        if not torch.cuda.is_available():
            return
        from kempnerforge.profiling.profiler import CUDATimer

        timer = CUDATimer()
        timer.start()
        # Do some GPU work
        x = torch.randn(1000, 1000, device="cuda")
        _ = x @ x
        timer.stop()
        elapsed = timer.elapsed_ms()
        assert elapsed > 0


# ---------------------------------------------------------------------------
# torch.compile correctness
# ---------------------------------------------------------------------------


class TestCompile:
    def test_compile_produces_same_output(self):
        """Compiled model should produce identical output to eager mode."""
        if not torch.cuda.is_available():
            return

        from kempnerforge.model.transformer import Transformer

        config = ModelConfig(
            dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32
        )
        model = Transformer(config).to(DEVICE).eval()

        tokens = torch.randint(0, 256, (1, 16), device=DEVICE)

        with torch.no_grad():
            eager_out = model(tokens)

        compiled = torch.compile(model)
        with torch.no_grad():
            compiled_out = compiled(tokens)

        assert torch.allclose(eager_out, compiled_out, atol=1e-4), (
            f"Compiled output differs: max diff={( eager_out - compiled_out).abs().max().item()}"
        )
