"""Unit tests for KempnerForge performance modules (MFU, memory, profiling)."""

from __future__ import annotations

import math

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

SMALL_CONFIG = ModelConfig(dim=256, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=128)


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
            dim=4096,
            n_layers=32,
            n_heads=32,
            vocab_size=32000,
            ffn_hidden_dim=11008,
            max_seq_len=2048,
        )
        flops = estimate_model_flops_per_token(config)
        # Llama 7B: ~41e9 FLOPS/token (6*7B + attention term)
        assert 30e9 < flops < 60e9

    def test_moe_uses_active_params_not_total(self):
        """MoE MFU counts only top_k experts per token, not all experts."""
        base = dict(dim=256, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=128)
        dense = ModelConfig(**base)
        # 8 experts, top_k=2 → 2 active experts per token
        moe = ModelConfig(**base, num_experts=8, moe_top_k=2)

        dense_flops = estimate_model_flops_per_token(dense)
        moe_flops = estimate_model_flops_per_token(moe)

        # MoE with top_k=2 should have ~2x the MLP flops of dense (2 experts active),
        # NOT 8x (all experts). So MoE flops should be more than dense but less than 4x.
        assert moe_flops > dense_flops, "MoE should have more flops than dense (extra experts)"
        assert moe_flops < 4 * dense_flops, "MoE flops too high — counting all experts, not active"

    def test_moe_shared_experts_add_flops(self):
        """Shared experts add to MoE active flops."""
        base = dict(dim=256, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=128)
        moe_no_shared = ModelConfig(**base, num_experts=8, moe_top_k=2, moe_shared_experts=0)
        moe_shared = ModelConfig(**base, num_experts=8, moe_top_k=2, moe_shared_experts=1)

        assert estimate_model_flops_per_token(moe_shared) > estimate_model_flops_per_token(
            moe_no_shared
        )

    def test_moe_frequency_affects_flops(self):
        """moe_frequency=2 → half the layers are MoE, less active flops than frequency=1."""
        base = dict(dim=256, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=128)
        all_moe = ModelConfig(**base, num_experts=8, moe_top_k=2, moe_frequency=1)
        half_moe = ModelConfig(**base, num_experts=8, moe_top_k=2, moe_frequency=2)

        assert estimate_model_flops_per_token(all_moe) > estimate_model_flops_per_token(half_moe)

    def test_moe_seq_len_override(self):
        """Explicit seq_len should override config.max_seq_len in flops calculation."""
        config = ModelConfig(
            dim=256,
            n_layers=4,
            n_heads=4,
            vocab_size=1000,
            max_seq_len=128,
            num_experts=4,
            moe_top_k=2,
        )
        flops_short = estimate_model_flops_per_token(config, seq_len=64)
        flops_long = estimate_model_flops_per_token(config, seq_len=256)
        assert flops_long > flops_short

    def test_gpu_peak_tflops_cpu_only(self, monkeypatch):
        """Without CUDA, get_gpu_peak_tflops returns the 1.0 dummy."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert get_gpu_peak_tflops() == 1.0

    def test_gpu_peak_tflops_unknown_hopper(self, monkeypatch):
        """Unknown GPU with compute capability >= 9 falls back to 989 TFLOPS."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=0: "Fake-Unknown-GPU-9000")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=0: (9, 0))
        assert get_gpu_peak_tflops() == 989.0

    def test_gpu_peak_tflops_unknown_ampere(self, monkeypatch):
        """Unknown GPU with compute capability 8.x falls back to 312 TFLOPS."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=0: "Fake-Unknown-GPU-9000")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=0: (8, 0))
        assert get_gpu_peak_tflops() == 312.0

    def test_gpu_peak_tflops_unknown_older(self, monkeypatch):
        """Unknown GPU with compute capability < 8 falls back to 100 TFLOPS."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=0: "Fake-Unknown-GPU-9000")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=0: (7, 5))
        assert get_gpu_peak_tflops() == 100.0

    def test_compute_mfu_zero_peak(self):
        """compute_mfu returns 0.0 when peak * num_gpus == 0 to avoid div-by-zero."""
        result = compute_mfu(SMALL_CONFIG, tokens_per_sec=1e6, num_gpus=1, gpu_peak_tflops=0.0)
        assert result == 0.0

    def test_compute_mfu_auto_detects_gpu_peak(self, monkeypatch):
        """compute_mfu(..., gpu_peak_tflops=None) auto-detects via get_gpu_peak_tflops."""
        monkeypatch.setattr("kempnerforge.metrics.mfu.get_gpu_peak_tflops", lambda device=0: 100.0)
        result = compute_mfu(SMALL_CONFIG, tokens_per_sec=1e6, num_gpus=1)
        assert math.isfinite(result)
        assert result > 0


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
        from kempnerforge.profiling.cuda_timer import CUDATimer

        timer = CUDATimer()
        timer.start()
        # Do some GPU work
        x = torch.randn(1000, 1000, device="cuda")
        _ = x @ x
        timer.stop()
        elapsed = timer.elapsed_ms()
        assert elapsed > 0

    def test_analyze_profiler_extracts_stats(self):
        """_analyze_profiler should return a dict with expected keys from mock profiler."""
        from unittest.mock import MagicMock

        from kempnerforge.profiling.profiler import _analyze_profiler

        # Create mock profiler events
        matmul_evt = MagicMock()
        matmul_evt.key = "aten::mm"
        matmul_evt.self_device_time_total = 5000  # 5000 us
        matmul_evt.flops = int(1e12)  # 1 TFLOP

        nccl_evt = MagicMock()
        nccl_evt.key = "nccl:allreduce"
        nccl_evt.self_device_time_total = 2000
        nccl_evt.flops = 0

        other_evt = MagicMock()
        other_evt.key = "aten::relu"
        other_evt.self_device_time_total = 1000
        other_evt.flops = 0

        mock_prof = MagicMock()
        mock_prof.key_averages.return_value = [matmul_evt, nccl_evt, other_evt]

        stats = _analyze_profiler(mock_prof)

        assert stats["matmul_time_us"] == 5000
        assert stats["comm_time_us"] == 2000
        assert stats["other_time_us"] == 1000
        assert stats["total_cuda_time_us"] == 8000
        assert stats["total_flops"] == int(1e12)
        assert "achieved_tflops" in stats
        assert "peak_tflops" in stats
        # Matmul should be 5000/8000 = 62.5%
        assert abs(stats["matmul_pct"] - 62.5) < 0.1

    def test_save_profiler_summary_creates_file(self, tmp_path):
        """_save_profiler_summary should write a summary.md file."""
        from unittest.mock import MagicMock

        from kempnerforge.profiling.profiler import _save_profiler_summary

        mock_evt = MagicMock()
        mock_evt.key = "aten::mm"
        mock_evt.self_device_time_total = 5000
        mock_evt.flops = int(1e12)
        mock_evt.count = 10

        mock_prof = MagicMock()
        mock_prof.key_averages.return_value = [mock_evt]

        stats = {
            "total_cuda_time_us": 8000,
            "matmul_time_us": 5000,
            "comm_time_us": 2000,
            "memory_time_us": 500,
            "other_time_us": 500,
            "matmul_pct": 62.5,
            "comm_pct": 25.0,
            "memory_pct": 6.25,
            "other_pct": 6.25,
            "total_flops": int(1e12),
            "achieved_tflops": 125.0,
            "peak_tflops": 989.0,
            "kernel_efficiency_pct": 12.6,
        }

        _save_profiler_summary(stats, mock_prof, str(tmp_path))

        summary = tmp_path / "summary.md"
        assert summary.exists()
        content = summary.read_text()
        assert "Profiling Summary" in content
        assert "MatMul/GEMM" in content
        assert "aten::mm" in content

    def test_cuda_timer_collection_disabled(self):
        """Disabled timer collection should be zero-cost no-ops."""
        from kempnerforge.profiling.cuda_timer import CUDATimerCollection

        timers = CUDATimerCollection(regions=["forward", "backward"], enabled=False)
        assert not timers.enabled
        timers.start("forward")
        timers.stop("forward")
        assert timers.elapsed_ms("forward") == 0.0
        report = timers.elapsed_all()
        assert report == {"forward": 0.0, "backward": 0.0}

    def test_cuda_timer_collection_enabled_no_gpu(self):
        """Enabled collection should create timers for each region."""
        if not torch.cuda.is_available():
            return
        from kempnerforge.profiling.cuda_timer import CUDATimerCollection

        timers = CUDATimerCollection(regions=["forward", "backward", "comm"])
        assert timers.enabled
        timers.start("forward")
        x = torch.randn(100, 100, device="cuda")
        _ = x @ x
        timers.stop("forward")
        report = timers.elapsed_all()
        assert report["forward"] > 0
        assert "backward" in report
        assert "comm" in report


# ---------------------------------------------------------------------------
# torch.compile correctness
# ---------------------------------------------------------------------------


class TestCompile:
    def test_compile_produces_same_output(self):
        """Compiled model should produce identical output to eager mode."""
        if not torch.cuda.is_available():
            return

        from kempnerforge.model.transformer import Transformer

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        model = Transformer(config).to(DEVICE).eval()

        tokens = torch.randint(0, 256, (1, 16), device=DEVICE)

        with torch.no_grad():
            eager_out = model(tokens)

        compiled = torch.compile(model)
        with torch.no_grad():
            compiled_out = compiled(tokens)

        assert torch.allclose(eager_out, compiled_out, atol=1e-4), (
            f"Compiled output differs: max diff={(eager_out - compiled_out).abs().max().item()}"
        )
