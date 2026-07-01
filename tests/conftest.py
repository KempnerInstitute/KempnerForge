"""Shared test fixtures for KempnerForge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.config.schema import (
    AdapterConfig,
    CheckpointConfig,
    JobConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    VisionEncoderConfig,
    VLMConfig,
)
from kempnerforge.config.video import VideoConfig

# ---------------------------------------------------------------------------
# CLI flags for opt-in test suites
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--e2e", action="store_true", default=False, help="Run end-to-end tests")
    parser.addoption("--slow", action="store_true", default=False, help="Include slow e2e tests")
    parser.addoption("--smoke", action="store_true", default=False, help="Run smoke tests")


def pytest_collection_modifyitems(config, items):
    run_e2e = config.getoption("--e2e")
    run_slow = config.getoption("--slow")
    run_smoke = config.getoption("--smoke")

    skip_e2e = pytest.mark.skip(reason="E2E tests disabled (pass --e2e to run)")
    skip_slow = pytest.mark.skip(reason="Slow tests disabled (pass --slow to run)")
    skip_smoke = pytest.mark.skip(reason="Smoke tests disabled (pass --smoke to run)")

    for item in items:
        if "e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)
        elif "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        elif "smoke" in item.keywords and not run_smoke:
            item.add_marker(skip_smoke)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

TINY_MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

SMALL_MODEL_CONFIG = ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=512, max_seq_len=128)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=512, max_seq_len=128)


# ---------------------------------------------------------------------------
# Job configs
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_job_config(tmp_path) -> JobConfig:
    """Minimal job config for fast testing."""
    return JobConfig(
        model=ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64),
        train=TrainConfig(
            batch_size=2,
            seq_len=32,
            max_steps=10,
            grad_accum_steps=1,
            compile_model=False,
        ),
        optimizer=OptimizerConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(warmup_steps=2),
        checkpoint=CheckpointConfig(dir=str(tmp_path / "checkpoints"), interval=5),
    )


# ---------------------------------------------------------------------------
# VLM configs / model
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_vlm_configs() -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig]:
    """Tiny Joint-Decoder VLM configs for CPU tests (random vision encoder)."""
    return (
        ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
        VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
        AdapterConfig(),
        VLMConfig(max_text_len=32),
    )


@pytest.fixture
def tiny_vlm_wrapper(tiny_vlm_configs):
    """A tiny CPU ``VLMWrapper`` built from ``tiny_vlm_configs`` (no checkpoint)."""
    from kempnerforge.model.vlm import build_vlm_wrapper

    return build_vlm_wrapper(*tiny_vlm_configs).eval()


@pytest.fixture
def tiny_video_configs(
    tiny_vlm_configs,
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig, VideoConfig]:
    """Tiny configs for a video VLM checkpoint (``frames_per_clip == 2``).

    Reuses the image VLM configs and adds a ``[video]`` section sized so the
    visual budget (2 frames x 8 tokens = 16) plus ``max_text_len`` (32) fits the
    tiny ``max_seq_len`` (64).
    """
    mc, vc, ac, lc = tiny_vlm_configs
    video = VideoConfig(max_frames=2, min_frames=1, frame_size=16)
    return (mc, vc, ac, lc, video)


@pytest.fixture
def tiny_video_vlm_wrapper(tiny_video_configs):
    """A tiny CPU video ``VLMWrapper`` (``frames_per_clip == 2``; no checkpoint)."""
    from kempnerforge.model.vlm import build_vlm_wrapper

    mc, vc, ac, lc, video = tiny_video_configs
    return build_vlm_wrapper(mc, vc, ac, lc, frames_per_clip=video.max_frames).eval()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def random_batch(tiny_model_config, device) -> dict[str, torch.Tensor]:
    """A random batch of input_ids and labels."""
    batch_size, seq_len = 2, 32
    return {
        "input_ids": torch.randint(
            0, tiny_model_config.vocab_size, (batch_size, seq_len), device=device
        ),
        "labels": torch.randint(
            0, tiny_model_config.vocab_size, (batch_size, seq_len), device=device
        ),
    }


@pytest.fixture
def mmap_data_dir(tmp_path) -> str:
    """Create a temp directory with small .npy token files."""
    tokens = np.arange(1000, dtype=np.uint16)
    for i in range(2):
        np.save(tmp_path / f"shard_{i}.npy", tokens[i * 500 : (i + 1) * 500])
    return str(tmp_path)
