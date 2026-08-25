"""Shared fixtures for the VLM-evaluation example tests.

Two jobs:

1. Path bootstrap: put the example root on ``sys.path`` so every test imports
   the adapter as the bare top-level module ``adapter`` — the same name the
   harness resolves when run as a script beside it.
2. Tiny VLM fixtures: this example is standalone (outside the main suite's
   ``testpaths``), so the tiny-model fixtures its tests use are duplicated here
   rather than shared from the repo-root ``tests/conftest.py``, which keeps its
   own copies for the core tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
