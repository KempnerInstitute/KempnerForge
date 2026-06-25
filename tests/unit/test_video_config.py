"""Unit tests for VideoConfig and its JobConfig wiring."""

from __future__ import annotations

import pytest

from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.video import VideoConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import JointDecoderConfig


class TestVideoConfig:
    def test_defaults(self):
        cfg = VideoConfig()
        assert cfg.split == "train"
        assert cfg.max_frames == 16
        assert cfg.min_frames == 4
        assert cfg.fps == 2.0
        assert cfg.frame_size == 224
        assert cfg.max_samples == 0

    def test_bad_split_rejected(self):
        with pytest.raises(ValueError, match="video.split"):
            VideoConfig(split="test")

    def test_min_greater_than_max_rejected(self):
        with pytest.raises(ValueError, match="must be <="):
            VideoConfig(min_frames=8, max_frames=4)

    def test_non_positive_fps_rejected(self):
        with pytest.raises(ValueError, match="video.fps must be positive"):
            VideoConfig(fps=0.0)

    def test_non_positive_frame_size_rejected(self):
        with pytest.raises(ValueError, match="video.frame_size must be positive"):
            VideoConfig(frame_size=0)

    def test_negative_max_samples_rejected(self):
        with pytest.raises(ValueError, match="video.max_samples"):
            VideoConfig(max_samples=-1)

    def test_min_frames_below_one_rejected(self):
        with pytest.raises(ValueError, match="video.min_frames must be >= 1"):
            VideoConfig(min_frames=0)

    def test_max_frames_below_one_rejected(self):
        with pytest.raises(ValueError, match="video.max_frames must be >= 1"):
            VideoConfig(max_frames=0)


class TestJobConfigVideoWiring:
    def _vlm_kwargs(self) -> dict:
        return {
            "model": ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
            "vision_encoder": VisionEncoderConfig(type="random"),
            "vlm": JointDecoderConfig(max_text_len=32),
        }

    def test_video_requires_vlm(self):
        with pytest.raises(ValueError, match=r"\[video\] is set but \[vlm\] is missing"):
            JobConfig(video=VideoConfig(data_root="/some/root"))

    def test_valid_video_job(self):
        cfg = JobConfig(video=VideoConfig(data_root="/some/root"), **self._vlm_kwargs())
        assert cfg.is_video is True
        assert cfg.is_vlm is True

    def test_is_video_false_without_section(self):
        cfg = JobConfig(**self._vlm_kwargs())
        assert cfg.is_video is False
        assert cfg.is_vlm is True

    def test_text_only_job_is_not_video(self):
        cfg = JobConfig()
        assert cfg.is_video is False
        assert cfg.is_vlm is False
