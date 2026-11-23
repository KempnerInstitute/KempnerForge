"""Unit tests for VideoConfig and its JobConfig wiring."""

from __future__ import annotations

import re

import pytest

from kempnerforge.config.data import DataConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.config.video import VideoConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import JointDecoderConfig

_DATA_HF_FIELDS = [
    ("hf_dataset_name", "org/image-captions", "set video.data_root and video.dataset_name instead"),
    ("hf_dataset_config", "captions", "no [video] counterpart"),
    ("hf_dataset_split", "validation", "set video.split instead"),
    ("hf_dataset_text_field", "caption", "no [video] counterpart"),
    ("hf_dataset_image_field", "jpg", "no [video] counterpart"),
    (
        "hf_dataset_prompt_field",
        "question",
        "set video.prompt instead, one instruction for every clip",
    ),
    ("hf_image_size", 448, "set video.frame_size instead"),
    ("hf_streaming", True, "no [video] counterpart"),
]


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

    def test_pluggable_defaults(self):
        cfg = VideoConfig()
        assert cfg.dataset_type == "webvid"
        assert cfg.dataset_name == "webvid-10M"
        assert cfg.sampling_policy == "uniform"

    def test_unknown_dataset_type_rejected(self):
        with pytest.raises(ValueError, match="video.dataset_type must be one of"):
            VideoConfig(dataset_type="bogus")

    def test_unknown_sampling_policy_rejected(self):
        with pytest.raises(ValueError, match="video.sampling_policy must be one of"):
            VideoConfig(sampling_policy="bogus")


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

    def _video_job(self, **data) -> JobConfig:
        return JobConfig(
            video=VideoConfig(data_root="/some/root"),
            train=TrainConfig(seq_len=64),
            data=DataConfig(tokenizer_path="gpt2", **data),
            **self._vlm_kwargs(),
        )

    @pytest.mark.parametrize(
        ("field", "value", "instead"), _DATA_HF_FIELDS, ids=[f for f, _, _ in _DATA_HF_FIELDS]
    )
    def test_a_data_hf_field_fails_validation(self, field, value, instead):
        """The video dataset reads none of them, so a run would drop the setting."""
        config = self._video_job(**{field: value})
        with pytest.raises(ValueError, match=re.escape(f"data.{field} = {value!r} ({instead})")):
            config.validate(world_size=1)

    def test_the_error_names_each_field_and_its_default(self):
        config = self._video_job(
            hf_dataset_name="org/image-captions",
            hf_dataset_split="validation",
            hf_dataset_prompt_field="question",
            hf_image_size=448,
        )
        with pytest.raises(ValueError) as info:
            config.validate(world_size=1)
        assert str(info.value) == (
            "[video] is set, so the dataset comes from [video] and these [data] fields would be "
            "ignored: data.hf_dataset_name = 'org/image-captions' (set video.data_root and "
            "video.dataset_name instead); data.hf_dataset_split = 'validation' (set video.split "
            "instead); data.hf_dataset_prompt_field = 'question' (set video.prompt instead, one "
            "instruction for every clip); data.hf_image_size = 448 (set video.frame_size "
            "instead). Leave them at their defaults (hf_dataset_name = None, hf_dataset_split = "
            "'train', hf_dataset_prompt_field = '', hf_image_size = 224)."
        )

    def test_fields_set_to_their_defaults_pass(self):
        """The fields are compared with their defaults, several of which are not empty."""
        self._video_job(
            hf_dataset_split="train",
            hf_dataset_text_field="text",
            hf_dataset_image_field="image",
            hf_image_size=224,
        ).validate(world_size=1)

    def test_an_image_run_keeps_them(self):
        JobConfig(
            train=TrainConfig(seq_len=64),
            data=DataConfig(
                hf_dataset_name="org/image-captions",
                hf_dataset_prompt_field="question",
                hf_image_size=448,
            ),
            **self._vlm_kwargs(),
        ).validate(world_size=1)

    def test_a_command_line_override_can_reset_a_toml_field(self, tmp_path):
        """The check reads the config after the CLI overrides land."""
        toml = tmp_path / "video.toml"
        toml.write_text(
            "[model]\ndim = 64\nn_layers = 2\nn_heads = 4\nn_kv_heads = 4\nvocab_size = 256\n"
            "max_seq_len = 64\n\n"
            '[train]\nseq_len = 64\n\n[data]\ntokenizer_path = "gpt2"\nhf_image_size = 448\n\n'
            '[vision_encoder]\ntype = "random"\n\n[vlm]\nmax_text_len = 32\n\n'
            '[video]\ndata_root = "root"\n'
        )
        with pytest.raises(ValueError, match=r"data\.hf_image_size = 448"):
            load_config(toml, cli_args=[]).validate(world_size=1)
        load_config(toml, cli_args=["--data.hf_image_size=224"]).validate(world_size=1)
