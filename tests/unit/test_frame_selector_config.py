"""Unit tests for FrameSelectorConfig (the [frame_selector] section)."""

from __future__ import annotations

import pytest

from kempnerforge.config.frame_selector import FrameSelectorConfig


class TestFrameSelectorConfig:
    def test_defaults(self):
        c = FrameSelectorConfig()
        assert c.type == "topk"
        assert c.scorer == "siglip2"
        assert c.scorer_path == "google/siglip2-so400m-patch14-224"
        assert c.candidate_frames == 128
        assert c.candidate_fps == 0.0
        assert c.query_source == "sample"
        assert c.mdp3_segment_size == 32

    def test_registers_selectors_on_construction(self):
        # __post_init__ late-imports the module, populating the registry.
        from kempnerforge.config.registry import registry

        FrameSelectorConfig()
        assert "mdp3" in registry.list_frame_selectors()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"type": "bogus"}, "Unknown frame_selector.type"),
            ({"scorer": "bogus"}, "frame_selector.scorer must be one of"),
            ({"scorer_path": ""}, "scorer_path must be non-empty"),
            ({"candidate_frames": 0}, "candidate_frames must be >= 1"),
            ({"candidate_fps": -1.0}, "candidate_fps must be non-negative"),
            ({"query_source": "bogus"}, "query_source must be one of"),
            ({"gumbel_tau": 0.0}, "gumbel_tau must be positive"),
            ({"mdp3_lambda": 0.0}, "mdp3_lambda must be positive"),
            ({"mdp3_segment_size": -1}, "mdp3_segment_size must be non-negative"),
            ({"mdp3_kernel": "bogus"}, "mdp3_kernel must be one of"),
        ],
    )
    def test_rejections(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FrameSelectorConfig(**kwargs)

    def test_segment_size_zero_allowed(self):
        # 0 is the plain-conditional-DPP ablation, not an error.
        assert FrameSelectorConfig(mdp3_segment_size=0).mdp3_segment_size == 0

    def test_extra_kwargs_exact(self):
        c = FrameSelectorConfig(
            candidate_frames=64,
            candidate_fps=2.0,
            gumbel_tau=0.5,
            seed=3,
            mdp3_lambda=0.1,
            mdp3_segment_size=16,
            mdp3_kernel="cosine",
        )
        assert c.extra_kwargs() == {
            "candidate_frames": 64,
            "candidate_fps": 2.0,
            "gumbel_tau": 0.5,
            "seed": 3,
            "mdp3_lambda": 0.1,
            "mdp3_segment_size": 16,
            "mdp3_kernel": "cosine",
        }
