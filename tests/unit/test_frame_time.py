"""Unit tests for FrameTimeEmbedding (per-frame timestamp encoding)."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.model.frame_time import FrameTimeEmbedding, build_time_embedding


class TestFrameTimeEmbedding:
    def test_output_shape(self):
        emb = FrameTimeEmbedding(dim=64, num_bands=8)
        out = emb(torch.zeros(2, 4))  # (B, F) -> (B, F, dim)
        assert out.shape == (2, 4, 64)

    def test_zero_init_is_zero(self):
        # Zero-init proj => the temporal signal starts at exactly zero, so adding
        # it is identity at step 0 (the CrossAttention warm-start convention).
        emb = FrameTimeEmbedding(dim=32, num_bands=8)
        out = emb(torch.tensor([[0.0, 1.0, 5.0, 10.0]]))
        assert torch.count_nonzero(out) == 0

    def test_grad_flows_from_zero_init(self):
        # Features are nonzero (cos(0)=1, etc.) so the proj gets a real gradient
        # even from zero-init and moves off zero during training.
        emb = FrameTimeEmbedding(dim=16, num_bands=4)
        emb(torch.tensor([[0.0, 1.0, 2.0, 3.0]])).sum().backward()
        assert emb.proj.weight.grad is not None
        assert torch.isfinite(emb.proj.weight.grad).all()
        assert torch.count_nonzero(emb.proj.weight.grad) > 0

    def test_distinguishes_timescales(self):
        # Same frame INDICES, different absolute times must produce different
        # embeddings — the whole point of encoding seconds rather than order.
        emb = FrameTimeEmbedding(dim=16, num_bands=8)
        with torch.no_grad():
            emb.proj.weight.normal_()
        short = emb(torch.tensor([[0.0, 0.5, 1.0, 1.5]]))  # 2s clip
        long = emb(torch.tensor([[0.0, 20.0, 40.0, 60.0]]))  # 60s clip
        assert not torch.allclose(short, long)

    def test_large_timestamps_need_fp32(self):
        # bf16 (8 mantissa bits) cannot resolve sub-second differences at large
        # t, which is why the FSDP wrap must NOT cast the fp32 timestamps to bf16
        # before the sinusoidal features (parallel.py: cast_forward_inputs=False).
        emb = FrameTimeEmbedding(dim=16, num_bands=8, min_period=0.5, max_period=256.0)
        with torch.no_grad():
            emb.proj.weight.normal_(std=0.1)
        t = torch.tensor([[256.0, 256.3]])  # large, sub-second apart
        # These two times collapse to a single value in bf16 ...
        t_bf16 = t.to(torch.bfloat16).to(torch.float32)
        assert t_bf16[0, 0] == t_bf16[0, 1]
        # ... so a bf16-cast input makes the two frames' embeddings identical,
        out_bf16 = emb(t_bf16)
        assert torch.allclose(out_bf16[:, 0], out_bf16[:, 1])
        # ... whereas the fp32 input keeps them distinct.
        out_fp32 = emb(t)
        assert not torch.allclose(out_fp32[:, 0], out_fp32[:, 1])

    def test_dtype_follows_proj(self):
        emb = FrameTimeEmbedding(dim=16, num_bands=4).to(torch.bfloat16)
        out = emb(torch.zeros(1, 3))  # float32 input, bf16 module
        assert out.dtype == torch.bfloat16

    def test_reset_parameters_rezeros(self):
        emb = FrameTimeEmbedding(dim=16, num_bands=4)
        with torch.no_grad():
            emb.proj.weight.fill_(1.0)
            emb.proj.bias.fill_(1.0)
        emb.reset_parameters()
        assert torch.count_nonzero(emb.proj.weight) == 0
        assert torch.count_nonzero(emb.proj.bias) == 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dim": 0}, "dim must be positive"),
            ({"dim": 16, "num_bands": 0}, "num_bands must be positive"),
            ({"dim": 16, "min_period": 0.0}, "min_period < max_period"),
            ({"dim": 16, "min_period": 10.0, "max_period": 5.0}, "min_period < max_period"),
        ],
    )
    def test_invalid_args_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FrameTimeEmbedding(**kwargs)


class TestTimeEmbeddingRegistry:
    """The registry + builder make the time embedding config-switchable."""

    def test_sinusoidal_registered(self):
        from kempnerforge.config.registry import registry

        assert "sinusoidal" in registry.list_time_embeddings()

    def test_build_none_config_returns_none(self):
        # Opt-in: no config (no [time_embedding] section) builds NO embedding, so
        # a default video model is identical to one built with no time embedding.
        assert build_time_embedding(None, dim=64) is None

    def test_build_from_config(self):
        from kempnerforge.config.time_embedding import TimeEmbeddingConfig

        m = build_time_embedding(TimeEmbeddingConfig(type="sinusoidal", num_bands=8), dim=32)
        assert isinstance(m, FrameTimeEmbedding)
        assert m.num_bands == 8

    def test_build_none_type_returns_none(self):
        from kempnerforge.config.time_embedding import TimeEmbeddingConfig

        assert build_time_embedding(TimeEmbeddingConfig(type="none"), dim=32) is None
