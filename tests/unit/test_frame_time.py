"""Unit tests for FrameTimeEmbedding (per-frame timestamp encoding)."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.model.frame_time import FrameTimeEmbedding


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
