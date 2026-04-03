"""Integration test: torch.compile correctness.

Verifies that compiled model produces identical (or nearly identical)
output compared to eager mode. Also tests that compilation doesn't
break the backward pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer


@pytest.mark.gpu
class TestCompileCorrectness:
    def test_compiled_forward_matches_eager(self):
        """Compiled model produces same logits as eager model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        device = torch.device("cuda")

        model = Transformer(config).to(device).eval()
        tokens = torch.randint(0, 256, (1, 16), device=device)

        with torch.no_grad():
            eager_out = model(tokens)

        compiled_model = torch.compile(model)
        with torch.no_grad():
            compiled_out = compiled_model(tokens)

        max_diff = (eager_out - compiled_out).abs().max().item()
        assert max_diff < 1e-4, f"Compiled output differs: max diff={max_diff}"

    def test_compiled_backward_produces_gradients(self):
        """Compiled model produces valid gradients."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        device = torch.device("cuda")

        model = Transformer(config).to(device)
        compiled_model = torch.compile(model)

        tokens = torch.randint(0, 256, (2, 16), device=device)
        labels = torch.randint(0, 256, (2, 16), device=device)

        logits = compiled_model(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_compiled_loss_matches_eager(self):
        """Loss from compiled model matches eager model on same input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        device = torch.device("cuda")

        model = Transformer(config).to(device).eval()
        tokens = torch.randint(0, 256, (2, 16), device=device)
        labels = torch.randint(0, 256, (2, 16), device=device)

        with torch.no_grad():
            eager_logits = model(tokens)
            eager_loss = F.cross_entropy(
                eager_logits.view(-1, eager_logits.size(-1)), labels.view(-1)
            )

        compiled_model = torch.compile(model)
        with torch.no_grad():
            compiled_logits = compiled_model(tokens)
            compiled_loss = F.cross_entropy(
                compiled_logits.view(-1, compiled_logits.size(-1)), labels.view(-1)
            )

        assert abs(eager_loss.item() - compiled_loss.item()) < 1e-4
