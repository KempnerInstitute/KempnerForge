"""Integration tests for a single training step.

Verifies that forward + backward + optimizer step produce correct
loss values and parameter updates for a small model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kempnerforge.config.schema import ModelConfig, OptimizerConfig
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


class TestTrainStep:
    def test_single_step_loss_finite(self):
        """A single training step produces a finite loss."""
        model = Transformer(CONFIG).to(DEVICE)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=DEVICE)

        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss.isfinite().item()

    def test_parameters_update_after_step(self):
        """Parameters should change after an optimizer step."""
        model = Transformer(CONFIG).to(DEVICE)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-2, fused=False))

        # Snapshot initial params
        param_before = {
            n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad
        }

        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=DEVICE)

        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        # At least some params should have changed
        changed = 0
        for n, p in model.named_parameters():
            if n in param_before and not torch.equal(p.data, param_before[n]):
                changed += 1

        assert changed > 0, "No parameters changed after optimizer step"

    def test_gradients_exist_after_backward(self):
        """All trainable parameters should have gradients after backward."""
        model = Transformer(CONFIG).to(DEVICE)

        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=DEVICE)

        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.isfinite().all(), f"Non-finite gradient for {name}"

    def test_zero_grad_clears_gradients(self):
        """zero_grad should clear all parameter gradients."""
        model = Transformer(CONFIG).to(DEVICE)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=DEVICE)

        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), f"Gradient not zeroed for {name}"

    @pytest.mark.gpu
    def test_bf16_training_step(self):
        """Training step in bf16 produces reasonable results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        model = Transformer(CONFIG).to(DEVICE).to(torch.bfloat16)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        tokens = torch.randint(0, 256, (2, 32), device=DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

        assert loss.isfinite().item()
        assert loss.item() > 0

    def test_loss_decreases_over_steps(self):
        """Loss should decrease when training on a fixed batch."""
        torch.manual_seed(42)
        model = Transformer(CONFIG).to(DEVICE)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Fixed batch for memorization
        tokens = torch.randint(0, 256, (4, 32), device=DEVICE)
        labels = tokens.clone()

        losses = []
        for _ in range(50):
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
