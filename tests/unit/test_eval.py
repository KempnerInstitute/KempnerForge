"""Unit tests for the evaluation module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kempnerforge.training.eval import run_eval

VOCAB_SIZE = 64
DIM = 32
SEQ_LEN = 16


class _TinyLM(nn.Module):
    """Minimal LM: embedding → linear → logits. Accepts token IDs like a real model."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))


def _make_eval_dataloader(
    batch_size: int = 4,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
    num_samples: int = 20,
) -> DataLoader:
    """Create a simple eval dataloader with random data."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(input_ids, labels)

    def collate_fn(batch):
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        return {"input_ids": inputs, "labels": targets}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def _loss_fn(logits, labels):
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


class TestRunEval:
    def test_returns_loss_and_perplexity(self):
        model = _TinyLM()
        dl = _make_eval_dataloader()
        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=3)

        assert "eval/loss" in metrics
        assert "eval/perplexity" in metrics
        assert metrics["eval/loss"] > 0
        assert metrics["eval/perplexity"] > 1.0

    def test_perplexity_equals_exp_loss(self):
        model = _TinyLM()
        dl = _make_eval_dataloader()
        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=3)
        expected_ppl = math.exp(min(metrics["eval/loss"], 20.0))
        assert abs(metrics["eval/perplexity"] - expected_ppl) < 1e-4

    def test_perfect_model_low_loss(self):
        """A model that always predicts the correct class should have low loss."""
        # All labels are token 0
        input_ids = torch.zeros(8, SEQ_LEN, dtype=torch.long)
        labels = torch.zeros(8, SEQ_LEN, dtype=torch.long)
        dataset = TensorDataset(input_ids, labels)

        def collate_fn(batch):
            inputs = torch.stack([b[0] for b in batch])
            targets = torch.stack([b[1] for b in batch])
            return {"input_ids": inputs, "labels": targets}

        dl = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        # Bias the output head to strongly predict class 0.
        # Also zero the embedding and set row 0 to ones so logits[0] is deterministically
        # positive regardless of inherited global RNG state (otherwise loss depends on
        # whether sum(embed(0)) lands above or below ~0.4, which is coin-flip-random).
        model = _TinyLM()
        with torch.no_grad():
            model.embed.weight.zero_()
            model.embed.weight[0, :] = 1.0
            model.head.weight.zero_()
            model.head.weight[0, :] = 10.0

        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=2)
        assert metrics["eval/loss"] < 0.1
        assert metrics["eval/perplexity"] < 1.2

    def test_model_restored_to_train_mode(self):
        """run_eval should restore model.train() after evaluation."""
        model = _TinyLM()
        model.train()
        dl = _make_eval_dataloader()
        run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=1)
        assert model.training is True

    def test_wraps_around_short_dataloader(self):
        """If dataloader has fewer batches than eval_steps, it should wrap around."""
        model = _TinyLM()
        dl = _make_eval_dataloader(num_samples=4, batch_size=4)
        # 1 batch available but eval_steps=5 — should wrap without error
        metrics = run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=5)
        assert metrics["eval/loss"] > 0

    def test_no_gradients_computed(self):
        """run_eval should not accumulate gradients."""
        model = _TinyLM()
        dl = _make_eval_dataloader()
        run_eval(model, dl, _loss_fn, torch.device("cpu"), eval_steps=2)
        for param in model.parameters():
            assert param.grad is None
