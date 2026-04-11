"""Unit tests for KV-cache and text generation."""

from __future__ import annotations

import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import KVCache
from kempnerforge.model.generate import generate, sample
from kempnerforge.model.transformer import Transformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> ModelConfig:
    return ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=128, max_seq_len=64)


def _tiny_model() -> Transformer:
    config = _tiny_config()
    model = Transformer(config)
    model.eval()
    return model


# ============================================================================
# KVCache tests
# ============================================================================


class TestKVCache:
    def test_update_shapes(self):
        cache = KVCache(
            batch_size=2, max_seq_len=32, n_kv_heads=2, head_dim=16,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        k_new = torch.randn(2, 2, 5, 16)
        v_new = torch.randn(2, 2, 5, 16)

        k_all, v_all = cache.update(k_new, v_new)
        assert k_all.shape == (2, 2, 5, 16)
        assert v_all.shape == (2, 2, 5, 16)
        assert cache.seq_len == 5

    def test_incremental_update(self):
        cache = KVCache(
            batch_size=1, max_seq_len=16, n_kv_heads=1, head_dim=8,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        # First update: 3 tokens (prefill)
        k1 = torch.randn(1, 1, 3, 8)
        v1 = torch.randn(1, 1, 3, 8)
        k_all, v_all = cache.update(k1, v1)
        assert k_all.shape == (1, 1, 3, 8)
        assert cache.seq_len == 3

        # Second update: 1 token (decode)
        k2 = torch.randn(1, 1, 1, 8)
        v2 = torch.randn(1, 1, 1, 8)
        k_all, v_all = cache.update(k2, v2)
        assert k_all.shape == (1, 1, 4, 8)
        assert cache.seq_len == 4

        # Verify concatenation correctness
        assert torch.allclose(k_all[:, :, :3, :], k1)
        assert torch.allclose(k_all[:, :, 3:4, :], k2)

    def test_initial_state(self):
        cache = KVCache(
            batch_size=1, max_seq_len=8, n_kv_heads=2, head_dim=4,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        assert cache.seq_len == 0
        assert cache.k.shape == (1, 2, 8, 4)
        assert cache.v.shape == (1, 2, 8, 4)


# ============================================================================
# Sampling tests
# ============================================================================


class TestSampling:
    def test_greedy(self):
        """Temperature 0 should always pick the highest-logit token."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        token = sample(logits, temperature=0, top_k=0, top_p=1.0)
        assert token.item() == 1  # index of 5.0

    def test_greedy_deterministic(self):
        """Greedy should be deterministic across calls."""
        logits = torch.randn(4, 128)
        t1 = sample(logits, temperature=0, top_k=0, top_p=1.0)
        t2 = sample(logits, temperature=0, top_k=0, top_p=1.0)
        assert torch.equal(t1, t2)

    def test_top_k_restricts(self):
        """Top-k should only sample from the top k tokens."""
        # Make one token have very high logit, rest are very low
        logits = torch.full((1, 100), -100.0)
        logits[0, 42] = 10.0
        logits[0, 7] = 9.0
        logits[0, 99] = 8.0

        # With top_k=1, should always pick token 42
        for _ in range(10):
            token = sample(logits.clone(), temperature=1.0, top_k=1, top_p=1.0)
            assert token.item() == 42

    def test_top_p_restricts(self):
        """Top-p should restrict to the nucleus of highest-probability tokens."""
        # Token 0 has >99% probability after softmax
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        for _ in range(10):
            token = sample(logits.clone(), temperature=1.0, top_k=0, top_p=0.5)
            assert token.item() == 0

    def test_temperature_scaling(self):
        """High temperature should make distribution more uniform (higher entropy)."""
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        # Low temperature: probabilities concentrate on max
        low_t = (logits / 0.1).softmax(dim=-1)
        # High temperature: probabilities more uniform
        high_t = (logits / 10.0).softmax(dim=-1)

        # Entropy of high-temperature should be greater
        entropy_low = -(low_t * low_t.log()).sum()
        entropy_high = -(high_t * high_t.log()).sum()
        assert entropy_high > entropy_low

    def test_batch_sampling(self):
        """Sampling should work with batched logits."""
        logits = torch.randn(8, 64)
        tokens = sample(logits, temperature=1.0, top_k=0, top_p=1.0)
        assert tokens.shape == (8,)
        assert (tokens >= 0).all() and (tokens < 64).all()


# ============================================================================
# Generation tests
# ============================================================================


class TestGenerate:
    def test_output_length(self):
        """Output should be prompt_len + max_new_tokens."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 5))
        output = generate(model, prompt, max_new_tokens=10, temperature=0)
        assert output.shape == (1, 15)

    def test_prompt_preserved(self):
        """The prompt should be preserved at the beginning of the output."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 8))
        output = generate(model, prompt, max_new_tokens=5, temperature=0)
        assert torch.equal(output[:, :8], prompt)

    def test_greedy_deterministic(self):
        """Greedy generation should be deterministic."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 4))
        out1 = generate(model, prompt, max_new_tokens=10, temperature=0)
        out2 = generate(model, prompt, max_new_tokens=10, temperature=0)
        assert torch.equal(out1, out2)

    def test_valid_token_ids(self):
        """All generated tokens should be valid vocab indices."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 4))
        output = generate(model, prompt, max_new_tokens=20, temperature=1.0)
        assert (output >= 0).all()
        assert (output < 128).all()

    def test_eos_stopping(self):
        """Generation should stop early when EOS is produced."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 3))

        # Generate once to discover what the model greedily predicts first
        full_output = generate(model, prompt, max_new_tokens=10, temperature=0)
        first_token = full_output[0, 3].item()

        # Now use that token as EOS — generation must stop after 1 token
        output = generate(
            model, prompt, max_new_tokens=50, temperature=0, eos_token_id=first_token,
        )
        generated_len = output.shape[1] - prompt.shape[1]
        assert generated_len == 1

    def test_batch_generation(self):
        """Generation should work with batch_size > 1."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (3, 6))
        output = generate(model, prompt, max_new_tokens=8, temperature=1.0)
        assert output.shape == (3, 14)

    def test_model_restored_to_train_mode(self):
        """generate() should restore model to its original training mode."""
        model = _tiny_model()
        model.train()
        assert model.training is True
        prompt = torch.randint(0, 128, (1, 4))
        generate(model, prompt, max_new_tokens=3, temperature=0)
        assert model.training is True

    def test_model_stays_in_eval(self):
        """If model was already in eval, it should stay in eval."""
        model = _tiny_model()
        model.eval()
        prompt = torch.randint(0, 128, (1, 4))
        generate(model, prompt, max_new_tokens=3, temperature=0)
        assert model.training is False

    def test_no_gradients(self):
        """Generation should not accumulate any gradients."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 4))
        generate(model, prompt, max_new_tokens=5, temperature=0)
        for param in model.parameters():
            assert param.grad is None

    def test_kv_cache_with_gqa(self):
        """Generation should work with GQA (n_kv_heads < n_heads)."""
        config = ModelConfig(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
            vocab_size=128, max_seq_len=64,
        )
        model = Transformer(config)
        model.eval()
        prompt = torch.randint(0, 128, (1, 5))
        output = generate(model, prompt, max_new_tokens=8, temperature=0)
        assert output.shape == (1, 13)

    def test_max_new_tokens_zero(self):
        """max_new_tokens=0 should return the prompt unchanged."""
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 5))
        output = generate(model, prompt, max_new_tokens=0, temperature=0)
        assert torch.equal(output, prompt)

    def test_rejects_exceeding_max_seq_len(self):
        """Should raise if prompt + max_new_tokens exceeds model max_seq_len."""
        # max_seq_len=64 from _tiny_config
        model = _tiny_model()
        prompt = torch.randint(0, 128, (1, 50))
        import pytest

        with pytest.raises(ValueError, match="exceeds model max_seq_len"):
            generate(model, prompt, max_new_tokens=20, temperature=0)
