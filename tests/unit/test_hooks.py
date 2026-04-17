"""Unit tests for activation extraction hooks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import Attention
from kempnerforge.model.hooks import ActivationStore, extract_representations, save_activations
from kempnerforge.model.position import precompute_rope_frequencies
from kempnerforge.model.transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=32)


@pytest.fixture
def small_model(small_config):
    return Transformer(small_config).to(DEVICE).eval()


@pytest.fixture
def small_dataset():
    """Simple map-style dataset for testing."""

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=16, seq_len=16, vocab_size=256):
            self._data = torch.randint(0, vocab_size, (n_samples, seq_len))

        def __len__(self):
            return self._data.shape[0]

        def __getitem__(self, idx):
            return {"input_ids": self._data[idx]}

    return _Dataset()


# ---------------------------------------------------------------------------
# ActivationStore basics
# ---------------------------------------------------------------------------


class TestActivationStore:
    def test_capture_single_layer(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        act = store.get("layers.0.attention")
        assert act is not None
        assert act.shape == (2, 16, 64)  # (batch, seq, dim)
        assert act.device == torch.device("cpu")
        store.disable()

    def test_capture_multiple_layers(self, small_model):
        layers = ["layers.0.attention", "layers.1.mlp", "layers.3.attention"]
        store = ActivationStore(small_model, layers=layers)
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        for name in layers:
            assert store.get(name) is not None
        store.disable()

    def test_capture_norm_layer(self, small_model):
        store = ActivationStore(small_model, layers=["norm"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        act = store.get("norm")
        assert act is not None
        assert act.shape == (2, 16, 64)
        store.disable()

    def test_enable_disable(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        assert not store.enabled

        store.enable()
        assert store.enabled
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        assert store.get("layers.0.attention") is not None

        store.disable()
        assert not store.enabled
        store.clear()
        small_model(tokens)
        assert store.get("layers.0.attention") is None

    def test_clear(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        assert store.get("layers.0.attention") is not None
        store.clear()
        assert store.get("layers.0.attention") is None
        assert store.activations == {}
        store.disable()

    def test_activations_property(self, small_model):
        layers = ["layers.0.attention", "layers.1.mlp"]
        store = ActivationStore(small_model, layers=layers)
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        acts = store.activations
        assert set(acts.keys()) == set(layers)
        store.disable()

    def test_layer_names_property(self, small_model):
        layers = ["layers.0.attention", "norm"]
        store = ActivationStore(small_model, layers=layers)
        assert store.layer_names == layers

    def test_invalid_layer_raises(self, small_model):
        store = ActivationStore(small_model, layers=["nonexistent_module"])
        with pytest.raises(ValueError, match="not found"):
            store.enable()

    def test_double_enable_is_idempotent(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        store.enable()  # Should not duplicate hooks
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        assert store.get("layers.0.attention") is not None
        store.disable()

    def test_output_is_detached(self, small_model):
        """Captured activations should not require grad."""
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        act = store.get("layers.0.attention")
        assert not act.requires_grad
        store.disable()

    def test_shapes_match_model_dim(self, small_config, small_model):
        """Captured shapes should match model dimension."""
        store = ActivationStore(small_model, layers=["layers.2.mlp"])
        store.enable()
        tokens = torch.randint(0, 256, (4, 8), device=DEVICE)
        small_model(tokens)
        act = store.get("layers.2.mlp")
        assert act.shape == (4, 8, small_config.dim)
        store.disable()


# ---------------------------------------------------------------------------
# Attention weight capture
# ---------------------------------------------------------------------------


class TestAttentionWeightCapture:
    @pytest.fixture
    def attention_setup(self):
        dim, n_heads, n_kv_heads = 64, 4, 4
        attn = Attention(dim, n_heads, n_kv_heads).to(DEVICE).eval()
        rope_cos, rope_sin = precompute_rope_frequencies(head_dim=dim // n_heads, max_seq_len=32)
        rope_cos = rope_cos[:16].to(DEVICE)
        rope_sin = rope_sin[:16].to(DEVICE)
        return attn, rope_cos, rope_sin

    def test_weights_not_captured_by_default(self, attention_setup):
        attn, rope_cos, rope_sin = attention_setup
        x = torch.randn(2, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is None

    def test_capture_attention_weights(self, attention_setup):
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is not None
        # Shape: (batch, n_heads, seq_q, seq_k)
        assert attn.last_attention_weights.shape == (2, 4, 16, 16)
        assert attn.last_attention_weights.device == torch.device("cpu")
        # Output shape should be unchanged
        assert out.shape == (2, 16, 64)

    def test_attention_weights_sum_to_one(self, attention_setup):
        """Attention weights should sum to ~1.0 along key dimension."""
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(2, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        weights = attn.last_attention_weights.float()
        # Each row sums to 1.0 (softmax over keys)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_are_causal(self, attention_setup):
        """Upper triangle (future positions) should be zero."""
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(1, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        weights = attn.last_attention_weights.float()
        # Check upper triangle is zero (strictly above diagonal)
        seq_len = weights.shape[-1]
        upper_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        upper_values = weights[0, 0][upper_mask]
        assert torch.allclose(upper_values, torch.zeros_like(upper_values), atol=1e-6)

    def test_capture_output_matches_sdpa(self, attention_setup):
        """Manual attention path should produce same output as SDPA."""
        attn, rope_cos, rope_sin = attention_setup
        x = torch.randn(2, 16, 64, device=DEVICE)

        # SDPA path
        attn.capture_attention_weights = False
        out_sdpa = attn(x, rope_cos, rope_sin)

        # Manual path
        attn.capture_attention_weights = True
        out_manual = attn(x, rope_cos, rope_sin)

        assert torch.allclose(out_sdpa, out_manual, atol=1e-4)

    def test_capture_disables_cleanly(self, attention_setup):
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(2, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is not None

        attn.capture_attention_weights = False
        attn.last_attention_weights = None
        attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is None


# ---------------------------------------------------------------------------
# extract_representations
# ---------------------------------------------------------------------------


class TestExtractRepresentations:
    def test_basic_extraction(self, small_model, small_dataset):
        layers = ["layers.0.attention", "layers.1.mlp"]
        reps = extract_representations(small_model, small_dataset, layers, DEVICE, batch_size=8)
        assert set(reps.keys()) == set(layers)
        for name in layers:
            assert reps[name].shape[0] == len(small_dataset)  # all samples
            assert reps[name].shape[2] == 64  # model dim

    def test_max_samples(self, small_model, small_dataset):
        reps = extract_representations(
            small_model,
            small_dataset,
            ["layers.0.attention"],
            DEVICE,
            batch_size=4,
            max_samples=8,
        )
        assert reps["layers.0.attention"].shape[0] == 8

    def test_model_restored_to_original_mode(self, small_model, small_dataset):
        small_model.train()
        assert small_model.training
        extract_representations(small_model, small_dataset, ["layers.0.attention"], DEVICE)
        assert small_model.training  # Should be restored to train mode

    def test_results_on_cpu(self, small_model, small_dataset):
        reps = extract_representations(small_model, small_dataset, ["layers.0.attention"], DEVICE)
        for tensor in reps.values():
            assert tensor.device == torch.device("cpu")

    def test_non_trivial_activations(self, small_model, small_dataset):
        """Activations should not be all zeros."""
        reps = extract_representations(small_model, small_dataset, ["layers.0.attention"], DEVICE)
        act = reps["layers.0.attention"]
        assert act.abs().sum() > 0


# ---------------------------------------------------------------------------
# save_activations
# ---------------------------------------------------------------------------


class TestSaveActivations:
    def test_save_and_load(self, tmp_path, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention", "norm"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)

        path = tmp_path / "activations.npz"
        save_activations(store.activations, path)
        store.disable()

        assert path.exists()
        loaded = np.load(str(path))
        assert "layers.0.attention" in loaded
        assert "norm" in loaded
        assert loaded["layers.0.attention"].shape == (2, 16, 64)

    def test_adds_npz_extension(self, tmp_path, small_model):
        store = ActivationStore(small_model, layers=["norm"])
        store.enable()
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        small_model(tokens)

        path = tmp_path / "acts"  # No extension
        save_activations(store.activations, path)
        store.disable()

        assert (tmp_path / "acts.npz").exists()

    def test_creates_parent_dirs(self, tmp_path, small_model):
        store = ActivationStore(small_model, layers=["norm"])
        store.enable()
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        small_model(tokens)

        path = tmp_path / "subdir" / "deep" / "acts.npz"
        save_activations(store.activations, path)
        store.disable()

        assert path.exists()


# ---------------------------------------------------------------------------
# Integration: ActivationStore with Transformer
# ---------------------------------------------------------------------------


class TestActivationStoreIntegration:
    def test_full_workflow(self, small_model):
        """Full workflow: enable → forward → capture → save → load."""
        layers = ["layers.0.attention", "layers.2.mlp", "norm"]
        store = ActivationStore(small_model, layers=layers)
        store.enable()

        tokens = torch.randint(0, 256, (4, 16), device=DEVICE)
        with torch.no_grad():
            small_model(tokens)

        for name in layers:
            act = store.get(name)
            assert act is not None
            assert act.shape[0] == 4
            assert act.shape[1] == 16
            assert act.shape[2] == 64

        store.disable()

    def test_multiple_forward_passes(self, small_model):
        """Each forward overwrites previous activations."""
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()

        tokens1 = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens1)
        act1 = store.get("layers.0.attention").clone()

        tokens2 = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens2)
        act2 = store.get("layers.0.attention")

        # Different inputs should produce different activations
        assert not torch.allclose(act1, act2)
        store.disable()

    def test_no_hooks_leak_after_disable(self, small_model):
        """After disable, no hooks remain on the model."""
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        store.disable()

        # Our hook handles should all have been removed
        assert len(store._hooks) == 0
