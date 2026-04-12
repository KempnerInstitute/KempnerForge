"""Unit tests for sequence packing (Phase 5).

Tests document-aware packing: doc_ids computation, cross-document attention
masking, boundary label masking, and loss function ignore_index handling.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.data.dataset import MemoryMappedDataset, _compute_packed_output
from kempnerforge.training.loss import chunked_cross_entropy_loss, cross_entropy_loss

# ---------------------------------------------------------------------------
# _compute_packed_output
# ---------------------------------------------------------------------------


class TestComputePackedOutput:
    def test_single_document_no_eos(self):
        """A sequence with no EOS tokens is one document (doc_id=0 everywhere)."""
        tokens = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        result = _compute_packed_output(tokens, eos_token_id=0)
        assert result["input_ids"].shape == (4,)
        assert result["labels"].shape == (4,)
        assert result["doc_ids"].shape == (4,)
        assert (result["doc_ids"] == 0).all()
        # No cross-document boundaries → no -100 labels
        assert (result["labels"] != -100).all()

    def test_two_documents(self):
        """Two documents separated by EOS: doc_ids increment after EOS."""
        # Doc A: [10, 20, EOS=0], Doc B: [30, 40]
        tokens = np.array([10, 20, 0, 30, 40], dtype=np.int64)
        result = _compute_packed_output(tokens, eos_token_id=0)

        # Full doc_ids: [0, 0, 0, 1, 1]
        # input_doc_ids = [0, 0, 0, 1]  (first 4)
        # label_doc_ids = [0, 0, 1, 1]  (last 4)
        assert result["doc_ids"].tolist() == [0, 0, 0, 1]

        # Cross-boundary at position 2: input_doc=0, label_doc=1 → label=-100
        assert result["labels"][0].item() == 20
        assert result["labels"][1].item() == 0  # EOS itself is a valid label
        assert result["labels"][2].item() == -100  # boundary masked
        assert result["labels"][3].item() == 40

    def test_three_documents(self):
        """Three docs: boundaries detected correctly."""
        # [A1, EOS, B1, B2, EOS, C1]
        tokens = np.array([10, 0, 20, 30, 0, 40], dtype=np.int64)
        result = _compute_packed_output(tokens, eos_token_id=0)

        # Full doc_ids: [0, 0, 1, 1, 1, 2]
        # input_doc_ids: [0, 0, 1, 1, 1] — length 5
        # label_doc_ids: [0, 1, 1, 1, 2] — length 5
        assert result["doc_ids"].tolist() == [0, 0, 1, 1, 1]

        # Boundaries at positions 1 (0→1) and 4 (1→2)
        boundary_mask = result["labels"] == -100
        assert boundary_mask.tolist() == [False, True, False, False, True]

    def test_eos_at_end(self):
        """EOS at the very end: no boundary because there's no next document."""
        # [A1, A2, EOS] — full sequence is one doc
        tokens = np.array([10, 20, 0], dtype=np.int64)
        result = _compute_packed_output(tokens, eos_token_id=0)
        # input_doc_ids: [0, 0], label_doc_ids: [0, 0]  (EOS is doc 0's last token)
        assert (result["labels"] != -100).all()

    def test_consecutive_eos(self):
        """Consecutive EOS tokens: each starts a new (empty) document."""
        tokens = np.array([10, 0, 0, 20], dtype=np.int64)
        result = _compute_packed_output(tokens, eos_token_id=0)
        # Full doc_ids: [0, 0, 1, 2]
        # input_doc_ids: [0, 0, 1]
        # label_doc_ids: [0, 1, 2]
        assert result["doc_ids"].tolist() == [0, 0, 1]
        # Boundaries at positions 1 (0→1) and 2 (1→2)
        assert result["labels"][1].item() == -100
        assert result["labels"][2].item() == -100

    def test_output_dtypes(self):
        tokens = np.array([10, 20, 0, 30, 40], dtype=np.int64)
        result = _compute_packed_output(tokens, eos_token_id=0)
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["doc_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# Attention mask for packed sequences
# ---------------------------------------------------------------------------


class TestPackedAttentionMask:
    def _build_mask(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """Build block-diagonal causal mask from doc_ids (matches Attention logic)."""
        seq_len = doc_ids.shape[-1]
        doc_mask = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)  # same-doc
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        return doc_mask & causal

    def test_single_document_equals_causal(self):
        """With one document, mask should be standard lower-triangular."""
        doc_ids = torch.tensor([0, 0, 0, 0])
        mask = self._build_mask(doc_ids)
        expected = torch.ones(4, 4, dtype=torch.bool).tril()
        assert torch.equal(mask, expected)

    def test_two_documents_blocks_cross_attention(self):
        """Tokens in doc B should not attend to tokens in doc A."""
        # [A, A, B, B]
        doc_ids = torch.tensor([0, 0, 1, 1])
        mask = self._build_mask(doc_ids)

        # Token 2 (first of doc B) should attend to: itself only
        assert mask[2].tolist() == [False, False, True, False]
        # Token 3 (second of doc B) should attend to: token 2 and 3
        assert mask[3].tolist() == [False, False, True, True]
        # Token 0 (doc A) still has standard causal
        assert mask[0].tolist() == [True, False, False, False]
        assert mask[1].tolist() == [True, True, False, False]

    def test_three_documents_fully_isolated(self):
        """Three docs: each doc's tokens only attend within their block."""
        # [A, B, B, C]
        doc_ids = torch.tensor([0, 1, 1, 2])
        mask = self._build_mask(doc_ids)

        # Token 0 (doc A): attends to self only
        assert mask[0].tolist() == [True, False, False, False]
        # Token 1 (doc B, first): attends to self only (no earlier B tokens)
        assert mask[1].tolist() == [False, True, False, False]
        # Token 2 (doc B, second): attends to tokens 1 and 2
        assert mask[2].tolist() == [False, True, True, False]
        # Token 3 (doc C): attends to self only
        assert mask[3].tolist() == [False, False, False, True]

    def test_every_token_attends_to_self(self):
        """Diagonal should always be True (every token attends to itself)."""
        doc_ids = torch.tensor([0, 1, 2, 3, 4])
        mask = self._build_mask(doc_ids)
        assert torch.diag(mask).all()

    def test_batched_mask(self):
        """Batch dimension should produce per-sample masks."""
        doc_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
        mask = self._build_mask(doc_ids)
        assert mask.shape == (2, 4, 4)
        # First sample: standard 2+2 block
        assert mask[0, 2, 0].item() is False  # cross-doc
        # Second sample: different blocking
        assert mask[1, 1, 0].item() is False  # cross-doc


# ---------------------------------------------------------------------------
# Loss with ignore_index
# ---------------------------------------------------------------------------


class TestLossIgnoreIndex:
    def test_cross_entropy_ignores_minus100(self):
        """Labels set to -100 should be excluded from the loss."""
        torch.manual_seed(42)
        logits = torch.randn(1, 4, 100)
        labels = torch.tensor([[5, -100, 10, 20]])

        loss = cross_entropy_loss(logits, labels)
        assert loss.item() > 0  # loss is computed on 3 valid tokens
        assert torch.isfinite(loss)

    def test_cross_entropy_all_masked_returns_zero(self):
        """If all labels are -100, loss should be 0."""
        logits = torch.randn(1, 4, 100)
        labels = torch.full((1, 4), -100, dtype=torch.long)
        loss = cross_entropy_loss(logits, labels)
        assert loss.item() == 0.0

    def test_cross_entropy_no_masked_unchanged(self):
        """Without -100 labels, result should match the old behavior."""
        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))

        loss_new = cross_entropy_loss(logits, labels)
        # Compute manually without ignore_index
        import torch.nn.functional as F

        loss_manual = F.cross_entropy(logits.view(-1, 100), labels.view(-1))
        torch.testing.assert_close(loss_new, loss_manual)

    def test_chunked_ce_ignores_minus100(self):
        """Chunked CE should also respect -100 labels."""
        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        labels[0, 3] = -100
        labels[1, 0] = -100

        loss = chunked_cross_entropy_loss(logits, labels, chunk_size=4)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_chunked_ce_all_masked(self):
        """Chunked CE with all -100 labels should return 0."""
        logits = torch.randn(2, 8, 100)
        labels = torch.full((2, 8), -100, dtype=torch.long)
        loss = chunked_cross_entropy_loss(logits, labels, chunk_size=4)
        assert loss.item() == 0.0

    def test_chunked_ce_matches_standard_with_ignore(self):
        """Chunked and standard CE should agree when some labels are -100."""
        torch.manual_seed(42)
        logits = torch.randn(4, 16, 256)
        labels = torch.randint(0, 256, (4, 16))
        # Mask ~25% of labels
        mask_positions = torch.rand(4, 16) < 0.25
        labels[mask_positions] = -100

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=16)
        torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Model forward with doc_ids
# ---------------------------------------------------------------------------


class TestModelWithPacking:
    @pytest.fixture
    def tiny_config(self):
        from kempnerforge.config.schema import ModelConfig

        return ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32)

    @pytest.fixture
    def tiny_model(self, tiny_config):
        from kempnerforge.model.transformer import Transformer

        return Transformer(tiny_config)

    def test_forward_with_doc_ids_shape(self, tiny_model):
        """Forward with doc_ids should produce same output shape as without."""
        tokens = torch.randint(0, 256, (2, 16))
        doc_ids = torch.zeros(2, 16, dtype=torch.long)  # all same doc
        out = tiny_model(tokens, doc_ids=doc_ids)
        assert out.shape == (2, 16, 256)

    def test_forward_without_doc_ids_unchanged(self, tiny_model):
        """Forward without doc_ids should behave identically to before."""
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (2, 16))
        out = tiny_model(tokens)
        assert out.shape == (2, 16, 256)

    def test_doc_ids_none_is_exact_noop(self, tiny_model):
        """doc_ids=None should produce identical output to not passing it."""
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (2, 8))

        out_without = tiny_model(tokens)
        out_with_none = tiny_model(tokens, doc_ids=None)
        torch.testing.assert_close(out_without, out_with_none)

    def test_single_doc_matches_causal(self, tiny_model):
        """doc_ids all-same-value should produce same result as standard causal."""
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (1, 8))
        doc_ids = torch.zeros(1, 8, dtype=torch.long)

        out_causal = tiny_model(tokens)
        out_packed = tiny_model(tokens, doc_ids=doc_ids)
        torch.testing.assert_close(out_causal, out_packed)

    def test_cross_doc_isolation(self, tiny_model):
        """Different doc_ids should produce different outputs for later tokens."""
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (1, 8))

        # All one document
        doc_ids_single = torch.zeros(1, 8, dtype=torch.long)
        out_single = tiny_model(tokens, doc_ids=doc_ids_single)

        # Split into two documents at position 4
        doc_ids_split = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        out_split = tiny_model(tokens, doc_ids=doc_ids_split)

        # First 4 tokens (doc A) should be identical in both cases
        torch.testing.assert_close(out_single[:, :4, :], out_split[:, :4, :])

        # Tokens 4-7 (doc B) should differ: in single-doc they attend to 0-3,
        # in split they only attend to 4-7
        assert not torch.allclose(out_single[:, 4:, :], out_split[:, 4:, :])

    def test_gradient_flows_with_doc_ids(self, tiny_model):
        """Backward pass should work with packed attention mask."""
        tokens = torch.randint(0, 256, (2, 8))
        doc_ids = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2]])

        out = tiny_model(tokens, doc_ids=doc_ids)
        loss = out.sum()
        loss.backward()

        for p in tiny_model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# MemoryMappedDataset with packing
# ---------------------------------------------------------------------------


class TestMemoryMappedDatasetPacking:
    @pytest.fixture
    def packed_data_dir(self, tmp_path):
        """Create .npy file with known EOS positions for packing tests."""
        # Tokens: [10, 20, EOS=0, 30, 40, 50, EOS=0, 60, 70, 80, 90, 100, ...]
        seq_len = 8  # +1 extra for label offset, so chunks are 8 tokens
        tokens = np.array(
            [10, 20, 0, 30, 40, 50, 0, 60, 70, 80, 90, 100, 0, 110, 120, 130],
            dtype=np.uint16,
        )
        np.save(tmp_path / "data.npy", tokens)
        return tmp_path, seq_len

    def test_packing_returns_doc_ids(self, packed_data_dir):
        data_dir, seq_len = packed_data_dir
        ds = MemoryMappedDataset(
            str(data_dir), seq_len=seq_len, pack_sequences=True, eos_token_id=0
        )
        sample = ds[0]
        assert "doc_ids" in sample
        assert sample["doc_ids"].shape == (seq_len - 1,)

    def test_packing_masks_boundaries(self, packed_data_dir):
        data_dir, seq_len = packed_data_dir
        ds = MemoryMappedDataset(
            str(data_dir), seq_len=seq_len, pack_sequences=True, eos_token_id=0
        )
        sample = ds[0]
        # First chunk: [10, 20, 0, 30, 40, 50, 0, 60]
        # doc_ids:     [0,  0,  0, 1,  1,  1,  1, 2]
        # input_ids:   [10, 20, 0, 30, 40, 50, 0]
        # labels:      [20, 0,  ?, 40, 50, 0,  ?]
        # Boundaries at positions 2 (doc 0→1) and 6 (doc 1→2)
        assert sample["labels"][2].item() == -100
        assert sample["labels"][6].item() == -100

    def test_no_packing_no_doc_ids(self, packed_data_dir):
        data_dir, seq_len = packed_data_dir
        ds = MemoryMappedDataset(str(data_dir), seq_len=seq_len)
        sample = ds[0]
        assert "doc_ids" not in sample

    def test_packing_requires_eos_token_id(self, tmp_path):
        tokens = np.arange(100, dtype=np.uint16)
        np.save(tmp_path / "data.npy", tokens)
        with pytest.raises(ValueError, match="eos_token_id"):
            MemoryMappedDataset(str(tmp_path), seq_len=16, pack_sequences=True)


# ---------------------------------------------------------------------------
# HuggingFaceDataset with packing
# ---------------------------------------------------------------------------


class TestHuggingFaceDatasetPacking:
    def _make_dataset(self, seq_len, documents, pack_sequences=False):
        from unittest.mock import MagicMock, patch

        from kempnerforge.data.dataset import HuggingFaceDataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode = lambda text, **kwargs: [ord(c) for c in text]

        with (
            patch("datasets.load_dataset", return_value=documents),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            ds = HuggingFaceDataset(
                dataset_name="mock/dataset",
                split="train",
                text_field="text",
                seq_len=seq_len,
                tokenizer_path="mock",
                pack_sequences=pack_sequences,
            )
        return ds

    def test_packing_returns_doc_ids(self):
        # "abcde" = [97,98,99,100,101] + EOS=0. chunk_size=5 → [97,98,99,100,101]
        ds = self._make_dataset(
            seq_len=4,
            documents=[{"text": "abcdefghij"}],
            pack_sequences=True,
        )
        sample = ds[0]
        assert "doc_ids" in sample
        assert sample["doc_ids"].shape == (4,)

    def test_no_packing_no_doc_ids(self):
        ds = self._make_dataset(
            seq_len=4,
            documents=[{"text": "abcdefghij"}],
            pack_sequences=False,
        )
        sample = ds[0]
        assert "doc_ids" not in sample

    def test_multi_doc_packing_masks_boundaries(self):
        # "ab" → [97,98] + EOS=0 → [97,98,0]
        # "cd" → [99,100] + EOS=0 → [99,100,0]
        # Buffer: [97,98,0,99,100,0], chunk_size=5 → first chunk [97,98,0,99,100]
        ds = self._make_dataset(
            seq_len=4,
            documents=[{"text": "ab"}, {"text": "cd"}, {"text": "ef"}],
            pack_sequences=True,
        )
        sample = ds[0]
        # Tokens: [97, 98, 0, 99, 100]
        # doc_ids: [0, 0, 0, 1, 1]
        # input_doc_ids: [0, 0, 0, 1]
        # label_doc_ids: [0, 0, 1, 1]
        # Boundary at position 2
        assert sample["labels"][2].item() == -100
        assert sample["labels"][0].item() != -100
        assert sample["labels"][1].item() != -100
        assert sample["labels"][3].item() != -100


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestPackingConfig:
    def test_pack_sequences_default_false(self):
        from kempnerforge.config.schema import DataConfig

        config = DataConfig()
        assert config.pack_sequences is False

    def test_pack_sequences_true(self):
        from kempnerforge.config.schema import DataConfig

        config = DataConfig(pack_sequences=True)
        assert config.pack_sequences is True
