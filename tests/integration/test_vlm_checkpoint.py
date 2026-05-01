"""Integration tests for VLM freeze metadata in CheckpointManager.

Covers save + load round-trip with canonical freeze metadata, mismatch
detection (raises by default), and the ``ignore_freeze_mismatch`` escape
hatch. Uses a tiny text-only model under CheckpointManager for speed;
the VLM freeze machinery lives on the manager, not the model wrapper,
so the tests don't need to build a full VLMWrapper here.
"""

from __future__ import annotations

import json

import pytest
import torch

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import CheckpointConfig, ModelConfig, OptimizerConfig
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.freeze import canonical_freeze_meta
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CheckpointManager requires CUDA in this test"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TINY_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


def _freeze(*pairs):
    from kempnerforge.config.vlm import FreezeSpec

    return canonical_freeze_meta([FreezeSpec(m, f) for (m, f) in pairs])


def _make_manager(tmp_path, *, ignore_freeze_mismatch=False):
    model = Transformer(_TINY_CONFIG).to(DEVICE)
    opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
    cfg = CheckpointConfig(
        dir=str(tmp_path / "ckpt"), interval=1, ignore_freeze_mismatch=ignore_freeze_mismatch
    )
    return CheckpointManager(cfg, model, opt)


class TestSaveFreezeMetadata:
    def test_metadata_json_contains_vlm_freeze(self, tmp_path):
        mgr = _make_manager(tmp_path)
        freeze = _freeze(("vision_encoder", True), ("adapter", False))
        mgr.save(step=1, extra={"vlm_freeze": freeze})

        meta_path = tmp_path / "ckpt" / "step_1" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["vlm_freeze"] == freeze
        assert meta["step"] == 1

    def test_metadata_json_omits_when_not_vlm(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.save(step=1)  # no extra
        meta_path = tmp_path / "ckpt" / "step_1" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "vlm_freeze" not in meta


class TestLoadFreezeMismatch:
    def test_match_ok(self, tmp_path):
        mgr = _make_manager(tmp_path)
        freeze = _freeze(("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": freeze})

        # Fresh manager loads back; matching freeze list does not raise.
        mgr2 = _make_manager(tmp_path)
        step, _, _ = mgr2.load(vlm_freeze_expected=freeze)
        assert step == 1

    def test_reorder_invariant(self, tmp_path):
        """Canonical form is sorted; two equivalent lists match regardless
        of construction order."""
        mgr = _make_manager(tmp_path)
        saved = _freeze(("vision_encoder", True), ("adapter", False))
        mgr.save(step=1, extra={"vlm_freeze": saved})

        mgr2 = _make_manager(tmp_path)
        current = _freeze(("adapter", False), ("vision_encoder", True))
        assert saved == current  # canonical_freeze_meta is sort-based
        step, _, _ = mgr2.load(vlm_freeze_expected=current)
        assert step == 1

    def test_semantic_mismatch_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.save(step=1, extra={"vlm_freeze": _freeze(("vision_encoder", True))})

        mgr2 = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="VLM freeze mismatch"):
            mgr2.load(
                vlm_freeze_expected=_freeze(("vision_encoder", False)),
            )

    def test_ignore_flag_allows_mismatch(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.save(step=1, extra={"vlm_freeze": _freeze(("vision_encoder", True))})

        mgr2 = _make_manager(tmp_path, ignore_freeze_mismatch=True)
        step, _, _ = mgr2.load(
            vlm_freeze_expected=_freeze(("vision_encoder", False)),
        )
        assert step == 1

    def test_no_expected_skips_check(self, tmp_path):
        """Text-only runs call load() without vlm_freeze_expected; the
        saved metadata should be ignored for the compare."""
        mgr = _make_manager(tmp_path)
        mgr.save(step=1, extra={"vlm_freeze": _freeze(("vision_encoder", True))})
        mgr2 = _make_manager(tmp_path)
        step, _, _ = mgr2.load()  # no vlm_freeze_expected
        assert step == 1

    def test_no_saved_freeze_skips_check(self, tmp_path):
        """Older checkpoints saved without vlm_freeze metadata should
        not trip the compare when the current run is VLM."""
        mgr = _make_manager(tmp_path)
        mgr.save(step=1)  # no extra
        mgr2 = _make_manager(tmp_path)
        step, _, _ = mgr2.load(vlm_freeze_expected=_freeze(("vision_encoder", True)))
        assert step == 1


class TestCrossArchLoad:
    """Cross-arch resume: filter both saved and expected freeze metadata
    to the intersection of module keys before comparing.

    A JD checkpoint may have only ``vision_encoder`` in its freeze list,
    while a future arch's config could add an extra alias by default.
    The intersection rule lets keys present on only one side drop out
    so the remaining shared keys compare cleanly. Real semantic
    mismatches on shared keys still raise.
    """

    def test_extra_expected_key_drops_out(self, tmp_path):
        """Saved {vision_encoder=True}; expected
        {vision_encoder=True, future_arch=False}. Load succeeds:
        future_arch is dropped from expected (not in saved)."""
        mgr = _make_manager(tmp_path)
        mgr.save(step=1, extra={"vlm_freeze": _freeze(("vision_encoder", True))})

        mgr2 = _make_manager(tmp_path)
        # Simulate a future-arch config's expected: extra alias not in saved.
        expected = _freeze(("future_arch", False), ("vision_encoder", True))
        step, _, _ = mgr2.load(vlm_freeze_expected=expected)
        assert step == 1

    def test_extra_saved_key_drops_out(self, tmp_path):
        """Saved {vision_encoder=True, future_arch=False}; expected
        {vision_encoder=True}. Load succeeds: future_arch is dropped
        from saved (not in expected)."""
        mgr = _make_manager(tmp_path)
        saved = _freeze(("future_arch", False), ("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": saved})

        mgr2 = _make_manager(tmp_path)
        expected = _freeze(("vision_encoder", True))
        step, _, _ = mgr2.load(vlm_freeze_expected=expected)
        assert step == 1

    def test_shared_key_semantic_mismatch_still_raises(self, tmp_path):
        """Even with cross-arch keys dropped, mismatch on a shared key
        (vision_encoder=True saved vs False expected) still raises."""
        mgr = _make_manager(tmp_path)
        saved = _freeze(("future_arch", False), ("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": saved})

        mgr2 = _make_manager(tmp_path)
        # Shared key vision_encoder differs in frozen state.
        mismatched = _freeze(("vision_encoder", False))
        with pytest.raises(ValueError, match="VLM freeze mismatch"):
            mgr2.load(vlm_freeze_expected=mismatched)

    def test_error_message_lists_dropped_keys(self, tmp_path):
        """When a shared-key mismatch raises during a cross-arch load,
        the error message should call out which keys were dropped from
        each side so the user can diagnose."""
        mgr = _make_manager(tmp_path)
        saved = _freeze(("future_arch", False), ("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": saved})

        mgr2 = _make_manager(tmp_path)
        mismatched = _freeze(("vision_encoder", False))
        with pytest.raises(ValueError) as exc_info:
            mgr2.load(vlm_freeze_expected=mismatched)
        # Error mentions cross-arch drops to help the user.
        assert "cross-arch" in str(exc_info.value)
        assert "future_arch" in str(exc_info.value)

    def test_disjoint_only_passes_quietly(self, tmp_path):
        """If saved and expected have NO shared keys, both filter to
        empty lists and the compare passes (degenerate but safe)."""
        mgr = _make_manager(tmp_path)
        mgr.save(step=1, extra={"vlm_freeze": _freeze(("vision_encoder", True))})

        mgr2 = _make_manager(tmp_path)
        # Wholly disjoint expected.
        weird_expected = _freeze(("future_arch", False))
        step, _, _ = mgr2.load(vlm_freeze_expected=weird_expected)
        assert step == 1
