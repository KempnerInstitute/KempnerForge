"""CPU unit test for the lmms-eval registration manifest.

Skipped when lmms-eval is absent (optional, undeclared dependency).
"""

from __future__ import annotations

import pytest

pytest.importorskip("lmms_eval")

from lmms_eval.models.registry_v2 import ModelManifest  # noqa: E402

from kempnerforge.eval.vlm.registry import MANIFEST  # noqa: E402


def test_manifest_is_well_formed():
    assert isinstance(MANIFEST, ModelManifest)
    assert MANIFEST.model_id == "kempnerforge_vlm"
    assert MANIFEST.chat_class_path == "kempnerforge.eval.vlm.adapter.KempnerForgeVLM"
    # Chat-only registration keeps resolution correct even under force_simple.
    assert MANIFEST.simple_class_path is None
