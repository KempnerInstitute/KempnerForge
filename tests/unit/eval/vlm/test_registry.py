"""CPU unit test for the lmms-eval registration manifest.

Runs against the fake ``lmms_eval`` injected by ``conftest.py`` (lmms-eval is an optional,
undeclared dependency), so it executes in CI and covers ``registry.py``. Real entry-point
resolution against the installed package is verified by the gated integration test.
"""

from __future__ import annotations

from lmms_eval.models.registry_v2 import ModelManifest

from kempnerforge.eval.vlm.registry import MANIFEST


def test_manifest_is_well_formed():
    assert isinstance(MANIFEST, ModelManifest)
    assert MANIFEST.model_id == "kempnerforge_vlm"
    assert MANIFEST.chat_class_path == "kempnerforge.eval.vlm.adapter.KempnerForgeVLM"
    # Chat-only registration keeps resolution correct even under force_simple.
    assert MANIFEST.simple_class_path is None
