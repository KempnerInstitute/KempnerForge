"""The example's run-only entry points start from their own paths.

``eval.py`` runs as ``__main__`` against a stub ``lmms_eval.evaluator`` and must hand
``simple_evaluate`` the adapter it imports from its sibling ``lmms_adapter.py``. Both
scripts must also start as plain subprocesses from where they live.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
import types
from pathlib import Path

import lmms_adapter
import pytest

from .test_adapter import _patch_loaders, _vlm_job_config

EXAMPLE_ROOT = Path(__file__).resolve().parents[3]


def test_eval_passes_the_sibling_adapter_to_simple_evaluate(
    monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper
):
    received: dict = {}
    evaluator = types.ModuleType("lmms_eval.evaluator")
    evaluator.simple_evaluate = lambda **kwargs: received.update(kwargs) or {"results": {}}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmms_eval.evaluator", evaluator)
    _patch_loaders(monkeypatch, _vlm_job_config(tiny_vlm_configs), tiny_vlm_wrapper)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval.py", "--config", "x", "--checkpoint", "y", "--tasks", "task_a,task_b"]
        + ["--device", "cpu", "--dtype", "float32"],
    )

    runpy.run_path(str(EXAMPLE_ROOT / "eval.py"), run_name="__main__")

    assert Path(lmms_adapter.__file__).resolve() == EXAMPLE_ROOT / "lmms_adapter.py"
    assert type(received["model"]) is lmms_adapter.KempnerForgeVLM
    assert received["tasks"] == ["task_a", "task_b"]


@pytest.mark.parametrize("script", ["eval.py", "scripts/prep_vlm_coco.py"])
def test_script_starts_from_its_path(script):
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert f"usage: {Path(script).name}" in result.stdout
