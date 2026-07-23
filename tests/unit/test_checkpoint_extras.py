"""Tests for ``load_train_state_extras``: extras round-trip, the UID trust
boundary, and that reading never applies the saved RNG state."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import torch

import kempnerforge.checkpoint.manager as manager_mod
from kempnerforge.checkpoint import build_train_state, load_train_state_extras


class _Payload:
    """Pickle-time side effect. If ``__reduce__`` runs, the marker file appears."""

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self._marker}",))


def _write_train_state(ckpt_dir: Path, extra: dict | None = None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = build_train_state(step=7, tokens_seen=128, extra=extra)
    torch.save(state, ckpt_dir / "train_state.pt")


class TestLoadTrainStateExtras:
    def test_returns_extras_and_strips_standard_keys(self, tmp_path: Path) -> None:
        _write_train_state(tmp_path / "step_7", extra={"wandb_run_id": "abc123", "phase_idx": 1})

        extras = load_train_state_extras(tmp_path / "step_7")

        assert extras == {"wandb_run_id": "abc123", "phase_idx": 1}

    def test_no_extras_returns_empty(self, tmp_path: Path) -> None:
        _write_train_state(tmp_path / "step_7")

        assert load_train_state_extras(tmp_path / "step_7") == {}

    def test_missing_train_state_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "step_7").mkdir()

        assert load_train_state_extras(tmp_path / "step_7") == {}

    def test_does_not_touch_global_rng(self, tmp_path: Path) -> None:
        """The saved RNG state must be ignored, not applied (restore_train_state applies it)."""
        _write_train_state(tmp_path / "step_7", extra={"wandb_run_id": "abc"})

        random.seed(1234)  # move to a state different from the one captured at save time
        before_python = random.getstate()
        before_torch = torch.random.get_rng_state()

        load_train_state_extras(tmp_path / "step_7")

        assert random.getstate() == before_python
        assert torch.equal(torch.random.get_rng_state(), before_torch)

    def test_foreign_owned_file_raises_before_unpickling(self, tmp_path: Path) -> None:
        """The UID gate fires before torch.load, so a planted payload never executes."""
        ckpt_dir = tmp_path / "step_42"
        ckpt_dir.mkdir()
        marker = tmp_path / "rce_marker"
        torch.save(
            {"step": 42, "tokens_seen": 0, "rng": {}, "payload": _Payload(marker)},
            ckpt_dir / "train_state.pt",
        )

        real_uid = os.getuid()
        orig_getuid = manager_mod.os.getuid
        try:
            manager_mod.os.getuid = lambda: real_uid + 12345
            with pytest.raises(PermissionError, match="Refusing to load"):
                load_train_state_extras(ckpt_dir)
        finally:
            manager_mod.os.getuid = orig_getuid

        assert not marker.exists(), "payload fired despite the ownership gate"
