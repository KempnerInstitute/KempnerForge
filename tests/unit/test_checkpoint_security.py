"""Tests for the train_state.pt trust boundary.

``train_state.pt`` is loaded with ``weights_only=False`` (full pickle),
because it carries scheduler state and an arbitrary caller-supplied
``extra`` dict. That means any ``__reduce__`` in the file runs at load
time. Shared-FS clusters (HolyLFS, Kempner lab storage) and imported
"pretrained" checkpoints both make the write side of that file
attacker-reachable, so the loader MUST at minimum refuse to execute
pickles planted by a different UID.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import kempnerforge.checkpoint.manager as manager_mod
from kempnerforge.checkpoint.manager import CheckpointManager, _load_train_state
from kempnerforge.config.schema import CheckpointConfig


class _Payload:
    """Pickle-time side effect. If ``__reduce__`` runs, the marker file appears."""

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self):
        # Tells pickle: on load, call os.system(cmd). os.system is a stand-in
        # for any arbitrary command the attacker chooses.
        return (os.system, (f"touch {self._marker}",))


def _write_malicious_train_state(path: Path, marker: Path) -> None:
    """Write a torch-format file that fires a side effect on load.

    ``torch.save`` wraps pickle, so a ``__reduce__`` on any object inside
    still runs when the file is opened with ``torch.load(weights_only=False)``.
    """
    train_state = {
        "step": 42,
        "tokens_seen": 1000,
        "rng": {},
        "payload": _Payload(marker),
    }
    torch.save(train_state, path)


def _write_benign_train_state(path: Path, step: int = 7, tokens_seen: int = 128) -> None:
    torch.save({"step": step, "tokens_seen": tokens_seen, "rng": {}}, path)


def _fake_ckpt_dir(tmp_path: Path, marker: Path) -> Path:
    """Build a ``step_42`` checkpoint dir with a malicious train_state.pt
    and point ``latest`` at it so ``_resolve_load_path`` picks it up."""
    ckpt_dir = tmp_path / "step_42"
    ckpt_dir.mkdir()
    _write_malicious_train_state(ckpt_dir / "train_state.pt", marker)
    (tmp_path / "latest").symlink_to("step_42")
    return ckpt_dir


def _make_manager(tmp_path: Path) -> CheckpointManager:
    config = CheckpointConfig(dir=str(tmp_path))
    model = MagicMock()
    model.state_dict.return_value = {}
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {}
    return CheckpointManager(config=config, model=model, optimizer=optimizer)


class TestLoadTrainStateOwnershipGate:
    def test_rejects_foreign_owned_file(self, tmp_path: Path) -> None:
        """_load_train_state refuses when st_uid != current uid."""
        marker = tmp_path / "rce_marker"
        path = tmp_path / "train_state.pt"
        _write_malicious_train_state(path, marker)

        real_uid = os.getuid()
        orig_getuid = manager_mod.os.getuid
        try:
            manager_mod.os.getuid = lambda: real_uid + 12345
            with pytest.raises(PermissionError, match="Refusing to load"):
                _load_train_state(path)
        finally:
            manager_mod.os.getuid = orig_getuid

        assert not marker.exists(), (
            "payload fired despite ownership gate — torch.load was reached before the check"
        )

    def test_accepts_own_file(self, tmp_path: Path) -> None:
        """_load_train_state loads when st_uid matches current uid."""
        path = tmp_path / "train_state.pt"
        _write_benign_train_state(path)

        loaded = _load_train_state(path)
        assert loaded["step"] == 7
        assert loaded["tokens_seen"] == 128

    def test_warns_on_group_writable(self, tmp_path: Path) -> None:
        """_load_train_state warns (but still loads) on group-writable files.

        Same-UID group-writable is the footgun this case addresses: a
        colleague in your lab group can plant the file. We warn instead of
        refusing because the same-UID assumption typically holds on HPC
        shared FS and the user deserves a heads-up, not a hard block.

        Asserts via a direct logger handler rather than pytest's caplog so
        the test is robust to other tests mutating logger propagation.
        """
        path = tmp_path / "train_state.pt"
        _write_benign_train_state(path, step=1, tokens_seen=1)
        path.chmod(path.stat().st_mode | stat.S_IWGRP)

        import logging

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        logger = logging.getLogger("kempnerforge.checkpoint.manager")
        prior_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            _load_train_state(path)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prior_level)

        assert any("group/world-writable" in r.getMessage() for r in records), (
            "expected a warning about group/world-writable train_state.pt"
        )

    def test_no_warning_on_private_mode(self, tmp_path: Path) -> None:
        """Files with a tight mode (600/640 without group write) don't warn."""
        path = tmp_path / "train_state.pt"
        _write_benign_train_state(path, step=1, tokens_seen=1)
        path.chmod(0o600)

        import logging

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        logger = logging.getLogger("kempnerforge.checkpoint.manager")
        prior_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            _load_train_state(path)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prior_level)

        assert not any("group/world-writable" in r.getMessage() for r in records)


class TestManagerLoadRejectsForeignCheckpoint:
    def test_load_raises_before_executing_pickle(self, tmp_path: Path) -> None:
        """CheckpointManager.load() blocks a foreign-owned pickle before it fires.

        Pre-fix, the pickle ran through ``torch.load(..., weights_only=False)``
        with no provenance check. Post-fix, ``_load_train_state`` raises
        ``PermissionError`` before ``torch.load`` is reached, so the
        ``__reduce__`` side effect never executes.

        ``exclude_keys`` skips the DCP model/optimizer load so we don't need
        a real checkpoint to reach the train_state.pt branch.
        """
        marker = tmp_path / "rce_marker_2"
        _fake_ckpt_dir(tmp_path, marker)

        real_uid = os.getuid()
        orig_getuid = manager_mod.os.getuid
        mgr = _make_manager(tmp_path)
        try:
            manager_mod.os.getuid = lambda: real_uid + 12345
            with pytest.raises(PermissionError, match="Refusing to load"):
                mgr.load(exclude_keys=["model", "optimizer"])
        finally:
            manager_mod.os.getuid = orig_getuid

        assert not marker.exists(), (
            "ownership gate did not block the load — payload fired despite the check"
        )
