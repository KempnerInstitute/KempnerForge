"""Unit tests for MemoryMappedDataset mmap lifecycle.

Covers the partial-open failure path (prior mmaps must be released so they
don't leak through exception tracebacks) and the success-path invariant
(mmaps stay open for the life of the dataset).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from kempnerforge.data.dataset import MemoryMappedDataset


def _write_npy_files(tmp_path, n_files: int, tokens_per_file: int = 1024) -> None:
    for i in range(n_files):
        arr = np.arange(tokens_per_file, dtype=np.uint32)
        np.save(tmp_path / f"shard_{i:03d}.npy", arr)


def test_partial_open_failure_closes_prior_mmaps(tmp_path):
    """If np.load raises partway through, the mmaps already opened must be closed.

    Without the fix, the prior mmaps stay live through any exception traceback
    (pytest frames, logger.exception, post-mortem debuggers), accumulating
    virtual-memory mappings on Lustre/NFS clusters under retry loops.
    """
    _write_npy_files(tmp_path, n_files=5)

    original = np.load
    calls = {"n": 0}
    opened_mmaps: list = []

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated Lustre hiccup")
        mm = original(*args, **kwargs)
        opened_mmaps.append(mm)
        return mm

    with (
        patch("kempnerforge.data.dataset.np.load", side_effect=flaky),
        pytest.raises(RuntimeError, match="Lustre hiccup"),
    ):
        MemoryMappedDataset(str(tmp_path), seq_len=128)

    assert len(opened_mmaps) == 2, f"expected 2 opens before failure, got {len(opened_mmaps)}"
    for mm in opened_mmaps:
        inner = getattr(mm, "_mmap", None)
        assert inner is not None
        assert inner.closed, "mmap was not closed after __init__ raised"


def test_close_is_idempotent_and_releases_mmaps(tmp_path):
    _write_npy_files(tmp_path, n_files=3)
    ds = MemoryMappedDataset(str(tmp_path), seq_len=128)

    inners = [mm._mmap for mm in ds._mmaps]
    assert all(not i.closed for i in inners)

    ds.close()
    assert all(i.closed for i in inners)

    # Second close is a no-op, not a crash.
    ds.close()


def test_successful_init_keeps_mmaps_open(tmp_path):
    """Regression guard: the fix must not close mmaps on the success path."""
    _write_npy_files(tmp_path, n_files=2)
    ds = MemoryMappedDataset(str(tmp_path), seq_len=128)
    assert all(not mm._mmap.closed for mm in ds._mmaps)
    sample = ds[0]
    assert "input_ids" in sample
    ds.close()


def test_partial_open_failure_on_bin_files(tmp_path):
    """Same leak guarantee applies to the .bin branch."""
    for i in range(4):
        (tmp_path / f"shard_{i:03d}.bin").write_bytes(np.arange(512, dtype=np.uint32).tobytes())

    original = np.memmap
    opened_mmaps: list = []
    calls = {"n": 0}

    def flaky_memmap(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated bin open failure")
        mm = original(*args, **kwargs)
        opened_mmaps.append(mm)
        return mm

    with (
        patch("kempnerforge.data.dataset.np.memmap", side_effect=flaky_memmap),
        pytest.raises(RuntimeError, match="bin open failure"),
    ):
        MemoryMappedDataset(str(tmp_path), seq_len=128, file_pattern="*.bin")

    assert len(opened_mmaps) == 1
    inner = getattr(opened_mmaps[0], "_mmap", None)
    assert inner is not None and inner.closed
