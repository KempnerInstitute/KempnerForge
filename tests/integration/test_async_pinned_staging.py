"""Integration test: ``async_with_pinned_mem`` snapshots a GPU state into pinned memory.

The saves go through the real ``dcp.async_save`` and ``DefaultStager``; a wrapper only
records what the stager returns, so the test can check that the snapshot sits in pinned
host memory, that a later save refills the same buffers, and that changing the state after
``save()`` returns does not reach the checkpoint. The background write is held until the
state has changed, so a snapshot that still points at the live state is always caught.
"""

from __future__ import annotations

import threading

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.staging import DefaultStager

from kempnerforge.checkpoint import async_save
from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.config.schema import AsyncCheckpointMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")


def test_pinned_mode_snapshots_into_reused_pinned_memory(tmp_path, monkeypatch):
    """Two saves through one checkpointer each snapshot the state into pinned memory and
    load back as it was when save() returned."""
    staged = []
    real_stage = DefaultStager.stage

    def recording_stage(self, state_dict, **kwargs):
        result = real_stage(self, state_dict, **kwargs)
        staged.append(result)
        return result

    monkeypatch.setattr(DefaultStager, "stage", recording_stage)
    release = threading.Event()
    real_write = FileSystemWriter.write_data

    def held_write(self, plan, planner):
        assert release.wait(timeout=60)
        return real_write(self, plan, planner)

    monkeypatch.setattr(FileSystemWriter, "write_data", held_write)
    ckpt = AsyncCheckpointer(mode=AsyncCheckpointMode.async_pinned)
    state = {
        "weight": torch.randn(256, 128, device="cuda"),
        "bias": torch.randn(128, device="cuda"),
        "step": torch.tensor(3.0, device="cuda"),
    }
    for step in (1, 2):
        release.clear()
        snapshot = {name: tensor.clone() for name, tensor in state.items()}
        checkpoint_id = str(tmp_path / f"step_{step}")
        try:
            ckpt.save(state, checkpoint_id=checkpoint_id)
            for tensor in state.values():
                tensor.add_(1.0)
        finally:
            release.set()
        ckpt.wait()
        assert staged[-1]["weight"].is_pinned() and staged[-1]["bias"].is_pinned()
        loaded = {name: torch.zeros_like(tensor) for name, tensor in state.items()}
        dcp.load(loaded, checkpoint_id=checkpoint_id)
        assert all(torch.equal(loaded[name], snapshot[name]) for name in state)
    if async_save._reuses_stager():
        assert staged[0]["weight"].data_ptr() == staged[1]["weight"].data_ptr()
