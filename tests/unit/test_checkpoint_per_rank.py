"""Unit tests for per-rank dataloader/RNG checkpoint state (schema v1 envelope).

Four simulated ranks with distinct loader cursors, prefetch depths, and RNG
streams prove that each rank saves and restores its own payload. Legacy
checkpoints, topology mismatches, and mid-save interruption cover the
compatibility and durability surface. Uses the mocked-DCP manager patterns
from test_checkpoint.py.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import kempnerforge.checkpoint.manager as mgr_mod
from kempnerforge.checkpoint.manager import _DCP_METADATA_FILE, CheckpointManager
from kempnerforge.checkpoint.state import (
    PER_RANK_KEY,
    PER_RANK_VERSION,
    build_per_rank_envelope,
    build_train_state,
    capture_rank_state,
    get_rng_state,
    restore_train_state,
    select_rank_state,
)
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig

WORLD = 4


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw() -> tuple[float, float, float]:
    """One draw from each generator; identifies the stream position exactly."""
    return (random.random(), float(np.random.random()), float(torch.rand(())))


def _mk_payloads(
    seed_base: int, cursor_base: int
) -> tuple[dict[int, dict], dict[int, tuple[float, float, float]]]:
    """Distinct, verifiable payloads for ranks 0..WORLD-1.

    Each payload holds that rank's RNG capture and a loader cursor with a
    rank-specific prefetch depth; ``expected[r]`` is the next draw rank r's
    streams must reproduce after restore.
    """
    payloads: dict[int, dict] = {}
    expected: dict[int, tuple[float, float, float]] = {}
    for r in range(WORLD):
        _seed_all(seed_base + r)
        payloads[r] = {
            "rng": get_rng_state(),
            "dataloader": {"cursor": cursor_base + r, "prefetch_depth": 1 + r},
        }
        expected[r] = _draw()
    return payloads, expected


def _mock_mgr(tmp_path, monkeypatch, **cfg_kwargs) -> CheckpointManager:
    """CheckpointManager with DCP calls mocked out (same seam as test_checkpoint)."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    config = CheckpointConfig(dir=str(tmp_path), keep_last_n=5, **cfg_kwargs)
    mgr = CheckpointManager(config, model, opt)
    monkeypatch.setattr(mgr._async_ckpt, "save", MagicMock())
    monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())
    return mgr


def _install_fake_world(monkeypatch, payloads: dict[int, dict], events: list) -> None:
    """Simulate a WORLD-rank job around rank 0's manager.

    ``all_gather_object`` fills the output list with the handcrafted payloads
    of every rank, exactly as the real collective would; events record the
    gather and the barrier with the group they ran on.
    """
    monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mgr_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mgr_mod.dist, "get_world_size", lambda: WORLD)
    monkeypatch.setattr(mgr_mod.dist, "new_group", lambda backend=None: "GLOO_PG")
    monkeypatch.setattr(
        mgr_mod.dist, "barrier", lambda group=None: events.append(("barrier", group))
    )

    def fake_all_gather(obj_list, obj, group=None):
        events.append(("gather", group))
        for r in range(WORLD):
            obj_list[r] = (r, payloads[r])

    monkeypatch.setattr(mgr_mod.dist, "all_gather_object", fake_all_gather)


def _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step: int) -> list:
    """Write a real ``step_N`` checkpoint whose envelope holds the payloads."""
    events: list = []
    with monkeypatch.context() as m:
        _install_fake_world(m, payloads, events)
        mgr = _mock_mgr(tmp_path, m)
        mgr.save(step=step, tokens_seen=step * 64)
    return events


def _load_as_rank(tmp_path, monkeypatch, rank: int, step: int, world_size: int = WORLD, **cfg):
    """Load a checkpoint as a simulated rank of a ``world_size`` job."""
    mgr = _mock_mgr(tmp_path, monkeypatch, **cfg)
    mgr._rank = rank
    mgr._world_size = world_size
    result = mgr.load(path=str(Path(tmp_path) / f"step_{step}"))
    return mgr, result


# ---------------------------------------------------------------------------
# Envelope schema (executable documentation of schema v1)
# ---------------------------------------------------------------------------


class TestPerRankEnvelopeSchema:
    def test_schema_v1_shape(self):
        """Schema v1: {'version': 1, 'world_size': W, 'ranks': {rank: payload}}."""
        entries = {0: {"rng": {}}, 1: {"rng": {}}}
        env = build_per_rank_envelope(entries, world_size=2)
        assert set(env) == {"version", "world_size", "ranks"}
        assert env["version"] == PER_RANK_VERSION == 1
        assert env["world_size"] == 2
        assert env["ranks"] is entries

    def test_build_refuses_missing_rank(self):
        with pytest.raises(ValueError, match=r"rank 0\.\.2"):
            build_per_rank_envelope({0: {}, 2: {}}, world_size=3)

    def test_build_refuses_extra_rank(self):
        with pytest.raises(ValueError, match="per-rank envelope"):
            build_per_rank_envelope({0: {}, 1: {}}, world_size=1)

    def test_capture_without_loader_has_rng_only(self):
        payload = capture_rank_state(None)
        assert set(payload) == {"rng"}
        assert "torch_cpu" in payload["rng"]

    def test_capture_with_stateful_loader(self):
        class Loader:
            def state_dict(self):
                return {"cursor": 11}

        payload = capture_rank_state(Loader())
        assert payload["dataloader"] == {"cursor": 11}

    def test_capture_ignores_stateless_loader(self):
        class Plain:
            pass

        assert "dataloader" not in capture_rank_state(Plain())

    def test_select_returns_each_ranks_own_entry(self):
        payloads, _ = _mk_payloads(seed_base=10, cursor_base=100)
        env = build_per_rank_envelope(payloads, world_size=WORLD)
        for r in range(WORLD):
            entry = select_rank_state(env, rank=r, world_size=WORLD)
            assert entry["dataloader"] == {"cursor": 100 + r, "prefetch_depth": 1 + r}

    def test_select_refuses_world_size_change(self):
        payloads, _ = _mk_payloads(seed_base=10, cursor_base=100)
        env = build_per_rank_envelope(payloads, world_size=WORLD)
        with pytest.raises(ValueError, match="world_size=4.*world_size=2") as exc:
            select_rank_state(env, rank=0, world_size=2)
        assert "ignore_rank_state_mismatch" in str(exc.value)

    def test_select_refuses_newer_version(self):
        env = {"version": PER_RANK_VERSION + 1, "world_size": 1, "ranks": {0: {}}}
        with pytest.raises(ValueError, match="newer"):
            select_rank_state(env, rank=0, world_size=1)

    @pytest.mark.parametrize(
        "envelope",
        [
            "not-a-dict",
            {"version": 1, "world_size": 1},  # no ranks
            {"version": 1, "world_size": 1, "ranks": [("0", {})]},  # ranks not a dict
            {"version": "1", "world_size": 1, "ranks": {0: {}}},  # version not int
            {"version": 0, "world_size": 1, "ranks": {0: {}}},  # version < 1
            {"version": 1, "world_size": "4", "ranks": {0: {}}},  # world_size not int
        ],
    )
    def test_select_refuses_malformed(self, envelope):
        with pytest.raises(ValueError):
            select_rank_state(envelope, rank=0, world_size=1)

    def test_select_refuses_missing_own_rank(self):
        env = {"version": 1, "world_size": 2, "ranks": {0: {"rng": {}}}}
        with pytest.raises(ValueError, match="no payload for rank 1"):
            select_rank_state(env, rank=1, world_size=2)

    def test_restore_keeps_envelope_out_of_extra(self):
        env = build_per_rank_envelope({0: {"rng": get_rng_state()}}, world_size=1)
        state = build_train_state(step=3, tokens_seen=96, extra={"phase_idx": 2}, per_rank=env)
        step, tokens, extra = restore_train_state(state)
        assert (step, tokens) == (3, 96)
        assert extra == {"phase_idx": 2}


# ---------------------------------------------------------------------------
# Four simulated ranks through the real manager save/load path
# ---------------------------------------------------------------------------


class TestFourRankSaveLoad:
    def test_each_rank_restores_its_own_loader_and_rng(self, tmp_path, monkeypatch):
        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)

        for r in range(WORLD):
            _seed_all(555)  # scramble, same on every rank, before each load
            mgr, (step, tokens, _) = _load_as_rank(tmp_path, monkeypatch, rank=r, step=7)
            assert (step, tokens) == (7, 448)
            assert mgr._pending_dataloader_state == {"cursor": 100 + r, "prefetch_depth": 1 + r}
            assert _draw() == expected[r], f"rank {r} did not restore its own RNG streams"

    def test_saved_entries_are_distinct_across_ranks(self, tmp_path, monkeypatch):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)
        raw = torch.load(tmp_path / "step_7" / "train_state.pt", weights_only=False)
        env = raw[PER_RANK_KEY]
        assert env["version"] == 1 and env["world_size"] == WORLD
        cursors = {env["ranks"][r]["dataloader"]["cursor"] for r in range(WORLD)}
        assert len(cursors) == WORLD
        cpu_states = [bytes(env["ranks"][r]["rng"]["torch_cpu"].numpy()) for r in range(WORLD)]
        assert len(set(cpu_states)) == WORLD, "rank RNG captures are not distinct"

    def test_two_consecutive_resumes(self, tmp_path, monkeypatch):
        first, exp_first = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, first, step=7)
        for r in (2, 3):
            _seed_all(555)
            mgr, (step, _, _) = _load_as_rank(tmp_path, monkeypatch, rank=r, step=7)
            assert step == 7
            assert mgr._pending_dataloader_state == {"cursor": 100 + r, "prefetch_depth": 1 + r}
            assert _draw() == exp_first[r]

        # The resumed run advances and saves again: a fresh envelope replaces
        # the first one entirely, and the second resume restores it.
        second, exp_second = _mk_payloads(seed_base=2000, cursor_base=200)
        _save_four_rank_ckpt(tmp_path, monkeypatch, second, step=8)
        for r in range(WORLD):
            _seed_all(555)
            mgr, (step, _, _) = _load_as_rank(tmp_path, monkeypatch, rank=r, step=8)
            assert step == 8
            assert mgr._pending_dataloader_state == {"cursor": 200 + r, "prefetch_depth": 1 + r}
            assert _draw() == exp_second[r]

    def test_save_contributes_own_payload_to_gather(self, tmp_path, monkeypatch):
        """The object each rank hands the collective is (global_rank, payload)."""
        contributed: list = []

        class Loader:
            def state_dict(self):
                return {"cursor": 41}

        with monkeypatch.context() as m:
            events: list = []
            _install_fake_world(m, dict(_mk_payloads(1, 1)[0]), events)

            def capturing_gather(obj_list, obj, group=None):
                contributed.append((obj, group))
                for r in range(WORLD):
                    obj_list[r] = (r, _mk_payloads(1, 1)[0][r])

            m.setattr(mgr_mod.dist, "all_gather_object", capturing_gather)
            mgr = _mock_mgr(tmp_path, m)
            mgr.save(step=1, dataloader=Loader())

        (obj, group) = contributed[0]
        assert group == "GLOO_PG", "gather must run on the dedicated gloo group"
        rank, payload = obj
        assert rank == 0
        assert payload["dataloader"] == {"cursor": 41}
        assert "torch_cpu" in payload["rng"]

    def test_collective_and_durability_ordering(self, tmp_path, monkeypatch):
        """gather -> train_state write -> latest commit -> end barrier."""
        payloads, _ = _mk_payloads(seed_base=1, cursor_base=1)
        events: list = []
        with monkeypatch.context() as m:
            _install_fake_world(m, payloads, events)
            mgr = _mock_mgr(tmp_path, m)

            orig_torch_save = torch.save

            def recording_save(obj, f, *args, **kwargs):
                events.append(("write", Path(str(f)).name))
                orig_torch_save(obj, f, *args, **kwargs)

            m.setattr(mgr_mod.torch, "save", recording_save)
            orig_commit = mgr._commit_latest
            m.setattr(
                mgr,
                "_commit_latest",
                lambda step, d: (events.append(("commit", step)), orig_commit(step, d)),
            )
            mgr.save(step=4)

        names = [e[0] for e in events]
        assert names == ["gather", "write", "commit", "barrier"]
        assert events[0][1] == "GLOO_PG" and events[-1][1] == "GLOO_PG"
        assert events[1][1] == "train_state.pt"


# ---------------------------------------------------------------------------
# Legacy compatibility and topology validation
# ---------------------------------------------------------------------------


class TestLegacyAndTopology:
    def _write_legacy_ckpt(self, tmp_path, step: int):
        """A checkpoint exactly as the pre-envelope manager wrote it."""
        d = Path(tmp_path) / f"step_{step}"
        d.mkdir(parents=True)
        _seed_all(777)
        state = {
            "step": step,
            "tokens_seen": 99,
            "rng": get_rng_state(),
            "dataloader": {"epoch": 2, "batches_yielded": 50},
        }
        expected = _draw()
        torch.save(state, d / "train_state.pt")
        return expected

    def test_legacy_checkpoint_restores_single_copy_on_every_rank(self, tmp_path, monkeypatch):
        expected = self._write_legacy_ckpt(tmp_path, step=3)
        for r in (0, 3):
            _seed_all(555)
            mgr, (step, tokens, _) = _load_as_rank(tmp_path, monkeypatch, rank=r, step=3)
            assert (step, tokens) == (3, 99)
            assert mgr._pending_dataloader_state == {"epoch": 2, "batches_yielded": 50}
            assert _draw() == expected  # rank 0's copy, on every rank, as before

    def test_new_checkpoint_keeps_legacy_single_copy_keys(self, tmp_path, monkeypatch):
        """Old readers keep working: the single-copy keys sit beside the envelope."""
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)
        raw = torch.load(tmp_path / "step_7" / "train_state.pt", weights_only=False)
        assert "rng" in raw and raw[PER_RANK_KEY]["version"] == 1
        # A legacy reader restores the flat keys and never looks at the
        # envelope; emulate it with the documented single-copy limitation.
        legacy_view = {k: v for k, v in raw.items() if k != PER_RANK_KEY}
        step, tokens, extra = restore_train_state(legacy_view)
        assert (step, tokens) == (7, 448)
        assert extra == {}

    def test_world_size_mismatch_refused_with_actionable_error(self, tmp_path, monkeypatch):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)
        with pytest.raises(ValueError, match="world_size=4.*world_size=2") as exc:
            _load_as_rank(tmp_path, monkeypatch, rank=1, step=7, world_size=2)
        assert "ignore_rank_state_mismatch" in str(exc.value)

    def test_world_size_mismatch_ignored_discards_loader_and_rng(self, tmp_path, monkeypatch):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)

        # Build the manager first: module init consumes torch RNG.
        mgr = _mock_mgr(tmp_path, monkeypatch, ignore_rank_state_mismatch=True)
        mgr._rank = 1
        mgr._world_size = 2
        _seed_all(555)
        ref = _draw()  # where seed 555's streams go when nothing restores RNG
        _seed_all(555)
        step, tokens, _ = mgr.load(path=str(Path(tmp_path) / "step_7"))
        assert (step, tokens) == (7, 448)
        assert mgr._pending_dataloader_state is None
        assert _draw() == ref, "RNG must stay untouched when per-rank state is discarded"

    def test_missing_rank_entry_refused(self, tmp_path, monkeypatch):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)
        path = tmp_path / "step_7" / "train_state.pt"
        raw = torch.load(path, weights_only=False)
        del raw[PER_RANK_KEY]["ranks"][2]
        torch.save(raw, path)
        with pytest.raises(ValueError, match="no payload for rank 2"):
            _load_as_rank(tmp_path, monkeypatch, rank=2, step=7)

    def test_single_process_round_trip(self, tmp_path, monkeypatch):
        """The default non-distributed path uses the envelope end to end."""

        class Loader:
            def state_dict(self):
                return {"cursor": 9}

        mgr = _mock_mgr(tmp_path, monkeypatch)
        _seed_all(31)
        mgr.save(step=2, tokens_seen=128, dataloader=Loader())
        expected = _draw()

        _seed_all(555)
        mgr2, (step, tokens, _) = _load_as_rank(tmp_path, monkeypatch, rank=0, step=2, world_size=1)
        assert (step, tokens) == (2, 128)
        assert mgr2._pending_dataloader_state == {"cursor": 9}
        assert _draw() == expected

    def test_world_one_checkpoint_refused_on_four_ranks(self, tmp_path, monkeypatch):
        class Loader:
            def state_dict(self):
                return {"cursor": 9}

        mgr = _mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=2, dataloader=Loader())
        with pytest.raises(ValueError, match="world_size=1"):
            _load_as_rank(tmp_path, monkeypatch, rank=3, step=2, world_size=WORLD)


# ---------------------------------------------------------------------------
# Mid-save interruption: durability ordering with the envelope present
# ---------------------------------------------------------------------------


class TestMidSaveInterruption:
    @staticmethod
    def _install_async_mock(mgr, monkeypatch):
        """AsyncCheckpointer contract: save(new) first makes the PREVIOUS flush
        durable, then dispatches the new one; wait() drains the last one."""
        st = {"prev": None}

        def _save(state_dict, checkpoint_id, process_group=None):
            if st["prev"] is not None:
                (Path(st["prev"]) / _DCP_METADATA_FILE).write_text("ok")
            st["prev"] = checkpoint_id

        def _wait():
            if st["prev"] is not None:
                (Path(st["prev"]) / _DCP_METADATA_FILE).write_text("ok")

        monkeypatch.setattr(mgr._async_ckpt, "save", _save)
        monkeypatch.setattr(mgr._async_ckpt, "wait", _wait)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())
        return st

    def test_interrupted_flush_resumes_from_last_durable_envelope(self, tmp_path, monkeypatch):
        """Crash mid-flush: train_state.pt (with envelope) of the torn step is
        already on disk, but `latest` and auto-resume stay on the last step
        whose DCP shards are durable, and that step's per-rank state restores."""

        class Loader:
            def __init__(self):
                self.cursor = 0

            def state_dict(self):
                return {"cursor": self.cursor}

        loader = Loader()
        model = torch.nn.Linear(4, 4)
        mgr = CheckpointManager(
            CheckpointConfig(
                dir=str(tmp_path), keep_last_n=3, async_mode=AsyncCheckpointMode.async_pinned
            ),
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
        )
        self._install_async_mock(mgr, monkeypatch)

        loader.cursor = 10
        _seed_all(31)
        mgr.save(step=10, tokens_seen=640, dataloader=loader)
        expected = _draw()

        # Second save makes step_10 durable + committed; step_20's flush stays
        # in flight and the process "crashes" here (wait() never runs).
        loader.cursor = 20
        mgr.save(step=20, tokens_seen=1280, dataloader=loader)

        # Durability ordering: the torn checkpoint's train_state.pt was
        # written before any commit, yet `latest` still names step_10.
        assert (tmp_path / "step_20" / "train_state.pt").exists()
        assert not (tmp_path / "step_20" / _DCP_METADATA_FILE).exists()
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_10").resolve()

        # Fresh process: auto-resume lands on step_10 and restores ITS state.
        mgr2 = _mock_mgr(tmp_path, monkeypatch)
        _seed_all(555)
        step, tokens, _ = mgr2.load()
        assert (step, tokens) == (10, 640)
        assert mgr2._pending_dataloader_state == {"cursor": 10}
        assert _draw() == expected
