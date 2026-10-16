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
    set_rng_state,
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


def _mock_mgr(tmp_path, monkeypatch, dp_info=None, **cfg_kwargs) -> CheckpointManager:
    """CheckpointManager with DCP calls mocked out (same seam as test_checkpoint)."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    config = CheckpointConfig(dir=str(tmp_path), keep_last_n=5, **cfg_kwargs)
    mgr = CheckpointManager(config, model, opt, dp_info=dp_info)
    monkeypatch.setattr(mgr._async_ckpt, "save", MagicMock())
    monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())
    return mgr


def _install_fake_world(
    monkeypatch, payloads: dict[int, dict], events: list, layout: dict | None = None
) -> None:
    """Simulate a WORLD-rank job around rank 0's manager.

    ``all_gather_object`` fills the output list with every rank's
    ``(global_rank, dp_coordinates, payload)``, exactly as the real collective
    would (coordinates from ``layout``, or None); events record the gather and
    the barrier with the group they ran on.
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
            obj_list[r] = (r, layout[r] if layout is not None else None, payloads[r])

    monkeypatch.setattr(mgr_mod.dist, "all_gather_object", fake_all_gather)


def _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step: int, layout=None) -> list:
    """Write a real ``step_N`` checkpoint whose envelope holds the payloads.

    The saving process is rank 0: its live RNG streams and loader are rank 0's,
    so the single-copy ``rng``/``dataloader`` keys hold ``payloads[0]``.
    """
    events: list = []

    class Rank0Loader:
        def state_dict(self):
            return dict(payloads[0]["dataloader"])

    with monkeypatch.context() as m:
        _install_fake_world(m, payloads, events, layout=layout)
        mgr = _mock_mgr(tmp_path, m, dp_info=layout[0] if layout is not None else None)
        set_rng_state(payloads[0]["rng"])
        loader = Rank0Loader() if "dataloader" in payloads[0] else None
        mgr.save(step=step, tokens_seen=step * 64, dataloader=loader)
    return events


def _install_fake_load_world(
    m, rank: int, layout: dict | None, ckpt_dir: Path, gathers: list, world: int = WORLD
):
    """Simulate global rank ``rank`` of a ``world``-rank job during ``load()``.

    Broadcasts deliver rank 0's values (the file-existence flag, then the
    train state rank 0 reads). The gather delivers every rank's current
    ``(global_rank, dp_coordinates)`` from ``layout`` and records what this
    rank contributed and on which group.
    """
    m.setattr(mgr_mod.dist, "is_initialized", lambda: True)
    m.setattr(mgr_mod.dist, "get_rank", lambda: rank)
    m.setattr(mgr_mod.dist, "get_world_size", lambda: world)
    m.setattr(mgr_mod.dist, "new_group", lambda backend=None: "GLOO_PG")
    ts_path = ckpt_dir / "train_state.pt"

    def fake_broadcast(obj_list, src=0):
        if obj_list[0] is False:
            obj_list[0] = ts_path.exists()
        elif obj_list[0] is None:
            obj_list[0] = torch.load(ts_path, weights_only=False)

    def fake_all_gather(obj_list, obj, group=None):
        gathers.append((obj, group))
        for r in range(world):
            obj_list[r] = (r, layout[r] if layout is not None else None)

    m.setattr(mgr_mod.dist, "broadcast_object_list", fake_broadcast)
    m.setattr(mgr_mod.dist, "all_gather_object", fake_all_gather)


def _load_in_layout(tmp_path, monkeypatch, rank: int, step: int, layout: dict | None, **cfg):
    """Resume as global rank ``rank`` of a WORLD-rank job with the given current
    data-parallel layout. Seeds every generator with 555 right before the load,
    so a discarded state leaves seed 555's streams in place."""
    gathers: list = []
    ckpt_dir = Path(tmp_path) / f"step_{step}"
    with monkeypatch.context() as m:
        _install_fake_load_world(m, rank, layout, ckpt_dir, gathers)
        mgr = _mock_mgr(tmp_path, m, dp_info=layout[rank] if layout is not None else None, **cfg)
        _seed_all(555)
        result = mgr.load(path=str(ckpt_dir), exact_resume=True)
    return mgr, result, gathers


def _load_as_rank(tmp_path, monkeypatch, rank: int, step: int, world_size: int = WORLD, **cfg):
    """Resume from a checkpoint as a simulated rank of a ``world_size`` job."""
    mgr = _mock_mgr(tmp_path, monkeypatch, **cfg)
    mgr._rank = rank
    mgr._world_size = world_size
    result = mgr.load(path=str(Path(tmp_path) / f"step_{step}"), exact_resume=True)
    return mgr, result


# ---------------------------------------------------------------------------
# Envelope schema (executable documentation of schema v1)
# ---------------------------------------------------------------------------


class TestPerRankEnvelopeSchema:
    def test_schema_v1_shape(self):
        """Schema v1: {'version': 1, 'world_size': W, 'data_parallel': layout | None,
        'ranks': {rank: payload}}."""
        entries = {0: {"rng": {}}, 1: {"rng": {}}}
        env = build_per_rank_envelope(entries, world_size=2)
        assert set(env) == {"version", "world_size", "data_parallel", "ranks"}
        assert env["version"] == PER_RANK_VERSION == 1
        assert env["world_size"] == 2
        assert env["data_parallel"] is None
        assert env["ranks"] is entries

    def test_schema_v1_records_a_layout(self):
        layout = {0: (0, 1), 1: (0, 1)}
        env = build_per_rank_envelope({0: {"rng": {}}, 1: {"rng": {}}}, 2, data_parallel=layout)
        assert env["data_parallel"] == {0: (0, 1), 1: (0, 1)}

    @pytest.mark.parametrize(
        "layout",
        [
            {0: (0, 2)},  # rank 1 missing
            {0: (0, 2), 1: (2, 2)},  # dp_rank out of range
            {0: (0, 2), 1: ("1", 2)},  # not an int
        ],
    )
    def test_build_refuses_malformed_layout(self, layout):
        with pytest.raises(ValueError, match="data-parallel layout"):
            build_per_rank_envelope({0: {"rng": {}}, 1: {"rng": {}}}, 2, data_parallel=layout)

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
        env = {"version": 1, "world_size": 2, "data_parallel": None, "ranks": {0: {"rng": {}}}}
        with pytest.raises(ValueError, match=r"no payload for ranks \[1\]"):
            select_rank_state(env, rank=1, world_size=2)

    def test_restore_keeps_envelope_out_of_extra(self):
        env = build_per_rank_envelope({0: {"rng": get_rng_state()}}, world_size=1)
        state = build_train_state(step=3, tokens_seen=96, extra={"phase_idx": 2}, per_rank=env)
        step, tokens, extra = restore_train_state(state)
        assert (step, tokens) == (3, 96)
        assert extra == {"phase_idx": 2}


# ---------------------------------------------------------------------------
# Whole-envelope validation before any entry is selected
# ---------------------------------------------------------------------------


def _defective_envelope(defect: str) -> dict:
    """A WORLD-rank envelope carrying exactly one described defect."""
    payloads, _ = _mk_payloads(seed_base=10, cursor_base=100)
    ranks: dict = dict(payloads)
    if defect == "empty_entry":
        ranks[1] = {}
    elif defect == "missing_entry":
        del ranks[2]
    elif defect == "extra_entry":
        ranks[WORLD] = dict(payloads[0])
    elif defect == "entry_not_dict":
        ranks[3] = None
    elif defect == "no_rng":
        del ranks[0]["rng"]
    elif defect == "rng_not_dict":
        ranks[2]["rng"] = "state"
    elif defect == "rng_empty":
        ranks[2]["rng"] = {}
    elif defect == "rng_without_torch_cpu":
        del ranks[1]["rng"]["torch_cpu"]
    elif defect == "dataloader_on_some_ranks":
        del ranks[1]["dataloader"]
    elif defect == "dataloader_not_dict":
        ranks[3]["dataloader"] = None
    elif defect == "str_rank_keys":
        ranks = {str(r): p for r, p in payloads.items()}
    else:
        raise AssertionError(defect)
    return {"version": PER_RANK_VERSION, "world_size": WORLD, "data_parallel": None, "ranks": ranks}


_ENVELOPE_DEFECTS = [
    "empty_entry",
    "missing_entry",
    "extra_entry",
    "entry_not_dict",
    "no_rng",
    "rng_not_dict",
    "rng_empty",
    "rng_without_torch_cpu",
    "dataloader_on_some_ranks",
    "dataloader_not_dict",
    "str_rank_keys",
]


def _corrupt_saved_entry(tmp_path, step: int, defect: str) -> None:
    """Rewrite rank 2's entry of a saved checkpoint (missing or empty)."""
    path = Path(tmp_path) / f"step_{step}" / "train_state.pt"
    raw = torch.load(path, weights_only=False)
    if defect == "missing_entry":
        del raw[PER_RANK_KEY]["ranks"][2]
    elif defect == "empty_entry":
        raw[PER_RANK_KEY]["ranks"][2] = {}
    else:
        raise AssertionError(defect)
    torch.save(raw, path)


class TestEnvelopeValidatedBeforeSelection:
    """A defect anywhere in the envelope is every rank's decision, not one rank's."""

    @pytest.mark.parametrize("defect", _ENVELOPE_DEFECTS)
    def test_every_rank_refuses_with_the_same_reason(self, defect):
        env = _defective_envelope(defect)
        reasons = []
        for r in range(WORLD):
            with pytest.raises(ValueError) as exc:
                select_rank_state(env, rank=r, world_size=WORLD)
            reasons.append(str(exc.value))
        assert len(set(reasons)) == 1, f"ranks disagree on {defect}: {reasons}"

    def test_rng_only_entries_are_complete(self):
        """A loader without state_dict leaves every entry rng-only; that is valid."""
        payloads, _ = _mk_payloads(seed_base=10, cursor_base=100)
        for p in payloads.values():
            del p["dataloader"]
        env = {"version": 1, "world_size": WORLD, "data_parallel": None, "ranks": payloads}
        for r in range(WORLD):
            assert set(select_rank_state(env, rank=r, world_size=WORLD)) == {"rng"}

    @staticmethod
    def _cuda_envelope(without_cuda: list[int]) -> dict:
        """Every entry carries a 'torch_cuda' capture except ranks ``without_cuda``."""
        payloads, _ = _mk_payloads(seed_base=10, cursor_base=100)
        for r, p in payloads.items():
            p["rng"].pop("torch_cuda", None)
            if r not in without_cuda:
                p["rng"]["torch_cuda"] = torch.zeros(16, dtype=torch.uint8)
        return {"version": 1, "world_size": WORLD, "data_parallel": None, "ranks": payloads}

    @pytest.mark.parametrize("without_cuda", [[], list(range(WORLD))], ids=["all", "none"])
    def test_torch_cuda_on_all_or_no_ranks_is_complete(self, without_cuda):
        env = self._cuda_envelope(without_cuda)
        for r in range(WORLD):
            assert select_rank_state(env, rank=r, world_size=WORLD) is env["ranks"][r]

    def test_torch_cuda_on_some_ranks_refused_on_every_rank(self):
        """A rank without the CUDA capture would resume with fresh CUDA streams."""
        env = self._cuda_envelope(without_cuda=[1])
        reasons = []
        for r in range(WORLD):
            with pytest.raises(ValueError, match=r"'torch_cuda' missing on ranks \[1\]") as exc:
                select_rank_state(env, rank=r, world_size=WORLD)
            reasons.append(str(exc.value))
        assert len(set(reasons)) == 1, f"ranks disagree: {reasons}"

    @pytest.mark.parametrize(
        ("defect", "reason"),
        [
            ("missing_entry", r"no payload for ranks \[2\]"),
            ("empty_entry", r"entry for rank 2 lacks a complete 'rng' capture"),
        ],
    )
    def test_load_refuses_on_every_rank(self, tmp_path, monkeypatch, defect, reason):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)
        _corrupt_saved_entry(tmp_path, step=7, defect=defect)
        for r in range(WORLD):
            with pytest.raises(ValueError, match=reason):
                _load_as_rank(tmp_path, monkeypatch, rank=r, step=7)

    @pytest.mark.parametrize("defect", ["missing_entry", "empty_entry"])
    def test_override_discards_on_every_rank(self, tmp_path, monkeypatch, defect):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)
        _corrupt_saved_entry(tmp_path, step=7, defect=defect)
        for r in range(WORLD):
            mgr = _mock_mgr(tmp_path, monkeypatch, ignore_rank_state_mismatch=True)
            mgr._rank = r
            mgr._world_size = WORLD
            _seed_all(555)
            ref = _draw()
            _seed_all(555)
            step, tokens, _ = mgr.load(path=str(Path(tmp_path) / "step_7"), exact_resume=True)
            assert (step, tokens) == (7, 448)
            assert mgr._pending_dataloader_state is None, f"rank {r} restored a loader cursor"
            assert _draw() == ref, f"rank {r} restored RNG from a defective envelope"


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

    def test_rng_only_entries_restore_rng_and_no_loader_state(self, tmp_path, monkeypatch):
        """Without a stateful loader every entry is rng-only: each rank restores its
        own RNG streams and no loader state is pending."""
        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        for p in payloads.values():
            del p["dataloader"]
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7)

        for r in range(WORLD):
            _seed_all(555)
            mgr, (step, _, _) = _load_as_rank(tmp_path, monkeypatch, rank=r, step=7)
            assert step == 7
            assert mgr._pending_dataloader_state is None
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
                    obj_list[r] = (r, (r, WORLD), _mk_payloads(1, 1)[0][r])

            m.setattr(mgr_mod.dist, "all_gather_object", capturing_gather)
            mgr = _mock_mgr(tmp_path, m, dp_info=(0, WORLD))
            mgr.save(step=1, dataloader=Loader())

        (obj, group) = contributed[0]
        assert group == "GLOO_PG", "gather must run on the dedicated gloo group"
        rank, dp_coords, payload = obj
        assert rank == 0
        assert dp_coords == (0, WORLD)
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
        step, tokens, _ = mgr.load(path=str(Path(tmp_path) / "step_7"), exact_resume=True)
        assert (step, tokens) == (7, 448)
        assert mgr._pending_dataloader_state is None
        assert _draw() == ref, "RNG must stay untouched when per-rank state is discarded"

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
# Data-parallel layout: same world size does not mean same data partition
# ---------------------------------------------------------------------------

# Hand-enumerated global rank -> (dp_rank, dp_size) for a 4-rank job. The
# device mesh is row-major over (pp, dp_replicate, dp_shard, ep, tp).
DP4 = {0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (3, 4)}  # dp_shard=4
DP2_TP2 = {0: (0, 2), 1: (0, 2), 2: (1, 2), 3: (1, 2)}  # dp_shard=2, tp=2
PP2_DP2 = {0: (0, 2), 1: (1, 2), 2: (0, 2), 3: (1, 2)}  # pp=2, dp_shard=2


class TestDataParallelLayout:
    def test_envelope_records_every_ranks_coordinates(self, tmp_path, monkeypatch):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=DP2_TP2)
        raw = torch.load(tmp_path / "step_7" / "train_state.pt", weights_only=False)
        assert raw[PER_RANK_KEY]["data_parallel"] == {0: (0, 2), 1: (0, 2), 2: (1, 2), 3: (1, 2)}

    def test_same_layout_restores_each_rank(self, tmp_path, monkeypatch):
        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=DP2_TP2)
        for r in range(WORLD):
            mgr, (step, tokens, _), gathers = _load_in_layout(
                tmp_path, monkeypatch, rank=r, step=7, layout=DP2_TP2
            )
            assert (step, tokens) == (7, 448)
            assert mgr._pending_dataloader_state == {"cursor": 100 + r, "prefetch_depth": 1 + r}
            assert _draw() == expected[r]
            assert gathers == [((r, DP2_TP2[r]), "GLOO_PG")]

    @pytest.mark.parametrize(
        ("saved", "current"),
        [
            (DP4, DP2_TP2),  # dp_size changes on every rank
            (DP2_TP2, PP2_DP2),  # ranks 0 and 3 keep their own coordinates
        ],
        ids=["dp4_to_dp2tp2", "dp2tp2_to_pp2dp2"],
    )
    def test_layout_change_refused_on_every_rank(self, tmp_path, monkeypatch, saved, current):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=saved)
        reasons = []
        for r in range(WORLD):
            with pytest.raises(ValueError, match="data-parallel layout") as exc:
                _load_in_layout(tmp_path, monkeypatch, rank=r, step=7, layout=current)
            reasons.append(str(exc.value))
        assert len(set(reasons)) == 1, f"ranks disagree: {reasons}"
        assert "ignore_rank_state_mismatch" in reasons[0]

    def test_layout_change_override_discards_on_every_rank(self, tmp_path, monkeypatch):
        payloads, _ = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=DP4)
        _seed_all(555)
        fresh = _draw()
        for r in range(WORLD):
            mgr, (step, tokens, _), _ = _load_in_layout(
                tmp_path,
                monkeypatch,
                rank=r,
                step=7,
                layout=DP2_TP2,
                ignore_rank_state_mismatch=True,
            )
            assert (step, tokens) == (7, 448)
            assert mgr._pending_dataloader_state is None, f"rank {r} restored a loader cursor"
            assert _draw() == fresh, f"rank {r} restored RNG across a layout change"

    def test_unrecorded_saved_layout_checks_world_size_only(self, tmp_path, monkeypatch):
        """A manager built without coordinates records none; the load cannot
        compare layouts and validates only the world size."""
        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=None)
        for r in range(WORLD):
            mgr, (step, _, _), _ = _load_in_layout(
                tmp_path, monkeypatch, rank=r, step=7, layout=DP2_TP2
            )
            assert step == 7
            assert mgr._pending_dataloader_state == {"cursor": 100 + r, "prefetch_depth": 1 + r}
            assert _draw() == expected[r]

    def test_unknown_current_layout_checks_world_size_only(self, tmp_path, monkeypatch):
        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=DP4)
        for r in range(WORLD):
            mgr, (step, _, _), gathers = _load_in_layout(
                tmp_path, monkeypatch, rank=r, step=7, layout=None
            )
            assert step == 7
            assert mgr._pending_dataloader_state == {"cursor": 100 + r, "prefetch_depth": 1 + r}
            assert _draw() == expected[r]
            assert gathers == [((r, None), "GLOO_PG")]

    @pytest.mark.parametrize(
        "saved_layout",
        [
            "absent",  # key missing from the envelope
            [(0, 4), (1, 4), (2, 4), (3, 4)],  # not a mapping
            {0: (0, 4), 1: (1, 4), 2: (2, 4)},  # rank 3 missing
            {0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (4, 4)},  # dp_rank out of range
            {0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (3, 0)},  # dp_size not positive
            {0: (0, 4), 1: (1, 4), 2: (2, 4), 3: ("3", 4)},  # not an int
            {0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (3, 4, 1)},  # not a pair
            {0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (3, 4), 4: (0, 4), "x": (0, 4)},  # extra keys
        ],
        ids=["absent", "list", "incomplete", "rank_range", "size", "str", "triple", "extra"],
    )
    @pytest.mark.parametrize("current", [DP4, None], ids=["current_dp4", "current_unknown"])
    def test_malformed_saved_layout_refused_on_every_rank(self, saved_layout, current):
        payloads, _ = _mk_payloads(seed_base=10, cursor_base=100)
        env = {"version": 1, "world_size": WORLD, "ranks": payloads}
        if saved_layout != "absent":
            env["data_parallel"] = saved_layout
        reasons = []
        for r in range(WORLD):
            with pytest.raises(ValueError, match="data-parallel layout") as exc:
                select_rank_state(env, rank=r, world_size=WORLD, data_parallel=current)
            reasons.append(str(exc.value))
        assert len(set(reasons)) == 1

    @pytest.mark.parametrize(
        "dp_info",
        [(4, 4), (-1, 2), (0, 0), ("0", 2), (0,), (0, 2, 1), (True, 2), 3],
        ids=["rank_eq_size", "negative", "size0", "str", "short", "long", "bool", "scalar"],
    )
    def test_constructor_rejects_invalid_coordinates(self, tmp_path, dp_info):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="dp_info"):
            CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt, dp_info=dp_info)

    def test_constructor_accepts_a_list_pair(self, tmp_path):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt, dp_info=[1, 2])
        assert mgr._dp_info == (1, 2)


# ---------------------------------------------------------------------------
# Loads that are not an exact resume keep the single-copy behavior
# ---------------------------------------------------------------------------


class TestLoadsThatAreNotAResume:
    """Only an exact resume restores per-rank state. Any other load restores
    the single-copy keys (rank 0's) on every rank, with no topology check and
    no extra collective, exactly as before the envelope existed."""

    @pytest.mark.parametrize(
        "world", [2, WORLD], ids=["world_size_change", "same_size_layout_change"]
    )
    def test_eval_style_load(self, tmp_path, monkeypatch, world):
        """Weights-only load by a manager built without dp_info and called
        without a resume context, as scripts/eval.py does. It cannot see the
        new run's layout, so a same-size layout change is just the same size."""
        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=DP4)
        ckpt_dir = tmp_path / "step_7"
        gathers: list = []
        for r in range(world):
            with monkeypatch.context() as m:
                _install_fake_load_world(m, r, None, ckpt_dir, gathers, world=world)
                mgr = _mock_mgr(tmp_path, m)
                _seed_all(555)
                step, tokens, _ = mgr.load(path=str(ckpt_dir), exclude_keys=["optimizer"])
            assert (step, tokens) == (7, 448)
            assert mgr._pending_dataloader_state == {"cursor": 100, "prefetch_depth": 1}
            assert _draw() == expected[0], f"rank {r} did not restore rank 0's copy"
        assert gathers == []

    @pytest.mark.parametrize(
        ("world", "layout"),
        [(2, {0: (0, 2), 1: (1, 2)}), (WORLD, DP2_TP2)],
        ids=["world_size_change", "same_size_layout_change"],
    )
    def test_load_path_warm_start(self, tmp_path, monkeypatch, world, layout):
        """Weights-only warm start through the training entry, by a manager that
        records its layout as build_checkpoint_manager does."""
        from kempnerforge.config.job import JobConfig
        from kempnerforge.training.entry import restore_checkpoint

        payloads, expected = _mk_payloads(seed_base=1000, cursor_base=100)
        _save_four_rank_ckpt(tmp_path, monkeypatch, payloads, step=7, layout=DP4)
        ckpt_dir = tmp_path / "step_7"
        gathers: list = []
        for r in range(world):
            config = JobConfig(
                checkpoint=CheckpointConfig(
                    dir=str(tmp_path / "new_run"),
                    load_path=str(ckpt_dir),
                    exclude_from_loading=["optimizer"],
                )
            )
            with monkeypatch.context() as m:
                _install_fake_load_world(m, r, layout, ckpt_dir, gathers, world=world)
                m.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())
                model = torch.nn.Linear(4, 4)
                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                mgr = CheckpointManager(config.checkpoint, model, opt, dp_info=layout[r])
                _seed_all(555)
                assert restore_checkpoint(config, model, None, mgr) == (7, 448)
            assert mgr._pending_dataloader_state == {"cursor": 100, "prefetch_depth": 1}
            assert _draw() == expected[0], f"rank {r} did not restore rank 0's copy"
        assert gathers == []


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
        step, tokens, _ = mgr2.load(exact_resume=True)
        assert (step, tokens) == (10, 640)
        assert mgr2._pending_dataloader_state == {"cursor": 10}
        assert _draw() == expected
