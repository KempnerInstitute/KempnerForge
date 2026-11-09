"""Every rank takes the same branch where the training loop's ranks could split.

A shutdown or a failed NCCL check seen on one rank stops every rank (#307), a rank
that the vote stops arms its own forced-exit timer, the loop destroys only the
control group it made, and a NaN loss on one rank skips the step on every rank
(#375).

Two gloo ranks run the real ``run_training_loop`` and only one of them sees the
event. ``NaNDetector.check_loss`` makes its flag on CUDA, so each rank swaps the
health module's ``torch`` for one that makes it on the CPU, as the
``test_resilience`` reduction tests do.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import socket
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2
PG_TIMEOUT_S = 30
DEADLINE_S = 90


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CpuTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def tensor(data, device=None, **kwargs):
        return torch.tensor(data, **kwargs)


def _init(rank: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=WORLD,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=PG_TIMEOUT_S),
    )


def _check_any_rank(rank: int, port: int, flags: tuple[bool, ...]) -> None:
    from kempnerforge.distributed.utils import any_rank, new_control_group

    _init(rank, port)
    try:
        group = new_control_group(PG_TIMEOUT_S)
        assert any_rank(flags[rank], group) is any(flags)
    finally:
        dist.destroy_process_group()


def _signal_on(signaled_step: int):
    """A text step body that sends its process SIGUSR1 in step ``signaled_step``."""
    from kempnerforge.training.loop import text_step

    def step_fn(session, step):
        result = text_step(session, step)
        if step == signaled_step:
            os.kill(os.getpid(), signal.SIGUSR1)
        return result

    return step_fn


def _group_lifetime(loop, session) -> str:
    """Run the loop with no control group, then with the caller's own, and report on both."""
    from kempnerforge.distributed.utils import any_rank, new_control_group

    groups = dist.distributed_c10d._world.pg_map
    before = len(groups)
    loop.run_training_loop(session)
    made = f"left={len(groups) - before} cleared={session.control_group is None}"
    mine = new_control_group(PG_TIMEOUT_S)
    session.control_group = mine
    loop.run_training_loop(session)
    return f"{made} kept={session.control_group is mine} works={any_rank(True, mine)}"


def _run_loop(rank: int, port: int, case: str, root: str, out_dir: str) -> None:
    """Run the loop on one rank and write how it ended, then wait for the other rank."""
    _init(rank, port)
    try:
        import kempnerforge.resilience.health as health
        import kempnerforge.training.loop as loop
        from kempnerforge.checkpoint.manager import CheckpointManager
        from tests.unit.test_train_entry import (
            FakeCheckpointManager,
            FakeShutdownHandler,
            _nan_on,
            make_config,
            make_session,
        )

        health.torch = _CpuTorch()
        config = make_config(max_steps=6)
        config.checkpoint.dir = root
        config.checkpoint.interval = 100
        signaled = rank == 1 and case in ("shutdown", "shutdown-dcp")
        overrides: dict = {"shutdown_handler": FakeShutdownHandler(2 if signaled else None)}
        if case == "nan" and rank == 0:
            overrides["step_fn"] = _nan_on({1})
        if case == "timer":
            from kempnerforge.resilience.signal_handler import ShutdownHandler

            overrides["shutdown_handler"] = ShutdownHandler(timeout_sec=300)
            overrides["shutdown_handler"].register()
            if rank == 1:
                overrides["step_fn"] = _signal_on(1)
        session = make_session(config, **overrides)
        session.runtime = dataclasses.replace(
            session.runtime, rank=rank, local_rank=rank, world_size=WORLD
        )
        if case == "shutdown-dcp":
            session.checkpointer = CheckpointManager(
                config.checkpoint, session.model, session.optimizer
            )
        if case == "health":
            config.train.nccl_health_check_interval = 2
            loop.check_nccl_health = lambda: rank != 1
        if case == "groups":
            outcome = _group_lifetime(loop, session)
        else:
            step, _ = loop.run_training_loop(session)
            ckpt = session.checkpointer
            saves = ckpt.saved_steps if isinstance(ckpt, FakeCheckpointManager) else None
            outcome = f"step={step} saves={saves} nans={session.nan_detector.state.total_nans}"
        if case == "timer":
            outcome += f" armed={session.shutdown_handler._timer is not None}"
            session.shutdown_handler.finish()
    except BaseException as e:
        outcome = f"{type(e).__name__}: {e}"
    Path(out_dir, f"rank{rank}").write_text(outcome)
    deadline = time.monotonic() + DEADLINE_S
    while len(list(Path(out_dir).iterdir())) < WORLD and time.monotonic() < deadline:
        time.sleep(0.2)
    dist.destroy_process_group()


def _outcomes(tmp_path: Path, case: str) -> dict[int, str]:
    """How the loop ended on each rank; fails the test when a rank is still running."""
    out_dir = tmp_path / "outcomes"
    out_dir.mkdir()
    context = mp.spawn(
        _run_loop,
        args=(_free_port(), case, str(tmp_path / "ckpt"), str(out_dir)),
        nprocs=WORLD,
        join=False,
    )
    deadline = time.monotonic() + DEADLINE_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            done = {int(p.name[4:]): p.read_text() for p in out_dir.iterdir()}
            pytest.fail(f"a rank is still running after {DEADLINE_S}s; finished: {done}")
    return {int(p.name[4:]): p.read_text() for p in out_dir.iterdir()}


class TestAnyRank:
    def test_without_a_group_it_is_the_local_flag(self):
        from kempnerforge.distributed.utils import any_rank

        assert any_rank(True, None) is True
        assert any_rank(False, None) is False

    def test_no_control_group_without_a_process_group(self):
        from kempnerforge.distributed.utils import new_control_group

        assert new_control_group(PG_TIMEOUT_S) is None

    @pytest.mark.parametrize("flags", [(False, False), (False, True), (True, True)], ids=str)
    def test_every_rank_gets_the_same_answer(self, flags):
        mp.spawn(_check_any_rank, args=(_free_port(), flags), nprocs=WORLD)


class TestLoopStopsOnEveryRank:
    def test_a_shutdown_seen_on_one_rank_stops_every_rank_at_that_step(self, tmp_path):
        outcomes = _outcomes(tmp_path, "shutdown")
        assert outcomes == {r: "step=2 saves=[2] nans=0" for r in range(WORLD)}

    def test_a_rank_the_vote_stops_arms_its_forced_exit_timer(self, tmp_path):
        """Only rank 1 gets SIGUSR1; rank 0 stops on the vote and is bounded by its own timer."""
        outcomes = _outcomes(tmp_path, "timer")
        assert outcomes == {r: "step=2 saves=[2] nans=0 armed=True" for r in range(WORLD)}

    def test_every_rank_writes_the_emergency_checkpoint(self, tmp_path):
        """The real manager: one DCP save over both ranks, committed as ``latest``."""
        outcomes = _outcomes(tmp_path, "shutdown-dcp")
        assert outcomes == {r: "step=2 saves=None nans=0" for r in range(WORLD)}
        ckpt = tmp_path / "ckpt"
        assert (ckpt / "latest").resolve() == (ckpt / "step_2").resolve()
        assert (ckpt / "step_2" / ".metadata").is_file()

    def test_an_unhealthy_rank_stops_and_fails_every_rank(self, tmp_path):
        """Every rank leaves the loop at the same step, saves nothing and raises."""
        outcomes = _outcomes(tmp_path, "health")
        failed = (
            "RuntimeError: NCCL health check failed at step 2 on at least one rank; "
            "every rank stopped without a final save"
        )
        assert outcomes == {r: failed for r in range(WORLD)}

    def test_the_loop_destroys_only_the_control_group_it_made(self, tmp_path):
        """A loop-made group is gone after the loop; a group the caller passed in still works."""
        outcomes = _outcomes(tmp_path, "groups")
        assert outcomes == {r: "left=0 cleared=True kept=True works=True" for r in range(WORLD)}

    def test_a_nan_on_one_rank_skips_the_step_on_every_rank(self, tmp_path):
        """Rank 0's NaN loss makes both ranks skip step 1, and both still finish (#375)."""
        outcomes = _outcomes(tmp_path, "nan")
        assert outcomes == {r: "step=6 saves=[6] nans=1" for r in range(WORLD)}
