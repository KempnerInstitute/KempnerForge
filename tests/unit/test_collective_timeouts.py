"""Tests for per-operation timeouts on distributed collectives.

The process-group default timeout (``nccl_timeout_sec``, 1800s) is sized
for training reduces on large tensors. Several init-path and coordination
collectives should fail faster than that so misconfiguration or a wedged
peer surfaces quickly. These tests verify that the per-op timeout actually
flows through to ``Work.wait`` rather than being silently ignored.
"""

from __future__ import annotations

import time
from datetime import timedelta

import pytest
import torch

_REAL_TORCH_ONES = torch.ones


def _cpu_ones(*args, **kwargs) -> torch.Tensor:
    """torch.ones with any ``device`` kwarg stripped.

    Lets tests on CPU-only hosts monkey-patch ``torch.ones`` so the CUDA
    branch inside ``check_nccl_health`` doesn't require a GPU. Uses the
    pre-patch ``torch.ones`` captured at module load time.
    """
    kwargs.pop("device", None)
    return _REAL_TORCH_ONES(*args, **kwargs)


class _FakeWork:
    """Stand-in for ``torch.distributed.Work`` that records wait() calls.

    ``raises``: raise RuntimeError on wait() (simulates timeout enforcement).
    ``returns``: return this value from wait() (False simulates legacy-
    backend timeout; True simulates success).
    ``sleep``: seconds to sleep before returning (simulates blocking wait).
    """

    def __init__(
        self,
        raises: BaseException | None = None,
        returns: bool = True,
        sleep: float = 0.0,
    ) -> None:
        self._raises = raises
        self._returns = returns
        self._sleep = sleep
        self.wait_calls: list[object] = []

    def wait(self, timeout=None):  # noqa: ANN001
        self.wait_calls.append(timeout)
        if self._raises is not None:
            raise self._raises
        if self._sleep > 0:
            time.sleep(self._sleep)
        return self._returns


class TestCheckNcclHealthTimeout:
    def test_passes_timedelta_to_work_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_nccl_health must forward timeout_sec as a timedelta to Work.wait.

        The buggy version called dist.all_reduce synchronously and ignored
        timeout_sec entirely. The fix uses async_op=True and wait(timeout=...).
        """
        from kempnerforge.resilience import health

        monkeypatch.setattr(health.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(health.dist, "get_world_size", lambda: 2)

        fake_work = _FakeWork(returns=True)

        def fake_all_reduce(tensor, *args, async_op: bool = False, **kwargs):  # noqa: ANN001
            assert async_op is True, (
                "check_nccl_health called dist.all_reduce synchronously — "
                "the caller-supplied timeout cannot be enforced"
            )
            tensor.fill_(2.0)
            return fake_work

        monkeypatch.setattr(health.dist, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(health.torch, "ones", lambda *a, **kw: _cpu_ones(*a, **kw))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        ok = health.check_nccl_health(timeout_sec=0.5)
        assert ok is True
        assert fake_work.wait_calls, "Work.wait was never called"
        forwarded = fake_work.wait_calls[0]
        assert isinstance(forwarded, timedelta), (
            f"Work.wait must receive a timedelta, got {type(forwarded).__name__}"
        )
        assert forwarded.total_seconds() == pytest.approx(0.5, rel=0.01)

    def test_returns_false_on_timeout_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Modern PyTorch: Work.wait raises RuntimeError on timeout — catch, return False."""
        from kempnerforge.resilience import health

        monkeypatch.setattr(health.dist, "is_initialized", lambda: True)
        fake_work = _FakeWork(raises=RuntimeError("Wait timeout: NCCL watchdog"))

        def fake_all_reduce(tensor, *args, async_op: bool = False, **kwargs):  # noqa: ANN001, ARG001
            return fake_work

        monkeypatch.setattr(health.dist, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(health.torch, "ones", lambda *a, **kw: _cpu_ones(*a, **kw))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        t0 = time.perf_counter()
        ok = health.check_nccl_health(timeout_sec=0.1)
        elapsed = time.perf_counter() - t0

        assert ok is False, "timeout must produce False, not True"
        assert elapsed < 1.0, (
            f"timeout was not honored: took {elapsed:.2f}s (likely synchronous all_reduce)"
        )

    def test_returns_false_on_wait_returning_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy backends: Work.wait returns False on timeout instead of raising."""
        from kempnerforge.resilience import health

        monkeypatch.setattr(health.dist, "is_initialized", lambda: True)
        fake_work = _FakeWork(returns=False)

        def fake_all_reduce(tensor, *args, async_op: bool = False, **kwargs):  # noqa: ANN001, ARG001
            return fake_work

        monkeypatch.setattr(health.dist, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(health.torch, "ones", lambda *a, **kw: _cpu_ones(*a, **kw))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        ok = health.check_nccl_health(timeout_sec=0.1)
        assert ok is False

    def test_returns_true_when_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No distributed → nothing to check → return True without touching work."""
        from kempnerforge.resilience import health

        monkeypatch.setattr(health.dist, "is_initialized", lambda: False)
        assert health.check_nccl_health(timeout_sec=0.1) is True


class TestBarrierWithTimeout:
    def test_raises_runtime_error_with_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_barrier_with_timeout surfaces a timeout with the reason string attached."""
        from kempnerforge.distributed import setup as dsetup

        monkeypatch.setattr(
            dsetup.dist,
            "barrier",
            lambda async_op=False: _FakeWork(  # noqa: ARG005
                raises=RuntimeError("Wait timeout: NCCL watchdog")
            ),
        )

        with pytest.raises(RuntimeError, match="DeviceMesh construction"):
            dsetup._barrier_with_timeout(5.0, reason="DeviceMesh construction")

    def test_raises_when_wait_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy-backend False return also translates to RuntimeError."""
        from kempnerforge.distributed import setup as dsetup

        monkeypatch.setattr(
            dsetup.dist,
            "barrier",
            lambda async_op=False: _FakeWork(returns=False),  # noqa: ARG005
        )

        with pytest.raises(RuntimeError, match="timed out after 5.0s"):
            dsetup._barrier_with_timeout(5.0, reason="init-check")

    def test_returns_normally_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: Work.wait returns True → no exception."""
        from kempnerforge.distributed import setup as dsetup

        fake_work = _FakeWork(returns=True)
        monkeypatch.setattr(
            dsetup.dist,
            "barrier",
            lambda async_op=False: fake_work,  # noqa: ARG005
        )

        dsetup._barrier_with_timeout(5.0, reason="smoke")
        assert fake_work.wait_calls
        assert isinstance(fake_work.wait_calls[0], timedelta)


class TestNcclAsyncErrorHandlingEnvGuard:
    def test_defaults_to_1_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_set_nccl_env forces TORCH_NCCL_ASYNC_ERROR_HANDLING=1 when unset."""
        from kempnerforge.distributed import setup as dsetup

        monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
        monkeypatch.setattr(dsetup, "_detect_ib_interface", lambda: None)
        dsetup._set_nccl_env()
        assert dsetup.os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "1"

    def test_preserves_explicit_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kempnerforge.distributed import setup as dsetup

        monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        monkeypatch.setattr(dsetup, "_detect_ib_interface", lambda: None)
        dsetup._set_nccl_env()
        assert dsetup.os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "1"

    def test_warns_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit "0" is user intent; do not override silently — warn loudly.

        Uses a direct logger handler rather than pytest's caplog so the test
        is robust to other tests mutating logger propagation.
        """
        import logging

        from kempnerforge.distributed import setup as dsetup

        monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
        monkeypatch.setattr(dsetup, "_detect_ib_interface", lambda: None)

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        log = logging.getLogger("kempnerforge.distributed.setup")
        prior_level = log.level
        log.setLevel(logging.WARNING)
        log.addHandler(handler)
        try:
            dsetup._set_nccl_env()
        finally:
            log.removeHandler(handler)
            log.setLevel(prior_level)

        assert dsetup.os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "0"
        assert any("TORCH_NCCL_ASYNC_ERROR_HANDLING=0" in r.getMessage() for r in records), (
            "expected warning when TORCH_NCCL_ASYNC_ERROR_HANDLING=0"
        )


_MIXED = "cpu:gloo,cuda:nccl"


class _FakeMesh:
    """The parts of a DeviceMesh ``init_distributed`` reads after building it."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.mesh_dim_names = names

    def __getitem__(self, name: str) -> _FakeMesh:
        return self

    def get_local_rank(self) -> int:
        return 0


class TestGroupTimeouts:
    def test_the_timeout_options_carry_the_timeout(self) -> None:
        """NCCL takes only its own options, set as torch sets them by default; Gloo ignores them."""
        import torch.distributed as dist

        from kempnerforge.distributed.setup import _timeout_options

        options = _timeout_options(timedelta(seconds=123))
        assert type(options) is dist.ProcessGroupNCCL.Options
        assert options._timeout == timedelta(seconds=123)
        assert options.is_high_priority_stream is False

    @pytest.mark.parametrize(
        ("dims", "sizes", "backend", "cuda", "given"),
        [
            ({"dp_shard": 2, "tp": 2}, {"dp_shard": 2, "tp": 2}, _MIXED, True, ("dp_shard", "tp")),
            ({"pp": 2, "dp_shard": 2}, {"pp": 2, "dp_shard": 2}, _MIXED, True, ("pp", "dp_shard")),
            ({"dp_shard": 2, "ep": 2}, {"dp_shard": 2, "ep": 2}, _MIXED, True, ("dp_shard", "ep")),
            (
                {"dp_replicate": 2, "dp_shard": 2},
                {"dp_replicate": 2, "dp_shard": 2},
                _MIXED,
                True,
                ("dp_replicate", "dp_shard"),
            ),
            ({"tp": 4}, {"dp_shard": 1, "tp": 4}, _MIXED, True, ("dp_shard",)),
            ({}, {"dp_shard": 4}, _MIXED, True, ()),
            ({}, {"dp_shard": 4}, "nccl", True, ()),
            ({"dp_shard": 2, "tp": 2}, {"dp_shard": 2, "tp": 2}, "nccl", True, ("dp_shard", "tp")),
            ({}, {"dp_shard": 4}, "gloo", True, ("dp_shard",)),
            ({"tp": 4}, {"dp_shard": 1, "tp": 4}, "gloo", True, ("dp_shard", "tp")),
            ({}, {"dp_shard": 4}, "gloo", False, ()),
        ],
    )
    def test_every_group_init_device_mesh_creates_gets_nccl_timeout_sec(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dims: dict[str, int],
        sizes: dict[str, int],
        backend: str,
        cuda: bool,
        given: tuple[str, ...],
    ) -> None:
        """init_device_mesh creates a group for each dimension smaller than the world, and for
        one as wide as the world only with CUDA and Gloo alone; it reuses the default group
        otherwise, so that dimension gets no options."""
        from kempnerforge.config.schema import DistributedConfig
        from kempnerforge.distributed import setup

        built: list[tuple[tuple[int, ...], tuple[str, ...], dict | None]] = []

        def fake_init_device_mesh(device_type, mesh_shape, mesh_dim_names, backend_override):
            built.append((mesh_shape, mesh_dim_names, backend_override))
            return _FakeMesh(mesh_dim_names)

        monkeypatch.setattr(setup, "get_world_info", lambda: (0, 0, 4))
        monkeypatch.setattr(setup, "_set_nccl_env", lambda: None)
        monkeypatch.setattr(setup.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(setup.dist, "get_backend", lambda group=None: backend)
        monkeypatch.setattr(setup.torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(setup.torch.cuda, "set_device", lambda device: None)
        monkeypatch.setattr(setup, "init_device_mesh", fake_init_device_mesh)
        monkeypatch.setattr(setup, "_barrier_with_timeout", lambda seconds, reason: None)
        monkeypatch.setattr(setup, "_set_seed", lambda seed, rank, pp_rank=0: None)
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", "29500")

        setup.init_distributed(DistributedConfig(nccl_timeout_sec=123, backend=backend, **dims))
        [(shape, names, overrides)] = built
        assert dict(zip(names, shape, strict=True)) == sizes
        if not given:
            assert overrides is None
            return
        assert overrides is not None and tuple(overrides) == given
        for named, options in overrides.values():
            assert named == backend
            assert options._timeout == timedelta(seconds=123)

    def test_the_checkpoint_sync_group_gets_the_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import torch.distributed as dist

        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.schema import CheckpointConfig

        created: list[dict] = []
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
        monkeypatch.setattr(dist, "new_group", lambda **kwargs: created.append(kwargs) or object())
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = CheckpointConfig(dir=str(tmp_path))
        CheckpointManager(config, model, optimizer, group_timeout_sec=123)
        CheckpointManager(config, model, optimizer)
        assert created == [
            {"backend": "gloo", "timeout": timedelta(seconds=123)},
            {"backend": "gloo"},
        ]

    def test_the_pipeline_stage_groups_get_the_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        import torch.distributed as dist

        from kempnerforge.distributed.pipeline_parallel import build_stage_group

        calls: list[tuple[list[list[int]], timedelta | None]] = []

        def subgroups(ranks: list[list[int]], timeout: timedelta | None = None):
            calls.append((ranks, timeout))
            return "stage group", []

        monkeypatch.setattr(dist, "new_subgroups_by_enumeration", subgroups)
        mesh = SimpleNamespace(mesh_dim_names=("pp", "dp_shard"), mesh=torch.arange(4).view(2, 2))
        assert build_stage_group(mesh, 123) == "stage group"
        assert build_stage_group(mesh) == "stage group"
        assert calls == [
            ([[0, 1], [2, 3]], timedelta(seconds=123)),
            ([[0, 1], [2, 3]], None),
        ]

    def test_training_passes_nccl_timeout_sec_to_the_checkpoint_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from kempnerforge.config.schema import DistributedConfig, JobConfig
        from kempnerforge.distributed import pipeline_parallel
        from kempnerforge.training import entry

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            entry, "CheckpointManager", lambda *args, **kwargs: seen.update(kwargs) or "manager"
        )
        monkeypatch.setattr(entry, "get_dp_info", lambda mesh: None)
        monkeypatch.setattr(
            pipeline_parallel,
            "build_stage_group",
            lambda mesh, timeout_sec=None: seen.update(stage_timeout_sec=timeout_sec) or "pg",
        )
        config = JobConfig(distributed=DistributedConfig(nccl_timeout_sec=123))
        runtime = SimpleNamespace(device_mesh=object())
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        pipeline = SimpleNamespace(rank=1)
        assert (
            entry.build_checkpoint_manager(config, runtime, model, optimizer, pipeline) == "manager"
        )
        assert seen["group_timeout_sec"] == 123
        assert seen["stage_timeout_sec"] == 123
        assert seen["process_group"] == "pg"
