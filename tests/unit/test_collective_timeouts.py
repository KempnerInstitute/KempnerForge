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

# Hold the real ``torch.ones`` before any test monkey-patches it, so the
# replacement wrapper can still build a tensor without recursion.
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
            tensor.fill_(2.0)  # world_size=2, simulates completed all-reduce
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


class TestCheckpointBroadcastTimeout:
    def test_broadcast_object_list_honors_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CheckpointManager.load raises with a clear message if the object
        broadcast does not complete within _TRAIN_STATE_BROADCAST_TIMEOUT_SEC."""
        from unittest.mock import MagicMock

        import kempnerforge.checkpoint.manager as mgr_mod
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.schema import CheckpointConfig

        # Build a step_1 dir with a benign train_state.pt so the load reaches
        # the broadcast path.
        tmp = pytest.importorskip("pathlib").Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = tmp(td)
            ckpt_dir = root / "step_1"
            ckpt_dir.mkdir()
            torch.save({"step": 1, "tokens_seen": 0, "rng": {}}, ckpt_dir / "train_state.pt")
            (root / "latest").symlink_to("step_1")

            config = CheckpointConfig(dir=str(root))
            model = MagicMock()
            model.state_dict.return_value = {}
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {}
            manager = CheckpointManager(config=config, model=model, optimizer=optimizer)

            # Force the "distributed" branches.
            monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
            monkeypatch.setattr(manager, "_rank", 0, raising=False)

            # First call: the existence-flag broadcast — let it succeed.
            # Second call: the object-list broadcast — simulate a hang by raising.
            call_count = {"n": 0}

            def fake_broadcast_object_list(obj_list, src=0, *, async_op: bool = False, **kwargs):  # noqa: ANN001, ARG001
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # existence probe — not async
                    return None
                assert async_op is True, (
                    "object_list broadcast on load must be async to enforce the timeout"
                )
                return _FakeWork(raises=RuntimeError("Wait timeout"))

            monkeypatch.setattr(mgr_mod.dist, "broadcast_object_list", fake_broadcast_object_list)

            with pytest.raises(RuntimeError, match="train_state broadcast timed out"):
                manager.load(exclude_keys=["model", "optimizer"])
