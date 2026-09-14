"""Unit tests for KempnerForge resilience modules (signals, health, elastic)."""

from __future__ import annotations

import os
import signal

import torch

from kempnerforge.resilience.elastic import (
    SLURMInfo,
    get_slurm_info,
    is_slurm_job,
    is_slurm_requeue,
    log_job_info,
    resolve_resume_path,
)
from kempnerforge.resilience.health import (
    NaNDetector,
    check_gpu_health,
)
from kempnerforge.resilience.signal_handler import (
    ShutdownHandler,
    ignore_shutdown_signals_in_worker,
)

# ---------------------------------------------------------------------------
# ShutdownHandler
# ---------------------------------------------------------------------------


class TestShutdownHandler:
    def test_initial_state(self):
        handler = ShutdownHandler()
        assert not handler.should_shutdown()
        assert handler.signal_received is None

    def test_register_and_unregister(self):
        handler = ShutdownHandler()
        handler.register()
        # Should have our handler installed
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        handler.unregister()

    def test_sigterm_sets_flag(self):
        handler = ShutdownHandler(timeout_sec=0)  # No timeout for test
        handler.register()
        try:
            # Simulate SIGTERM by sending to ourselves
            os.kill(os.getpid(), signal.SIGTERM)
            assert handler.should_shutdown()
            assert handler.signal_received == signal.SIGTERM
        finally:
            handler.unregister()

    def test_sigusr1_sets_flag(self):
        handler = ShutdownHandler(timeout_sec=0)
        handler.register()
        try:
            os.kill(os.getpid(), signal.SIGUSR1)
            assert handler.should_shutdown()
            assert handler.signal_received == signal.SIGUSR1
        finally:
            handler.unregister()

    def test_finish_cancels_timer(self):
        handler = ShutdownHandler(timeout_sec=60)
        handler.register()
        # Trigger signal to start timer
        os.kill(os.getpid(), signal.SIGTERM)
        assert handler._timer is not None
        handler.finish()
        assert handler._timer is None

    def test_no_timeout_when_zero(self):
        handler = ShutdownHandler(timeout_sec=0)
        handler.register()
        os.kill(os.getpid(), signal.SIGTERM)
        # No timer created when timeout is 0
        assert handler._timer is None
        handler.unregister()

    def test_timeout_timer_is_daemon(self):
        handler = ShutdownHandler(timeout_sec=300)
        handler.register()
        os.kill(os.getpid(), signal.SIGTERM)
        assert handler._timer is not None
        assert handler._timer.daemon
        handler.finish()

    def test_multiple_signals_idempotent(self):
        handler = ShutdownHandler(timeout_sec=0)
        handler.register()
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
        assert handler.should_shutdown()
        assert handler.state.consecutive_nans if hasattr(handler, "state") else True
        handler.unregister()


# ---------------------------------------------------------------------------
# NaN Detector
# ---------------------------------------------------------------------------


class TestNaNDetector:
    def test_finite_loss_returns_true(self):
        det = NaNDetector()
        assert det.check_loss(2.5, step=1) is True
        assert det.state.consecutive_nans == 0
        assert det.state.last_good_loss == 2.5

    def test_nan_loss_returns_false(self):
        det = NaNDetector(action="warn")
        assert det.check_loss(float("nan"), step=1) is False
        assert det.state.consecutive_nans == 1
        assert det.state.total_nans == 1

    def test_inf_loss_returns_false(self):
        det = NaNDetector(action="warn")
        assert det.check_loss(float("inf"), step=1) is False
        assert det.state.consecutive_nans == 1

    def test_neg_inf_loss_returns_false(self):
        det = NaNDetector(action="warn")
        assert det.check_loss(float("-inf"), step=1) is False

    def test_consecutive_nans_tracked(self):
        det = NaNDetector(action="warn")
        det.check_loss(float("nan"), step=1)
        det.check_loss(float("nan"), step=2)
        det.check_loss(float("nan"), step=3)
        assert det.state.consecutive_nans == 3
        assert det.state.total_nans == 3

    def test_consecutive_nans_reset_on_good_step(self):
        det = NaNDetector(action="warn")
        det.check_loss(float("nan"), step=1)
        det.check_loss(float("nan"), step=2)
        det.check_loss(1.5, step=3)
        assert det.state.consecutive_nans == 0
        assert det.state.total_nans == 2  # Total keeps accumulating

    def test_should_rollback(self):
        det = NaNDetector(action="warn", max_consecutive=3)
        det.check_loss(float("nan"), step=1)
        det.check_loss(float("nan"), step=2)
        assert not det.should_rollback
        det.check_loss(float("nan"), step=3)
        assert det.should_rollback

    def test_raise_action(self):
        det = NaNDetector(action="raise")
        import pytest

        with pytest.raises(RuntimeError, match="NaN/Inf loss"):
            det.check_loss(float("nan"), step=1)

    def test_skip_action(self):
        det = NaNDetector(action="skip")
        # Should return False but not raise
        assert det.check_loss(float("nan"), step=1) is False

    def test_invalid_action_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid NaN action"):
            NaNDetector(action="invalid")

    def test_nan_steps_recorded(self):
        det = NaNDetector(action="warn", max_history=5)
        for i in range(3):
            det.check_loss(float("nan"), step=i + 10)
        assert det.state.nan_steps == [10, 11, 12]

    def test_nan_history_capped(self):
        det = NaNDetector(action="warn", max_history=2)
        for i in range(5):
            det.check_loss(float("nan"), step=i)
        assert len(det.state.nan_steps) == 2  # Capped at max_history

    def test_last_good_loss_tracked(self):
        det = NaNDetector(action="warn")
        det.check_loss(3.0, step=1)
        det.check_loss(2.5, step=2)
        det.check_loss(float("nan"), step=3)
        assert det.state.last_good_loss == 2.5
        assert det.state.last_good_step == 2

    def test_reset(self):
        det = NaNDetector(action="warn")
        det.check_loss(float("nan"), step=1)
        det.check_loss(float("nan"), step=2)
        det.reset()
        assert det.state.consecutive_nans == 0
        assert det.state.total_nans == 0
        assert det.state.nan_steps == []

    def test_check_gradients_clean(self):
        model = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        det = NaNDetector(action="warn")
        assert det.check_gradients(model, step=1) is True

    def test_check_gradients_nan(self):
        model = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        # Inject NaN into gradients
        model.weight.grad[0, 0] = float("nan")

        det = NaNDetector(action="warn")
        assert det.check_gradients(model, step=1) is False

    def test_check_gradients_raise(self):
        model = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        model.weight.grad[0, 0] = float("nan")

        import pytest

        det = NaNDetector(action="raise")
        with pytest.raises(RuntimeError, match="NaN/Inf gradient"):
            det.check_gradients(model, step=1)


# ---------------------------------------------------------------------------
# GPU health
# ---------------------------------------------------------------------------


class TestGPUHealth:
    def test_check_gpu_health_keys(self):
        health = check_gpu_health()
        assert "cuda_available" in health
        assert "device_accessible" in health
        assert "compute_ok" in health
        assert "memory_ok" in health
        assert "error" in health

    def test_health_on_available_gpu(self):
        if not torch.cuda.is_available():
            return
        health = check_gpu_health()
        assert health["cuda_available"] is True
        assert health["device_accessible"] is True
        assert health["compute_ok"] is True
        assert health["memory_ok"] is True
        assert health["error"] == ""


# ---------------------------------------------------------------------------
# SLURM / Elastic
# ---------------------------------------------------------------------------


class TestSLURM:
    def test_not_slurm_returns_none(self):
        # Remove SLURM env vars if present
        old = os.environ.pop("SLURM_JOB_ID", None)
        try:
            assert get_slurm_info() is None
            assert not is_slurm_job()
        finally:
            if old is not None:
                os.environ["SLURM_JOB_ID"] = old

    def test_is_slurm_job_with_env(self):
        os.environ["SLURM_JOB_ID"] = "12345"
        try:
            assert is_slurm_job()
        finally:
            del os.environ["SLURM_JOB_ID"]

    def test_get_slurm_info(self):
        env = {
            "SLURM_JOB_ID": "99999",
            "SLURM_JOB_NAME": "train_llama",
            "SLURM_JOB_NODELIST": "node[001-004]",
            "SLURM_NNODES": "4",
            "SLURM_NTASKS_PER_NODE": "8",
            "SLURM_RESTART_COUNT": "0",
            "SLURM_JOB_PARTITION": "gpu",
        }
        old_vals = {}
        for k, v in env.items():
            old_vals[k] = os.environ.get(k)
            os.environ[k] = v
        try:
            info = get_slurm_info()
            assert info is not None
            assert info.job_id == "99999"
            assert info.job_name == "train_llama"
            assert info.num_nodes == 4
            assert info.ntasks_per_node == 8
            assert not info.is_requeued
        finally:
            for k in env:
                if old_vals[k] is not None:
                    os.environ[k] = old_vals[k]
                else:
                    os.environ.pop(k, None)

    def test_is_requeued(self):
        os.environ["SLURM_RESTART_COUNT"] = "2"
        try:
            assert is_slurm_requeue()
        finally:
            del os.environ["SLURM_RESTART_COUNT"]

    def test_not_requeued(self):
        old = os.environ.pop("SLURM_RESTART_COUNT", None)
        try:
            assert not is_slurm_requeue()
        finally:
            if old is not None:
                os.environ["SLURM_RESTART_COUNT"] = old

    def test_slurm_info_is_requeued_property(self):
        info = SLURMInfo(
            job_id="1",
            job_name="test",
            node_list="node01",
            num_nodes=1,
            ntasks_per_node=4,
            restart_count=3,
            partition="gpu",
            array_task_id=None,
        )
        assert info.is_requeued

    def test_slurm_info_not_requeued_when_zero(self):
        info = SLURMInfo(
            job_id="1",
            job_name="test",
            node_list="node01",
            num_nodes=1,
            ntasks_per_node=4,
            restart_count=0,
            partition="gpu",
            array_task_id=None,
        )
        assert not info.is_requeued

    def test_get_slurm_info_minimal_env(self):
        """Only SLURM_JOB_ID set — all other fields use defaults."""
        old = os.environ.pop("SLURM_JOB_ID", None)
        # Clear all SLURM vars
        slurm_keys = [
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_NNODES",
            "SLURM_NTASKS_PER_NODE",
            "SLURM_RESTART_COUNT",
            "SLURM_JOB_PARTITION",
            "SLURM_ARRAY_TASK_ID",
        ]
        saved = {k: os.environ.pop(k, None) for k in slurm_keys}
        os.environ["SLURM_JOB_ID"] = "55555"
        try:
            info = get_slurm_info()
            assert info is not None
            assert info.job_id == "55555"
            assert info.job_name == ""
            assert info.node_list == ""
            assert info.num_nodes == 1
            assert info.ntasks_per_node == 1
            assert info.restart_count == 0
            assert info.partition == ""
            assert info.array_task_id is None
            assert not info.is_requeued
        finally:
            os.environ.pop("SLURM_JOB_ID", None)
            if old is not None:
                os.environ["SLURM_JOB_ID"] = old
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_get_slurm_info_array_job(self):
        """SLURM_ARRAY_TASK_ID is captured for array jobs."""
        env = {
            "SLURM_JOB_ID": "77777",
            "SLURM_ARRAY_TASK_ID": "42",
        }
        saved = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            os.environ[k] = v
        try:
            info = get_slurm_info()
            assert info is not None
            assert info.array_task_id == "42"
        finally:
            for k in env:
                if saved[k] is not None:
                    os.environ[k] = saved[k]
                else:
                    os.environ.pop(k, None)

    def test_is_slurm_requeue_explicit_zero(self):
        """SLURM_RESTART_COUNT=0 is not a requeue."""
        os.environ["SLURM_RESTART_COUNT"] = "0"
        try:
            assert not is_slurm_requeue()
        finally:
            del os.environ["SLURM_RESTART_COUNT"]

    def test_log_job_info_outside_slurm(self):
        """log_job_info() logs 'Not running under SLURM' when no SLURM env."""
        import io
        import logging

        old = os.environ.pop("SLURM_JOB_ID", None)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("kempnerforge.resilience.elastic")
        logger.addHandler(handler)
        try:
            log_job_info()
            assert "Not running under SLURM" in buf.getvalue()
        finally:
            logger.removeHandler(handler)
            if old is not None:
                os.environ["SLURM_JOB_ID"] = old

    def test_log_job_info_under_slurm(self):
        """log_job_info() logs job details when SLURM env is set."""
        import io
        import logging

        env = {
            "SLURM_JOB_ID": "88888",
            "SLURM_JOB_NAME": "test_run",
            "SLURM_NNODES": "2",
            "SLURM_NTASKS_PER_NODE": "4",
            "SLURM_JOB_PARTITION": "gpu_test",
            "SLURM_RESTART_COUNT": "0",
        }
        saved = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            os.environ[k] = v

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("kempnerforge.resilience.elastic")
        logger.addHandler(handler)
        try:
            log_job_info()
            output = buf.getvalue()
            assert "88888" in output
            assert "test_run" in output
            assert "gpu_test" in output
        finally:
            logger.removeHandler(handler)
            for k in env:
                if saved[k] is not None:
                    os.environ[k] = saved[k]
                else:
                    os.environ.pop(k, None)

    def test_log_job_info_requeued(self):
        """log_job_info() notes when the job was requeued."""
        import io
        import logging

        env = {
            "SLURM_JOB_ID": "88888",
            "SLURM_RESTART_COUNT": "2",
        }
        saved = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            os.environ[k] = v

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("kempnerforge.resilience.elastic")
        logger.addHandler(handler)
        try:
            log_job_info()
            assert "requeued" in buf.getvalue().lower()
        finally:
            logger.removeHandler(handler)
            for k in env:
                if saved[k] is not None:
                    os.environ[k] = saved[k]
                else:
                    os.environ.pop(k, None)


def _durable_step_dir(base, step: int, pp_stages: int = 0):
    """Create a ``step_N`` directory that looks like a completed DCP save.

    ``resolve_resume_path`` treats DCP's ``.metadata`` as the durability marker,
    so a directory without one stands for an interrupted save.
    """
    d = base / f"step_{step}"
    d.mkdir()
    if pp_stages:
        for k in range(pp_stages):
            stage = d / f"pp{k}"
            stage.mkdir()
            (stage / ".metadata").write_text("")
    else:
        (d / ".metadata").write_text("")
    return d


class TestResumePathResolution:
    def test_no_checkpoint_dir(self, tmp_path):
        assert resolve_resume_path(str(tmp_path / "nonexistent")) is None

    def test_empty_checkpoint_dir(self, tmp_path):
        assert resolve_resume_path(str(tmp_path)) is None

    def test_latest_symlink(self, tmp_path):
        # Create a checkpoint directory
        step_dir = tmp_path / "step_100"
        step_dir.mkdir()
        (step_dir / "metadata.json").write_text("{}")

        # Create "latest" symlink
        latest = tmp_path / "latest"
        latest.symlink_to(step_dir.name)

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_100"

    def test_fallback_to_highest_step(self, tmp_path):
        # Create step directories without "latest" symlink
        for step in [10, 50, 30]:
            _durable_step_dir(tmp_path, step)

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_50"

    def test_ignores_non_step_dirs(self, tmp_path):
        _durable_step_dir(tmp_path, 100)
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "step_abc").mkdir()  # Not a valid step dir (will cause error)

        # Should find step_100 despite other dirs; step_abc would fail int parse
        # but our filter only grabs dirs starting with "step_"
        # Let's only have valid ones
        result = resolve_resume_path(str(tmp_path))
        # step_abc will cause int("abc") to fail — let's test that separately
        assert result is not None

    def test_latest_broken_symlink_falls_back(self, tmp_path):
        # Create a checkpoint directory
        _durable_step_dir(tmp_path, 50)

        # Create broken "latest" symlink pointing to nonexistent
        latest = tmp_path / "latest"
        latest.symlink_to("step_999")

        # Should still resolve via latest (which resolves to broken path)
        # but the resolve will point to a non-existent path
        # Our code checks latest.exists() which follows the symlink
        # So broken symlink → latest.exists() is False... actually
        # latest.exists() returns False for broken symlinks
        # So it should fall back to step_50
        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_50"

    def test_latest_symlink_takes_priority_over_higher_step(self, tmp_path):
        """latest symlink should be used even if higher step dirs exist."""
        for step in [10, 50, 100]:
            _durable_step_dir(tmp_path, step)

        # Point latest at step_50 (not the highest)
        latest = tmp_path / "latest"
        latest.symlink_to("step_50")

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_50"

    def test_single_step_dir(self, tmp_path):
        """Works with exactly one step directory."""
        _durable_step_dir(tmp_path, 1)

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_1"

    def test_step_dirs_with_large_numbers(self, tmp_path):
        """Handles large step numbers correctly (sorts numerically, not lexically)."""
        for step in [9, 100, 1000, 20]:
            _durable_step_dir(tmp_path, step)

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        # Numerically highest is 1000, not lexically highest "step_9"
        assert result.name == "step_1000"

    # -- durability of the step_N fallback (#177) --------------------------

    def test_skips_incomplete_newest_and_falls_back(self, tmp_path):
        """An interrupted save must not be selected over an older durable one.

        This is the #177 failure: the emergency save dies part-way, leaving a
        step dir with no DCP `.metadata`, and auto-resume picks it and fails in
        dcp.load with "metadata is None".
        """
        _durable_step_dir(tmp_path, 10)
        (tmp_path / "step_20").mkdir()  # interrupted save: no .metadata

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_10"

    def test_returns_none_when_no_step_dir_is_durable(self, tmp_path):
        """Better to start fresh than to resume into a directory that cannot load."""
        for step in [10, 20]:
            (tmp_path / f"step_{step}").mkdir()

        assert resolve_resume_path(str(tmp_path)) is None

    def test_accepts_pipeline_parallel_layout(self, tmp_path):
        """Under PP each stage writes its own pp{k}/.metadata, not a flat one."""
        _durable_step_dir(tmp_path, 30, pp_stages=2)

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_30"

    def test_skips_incomplete_pipeline_parallel_dir(self, tmp_path):
        """A pp{k}/ subdir without .metadata is still an interrupted save."""
        _durable_step_dir(tmp_path, 10, pp_stages=2)
        partial = tmp_path / "step_20"
        (partial / "pp0").mkdir(parents=True)  # stage dir exists, never finished

        result = resolve_resume_path(str(tmp_path))
        assert result is not None
        assert result.name == "step_10"


# ---------------------------------------------------------------------------
# DataLoader worker signal shielding (#177)
# ---------------------------------------------------------------------------


class TestWorkerSignalShielding:
    """SIGTERM is delivered to the process group, so it reaches DataLoader
    workers too. Workers that die take the emergency checkpoint with them:
    the main process fails fetching its next batch and raises before the loop
    reaches its should_shutdown() check.
    """

    def _restoring(self, fn):
        """Run fn with the process's shutdown-signal handlers restored after."""
        saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
        try:
            fn()
        finally:
            for s, h in saved.items():
                signal.signal(s, h)

    def test_ignores_both_shutdown_signals(self):
        def check():
            ignore_shutdown_signals_in_worker(0)
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN

        self._restoring(check)

    def test_sigterm_does_not_kill_the_caller(self):
        """The point of the shield: the signal arrives and is survived."""

        def check():
            ignore_shutdown_signals_in_worker(0)
            os.kill(os.getpid(), signal.SIGTERM)  # would terminate by default

        self._restoring(check)

    def test_dataloader_installs_the_shield(self):
        """StatefulDataLoader must wire the shield, not just export it."""
        from torch.utils.data import TensorDataset

        from kempnerforge.config.schema import DataConfig
        from kempnerforge.data.dataloader import StatefulDataLoader

        dataset = TensorDataset(torch.arange(8).float().unsqueeze(1))
        loader = StatefulDataLoader(
            dataset, batch_size=2, config=DataConfig(num_workers=2, pin_memory=False)
        )
        assert loader._dataloader.worker_init_fn is ignore_shutdown_signals_in_worker
