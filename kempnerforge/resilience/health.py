"""GPU health monitoring and NaN detection.

Provides utilities for detecting common failures during training:
  - NaN/Inf in loss or gradients
  - GPU availability and basic health
  - NCCL liveness via lightweight collectives
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaN / Inf detection
# ---------------------------------------------------------------------------


@dataclass
class NaNState:
    """Tracks NaN/Inf occurrences across training steps."""

    consecutive_nans: int = 0
    total_nans: int = 0
    last_good_loss: float = float("inf")
    last_good_step: int = 0
    nan_steps: list[int] = field(default_factory=list)


class NaNDetector:
    """Detects and tracks NaN/Inf values in loss and gradients.

    Supports three responses to NaN:
      - ``"warn"``: Log a warning and continue.
      - ``"skip"``: Skip the optimizer step (zero gradients).
      - ``"raise"``: Raise a ``RuntimeError``.

    If consecutive NaN count exceeds ``max_consecutive``, the detector
    signals that a checkpoint rollback is recommended.

    Args:
        action: What to do when NaN is detected.
        max_consecutive: Consecutive NaN steps before recommending rollback.
        max_history: Number of NaN step indices to retain.
    """

    def __init__(
        self,
        action: str = "warn",
        max_consecutive: int = 5,
        max_history: int = 100,
    ) -> None:
        if action not in ("warn", "skip", "raise"):
            raise ValueError(f"Invalid NaN action: {action!r} (expected warn/skip/raise)")
        self.action = action
        self.max_consecutive = max_consecutive
        self.max_history = max_history
        self.state = NaNState()

    def check_loss(self, loss: float, step: int) -> bool:
        """Check a loss value for NaN/Inf.

        When running distributed, all-reduces a NaN flag so ALL ranks agree
        on whether to skip. Prevents rank desync where one rank sees NaN and
        skips its optimizer step while others proceed normally.

        Args:
            loss: The scalar loss value to check.
            step: Current training step.

        Returns:
            True if the loss is valid (finite) on ALL ranks, False if any rank has NaN/Inf.

        Raises:
            RuntimeError: If action is "raise" and NaN is detected.
        """
        local_nan = not _is_finite(loss)

        # Sync NaN flag across all ranks to prevent desync.
        # One tiny all-reduce (4 bytes) — negligible vs gradient sync.
        if dist.is_initialized():
            nan_flag = torch.tensor([1.0 if local_nan else 0.0], device="cuda")
            dist.all_reduce(nan_flag)
            any_nan = nan_flag.item() > 0
        else:
            any_nan = local_nan

        if not any_nan:
            self.state.consecutive_nans = 0
            self.state.last_good_loss = loss
            self.state.last_good_step = step
            return True

        # NaN detected (on this rank or another)
        self.state.consecutive_nans += 1
        self.state.total_nans += 1
        if len(self.state.nan_steps) < self.max_history:
            self.state.nan_steps.append(step)

        if local_nan:
            msg = (
                f"NaN/Inf loss at step {step} "
                f"(consecutive={self.state.consecutive_nans}, "
                f"total={self.state.total_nans}, "
                f"last_good_loss={self.state.last_good_loss:.4f} "
                f"at step {self.state.last_good_step})"
            )
        else:
            msg = (
                f"NaN/Inf detected on another rank at step {step} "
                f"(consecutive={self.state.consecutive_nans}, "
                f"total={self.state.total_nans})"
            )

        if self.action == "raise":
            raise RuntimeError(msg)
        elif self.action == "skip":
            logger.warning(f"{msg} — skipping optimizer step")
        else:
            logger.warning(msg)

        return False

    def check_gradients(self, model: torch.nn.Module, step: int) -> bool:
        """Check model gradients for NaN/Inf before optimizer step.

        Args:
            model: The model to check.
            step: Current training step.

        Returns:
            True if all gradients are finite.
        """
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                msg = f"NaN/Inf gradient in {name} at step {step}"
                if self.action == "raise":
                    raise RuntimeError(msg)
                logger.warning(msg)
                return False
        return True

    @property
    def should_rollback(self) -> bool:
        """Whether consecutive NaN count suggests a checkpoint rollback."""
        return self.state.consecutive_nans >= self.max_consecutive

    def reset(self) -> None:
        """Reset NaN tracking state (e.g., after a rollback)."""
        self.state = NaNState()


# ---------------------------------------------------------------------------
# GPU health
# ---------------------------------------------------------------------------


def check_gpu_health(device: int = 0) -> dict[str, bool | str]:
    """Run basic GPU health checks.

    Performs:
      1. CUDA availability check
      2. Small test computation on the device
      3. Memory allocation test

    Returns:
        Dict with health check results.
    """
    result: dict[str, bool | str] = {
        "cuda_available": torch.cuda.is_available(),
        "device_accessible": False,
        "compute_ok": False,
        "memory_ok": False,
        "error": "",
    }

    if not result["cuda_available"]:
        result["error"] = "CUDA not available"
        return result

    try:
        # Check device is accessible
        torch.cuda.set_device(device)
        result["device_accessible"] = True

        # Test computation
        x = torch.ones(16, device=f"cuda:{device}")
        y = x + x
        assert y.sum().item() == 32.0
        result["compute_ok"] = True
        del x, y

        # Test memory allocation (1MB)
        buf = torch.empty(256 * 1024, dtype=torch.float32, device=f"cuda:{device}")
        del buf
        result["memory_ok"] = True

    except (RuntimeError, AssertionError) as e:
        result["error"] = str(e)
        logger.error(f"GPU health check failed on device {device}: {e}")

    return result


def check_nccl_health(timeout_sec: float = 10.0) -> bool:
    """Check NCCL communication health via a lightweight all-reduce.

    The all-reduce runs with ``async_op=True`` so ``work.wait(timeout=...)``
    enforces the caller's bound rather than falling back to the
    process-group default timeout (``nccl_timeout_sec``, 1800s). Without
    that, this function would sit for 30 minutes on a single stuck peer
    regardless of the ``timeout_sec`` argument.

    Args:
        timeout_sec: Per-operation timeout for the collective. Returns
            False if the all-reduce does not complete within this budget.

    Returns:
        True on success, False on timeout, error, or world-size mismatch.
    """
    if not dist.is_initialized():
        return True  # No distributed, nothing to check

    try:
        tensor = torch.ones(1, device="cuda")
        work = dist.all_reduce(tensor, async_op=True)
        # In current PyTorch Work.wait(timeout=...) raises RuntimeError on
        # timeout; older/alternate backends may return False instead. Handle
        # both so the timeout is honored regardless of version.
        try:
            done = work.wait(timeout=timedelta(seconds=timeout_sec))
        except RuntimeError as e:
            logger.warning(f"NCCL health check timed out after {timeout_sec}s: {e}")
            return False
        if done is False:
            logger.warning(f"NCCL health check timed out after {timeout_sec}s")
            return False
        torch.cuda.synchronize()
        expected = dist.get_world_size()
        return abs(tensor.item() - expected) < 1e-5
    except RuntimeError as e:
        logger.error(f"NCCL health check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite(value: float) -> bool:
    """Check if a float value is finite (not NaN/Inf)."""
    import math

    return math.isfinite(value)
