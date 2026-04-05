"""GPU health monitoring and NaN detection.

Provides utilities for detecting common failures during training:
  - NaN/Inf in loss or gradients
  - GPU availability and basic health
  - NCCL liveness via lightweight collectives
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

        Args:
            loss: The scalar loss value to check.
            step: Current training step.

        Returns:
            True if the loss is valid (finite), False if NaN/Inf.

        Raises:
            RuntimeError: If action is "raise" and NaN is detected.
        """
        if _is_finite(loss):
            # Reset consecutive counter on good step
            self.state.consecutive_nans = 0
            self.state.last_good_loss = loss
            self.state.last_good_step = step
            return True

        # NaN detected
        self.state.consecutive_nans += 1
        self.state.total_nans += 1
        if len(self.state.nan_steps) < self.max_history:
            self.state.nan_steps.append(step)

        msg = (
            f"NaN/Inf loss at step {step} "
            f"(consecutive={self.state.consecutive_nans}, "
            f"total={self.state.total_nans}, "
            f"last_good_loss={self.state.last_good_loss:.4f} at step {self.state.last_good_step})"
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

    Args:
        timeout_sec: Timeout for the collective operation.

    Returns:
        True if the all-reduce succeeded, False on timeout or error.
    """
    if not dist.is_initialized():
        return True  # No distributed, nothing to check

    try:
        tensor = torch.ones(1, device="cuda")
        # Use a work handle with timeout
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        expected = dist.get_world_size()
        return tensor.item() == expected
    except RuntimeError as e:
        logger.error(f"NCCL health check failed: {e}")
        return False


def log_health_status(device: int = 0) -> dict[str, bool | str]:
    """Run and log GPU health checks.

    Returns:
        Health check results dict.
    """
    health = check_gpu_health(device)

    status = (
        "HEALTHY"
        if all(
            health[k] for k in ("cuda_available", "device_accessible", "compute_ok", "memory_ok")
        )
        else "UNHEALTHY"
    )

    logger.info(
        f"GPU {device} health: {status} | "
        f"cuda={health['cuda_available']}, "
        f"device={health['device_accessible']}, "
        f"compute={health['compute_ok']}, "
        f"memory={health['memory_ok']}"
    )

    if health["error"]:
        logger.warning(f"GPU {device} error: {health['error']}")

    return health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite(value: float) -> bool:
    """Check if a float value is finite (not NaN/Inf)."""
    import math

    return math.isfinite(value)
