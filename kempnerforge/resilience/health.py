"""GPU health monitoring and NaN detection.

Provides utilities for detecting common failures during training:
  - NaN/Inf in loss, gradients or parameters
  - GPU availability and basic health
  - NCCL liveness via lightweight collectives
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


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

    Once the consecutive NaN count reaches ``max_consecutive``, the detector
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

    def check_loss(self, loss: float, step: int, grad_norm: float | None = None) -> bool:
        """Check a step's loss, and its gradient norm when given, for NaN/Inf.

        When running distributed, all-reduces a NaN flag so ALL ranks agree
        on whether to skip. Prevents rank desync where one rank sees NaN and
        skips its optimizer step while others proceed normally. The flag is on
        the current CUDA device, or on the CPU when CUDA is unavailable, so a
        gloo process group without CUDA reduces it as well.

        A finite loss does not guarantee finite gradients: a micro-batch whose
        labels are all masked has an exact zero cross-entropy loss whatever its
        activations, so the caller passes the gradient norm as well. Each rank's
        norm goes into the same reduced flag as the loss, so every rank skips the
        step together even when a non-finite gradient reaches only some ranks'
        norms.

        Args:
            loss: The scalar loss value to check.
            step: Current training step.
            grad_norm: The step's gradient norm, or ``None`` to check the loss alone.

        Returns:
            True if the loss and the gradient norm are finite on ALL ranks, False
            if any rank has NaN/Inf in either.

        Raises:
            RuntimeError: If action is "raise" and NaN is detected.
        """
        bad_loss = not _is_finite(loss)
        bad_grad_norm = grad_norm is not None and not _is_finite(grad_norm)
        local_nan = bad_loss or bad_grad_norm
        any_nan = _any_rank_flagged(local_nan)

        if not any_nan:
            self.state.consecutive_nans = 0
            self.state.last_good_loss = loss
            self.state.last_good_step = step
            return True

        self.state.consecutive_nans += 1
        self.state.total_nans += 1
        if len(self.state.nan_steps) < self.max_history:
            self.state.nan_steps.append(step)

        if local_nan:
            msg = (
                f"NaN/Inf {'loss' if bad_loss else 'gradient norm'} at step {step} "
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

        Each rank checks the gradients it holds (the local shard of a
        ``DTensor``). When running distributed, one all-reduce of a flag, as in
        :meth:`check_loss`, gives every rank the same verdict, so this is a
        collective that every rank must call, and the ranks all skip the step or
        all take it. Unlike :meth:`check_loss` it leaves :attr:`state` as it is.

        Args:
            model: The model to check, or this rank's pipeline stage of it.
            step: Current training step.

        Returns:
            True if all gradients are finite on ALL ranks, False if any rank
            holds a NaN/Inf gradient.

        Raises:
            RuntimeError: If action is "raise" and any rank holds a NaN/Inf gradient.
        """
        bad_name: str | None = None
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            local = grad.to_local() if isinstance(grad, DTensor) else grad
            if not torch.isfinite(local).all():
                bad_name = name
                break
        if not _any_rank_flagged(bad_name is not None):
            return True
        if bad_name is not None:
            msg = f"NaN/Inf gradient in {bad_name} at step {step}"
        else:
            msg = f"NaN/Inf gradient detected on another rank at step {step}"
        if self.action == "raise":
            raise RuntimeError(msg)
        logger.warning(msg)
        return False

    @property
    def should_rollback(self) -> bool:
        """Whether consecutive NaN count suggests a checkpoint rollback."""
        return self.state.consecutive_nans >= self.max_consecutive

    def reset(self) -> None:
        """Reset NaN tracking state (e.g., after a rollback)."""
        self.state = NaNState()


def params_are_finite(model: torch.nn.Module, device: torch.device | str) -> bool:
    """Check that no rank holds a NaN/Inf parameter of ``model``.

    Each rank checks the parameters it holds (the local shard of a
    ``DTensor``). When running distributed, one all-reduce of a flag over
    the default process group gives every rank the same answer, so this is
    a collective that every rank must call.

    Args:
        model: The model, or this rank's pipeline stage of it.
        device: Device for the all-reduced flag; it must suit the default
            process group's backend (CUDA under NCCL).

    Returns:
        True if every parameter is finite on every rank.
    """
    with torch.no_grad():
        nonfinite = torch.zeros(1, device=device)
        for p in model.parameters():
            local = p.to_local() if isinstance(p, DTensor) else p
            nonfinite += (~torch.isfinite(local).all()).to(device)
    if dist.is_initialized():
        dist.all_reduce(nonfinite)
    return nonfinite.item() == 0


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
        torch.cuda.set_device(device)
        result["device_accessible"] = True

        x = torch.ones(16, device=f"cuda:{device}")
        y = x + x
        assert y.sum().item() == 32.0
        result["compute_ok"] = True
        del x, y

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
        return True

    try:
        tensor = torch.ones(1, device="cuda")
        work = dist.all_reduce(tensor, async_op=True)
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


def _any_rank_flagged(flag: bool) -> bool:
    """Whether ``flag`` is True on any rank of the default process group.

    When running distributed this is one all-reduce of a one-element flag, so
    every rank must call it. The flag is on the current CUDA device when CUDA
    is available, for NCCL, and on the CPU otherwise, for gloo.
    """
    if not dist.is_initialized():
        return flag
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reduced = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(reduced)
    return reduced.item() > 0


def _is_finite(value: float) -> bool:
    """Check if a float value is finite (not NaN/Inf)."""
    import math

    return math.isfinite(value)
