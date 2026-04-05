"""Graceful shutdown via signal handling.

Handles SIGTERM (SLURM preemption / graceful shutdown) and SIGUSR1
(SLURM requeue) by setting a flag that the training loop checks after
each step. On signal, the loop saves an emergency checkpoint and exits.

Timeout protection ensures the process exits even if graceful shutdown
stalls (e.g., stuck in NCCL collective).
"""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType

logger = logging.getLogger(__name__)

# Signals we intercept for graceful shutdown
_SHUTDOWN_SIGNALS = (signal.SIGTERM, signal.SIGUSR1)


class ShutdownHandler:
    """Cooperative shutdown handler for long-running training jobs.

    Register this handler before the training loop. The training loop
    checks ``should_shutdown()`` after each step and takes appropriate
    action (save checkpoint, clean up, exit).

    If the graceful shutdown exceeds ``timeout_sec``, a forced exit is
    triggered via ``os._exit`` to avoid hanging on stuck collectives.

    Usage::

        handler = ShutdownHandler(timeout_sec=120)
        handler.register()

        for step in range(max_steps):
            train_step()
            if handler.should_shutdown():
                save_checkpoint()
                handler.finish()
                break

    Args:
        timeout_sec: Maximum seconds allowed for graceful shutdown before
            forced exit. Set to 0 to disable the timeout.
    """

    def __init__(self, timeout_sec: float = 120.0) -> None:
        self._shutdown_requested = False
        self._signal_received: signal.Signals | None = None
        self._timeout_sec = timeout_sec
        self._timer: threading.Timer | None = None
        self._original_handlers: dict[signal.Signals, signal._HANDLER] = {}

    @property
    def shutdown_requested(self) -> bool:
        """Whether a shutdown signal has been received."""
        return self._shutdown_requested

    @property
    def signal_received(self) -> signal.Signals | None:
        """The signal that triggered shutdown, or None."""
        return self._signal_received

    def should_shutdown(self) -> bool:
        """Check if the training loop should exit.

        Call this after each training step.
        """
        return self._shutdown_requested

    def register(self) -> None:
        """Register signal handlers for SIGTERM and SIGUSR1.

        Must be called from the main thread.
        """
        for sig in _SHUTDOWN_SIGNALS:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        logger.info("Shutdown handler registered (SIGTERM, SIGUSR1)")

    def unregister(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()
        self._cancel_timer()

    def finish(self) -> None:
        """Call after graceful shutdown is complete.

        Cancels the forced-exit timer and restores signal handlers.
        """
        self._cancel_timer()
        self.unregister()
        logger.info("Graceful shutdown complete")

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler — sets the shutdown flag and starts the timeout."""
        sig = signal.Signals(signum)
        self._shutdown_requested = True
        self._signal_received = sig
        logger.warning(f"Received {sig.name} — requesting graceful shutdown")

        # Start a timer for forced exit
        if self._timeout_sec > 0 and self._timer is None:
            self._timer = threading.Timer(self._timeout_sec, self._force_exit)
            self._timer.daemon = True
            self._timer.start()
            logger.info(f"Forced exit in {self._timeout_sec}s if shutdown not complete")

    def _force_exit(self) -> None:
        """Force-exit the process if graceful shutdown takes too long."""
        import os

        logger.error(f"Graceful shutdown timed out after {self._timeout_sec}s — forcing exit")
        os._exit(1)

    def _cancel_timer(self) -> None:
        """Cancel the forced-exit timer if active."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
