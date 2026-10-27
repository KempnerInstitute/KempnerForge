"""Tracking calls bounded in time, so a slow or unreachable server never stalls training.

``_BestEffortWorker`` runs calls in order off the training thread, each under a deadline;
``_RunOwner`` starts and later ends a run on one dedicated thread; ``_RateLimitedWarning``
keeps their failures to occasional warnings.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.tracker import _scrub_uris

logger = get_logger(__name__)

_WARN_INTERVAL_S = 300.0  # minimum gap between repeats of a warning


class _RateLimitedWarning:
    """Warns on the first event, then at most once per interval, counting what it held back."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._last: float | None = None
        self._held_back = 0

    def __call__(self, message: str) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < self._interval:
                self._held_back += 1
                return
            held_back, self._held_back, self._last = self._held_back, 0, now
        suffix = f" ({held_back} similar warnings suppressed)" if held_back else ""
        logger.warning(_scrub_uris(message) + suffix)


class _BestEffortWorker:
    """Runs tracking calls off the training thread so a slow server cannot stall it.

    One daemon thread takes calls in submission order. Each call runs on a short-lived thread
    that the worker waits on for at most the call's deadline; a thread cannot be cancelled, so a
    call past its deadline is abandoned and the next one proceeds. At most ``max_threads`` call
    threads are alive at once, abandoned ones included; beyond that, calls are dropped until one
    returns. ``submit`` never blocks (a full queue drops the new call), and failures, timeouts
    and drops only emit rate-limited warnings.
    """

    def __init__(
        self,
        name: str,
        *,
        max_queue: int,
        max_threads: int,
        call_deadline: float,
        drain_deadline: float,
    ) -> None:
        self._name = name
        self._max_queue = max_queue
        self._max_threads = max_threads
        self._call_deadline = call_deadline
        self._drain_deadline = drain_deadline
        self._calls: list[threading.Thread] = []  # call threads not yet seen to finish
        self._items: deque[tuple[str, Callable[[], Any], float]] = deque()
        self._cv = threading.Condition()
        self._busy = False  # a dequeued call is still being waited on
        self._closed = False
        self._thread = threading.Thread(target=self._consume, name=f"{name}-tracking", daemon=True)
        self._warn_call = _RateLimitedWarning(_WARN_INTERVAL_S)
        self._warn_drop = _RateLimitedWarning(_WARN_INTERVAL_S)

    def start(self) -> None:
        self._thread.start()

    def submit(self, what: str, fn: Callable[[], Any], deadline: float | None = None) -> bool:
        """Queue ``fn``; False if it was dropped (queue full or worker closed)."""
        with self._cv:
            if self._closed:
                return False
            accepted = len(self._items) < self._max_queue
            if accepted:
                limit = self._call_deadline if deadline is None else deadline
                self._items.append((what, fn, limit))
                self._cv.notify_all()
        if not accepted:
            self._warn_drop(
                f"{self._name} tracking queue full ({self._max_queue}); dropping {what}"
            )
        return accepted

    def flush(self, timeout: float) -> bool:
        """Wait until every queued call has been attempted; False if ``timeout`` ran out first."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._items or self._busy:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
        return True

    def close(self) -> bool:
        """Stop accepting calls and deliver the queued ones within the drain deadline."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        drained = self.flush(self._drain_deadline)
        if not drained:
            with self._cv:
                lost = len(self._items) + int(self._busy)
                self._items.clear()
            logger.warning(f"{self._name}: {lost} tracking call(s) not delivered before shutdown")
        return drained

    def _consume(self) -> None:
        while True:
            with self._cv:
                while not self._items:
                    if self._closed:
                        return
                    self._cv.wait()
                what, fn, deadline = self._items.popleft()
                self._busy = True
            try:
                self._run(what, fn, deadline)
            except Exception as e:  # the worker must outlive any single call
                self._warn_call(f"{self._name} {what} not run ({e}); points may be lost")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _run(self, what: str, fn: Callable[[], Any], deadline: float) -> None:
        self._calls = [thread for thread in self._calls if thread.is_alive()]
        if len(self._calls) >= self._max_threads:
            self._warn_drop(
                f"{self._name}: {len(self._calls)} abandoned tracking calls have not returned; "
                f"dropping {what}"
            )
            return
        failure: list[Exception] = []

        def call() -> None:
            try:
                fn()
            except Exception as e:  # reported by the worker, not raised in a stray thread
                failure.append(e)

        thread = threading.Thread(target=call, name=f"{self._name}-call", daemon=True)
        thread.start()
        self._calls.append(thread)
        thread.join(deadline)
        if thread.is_alive():
            self._warn_call(
                f"{self._name} {what} still running after {deadline:g}s; abandoning it, "
                "points may be lost"
            )
        elif failure:
            self._warn_call(f"{self._name} {what} failed ({failure[0]}); points may be lost")


class _RunOwner:
    """Starts and later ends a tracking run on one dedicated thread, waiting on each for a deadline.

    mlflow keeps the active run, and the system metrics monitor tied to it, per thread, so the end
    must run on the thread that ran the start. A caller that stops waiting is never blocked again:
    a start that returns after its deadline is ended right away, and a late end is abandoned.
    """

    def __init__(self, name: str, start: Callable[[], Any], end: Callable[[], Any]) -> None:
        self._start_fn = start
        self._end_fn = end
        self._started = threading.Event()  # start returned or raised
        self._end_requested = threading.Event()
        self._ended = threading.Event()  # end returned or raised, or there was nothing to end
        self._result: Any = None
        self._start_error: Exception | None = None
        self._end_error: Exception | None = None
        self._thread = threading.Thread(target=self._own, name=f"{name}-run", daemon=True)

    def start(self, deadline: float) -> Any:
        """Return what ``start`` returned; re-raise its error, or TimeoutError past ``deadline``."""
        self._thread.start()
        if not self._started.wait(deadline):
            self._end_requested.set()  # the start still running is ended as soon as it returns
            raise TimeoutError(f"no answer from the tracking server within {deadline:g}s")
        if self._start_error is not None:
            raise self._start_error
        return self._result

    def end(self, deadline: float) -> None:
        """Run ``end`` on the owner thread; re-raise its error, or TimeoutError past deadline."""
        self._end_requested.set()
        if not self._ended.wait(deadline):
            raise TimeoutError(f"no answer from the tracking server within {deadline:g}s")
        if self._end_error is not None:
            raise self._end_error

    def _own(self) -> None:
        try:
            self._result = self._start_fn()
        except Exception as e:  # raised again by start(), on the caller's thread
            self._start_error = e
        self._started.set()
        if self._start_error is None:
            self._end_requested.wait()
            try:
                self._end_fn()
            except Exception as e:  # raised again by end()
                self._end_error = e
        self._ended.set()
