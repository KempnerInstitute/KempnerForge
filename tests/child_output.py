"""Excerpt a child process's output for a test failure message."""

from __future__ import annotations

TRACEBACK = "Traceback (most recent call last):"


def traceback_or_tail(output: str, tail: int) -> str:
    """Return ``output`` from ``tail`` characters before its first traceback to the end.

    Without a traceback, return its last ``tail`` characters. torchrun prints its own
    teardown after a failed child's traceback, often more than a fixed tail holds, so a
    tail alone can hide the child's error. A child that dies without a Python traceback
    (``sys.exit`` with a message, a fatal signal) leaves torchrun's traceback first, and
    the ``tail`` characters before it keep the child's last words.
    """
    start = output.find(TRACEBACK)
    return output[max(0, start - tail) :] if start != -1 else output[-tail:]
