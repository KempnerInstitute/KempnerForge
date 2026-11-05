"""Excerpt a child process's output for a test failure message."""

from __future__ import annotations

TRACEBACK = "Traceback (most recent call last):"


def traceback_or_tail(output: str, tail: int) -> str:
    """Return ``output`` from its first traceback onward, or its last ``tail`` characters.

    torchrun prints its own teardown after a failed child's traceback, often more
    than a fixed tail holds, so a tail alone can hide the child's error.
    """
    start = output.find(TRACEBACK)
    return output[start:] if start != -1 else output[-tail:]
