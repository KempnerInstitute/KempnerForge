"""Tests for ``tests.child_output``, which the e2e and smoke failure messages use."""

from __future__ import annotations

from tests.child_output import traceback_or_tail

CHILD_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "scripts/train.py", line 42, in <module>\n'
    "ValueError: the child's root cause\n"
)


class TestTracebackOrTail:
    def test_keeps_tail_characters_before_the_first_traceback_and_all_after_it(self):
        logs = "[step 1] loss=5.5\n" * 200
        teardown = "torch.distributed.elastic teardown\n" * 200
        output = logs + CHILD_ERROR + CHILD_ERROR + teardown
        excerpt = traceback_or_tail(output, 2000)
        assert excerpt == logs[-2000:] + CHILD_ERROR + CHILD_ERROR + teardown
        assert "the child's root cause" not in output[-2000:]

    def test_falls_back_to_the_tail_without_a_traceback(self):
        output = "".join(f"[step {i}] loss=5.5\n" for i in range(500))
        assert traceback_or_tail(output, 300) == output[-300:]
        assert traceback_or_tail("short", 300) == "short"
