"""Picklable traceback frames for DCP's cross-rank error reports.

DCP wraps a rank's failure as ``(exception, traceback.extract_tb(...))`` and gathers it to the
coordinator, which pickles it. From Python 3.13 each ``traceback.FrameSummary`` holds the
frame's code object, which cannot be pickled, so the rank's own error is replaced by
``TypeError: cannot pickle code objects`` and the other ranks lose it.
``register_frame_summary_pickling`` makes those frames picklable without their code object.
"""

from __future__ import annotations

import copyreg
import traceback
from typing import Any


def reduce_frame_summary(frame: traceback.FrameSummary) -> tuple[Any, ...]:
    """Reduce ``frame`` for pickling without its code object.

    The frame's source lines are resolved first, on the process that raised, so the unpickled
    copy formats as the original does. The copy is a plain ``FrameSummary`` whose code object
    is ``None``; unpickling it needs no registration.

    Args:
        frame: The frame to reduce.

    Returns:
        A ``copyreg`` reduce value that rebuilds the frame from its slots.
    """
    _ = frame.line
    slots = {
        name: None if name == "_code" else getattr(frame, name)
        for name in traceback.FrameSummary.__slots__
        if hasattr(frame, name)
    }
    return copyreg.__newobj__, (type(frame),), (None, slots)


def register_frame_summary_pickling() -> None:
    """Register ``reduce_frame_summary`` as the process-wide pickler of ``FrameSummary``.

    Idempotent. A no-op on Python versions whose ``FrameSummary`` holds no code object, and
    when another reducer is already registered for it.
    """
    if "_code" not in traceback.FrameSummary.__slots__:
        return
    if traceback.FrameSummary in copyreg.dispatch_table:
        return
    copyreg.pickle(traceback.FrameSummary, reduce_frame_summary)
