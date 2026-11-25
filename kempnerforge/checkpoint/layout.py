"""Where a checkpoint keeps its DCP shards: one index at its top, or one per pipeline stage.

A save without pipeline parallelism writes one DCP index, ``.metadata``, at the top of its
``step_N`` directory: the flat layout. A save under pipeline parallelism writes one per stage,
in ``pp0/`` to ``pp{n-1}/``, and records ``n`` as ``pp`` in the checkpoint's ``metadata.json``.
A checkpoint saved before that record existed shows its degree only by the stage directories it
holds.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DCP_INDEX_FILE = ".metadata"
METADATA_FILE = "metadata.json"
PP_DEGREE_KEY = "pp"
_STAGE_DIR = re.compile(r"pp(\d+)")


@dataclass(frozen=True)
class SavedLayout:
    """How a checkpoint lays out its DCP shards.

    ``stages`` is 0 for the flat layout, and otherwise the number of pipeline stages, each with
    its own DCP index in ``pp{k}/``. ``recorded`` is whether the checkpoint's ``metadata.json``
    records ``stages``; when it does not, ``stages`` is the number of stage directories the
    checkpoint holds.
    """

    stages: int
    recorded: bool = False

    @property
    def flat(self) -> bool:
        """Whether the checkpoint was saved without pipeline parallelism."""
        return self.stages == 0

    def describe(self) -> str:
        """The layout in words, as the errors that refuse it name it."""
        if self.flat:
            return "saved without pipeline parallelism, with one DCP index at its top"
        dirs = "pp0/" if self.stages == 1 else f"pp0/ to pp{self.stages - 1}/"
        if self.recorded:
            return (
                f"saved with pipeline parallelism at pp={self.stages}, with one DCP index per "
                f"stage in {dirs}"
            )
        return (
            f"saved with pipeline parallelism before the degree was recorded, with "
            f"{self.stages} stage directories ({dirs})"
        )


def stage_dir(ckpt_dir: Path, stage: int) -> Path:
    """The directory that holds pipeline stage ``stage``'s DCP shards in ``ckpt_dir``."""
    return ckpt_dir / f"pp{stage}"


def _found_stage_dirs(ckpt_dir: Path) -> list[Path]:
    if not ckpt_dir.is_dir():
        return []
    found = [
        (int(match[1]), d)
        for d in ckpt_dir.iterdir()
        if d.is_dir() and (match := _STAGE_DIR.fullmatch(d.name))
    ]
    return [d for _, d in sorted(found)]


def _read_record(ckpt_dir: Path) -> Mapping[str, Any]:
    path = ckpt_dir / METADATA_FILE
    try:
        meta = json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except Exception as e:
        logger.warning(f"Could not read the pipeline degree from {path}: {e}")
        return {}
    return meta if isinstance(meta, dict) else {}


def _recorded_degree(meta: Mapping[str, Any], path: Path) -> int | None:
    degree = meta.get(PP_DEGREE_KEY)
    if degree is None:
        return None
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        logger.warning(f"{path} records {PP_DEGREE_KEY}={degree!r}, which is not a pipeline degree")
        return None
    return degree


def saved_layout(ckpt_dir: Path, meta: Mapping[str, Any] | None = None) -> SavedLayout | None:
    """How ``ckpt_dir`` lays out its DCP shards, None when it holds no DCP index.

    The degree ``metadata.json`` records comes first. Without one, a ``.metadata`` at the top
    makes the layout flat, and ``pp{k}/`` directories make it a stage layout of as many stages,
    whether or not each holds its index yet. A record of something other than a positive
    integer is logged and counts as none.

    Args:
        ckpt_dir: The checkpoint directory.
        meta: The JSON object the checkpoint's ``metadata.json`` holds, when the caller has
            read it. Without it the file is read here, and one that cannot be read is logged
            and counts as recording no degree.
    """
    record = _read_record(ckpt_dir) if meta is None else meta
    recorded = _recorded_degree(record, ckpt_dir / METADATA_FILE)
    if recorded is not None:
        return SavedLayout(recorded, recorded=True)
    if (ckpt_dir / DCP_INDEX_FILE).exists():
        return SavedLayout(0)
    found = _found_stage_dirs(ckpt_dir)
    return SavedLayout(len(found)) if found else None


def stage_dirs(ckpt_dir: Path, layout: SavedLayout) -> list[Path]:
    """The stage directories of ``layout`` in ``ckpt_dir``, in stage order.

    These are ``pp0/`` to ``pp{n-1}/`` when ``metadata.json`` records ``n``, whether or not
    each exists, and otherwise the ``pp{k}/`` directories ``ckpt_dir`` holds.
    """
    if layout.recorded:
        return [stage_dir(ckpt_dir, k) for k in range(layout.stages)]
    return _found_stage_dirs(ckpt_dir)
