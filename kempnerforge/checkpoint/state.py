"""Training state assembly for checkpointing.

Collects the full training state — model, optimizer, scheduler, dataloader,
training metadata, and RNG states — into a single dict for DCP save/load.

RNG state capture ensures exact reproducibility on resume.
"""

from __future__ import annotations

import logging
import random
from typing import Any, TypeGuard

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Versioned per-rank state envelope, stored in train_state.pt under this key.
# Schema v1:
#   {"version": 1, "world_size": W, "data_parallel": layout | None,
#    "ranks": {global_rank: payload}}
# where each payload carries that rank's own "rng" capture and, when its
# loader is stateful, its "dataloader" state (every rank or none). ``layout``
# maps every global rank to the (dp_rank, dp_size) data partition its loader
# read, or is None when the saving manager was not given it. The legacy
# single-copy "rng"/"dataloader" keys (always rank 0's) are still written
# beside the envelope, so readers that predate it keep loading; their
# documented limitation is that they restore rank 0's copy on every rank.
PER_RANK_KEY = "per_rank"
PER_RANK_VERSION = 1

# Streams get_rng_state() always captures; a per-rank entry missing one would
# silently resume that rank with a fresh stream.
_RNG_STREAMS = ("python", "numpy", "torch_cpu")

_MISMATCH_HINT = (
    "set checkpoint.ignore_rank_state_mismatch=true to discard the saved "
    "loader/RNG state and continue from model/optimizer/scheduler state "
    "(data order and random streams restart fresh; model/optimizer "
    "resharding is unaffected)"
)


def get_rng_state() -> dict[str, Any]:
    """Capture all RNG states for reproducibility on resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore all RNG states from a checkpoint."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def capture_rank_state(dataloader: Any | None = None) -> dict[str, Any]:
    """Capture this rank's own resume payload: RNG streams and loader cursor."""
    payload: dict[str, Any] = {"rng": get_rng_state()}
    if dataloader is not None and hasattr(dataloader, "state_dict"):
        payload["dataloader"] = dataloader.state_dict()
    return payload


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def dp_coordinates(value: Any) -> tuple[int, int] | None:
    """``value`` as a ``(dp_rank, dp_size)`` tuple, or None if it is not one.

    Valid: a pair of ints with ``0 <= dp_rank < dp_size``.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    dp_rank, dp_size = value
    if not (_is_int(dp_rank) and _is_int(dp_size)) or not 0 <= dp_rank < dp_size:
        return None
    return (dp_rank, dp_size)


def _ranks_text(ranks: list[Any]) -> str:
    shown = ", ".join(repr(r) for r in ranks[:8])
    return f"[{shown}{', ...' if len(ranks) > 8 else ''}]"


def _layout_problem(layout: Any, world_size: int) -> str | None:
    """Why ``layout`` is not a complete data-parallel layout, or None if it is."""
    if not isinstance(layout, dict):
        return (
            f"expected a mapping of global rank to (dp_rank, dp_size), got {type(layout).__name__}"
        )
    missing = [r for r in range(world_size) if r not in layout]
    if missing:
        return f"no coordinates for ranks {_ranks_text(missing)}"
    unexpected = [k for k in layout if not (_is_int(k) and 0 <= k < world_size)]
    if unexpected:
        return f"keys {_ranks_text(unexpected)} outside world_size={world_size}"
    for r in range(world_size):
        if dp_coordinates(layout[r]) is None:
            return (
                f"rank {r} has {layout[r]!r}, expected ints (dp_rank, dp_size) "
                f"with 0 <= dp_rank < dp_size"
            )
    return None


def build_per_rank_envelope(
    entries: dict[int, dict[str, Any]],
    world_size: int,
    data_parallel: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the versioned per-rank envelope from every rank's payload.

    Requires exactly one payload per global rank ``0..world_size - 1``: a
    gather that lost a rank would otherwise save an envelope the load side
    could never restore completely. ``data_parallel`` is the complete
    global rank -> ``(dp_rank, dp_size)`` layout, or None when unknown.
    """
    if sorted(entries) != list(range(world_size)):
        raise ValueError(
            f"per-rank envelope needs one payload per rank 0..{world_size - 1}, "
            f"got ranks {sorted(entries)}"
        )
    layout = None
    if data_parallel is not None:
        problem = _layout_problem(data_parallel, world_size)
        if problem is not None:
            raise ValueError(f"per-rank envelope data-parallel layout is malformed: {problem}")
        layout = {r: dp_coordinates(data_parallel[r]) for r in range(world_size)}
    return {
        "version": PER_RANK_VERSION,
        "world_size": world_size,
        "data_parallel": layout,
        "ranks": entries,
    }


def _check_envelope(
    envelope: Any, world_size: int, data_parallel: dict[int, tuple[int, int]] | None
) -> None:
    """Refuse an envelope that every rank cannot restore alike.

    Inspects the whole envelope before any rank selects its entry, and reads
    only inputs every rank shares (the broadcast envelope, the world size, the
    gathered current layout), so all ranks reach the same verdict.
    """
    if not isinstance(envelope, dict) or not isinstance(envelope.get("ranks"), dict):
        raise ValueError(
            "per-rank state envelope is malformed; expected "
            "{'version', 'world_size', 'data_parallel', 'ranks'}"
        )
    version = envelope.get("version")
    if not _is_int(version) or version < 1:
        raise ValueError(f"per-rank state envelope has invalid version {version!r}")
    if version > PER_RANK_VERSION:
        raise ValueError(
            f"per-rank state envelope version {version} is newer than this code "
            f"reads (<= {PER_RANK_VERSION}); resume with the KempnerForge version "
            f"that saved the checkpoint"
        )
    saved_world = envelope.get("world_size")
    if not _is_int(saved_world) or saved_world < 1:
        raise ValueError(f"per-rank state envelope has invalid world_size {saved_world!r}")
    if saved_world != world_size:
        raise ValueError(
            f"checkpoint per-rank state was saved with world_size={saved_world} but "
            f"this run has world_size={world_size}. Exact dataloader/RNG continuity "
            f"exists only on the saved topology: resume with {saved_world} ranks, or "
            f"{_MISMATCH_HINT}"
        )

    ranks = envelope["ranks"]
    missing = [r for r in range(world_size) if r not in ranks]
    unexpected = [k for k in ranks if not (_is_int(k) and 0 <= k < world_size)]
    if missing:
        raise ValueError(
            f"per-rank state envelope is incomplete: no payload for ranks "
            f"{_ranks_text(missing)} of world_size={world_size}"
        )
    if unexpected:
        raise ValueError(
            f"per-rank state envelope has payloads for ranks {_ranks_text(unexpected)} "
            f"outside world_size={world_size}"
        )
    for r in range(world_size):
        entry = ranks[r]
        rng = entry.get("rng") if isinstance(entry, dict) else None
        if not isinstance(rng, dict) or any(k not in rng for k in _RNG_STREAMS):
            raise ValueError(
                f"per-rank state envelope entry for rank {r} lacks a complete 'rng' capture "
                f"({', '.join(_RNG_STREAMS)})"
            )
        if "dataloader" in entry and not isinstance(entry["dataloader"], dict):
            raise ValueError(
                f"per-rank state envelope entry for rank {r} has a non-dict 'dataloader' state"
            )
    without_loader = [r for r in range(world_size) if "dataloader" not in ranks[r]]
    if without_loader and len(without_loader) < world_size:
        raise ValueError(
            f"per-rank state envelope has no 'dataloader' state for ranks "
            f"{_ranks_text(without_loader)} while the other ranks have one"
        )

    if "data_parallel" not in envelope:
        raise ValueError("per-rank state envelope has no data-parallel layout entry")
    saved_layout = envelope["data_parallel"]
    if saved_layout is None:
        return
    problem = _layout_problem(saved_layout, world_size)
    if problem is not None:
        raise ValueError(f"per-rank state envelope data-parallel layout is malformed: {problem}")
    if data_parallel is None:
        return
    changed = [r for r in range(world_size) if dp_coordinates(saved_layout[r]) != data_parallel[r]]
    if changed:
        examples = "; ".join(
            f"rank {r} saved {tuple(saved_layout[r])} now {data_parallel[r]}" for r in changed[:4]
        )
        more = f"; {len(changed) - 4} more" if len(changed) > 4 else ""
        raise ValueError(
            f"checkpoint per-rank state was saved under a different data-parallel "
            f"layout: {len(changed)} of {world_size} ranks now read another data "
            f"partition, as (dp_rank, dp_size): {examples}{more}. A loader cursor is "
            f"valid only on the partition it was saved on: resume with the saved "
            f"data-parallel layout, or {_MISMATCH_HINT}"
        )


def select_rank_state(
    envelope: Any,
    rank: int,
    world_size: int,
    data_parallel: dict[int, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Return ``rank``'s own payload from a per-rank envelope, or refuse.

    Refuses with ``ValueError`` rather than guessing when the envelope is
    malformed or incomplete, written by a newer schema version, or saved
    under a different world size or data-parallel layout: a loader/RNG
    payload restored onto the wrong rank or data partition silently changes
    the data order and random streams of the resumed run. The whole envelope
    is checked before ``rank``'s entry is selected, so every rank reaches the
    same verdict.

    Args:
        envelope: The broadcast ``per_rank`` envelope.
        rank: This process's global rank.
        world_size: This run's world size.
        data_parallel: This run's complete global rank -> ``(dp_rank, dp_size)``
            layout, or None when unknown (the layout check is then skipped).
    """
    _check_envelope(envelope, world_size, data_parallel)
    return envelope["ranks"][rank]


def build_train_state(
    step: int,
    tokens_seen: int,
    scheduler: Any | None = None,
    dataloader: Any | None = None,
    extra: dict | None = None,
    per_rank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the non-distributed portion of the training state.

    Model and optimizer state are handled by DCP directly.
    This function captures everything else needed for exact resumption.

    Args:
        step: Current training step.
        tokens_seen: Total tokens processed so far.
        scheduler: LR scheduler (must have state_dict()).
        dataloader: Stateful dataloader (must have state_dict()); writes the
            legacy single-copy ``dataloader`` key (the caller's own loader,
            rank 0 in distributed runs).
        extra: Additional metadata to include.
        per_rank: Versioned per-rank envelope (``build_per_rank_envelope``)
            holding every rank's own loader/RNG payload.

    Returns:
        Dict with training state, scheduler state, dataloader state, and RNG states.
    """
    state: dict[str, Any] = {
        "step": step,
        "tokens_seen": tokens_seen,
        "rng": get_rng_state(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if dataloader is not None and hasattr(dataloader, "state_dict"):
        state["dataloader"] = dataloader.state_dict()

    if extra:
        state.update(extra)

    if per_rank is not None:
        state[PER_RANK_KEY] = per_rank

    return state


def restore_train_state(
    state: dict[str, Any],
    scheduler: Any | None = None,
    dataloader: Any | None = None,
) -> tuple[int, int, dict[str, Any]]:
    """Restore the non-distributed portion of the training state.

    Args:
        state: Training state dict (from build_train_state).
        scheduler: LR scheduler to restore.
        dataloader: Stateful dataloader to restore.

    Restores whatever ``rng``/``dataloader`` keys ``state`` carries. In
    distributed runs the manager hands each rank its own view of the broadcast
    train state (its per-rank envelope entry overlaid on these keys) before
    calling this.

    Returns:
        Tuple of (step, tokens_seen, extra) where extra contains any
        additional keys saved via build_train_state(extra=...).
    """
    step = state.get("step", 0)
    tokens_seen = state.get("tokens_seen", 0)

    if "rng" in state:
        set_rng_state(state["rng"])
        logger.info("Restored RNG states")

    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
        logger.info("Restored scheduler state")

    if dataloader is not None and "dataloader" in state and hasattr(dataloader, "load_state_dict"):
        dataloader.load_state_dict(state["dataloader"])
        logger.info("Restored dataloader state")

    _standard_keys = {"step", "tokens_seen", "rng", "scheduler", "dataloader", PER_RANK_KEY}
    extra = {k: v for k, v in state.items() if k not in _standard_keys}

    return step, tokens_seen, extra
