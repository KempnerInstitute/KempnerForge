"""Training state assembly for checkpointing.

Collects the full training state — model, optimizer, scheduler, dataloader,
training metadata, and RNG states — into a single dict for DCP save/load.

RNG state capture ensures exact reproducibility on resume.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Versioned per-rank state envelope, stored in train_state.pt under this key.
# Schema v1:
#   {"version": 1, "world_size": W, "ranks": {global_rank: payload}}
# where each payload carries that rank's own "rng" capture and, when its
# loader is stateful, its "dataloader" state. The legacy single-copy
# "rng"/"dataloader" keys (always rank 0's) are still written beside the
# envelope, so readers that predate it keep loading; their documented
# limitation is that they restore rank 0's copy on every rank.
PER_RANK_KEY = "per_rank"
PER_RANK_VERSION = 1


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


def build_per_rank_envelope(entries: dict[int, dict[str, Any]], world_size: int) -> dict[str, Any]:
    """Assemble the versioned per-rank envelope from every rank's payload.

    Requires exactly one payload per global rank ``0..world_size - 1``: a
    gather that lost a rank would otherwise save an envelope the load side
    could never restore completely.
    """
    if sorted(entries) != list(range(world_size)):
        raise ValueError(
            f"per-rank envelope needs one payload per rank 0..{world_size - 1}, "
            f"got ranks {sorted(entries)}"
        )
    return {"version": PER_RANK_VERSION, "world_size": world_size, "ranks": entries}


def select_rank_state(envelope: Any, rank: int, world_size: int) -> dict[str, Any]:
    """Return ``rank``'s own payload from a per-rank envelope, or refuse.

    Refuses with ``ValueError`` rather than guessing when the envelope is
    malformed, written by a newer schema version, or saved under a different
    topology: a loader/RNG payload restored onto the wrong rank silently
    changes the data order and random streams of the resumed run.
    """
    if not isinstance(envelope, dict) or not isinstance(envelope.get("ranks"), dict):
        raise ValueError(
            "per-rank state envelope is malformed; expected {'version', 'world_size', 'ranks'}"
        )
    version = envelope.get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"per-rank state envelope has invalid version {version!r}")
    if version > PER_RANK_VERSION:
        raise ValueError(
            f"per-rank state envelope version {version} is newer than this code "
            f"reads (<= {PER_RANK_VERSION}); resume with the KempnerForge version "
            f"that saved the checkpoint"
        )
    saved_world = envelope.get("world_size")
    if not isinstance(saved_world, int):
        raise ValueError(f"per-rank state envelope has invalid world_size {saved_world!r}")
    if saved_world != world_size:
        raise ValueError(
            f"checkpoint per-rank state was saved with world_size={saved_world} but "
            f"this run has world_size={world_size}. Exact dataloader/RNG continuity "
            f"exists only on the saved topology: resume with {saved_world} ranks, or "
            f"set checkpoint.ignore_rank_state_mismatch=true to discard the saved "
            f"loader/RNG state and continue from model/optimizer/scheduler state "
            f"(data order and random streams restart fresh; model/optimizer "
            f"resharding is unaffected)"
        )
    entry = envelope["ranks"].get(rank)
    if not isinstance(entry, dict):
        raise ValueError(
            f"per-rank state envelope has no payload for rank {rank} "
            f"(saved ranks: {sorted(envelope['ranks'])})"
        )
    return entry


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
