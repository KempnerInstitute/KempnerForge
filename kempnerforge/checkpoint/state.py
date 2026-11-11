"""Training state assembly for checkpointing.

Collects the full training state — model, optimizer, scheduler, dataloader,
training metadata, and RNG states — into a single dict for DCP save/load.

RNG state capture ensures exact reproducibility on resume.

Dataloader and RNG state are per-rank. ``train_state.pt`` carries every rank's
own payload under ``PER_RANK_KEY`` as a versioned envelope::

    {"version": 1, "world_size": W, "data_parallel": layout | None,
     "ranks": {global_rank: payload}}

A payload holds that rank's own ``"rng"`` capture and, when its loader is
stateful, its ``"dataloader"`` state (every rank or none). Every rank must
capture the same RNG streams: ``"torch_cuda"`` is captured only where CUDA is
available, and an entry missing a stream would silently resume that rank with a
fresh one. ``layout`` maps every global rank to the ``(dp_rank, dp_size)`` data
partition its loader read, or is None when the saving manager was not given it.
The legacy single-copy ``"rng"``/``"dataloader"`` keys (always rank 0's) are
still written beside the envelope, so readers that predate it keep loading,
with their documented limitation that they restore rank 0's copy everywhere.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

import numpy as np
import torch

logger = logging.getLogger(__name__)

PER_RANK_KEY = "per_rank"
PER_RANK_VERSION = 1

_RNG_STREAMS = ("python", "numpy", "torch_cpu")

_MISMATCH_HINT = (
    "set checkpoint.ignore_rank_state_mismatch=true to discard the saved "
    "loader/RNG state and continue from model/optimizer/scheduler state "
    "(data order and random streams restart fresh; model/optimizer "
    "resharding is unaffected)"
)

_GROUP_MODULES_KEY = "scheduler_group_modules"
_PER_GROUP_KEYS = ("base_lrs", "_last_lr", "lr_lambdas")
_PER_GROUP_RATE_KEYS = ("base_lrs", "_last_lr")
_SAME_RATE_REL_TOL = 1e-9


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
    could never restore completely. The ``ValueError`` for an incomplete set
    names ``world_size``, the ranks the gather returned and both ways out, so
    a caller driving a manager as another rank can fix the world it simulated.
    ``data_parallel`` is the complete global rank -> ``(dp_rank, dp_size)``
    layout, or None when unknown.
    """
    if sorted(entries) != list(range(world_size)):
        raise ValueError(
            f"per-rank envelope needs one payload per rank 0..{world_size - 1} "
            f"(world_size={world_size}), got ranks {sorted(entries)}. Every rank must "
            f"reach save() and the gather must return each one's payload; a caller "
            f"driving a manager as another rank must give it that rank's world size "
            f"and a gather covering the whole world"
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
    rng_keys = [set(ranks[r]["rng"]) for r in range(world_size)]
    shared = rng_keys[0].intersection(*rng_keys[1:])
    gaps = []
    for k in sorted(rng_keys[0].union(*rng_keys[1:]) - shared, key=repr):
        lacking = [r for r in range(world_size) if k not in rng_keys[r]]
        gaps.append(f"{k!r} missing on ranks {_ranks_text(lacking)}")
    if gaps:
        raise ValueError(
            f"per-rank state envelope 'rng' captures differ across ranks: {'; '.join(gaps)}. "
            f"Every rank must capture the same RNG streams"
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


def _group_modules(scheduler: Any) -> list[str | None] | None:
    """Return the module each param group of ``scheduler``'s optimizer names, in group order.

    ``build_optimizer`` names the module of each group it gives a module of
    ``optimizer.lr_multipliers`` under ``"module"``; a group without one holds parameters
    outside those modules. None when ``scheduler`` has no optimizer.
    """
    param_groups = getattr(getattr(scheduler, "optimizer", None), "param_groups", None)
    if param_groups is None:
        return None
    return [group.get("module") for group in param_groups]


def _multiplier(module: str | None, lr_multipliers: Mapping[str, float] | None) -> float:
    """Return the multiplier of ``module``'s param groups, 1 for a group that names none.

    Raises:
        ValueError: ``lr_multipliers`` does not list ``module``.
    """
    if module is None:
        return 1.0
    if lr_multipliers is None or module not in lr_multipliers:
        raise ValueError(
            f"The checkpoint's scheduler state is laid out by module for other param groups, "
            f"which needs the multiplier of {module!r}, but lr_multipliers={lr_multipliers}"
        )
    return lr_multipliers[module]


def _layout_sources(
    saved_modules: Sequence[str | None],
    current: Sequence[str | None],
    first: dict[str | None, int],
    lr_multipliers: Mapping[str, float] | None,
) -> list[tuple[int, tuple[float, float] | None]]:
    """Return, for each current param group, the saved group whose entries it takes and the
    multipliers its rates are scaled from and to.

    A group of a module with saved groups takes the first of them as it is. Any other takes
    the first saved group, its rates divided by that group's multiplier and multiplied by its
    own, so it runs at the saved peak times its multiplier.

    Raises:
        ValueError: The state holds no saved group, or ``lr_multipliers`` lacks a module the
            scaling needs.
    """
    sources: list[tuple[int, tuple[float, float] | None]] = []
    for module in current:
        if module in first:
            sources.append((first[module], None))
            continue
        if not saved_modules:
            raise ValueError(
                f"The checkpoint's scheduler state holds no param group to lay out for this "
                f"run's groups of {current}"
            )
        scale = (_multiplier(saved_modules[0], lr_multipliers), _multiplier(module, lr_multipliers))
        sources.append((0, scale))
    return sources


def _scheduler_state_for(
    scheduler: Any,
    saved: dict[str, Any],
    saved_modules: object,
    lr_multipliers: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return the scheduler state ``saved`` laid out for the param groups ``scheduler`` holds.

    A scheduler state lists some entries per param group, such as its base rate, in the order
    of the groups it was saved with, and ``saved_modules`` names the module of each of those
    groups; without it, as in a state saved when no group named one, no saved group has one.
    Which groups ``build_optimizer`` makes for ``optimizer.lr_multipliers`` depends on which
    parameters are trainable and on ``optimizer.weight_decay_scope``, since it leaves out
    groups with no parameters, while every group of one module has that module's rate. So
    when the saved groups' modules differ from the current ones, in name or in number, the
    saved schedule carries on in every group, as it does when they match: each current group
    takes the entries of the first saved group of its module, groups that name no module
    matching each other, and a group whose module has no saved group takes those of the first
    saved group, its rates scaled from that group's multiplier in ``lr_multipliers`` to its
    own, so it runs at the saved peak times its multiplier; a group that names no module has a
    multiplier of 1. Otherwise ``saved`` is returned as it is, and so it is when the state
    lists no base rates, or when no group names a module and ``scheduler`` is not a
    ``LambdaLR``, which keeps the positional load such a scheduler always had.

    Raises:
        ValueError: ``saved_modules`` does not name one module per saved group; the groups
            differ and ``scheduler`` is not a ``LambdaLR``, whose per-group entries this lays
            out; the saved groups of one module have different base rates; or a group's module
            has no saved group and the state holds none, or ``lr_multipliers`` lacks the
            multiplier the scaling needs.
    """
    current = _group_modules(scheduler)
    base_lrs = saved.get("base_lrs")
    names_module = saved_modules is not None or any(m is not None for m in current or ())
    is_lambda = isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    if not isinstance(base_lrs, list) or not (names_module or is_lambda):
        return saved
    if saved_modules is None:
        saved_modules = [None] * len(base_lrs)
    entries = {key: saved[key] for key in _PER_GROUP_KEYS if key in saved}
    if (
        not isinstance(saved_modules, list)
        or not all(module is None or isinstance(module, str) for module in saved_modules)
        or any(
            not isinstance(values, list) or len(values) != len(saved_modules)
            for values in entries.values()
        )
    ):
        raise ValueError(
            f"The checkpoint's scheduler state does not hold one entry per param group of the "
            f"modules it records, {saved_modules!r}"
        )
    if saved_modules == current:
        return saved
    if current is None or not is_lambda:
        raise ValueError(
            f"The checkpoint's scheduler state belongs to param groups of the modules "
            f"{saved_modules}, but this run's groups are of {current}, and only the state of a "
            f"LambdaLR, which every registered scheduler builds, can be laid out for other groups"
        )
    first: dict[str | None, int] = {}
    for index, module in enumerate(saved_modules):
        lead = first.setdefault(module, index)
        if base_lrs[lead] != base_lrs[index]:
            raise ValueError(
                f"The checkpoint's scheduler state gives the param groups of {module!r} the base "
                f"rates {base_lrs[lead]} and {base_lrs[index]}, so it cannot be laid out for "
                f"this run's groups of {current}"
            )
    sources = _layout_sources(saved_modules, current, first, lr_multipliers)
    logger.info(
        f"Laying out the scheduler state saved for param groups of {saved_modules} for this "
        f"run's groups of {current} by module"
    )
    return {
        **saved,
        **{
            key: [
                values[index]
                if scale is None or key not in _PER_GROUP_RATE_KEYS
                else values[index] / scale[0] * scale[1]
                for index, scale in sources
            ]
            for key, values in entries.items()
        },
    }


def _module_peaks(scheduler: Any) -> dict[str | None, Any]:
    """Return the base rate of each module's first param group, when a group names a module."""
    modules = _group_modules(scheduler) or []
    base_lrs = getattr(scheduler, "base_lrs", None)
    if (
        not isinstance(base_lrs, list)
        or len(base_lrs) != len(modules)
        or all(module is None for module in modules)
    ):
        return {}
    peaks: dict[str | None, Any] = {}
    for module, base_lr in zip(modules, base_lrs, strict=True):
        peaks.setdefault(module, base_lr)
    return peaks


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
        scheduler: LR scheduler (must have state_dict()). When a param group of its optimizer
            names a module, as ``build_optimizer`` names those of ``optimizer.lr_multipliers``,
            the module of every group is recorded under ``scheduler_group_modules``.
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
        modules = _group_modules(scheduler)
        if modules is not None and any(module is not None for module in modules):
            state[_GROUP_MODULES_KEY] = modules

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
    lr_multipliers: Mapping[str, float] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    """Restore the non-distributed portion of the training state.

    Args:
        state: Training state dict (from build_train_state).
        scheduler: LR scheduler to restore. The saved schedule carries on in every param
            group, peak included, whatever the rates the optimizer was built with. Its state is
            laid out by module for groups other than those it was saved with (see
            ``build_train_state``): each group of a module the checkpoint holds resumes at that
            module's rates, and any other at the saved peak times its multiplier. Nothing is
            restored when that fails. When a group names a module, the peak rate of each
            module's groups is logged, with a warning when one differs from the rate the
            optimizer was built with, as when ``optimizer.lr`` changed since the save.
        dataloader: Stateful dataloader to restore.
        lr_multipliers: The ``optimizer.lr_multipliers`` the scheduler's groups were built
            with, which must be those of the save. Needed only for a group of a module the
            checkpoint holds no group of.

    Restores whatever ``rng``/``dataloader`` keys ``state`` carries. On an
    exact resume the manager hands each rank its own view of the broadcast
    train state (its per-rank envelope entry overlaid on these keys) before
    calling this; any other load passes the broadcast state as is, so rank 0's
    single copy restores on every rank.

    Returns:
        Tuple of (step, tokens_seen, extra) where extra contains any
        additional keys saved via build_train_state(extra=...).

    Raises:
        ValueError: The scheduler state cannot be laid out for the scheduler's param groups.
    """
    step = state.get("step", 0)
    tokens_seen = state.get("tokens_seen", 0)
    scheduler_state = None
    if scheduler is not None and "scheduler" in state:
        scheduler_state = _scheduler_state_for(
            scheduler, state["scheduler"], state.get(_GROUP_MODULES_KEY), lr_multipliers
        )

    if "rng" in state:
        set_rng_state(state["rng"])
        logger.info("Restored RNG states")

    if scheduler is not None and scheduler_state is not None:
        built = _module_peaks(scheduler)
        scheduler.load_state_dict(scheduler_state)
        peaks = _module_peaks(scheduler)
        if peaks:
            logger.info(
                f"Restored scheduler state; each module's param groups run the restored "
                f"schedule from peak lr {peaks}"
            )
            if any(
                not math.isclose(peak, built.get(module, peak), rel_tol=_SAME_RATE_REL_TOL)
                for module, peak in peaks.items()
            ):
                logger.warning(
                    f"The restored schedule runs the param groups from peak lr {peaks}, while "
                    f"the optimizer was built at {built}: a load that restores train_state.pt "
                    f"continues the saved schedule, its peak included, so a changed "
                    f"optimizer.lr does not apply. A stage with a new optimizer.lr needs a warm "
                    f"start from the weights alone."
                )
        else:
            logger.info("Restored scheduler state")

    if dataloader is not None and "dataloader" in state and hasattr(dataloader, "load_state_dict"):
        dataloader.load_state_dict(state["dataloader"])
        logger.info("Restored dataloader state")

    _standard_keys = {
        "step",
        "tokens_seen",
        "rng",
        "scheduler",
        "dataloader",
        PER_RANK_KEY,
        _GROUP_MODULES_KEY,
    }
    extra = {k: v for k, v in state.items() if k not in _standard_keys}

    return step, tokens_seen, extra
