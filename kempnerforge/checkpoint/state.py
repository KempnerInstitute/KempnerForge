"""Training state assembly for checkpointing.

Collects the full training state — model, optimizer, scheduler, dataloader,
training metadata, and RNG states — into a single dict for DCP save/load.

RNG state capture ensures exact reproducibility on resume.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

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
        dataloader: Stateful dataloader (must have state_dict()).
        extra: Additional metadata to include.

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

    _standard_keys = {"step", "tokens_seen", "rng", "scheduler", "dataloader", _GROUP_MODULES_KEY}
    extra = {k: v for k, v in state.items() if k not in _standard_keys}

    return step, tokens_seen, extra
