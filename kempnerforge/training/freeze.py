"""Parameter freezing helpers.

``freeze_params`` is the primitive: toggle ``requires_grad`` on parameters
whose fully-qualified name matches any of the provided fnmatch patterns.

``apply_freeze_specs`` consumes a list of ``FreezeSpec`` entries (e.g. from
``VLMConfig.freeze``) and resolves each spec's ``module`` field against a
pattern map (typically ``DEFAULT_MODULE_PATTERNS`` or an arch-specific
override on the config). A raw fnmatch pattern is passed through unchanged
when it is not a known alias.

``canonical_freeze_meta`` produces a stable, reorder-invariant
serialization of a freeze-spec list for checkpoint metadata, so a
checkpoint saved with ``[A, B]`` matches one loaded with ``[B, A]`` as
long as the effective mask is identical.

``effective_freeze`` resolves the active freeze-spec list at a given
training step from a ``base`` list (always-on) and a list of
``FreezeStage`` step-boundary transitions. Used at save (records the
post-transition state in metadata), at load (computes the expected
metadata for the compare), and at the training-loop hook site
(applies stage transitions when ``step`` reaches them).

``freeze_meta_at_step`` composes the last two into the single expression
both the save and the resume path use.

``trainable_during_run`` names every parameter that trains at some step of
a run, and ``trainable_for_the_run`` makes those parameters trainable while
the optimizer is built and a checkpoint's optimizer state is loaded, so a
module that a stage unfreezes is in the optimizer from the start.
"""

from __future__ import annotations

import contextlib
import fnmatch
from collections.abc import Iterable, Iterator, Mapping

import torch.nn as nn

from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig, is_parameter_pattern


def _always_frozen(model: nn.Module) -> set[int]:
    """The ids of the parameters that a module of ``model`` names in ``always_frozen``."""
    return {
        id(getattr(module, name))
        for module in model.modules()
        for name in getattr(module, "always_frozen", ())
    }


def freeze_params(
    model: nn.Module,
    patterns: Iterable[str],
    *,
    frozen: bool = True,
) -> int:
    """Toggle ``requires_grad`` on parameters matching any fnmatch pattern.

    Only parameters whose current state differs from the target are flipped,
    so calling this twice with the same arguments is idempotent. Returns the
    number of elements (``param.numel()`` summed) that were actually flipped.
    A parameter that its module names in ``always_frozen``, such as the sigmoid
    router's ``expert_bias``, is never unfrozen. A model compiled with
    ``torch.compile`` is matched through the module it wraps.
    """
    pats = list(patterns)
    if not pats:
        return 0
    target_requires_grad = not frozen
    stay_frozen = _always_frozen(model)
    n = 0
    for name, param in getattr(model, "_orig_mod", model).named_parameters():
        if target_requires_grad and id(param) in stay_frozen:
            continue
        if (
            any(fnmatch.fnmatch(name, pat) for pat in pats)
            and param.requires_grad != target_requires_grad
        ):
            param.requires_grad = target_requires_grad
            n += param.numel()
    return n


def apply_freeze_specs(
    model: nn.Module,
    specs: Iterable[FreezeSpec],
    pattern_map: Mapping[str, list[str]],
) -> dict[str, int]:
    """Apply a list of freeze specs to ``model``.

    For each spec, the ``module`` field is looked up in ``pattern_map``; if
    present, its pattern list is used. Otherwise ``module`` is treated as a
    raw fnmatch pattern. Returns ``{spec.module: n_params_flipped}``.
    """
    totals: dict[str, int] = {}
    for spec in specs:
        patterns = pattern_map.get(spec.module, [spec.module])
        totals[spec.module] = freeze_params(model, patterns, frozen=spec.frozen)
    return totals


def canonical_freeze_meta(specs: Iterable[FreezeSpec]) -> list[dict[str, object]]:
    """Return a sorted, deduplicated serialization of freeze specs.

    The output is safe to JSON-encode and compare across runs: two
    semantically equivalent freeze-spec lists (same ``(module, frozen)``
    pairs, any order or duplicates) produce byte-equal JSON.
    """
    as_tuples = sorted({(s.module, s.frozen) for s in specs})
    return [{"module": m, "frozen": f} for (m, f) in as_tuples]


def effective_freeze(
    step: int,
    base: Iterable[FreezeSpec],
    schedule: Iterable[FreezeStage],
    valid_modules: set[str] | None = None,
) -> list[FreezeSpec]:
    """Compute the active freeze-spec list at ``step``.

    Resolution rule:

    - Start from ``base`` (build-time freeze list).
    - For each ``FreezeStage`` with ``start_step <= step``, in
      ascending ``start_step`` order, override ``base`` entries on
      conflicting ``module`` keys (last-write-wins). Stages with
      ``start_step > step`` are ignored.
    - A key moves to the end of the list each time it is written, so
      the specs come in the order of their last write. Applied in that
      order, they leave each parameter as applying ``base`` and every
      stage up to ``step`` one after another does, also when keys
      select overlapping parameters.

    Module-key validation:

    - When ``valid_modules`` is provided, every ``spec.module``
      referenced in ``base`` or in any applied stage must appear in
      the set or be a parameter pattern (see ``is_parameter_pattern``),
      as the build accepts; otherwise ``ValueError``. This catches typos
      at load time rather than at the next step boundary.

    Args:
        step: The step the freeze is wanted at.
        base: The build-time freeze list.
        schedule: The freeze stages, in any order.
        valid_modules: The module aliases a ``spec.module`` may name besides a parameter
            pattern, or ``None`` to skip the check.

    Returns:
        The active specs, one per module key, in the order of their last write.

    Raises:
        ValueError: A spec names neither a module of ``valid_modules`` nor a parameter
            pattern.
    """
    by_module: dict[str, FreezeSpec] = {}

    def _check(spec: FreezeSpec, where: str) -> None:
        if valid_modules is None or spec.module in valid_modules:
            return
        if is_parameter_pattern(spec.module):
            return
        raise ValueError(
            f"FreezeSpec.module={spec.module!r} (from {where}) is neither a known module "
            f"alias {sorted(valid_modules)} nor a parameter pattern, which holds a dot or "
            "one of *?["
        )

    for spec in base:
        _check(spec, "base")
        by_module.pop(spec.module, None)
        by_module[spec.module] = spec

    for stage in sorted(schedule, key=lambda s: s.start_step):
        if stage.start_step > step:
            break
        for spec in stage.specs:
            _check(spec, f"FreezeStage(start_step={stage.start_step})")
            by_module.pop(spec.module, None)
            by_module[spec.module] = spec

    return list(by_module.values())


def unmatched_patterns(
    model: nn.Module,
    specs: Iterable[FreezeSpec],
    pattern_map: Mapping[str, list[str]],
) -> list[str]:
    """Return the parameter patterns among ``specs`` that match no parameter of ``model``.

    A spec whose ``module`` is a key of ``pattern_map`` is an alias and is not checked,
    since an alias such as ``vision_encoder`` can name a module without parameters.
    Names are matched as ``freeze_params`` matches them.

    Args:
        model: The model the specs apply to.
        specs: The freeze specs to check.
        pattern_map: The module aliases and their patterns.

    Returns:
        The patterns, sorted and without repeats, that freeze or unfreeze nothing.
    """
    names = [name for name, _ in getattr(model, "_orig_mod", model).named_parameters()]
    return sorted(
        {
            spec.module
            for spec in specs
            if spec.module not in pattern_map
            and not any(fnmatch.fnmatch(name, spec.module) for name in names)
        }
    )


def freeze_meta_at_step(step: int, config: VLMConfig) -> list[dict[str, object]]:
    """Canonical freeze metadata for ``config`` at ``step``.

    Save and resume must derive this identically — otherwise the
    ``vlm_freeze_expected`` compare rejects checkpoints this code wrote — so
    both go through here. Passing the config's alias set means a TOML typo
    raises instead of silently no-op'ing the freeze.
    """
    valid_modules = set(config.module_patterns.keys())
    return canonical_freeze_meta(
        effective_freeze(step, config.freeze, config.freeze_schedule, valid_modules)
    )


def _apply_to_names(
    requires_grad: dict[str, bool],
    specs: Iterable[FreezeSpec],
    pattern_map: Mapping[str, list[str]],
) -> None:
    """Set ``requires_grad`` by parameter name, as ``apply_freeze_specs`` sets it on a model."""
    for spec in specs:
        patterns = pattern_map.get(spec.module, [spec.module])
        for name in requires_grad:
            if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                requires_grad[name] = not spec.frozen


def trainable_during_run(model: nn.Module, config: VLMConfig) -> set[str]:
    """Return the names of the parameters of ``model`` that train at some step of a run.

    Covers the build freeze, each ``FreezeStage`` from its ``start_step``, and the freeze a
    resume applies at any step. A parameter that its module names in ``always_frozen`` never
    trains, so it is left out. ``model`` is not changed.

    Args:
        model: The model, with each parameter's ``requires_grad`` as built.
        config: The run's VLM config.

    Returns:
        The names of the parameters that require grad at some step of the run.
    """
    patterns = config.module_patterns
    built = {name: param.requires_grad for name, param in model.named_parameters()}
    _apply_to_names(built, config.freeze, patterns)
    trainable = {name for name, on in built.items() if on}
    resumed_before_stages = dict(built)
    _apply_to_names(resumed_before_stages, effective_freeze(0, config.freeze, []), patterns)
    trainable.update(name for name, on in resumed_before_stages.items() if on)
    running = dict(built)
    for stage in sorted(config.freeze_schedule, key=lambda s: s.start_step):
        _apply_to_names(running, stage.specs, patterns)
        resumed = dict(built)
        _apply_to_names(
            resumed,
            effective_freeze(stage.start_step, config.freeze, config.freeze_schedule),
            patterns,
        )
        trainable.update(name for name, on in running.items() if on)
        trainable.update(name for name, on in resumed.items() if on)
    stay_frozen = _always_frozen(model)
    return {
        name
        for name, param in model.named_parameters()
        if name in trainable and id(param) not in stay_frozen
    }


@contextlib.contextmanager
def trainable_for_the_run(model: nn.Module, config: VLMConfig) -> Iterator[list[str]]:
    """Make every parameter that trains at some step of the run require grad, then restore.

    Build the optimizer, and load a checkpoint, inside this context so that a parameter a
    ``FreezeStage`` unfreezes later is in the optimizer from the start.

    Args:
        model: The model, compiled with ``torch.compile`` or not.
        config: The run's VLM config.

    Yields:
        The names of the parameters it made require grad.
    """
    inner = getattr(model, "_orig_mod", model)
    names = trainable_during_run(inner, config)
    flipped = [
        (name, p) for name, p in inner.named_parameters() if name in names and not p.requires_grad
    ]
    for _, param in flipped:
        param.requires_grad_(True)
    try:
        yield [name for name, _ in flipped]
    finally:
        for _, param in flipped:
            param.requires_grad_(False)
