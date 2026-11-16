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
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping

import torch.nn as nn

from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig, is_parameter_pattern


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
    """
    pats = list(patterns)
    if not pats:
        return 0
    target_requires_grad = not frozen
    n = 0
    for name, param in model.named_parameters():
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
