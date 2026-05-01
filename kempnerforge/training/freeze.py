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
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping

import torch.nn as nn

from kempnerforge.config.vlm import FreezeSpec, FreezeStage


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

    Module-key validation:

    - When ``valid_modules`` is provided, every ``spec.module``
      referenced in ``base`` or in any applied stage must appear in
      the set; otherwise ``ValueError``. This catches typos at load
      time rather than at the next step boundary.

    Returns the list of active specs (one per module key).
    """
    by_module: dict[str, FreezeSpec] = {}

    def _check(spec: FreezeSpec, where: str) -> None:
        if valid_modules is not None and spec.module not in valid_modules:
            raise ValueError(
                f"FreezeSpec.module={spec.module!r} (from {where}) "
                f"not in known module aliases {sorted(valid_modules)}"
            )

    for spec in base:
        _check(spec, "base")
        by_module[spec.module] = spec

    for stage in sorted(schedule, key=lambda s: s.start_step):
        if stage.start_step > step:
            break
        for spec in stage.specs:
            _check(spec, f"FreezeStage(start_step={stage.start_step})")
            by_module[spec.module] = spec

    return list(by_module.values())
