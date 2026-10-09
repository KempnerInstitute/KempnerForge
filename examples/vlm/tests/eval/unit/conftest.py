"""Hermetic fake ``lmms_eval`` for the VLM-eval unit tests, scoped to this directory.

``lmms-eval`` is an optional, undeclared dependency, so ``lmms_adapter.py`` cannot be
imported without it and these tests would otherwise skip wherever it is absent. This
conftest installs a faithful in-repo fake (``_fake_lmms_eval``) into ``sys.modules`` so the
tests always run and exercise our code. The fake is installed unconditionally (hermetic):
unit-test behavior is identical with or without real lmms-eval present. Real-package
fidelity is pinned separately by the gated contract test in
``../integration/test_lmms_eval_contract.py``.

Installing it has to happen at import time, because the test modules here bind
``lmms_adapter`` while they are collected, and the adapter binds ``lmms_eval`` at its own
module scope. Leaving it installed would hand the fake to every other test sharing the
session, so collection's end lifts the overlay and the autouse fixture reapplies it for one
test at a time. ``tests/test_configs.py`` pins the absence.

The integration tier binds the real package and still wants a session of its own (see the
README): while a test here runs, the overlay redirects the ``sys.modules`` entries that its
``monkeypatch.setattr`` targets resolve through.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

from . import _fake_lmms_eval

_ADAPTER_MODULES = ("lmms_adapter",)
_FAKE_MODULES = _fake_lmms_eval.build_modules()
_MANAGED = (*_FAKE_MODULES.keys(), *_ADAPTER_MODULES)

# What ``sys.modules`` held before this conftest touched it, and what this tier needs in
# its place. The tier's own entries are captured once collection has imported them.
_ORIGINALS: dict[str, ModuleType | None] = {name: sys.modules.get(name) for name in _MANAGED}
_TIER: dict[str, ModuleType | None] = {}


def _apply(table: dict[str, ModuleType | None]) -> None:
    for name, module in table.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


# Install the fakes and evict any real-bound adapter, so it re-imports against them when
# the test modules below are collected.
sys.modules.update(_FAKE_MODULES)
for _name in _ADAPTER_MODULES:
    sys.modules.pop(_name, None)


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Collection is over, so every module here is imported: stop sharing the overlay."""
    del session, config, items
    _TIER.update({name: sys.modules.get(name) for name in _MANAGED})
    _apply(_ORIGINALS)


@pytest.fixture(autouse=True)
def _lmms_eval_doubles() -> Iterator[None]:
    """Hold the overlay for one test in this directory, then lift it again."""
    _apply(_TIER)
    yield
    _apply(_ORIGINALS)
