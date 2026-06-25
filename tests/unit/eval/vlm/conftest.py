"""Hermetic fake ``lmms_eval`` for the VLM-eval unit tests.

``lmms-eval`` is an optional, undeclared dependency, so
``adapter.py``/``registry.py`` cannot be imported without it and these tests would skip in
CI (no coverage). This conftest installs a faithful in-repo fake (``_fake_lmms_eval``) into
``sys.modules`` at import time so the tests always run and exercise our code. The fake is
installed unconditionally (hermetic): unit-test behavior is identical with or without real
lmms-eval present. Real-package fidelity is pinned separately by the gated contract test in
``tests/integration/`` (``test_lmms_eval_contract``).

CI runs ``tests/unit/`` and ``tests/integration/`` as separate jobs, so the injected fake
never reaches the real-lmms-eval integration tests. For a combined local ``pytest tests/``
run, ``tests/integration/`` is collected before ``tests/unit/`` (default ordering) and binds
the real package first; the saved originals are restored at session end.
"""

from __future__ import annotations

import sys

import pytest

from . import _fake_lmms_eval

_ADAPTER_MODULES = ("kempnerforge.eval.vlm.adapter", "kempnerforge.eval.vlm.registry")
_FAKE_MODULES = _fake_lmms_eval.build_modules()
_MANAGED = (*_FAKE_MODULES.keys(), *_ADAPTER_MODULES)
_SAVED = {name: sys.modules.get(name) for name in _MANAGED}

# Install the fakes and evict any real-bound adapter/registry so they re-import against the
# fakes when the test modules import them.
sys.modules.update(_FAKE_MODULES)
for _name in _ADAPTER_MODULES:
    sys.modules.pop(_name, None)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session, exitstatus
    for name in _MANAGED:
        saved = _SAVED[name]
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
