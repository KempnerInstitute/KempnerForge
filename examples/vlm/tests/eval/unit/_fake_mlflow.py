"""Strict fakes of the ``mlflow`` and ``pandas`` surface the eval publisher touches.

Neither package is needed to run the tests: they are installed into ``sys.modules`` per test.
``FakeMlflow`` offers ``MlflowClient(tracking_uri=...)`` with ``log_batch``, ``log_table`` and
``log_dict`` (signatures as MLflow's), ``entities.Metric`` and ``get_active_model_id``; any
other attribute raises AssertionError, so a publisher that starts, ends or edits a run fails
its test. ``FakePandas`` is a ``DataFrame`` that records its rows and columns.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from collections.abc import Callable, Mapping
from typing import Any

import pytest


class _Metric:
    """Mirrors ``mlflow.entities.Metric``, logged-model linkage included."""

    def __init__(self, key, value, timestamp, step, model_id=None):
        self.key, self.value, self.timestamp, self.step = key, value, timestamp, step
        self.model_id = model_id


class FakeMlflow(types.ModuleType):
    """Stand-in ``mlflow`` recording what a publisher does.

    ``fail`` maps a client method name to the exception that call raises, and ``on_call``
    is told each method name just before it runs.
    """

    def __init__(
        self,
        fail: Mapping[str, Exception] | None = None,
        on_call: Callable[[str], None] | None = None,
    ):
        super().__init__("mlflow")
        self.events: list[str] = []
        self.client_uris: list[Any] = []
        self.batches: list[types.SimpleNamespace] = []
        self.tables: list[types.SimpleNamespace] = []
        self.dicts: list[types.SimpleNamespace] = []
        entities = types.ModuleType("mlflow.entities")
        entities.Metric = _Metric  # type: ignore[attr-defined]
        self.entities_module = entities
        fake = self
        failures = dict(fail or {})

        def enter(method: str) -> None:
            fake.events.append(method)
            if on_call is not None:
                on_call(method)
            if method in failures:
                raise failures[method]

        class MlflowClient:
            def __init__(self, tracking_uri=None, registry_uri=None):
                enter("client")
                fake.client_uris.append(tracking_uri)
                self._uri = tracking_uri

            def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=None):
                enter("log_batch")
                fake.batches.append(
                    types.SimpleNamespace(
                        run_id=run_id, metrics=list(metrics), tracking_uri=self._uri
                    )
                )

            def log_table(self, run_id, data, artifact_file):
                enter("log_table")
                fake.tables.append(
                    types.SimpleNamespace(run_id=run_id, data=data, artifact_file=artifact_file)
                )

            def log_dict(self, run_id, dictionary, artifact_file):
                enter("log_dict")
                fake.dicts.append(
                    types.SimpleNamespace(
                        run_id=run_id, dictionary=dictionary, artifact_file=artifact_file
                    )
                )

            def __getattr__(self, name):
                raise AssertionError(f"the publisher touched MlflowClient.{name}")

        self.MlflowClient = MlflowClient

    def get_active_model_id(self) -> None:
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"the publisher touched mlflow.{name}")


class FakePandas(types.ModuleType):
    """Stand-in ``pandas`` whose ``DataFrame`` keeps what it was built from."""

    class DataFrame:
        def __init__(self, data: Any, columns: list[str]):
            self.columns = list(columns)
            self.rows = [tuple(row) for row in data]

    def __init__(self):
        super().__init__("pandas")


def install_mlflow(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> FakeMlflow:
    fake = FakeMlflow(**kwargs)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.entities", fake.entities_module)
    return fake


def install_pandas(monkeypatch: pytest.MonkeyPatch) -> FakePandas:
    fake = FakePandas()
    monkeypatch.setitem(sys.modules, "pandas", fake)
    return fake


def uninstall(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    """Make ``import <module>`` raise ImportError, whether or not it is installed."""
    monkeypatch.setitem(sys.modules, module, None)


class _BrokenFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, module: str, error: Exception):
        self._module, self._error = module, error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._module:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise self._error


def break_import(monkeypatch: pytest.MonkeyPatch, module: str, error: Exception) -> None:
    """Make ``import <module>`` run and then raise ``error``, as a broken install does."""
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenFinder(module, error), *sys.meta_path])
