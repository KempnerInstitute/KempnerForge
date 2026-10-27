"""Fakes shared by the unit tests of the MLflow checkpoint models (no server, no mlflow needed)."""

from __future__ import annotations

import itertools
import os
import sys
import threading
import types

import pytest

EXTERNAL_TAG = "mlflow.model.isExternal"
PAGE_LIMIT = 50


class FakeModelStore:
    """An MLflow server's runs and logged models, reached through ``client(tracking_uri)``.

    The client has the five ``MlflowClient`` calls a checkpoint model needs. ``calls`` lists
    each as (call, tracking URI, thread id), ``page_sizes`` the ``max_results`` of each search,
    and ``uploads`` the files uploaded per model id. ``fail(call, error)`` makes a call raise;
    ``hold(call)`` parks it, after setting ``entered``, until ``release`` is set.
    """

    def __init__(self) -> None:
        self.runs = {"run-xyz": "7"}
        self.models: list[types.SimpleNamespace] = []
        self.calls: list[tuple[str, str | None, int]] = []
        self.page_sizes: list[int | None] = []
        self.uploads: dict[str, dict[str, str]] = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self._failures: dict[str, Exception] = {}
        self._held: set[str] = set()
        self._clock = itertools.count(1000)

    def add(self, name, run_id="run-xyz", status="READY", experiment_id="7"):
        """Put a model on the server, as an earlier registration would have; return it."""
        model_id = f"m-{len(self.models) + 1}"
        model = types.SimpleNamespace(
            model_id=model_id,
            name=name,
            experiment_id=experiment_id,
            source_run_id=run_id,
            status=status,
            creation_timestamp=next(self._clock),
            artifact_location=f"store://models/{model_id}/artifacts",
            tags={},
            params={},
            model_type=None,
        )
        self.models.append(model)
        return model

    def fail(self, call: str, error: Exception) -> None:
        self._failures[call] = error

    def recover(self) -> None:
        self._failures.clear()

    def hold(self, call: str) -> None:
        self._held.add(call)

    def named(self, call: str) -> list[tuple[str, str | None, int]]:
        return [c for c in self.calls if c[0] == call]

    def client(self, tracking_uri=None):
        return _FakeModelClient(self, tracking_uri)

    def _enter(self, call: str, tracking_uri) -> None:
        self.calls.append((call, tracking_uri, threading.get_ident()))
        if call in self._held:
            self.entered.set()
            assert self.release.wait(10), f"{call} was never released"
        if call in self._failures:
            raise self._failures[call]


class _Page(list):
    def __init__(self, items, token):
        super().__init__(items)
        self.token = token


class _FakeModelClient:
    def __init__(self, store: FakeModelStore, tracking_uri) -> None:
        self._store = store
        self._uri = tracking_uri

    def get_run(self, run_id):
        self._store._enter("get_run", self._uri)
        info = types.SimpleNamespace(run_id=run_id, experiment_id=self._store.runs[run_id])
        return types.SimpleNamespace(info=info)

    def search_logged_models(
        self,
        experiment_ids,
        filter_string=None,
        datasets=None,
        max_results=None,
        order_by=None,
        page_token=None,
    ):
        store = self._store
        store._enter("search_logged_models", self._uri)
        store.page_sizes.append(max_results)
        if max_results is None or max_results > PAGE_LIMIT:
            raise RuntimeError(f"INVALID_PARAMETER_VALUE: max_results must be at most {PAGE_LIMIT}")
        assert filter_string is not None and filter_string.startswith("name = '")
        name = filter_string.removeprefix("name = '").removesuffix("'")
        found = [m for m in store.models if m.name == name and m.experiment_id in experiment_ids]
        start = int(page_token or 0)
        end = start + max_results
        return _Page(found[start:end], str(end) if end < len(found) else None)

    def create_logged_model(
        self, experiment_id, name=None, source_run_id=None, tags=None, params=None, model_type=None
    ):
        store = self._store
        store._enter("create_logged_model", self._uri)
        model = store.add(name, run_id=source_run_id, status="PENDING", experiment_id=experiment_id)
        model.tags, model.params, model.model_type = (
            dict(tags or {}),
            dict(params or {}),
            model_type,
        )
        return model

    def log_model_artifacts(self, model_id, local_dir):
        self._store._enter("log_model_artifacts", self._uri)
        files = {}
        for name in os.listdir(local_dir):
            with open(os.path.join(local_dir, name)) as f:
                files[name] = f.read()
        self._store.uploads[model_id] = files

    def finalize_logged_model(self, model_id, status):
        self._store._enter("finalize_logged_model", self._uri)
        (model,) = [m for m in self._store.models if m.model_id == model_id]
        model.status = str(status)
        return model


class _FakeModelSpec:
    """Stands in for the MLmodel descriptor ``get_external_mlflow_model_spec`` builds."""

    def __init__(self, model) -> None:
        self._model = model

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(
                f"model_id: {self._model.model_id}\nrun_id: {self._model.source_run_id}\n"
                f"metadata:\n  {EXTERNAL_TAG}: true\n"
            )


@pytest.fixture
def model_store(monkeypatch) -> FakeModelStore:
    """A fresh fake model store, with the mlflow modules that describe an external model."""
    modules = {
        "mlflow.models.model": {"MLMODEL_FILE_NAME": "MLmodel"},
        "mlflow.models.utils": {"get_external_mlflow_model_spec": _FakeModelSpec},
        "mlflow.utils.mlflow_tags": {"MLFLOW_MODEL_IS_EXTERNAL": EXTERNAL_TAG},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    return FakeModelStore()
