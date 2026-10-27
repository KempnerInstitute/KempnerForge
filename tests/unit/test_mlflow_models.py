"""The MLflow logged model of a checkpoint: naming, and find-or-create over a fake store."""

from __future__ import annotations

import errno
import inspect
import os
import shutil

import pytest

from kempnerforge.metrics.mlflow_tracking.models import (
    checkpoint_model_name,
    find_or_create_checkpoint_model,
)
from tests.unit.conftest import EXTERNAL_TAG, PAGE_LIMIT


def _find_or_create(store):
    return find_or_create_checkpoint_model(
        store.client("uri-a"), "run-xyz", "run-a-step_5", 5, "ckpts/step_5"
    )


class TestCheckpointModelName:
    @pytest.mark.parametrize(
        ("run_label", "name"),
        [
            ("7b-debug", "7b-debug-step_5"),
            ("lr 3e-4 (v2)", "lr 3e-4 (v2)-step_5"),
            ("a.b/c:d%e\"f'g", "a_b_c_d_e_f_g-step_5"),
        ],
        ids=["plain", "spaces", "rejected-characters"],
    )
    def test_is_the_run_label_and_the_step(self, run_label, name):
        assert checkpoint_model_name(run_label, 5) == name


class TestFindOrCreateCheckpointModel:
    def test_creates_a_ready_external_model_of_the_checkpoint(self, model_store):
        model_id = _find_or_create(model_store)
        (model,) = model_store.models
        assert model_id == model.model_id
        assert (model.name, model.source_run_id, model.experiment_id) == (
            "run-a-step_5",
            "run-xyz",
            "7",
        )
        assert model.model_type == "checkpoint"
        assert model.params == {"step": "5", "checkpoint_dir": "ckpts/step_5"}
        assert model.tags == {EXTERNAL_TAG: "true"}
        assert model.status == "READY"
        mlmodel = model_store.uploads[model_id]["MLmodel"]
        assert f"model_id: {model_id}" in mlmodel and f"{EXTERNAL_TAG}: true" in mlmodel
        assert [call for call, _, _ in model_store.calls] == [
            "get_run",
            "search_logged_models",
            "create_logged_model",
            "log_model_artifacts",
            "finalize_logged_model",
            "search_logged_models",
        ]
        assert {uri for _, uri, _ in model_store.calls} == {"uri-a"}

    def test_settles_on_a_model_made_earlier_while_it_made_its_own(self, model_store, monkeypatch):
        rival: list = []
        add = model_store.add

        def rival_first(name, **kwargs):
            if kwargs.get("status") == "PENDING" and not rival:
                rival.append(add(name))
            return add(name, **kwargs)

        monkeypatch.setattr(model_store, "add", rival_first)
        assert _find_or_create(model_store) == rival[0].model_id
        assert [m.status for m in model_store.models] == ["READY", "READY"]

    def test_keeps_its_own_model_when_the_search_does_not_list_it_yet(self, model_store):
        client = model_store.client("uri-a")
        search = client.search_logged_models
        searched: list = []

        def lagging(*args, **kwargs):
            searched.append(kwargs)
            page = search(*args, **kwargs)
            return page if len(searched) == 1 else type(page)([], None)

        client.search_logged_models = lagging
        model_id = find_or_create_checkpoint_model(
            client, "run-xyz", "run-a-step_5", 5, "ckpts/step_5"
        )
        (model,) = model_store.models
        assert model_id == model.model_id and model.status == "READY"
        assert len(searched) == 2

    def test_a_temp_dir_that_cannot_be_removed_still_leaves_a_ready_model(
        self, model_store, monkeypatch
    ):
        left: list = []

        def busy(path, *args, **kwargs):
            left.append(path)
            raise OSError(errno.ENOTEMPTY, "Directory not empty", path)

        monkeypatch.setattr(os, "rmdir", busy)
        model_id = _find_or_create(model_store)
        monkeypatch.undo()
        for path in left:
            if os.path.isabs(path):
                shutil.rmtree(path, ignore_errors=True)
        (model,) = model_store.models
        assert model_id == model.model_id and model.status == "READY"
        assert left

    def test_reuses_the_runs_ready_model_without_creating_one(self, model_store):
        model = model_store.add("run-a-step_5")
        assert _find_or_create(model_store) == model.model_id
        assert model_store.named("create_logged_model") == []
        assert len(model_store.models) == 1

    def test_takes_the_earliest_of_several_ready_models(self, model_store):
        first = model_store.add("run-a-step_5")
        model_store.add("run-a-step_5")
        model_store.models.reverse()
        assert _find_or_create(model_store) == first.model_id

    @pytest.mark.parametrize(
        ("held", "why"),
        [
            ({"status": "PENDING"}, "left unfinished"),
            ({"status": "FAILED"}, "failed"),
            ({"run_id": "other-run"}, "another run's"),
            ({"experiment_id": "8"}, "another experiment's"),
        ],
        ids=["pending", "failed", "other-run", "other-experiment"],
    )
    def test_never_reuses_a_model_that_is_not_the_runs_ready_one(self, model_store, held, why):
        model_store.runs["other-run"] = "7"
        stale = model_store.add("run-a-step_5", **held)
        model_id = _find_or_create(model_store)
        assert model_id != stale.model_id, why
        assert len(model_store.named("create_logged_model")) == 1

    def test_reads_every_page_and_never_asks_for_more_than_the_limit(self, model_store):
        model_store.runs["other-run"] = "7"
        for _ in range(2 * PAGE_LIMIT + 3):
            model_store.add("run-a-step_5", run_id="other-run")
        mine = model_store.add("run-a-step_5")
        assert _find_or_create(model_store) == mine.model_id
        assert model_store.page_sizes == [PAGE_LIMIT] * 3
        assert model_store.named("create_logged_model") == []

    @pytest.mark.parametrize("model_id", ["", "  ", None])
    def test_a_blank_model_id_from_the_server_is_an_error(self, model_store, monkeypatch, model_id):
        add = model_store.add

        def add_blank(*args, **kwargs):
            model = add(*args, **kwargs)
            model.model_id = model_id
            return model

        monkeypatch.setattr(model_store, "add", add_blank)
        with pytest.raises(ValueError, match="no model id"):
            _find_or_create(model_store)

    @pytest.mark.parametrize(
        "call",
        [
            "get_run",
            "search_logged_models",
            "create_logged_model",
            "log_model_artifacts",
            "finalize_logged_model",
        ],
    )
    def test_server_errors_propagate(self, model_store, call):
        model_store.fail(call, RuntimeError(f"{call} refused"))
        with pytest.raises(RuntimeError, match=f"{call} refused"):
            _find_or_create(model_store)
        assert all(m.status != "READY" for m in model_store.models)

    def test_the_calls_it_makes_exist_in_the_installed_mlflow(self):
        mlflow = pytest.importorskip("mlflow")
        from mlflow.models.model import MLMODEL_FILE_NAME
        from mlflow.models.utils import get_external_mlflow_model_spec
        from mlflow.utils.mlflow_tags import MLFLOW_MODEL_IS_EXTERNAL

        def parameters(fn):
            return set(inspect.signature(fn).parameters)

        client = mlflow.MlflowClient
        assert "run_id" in parameters(client.get_run)
        assert {"experiment_ids", "filter_string", "max_results", "page_token"} <= parameters(
            client.search_logged_models
        )
        assert {
            "experiment_id",
            "name",
            "source_run_id",
            "tags",
            "params",
            "model_type",
        } <= parameters(client.create_logged_model)
        assert {"model_id", "local_dir"} <= parameters(client.log_model_artifacts)
        assert {"model_id", "status"} <= parameters(client.finalize_logged_model)
        assert callable(get_external_mlflow_model_spec)
        assert (MLMODEL_FILE_NAME, MLFLOW_MODEL_IS_EXTERNAL) == ("MLmodel", EXTERNAL_TAG)
