"""The MLflow logged model of a checkpoint: naming, and find-or-create over a fake store."""

from __future__ import annotations

import errno
import inspect
import os
import shutil
import threading

import pytest

from kempnerforge.metrics.mlflow_models import (
    checkpoint_model_name,
    find_or_create_checkpoint_model,
)
from tests.unit.conftest import EXTERNAL_TAG, PAGE_LIMIT


def _find_or_create(store, client=None):
    return find_or_create_checkpoint_model(
        client or store.client("uri-a"), "run-xyz", "run-a-step_5", 5, "ckpts/step_5"
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
            "get_logged_model",
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
            ({"status": "FAILED"}, "failed"),
            ({"run_id": "other-run"}, "another run's"),
            ({"experiment_id": "8"}, "another experiment's"),
        ],
        ids=["failed", "other-run", "other-experiment"],
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
            "get_logged_model",
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
        from mlflow.entities import LoggedModelStatus
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
        assert "model_id" in parameters(client.get_logged_model)
        assert callable(get_external_mlflow_model_spec)
        assert (MLMODEL_FILE_NAME, MLFLOW_MODEL_IS_EXTERNAL) == ("MLmodel", EXTERNAL_TAG)
        assert (
            LoggedModelStatus.UNSPECIFIED,
            LoggedModelStatus.PENDING,
            LoggedModelStatus.READY,
            LoggedModelStatus.FAILED,
        ) == ("UNSPECIFIED", "PENDING", "READY", "FAILED")


class _Race:
    """Callers A and B of one checkpoint model: both search, then A creates first and parks.

    B runs in the test's thread through ``client_b``. Its first search finds nothing, and when
    it goes to create, A starts in thread ``a``, finds nothing either, creates the first model
    and parks just before uploading its descriptor until ``resume_a`` is called. ``returned``
    collects the id each caller returns.
    """

    def __init__(self, store) -> None:
        self.client_a, self.client_b = store.client("uri-a"), store.client("uri-a")
        self.returned: dict[str, str] = {}
        self._created, self._resumed = threading.Event(), threading.Event()
        upload_a, create_b = self.client_a.log_model_artifacts, self.client_b.create_logged_model

        def a_parks_after_creating(model_id, local_dir):
            self._created.set()
            assert self._resumed.wait(10)
            upload_a(model_id, local_dir)

        def a_creates_first(*args, **kwargs):
            self.a.start()
            assert self._created.wait(10)
            return create_b(*args, **kwargs)

        self.client_a.log_model_artifacts = a_parks_after_creating
        self.client_b.create_logged_model = a_creates_first
        self.a = threading.Thread(
            target=lambda: self.returned.update(a=_find_or_create(store, self.client_a))
        )

    def a_parked(self) -> bool:
        return self.a.is_alive() and not self._resumed.is_set()

    def resume_a(self) -> None:
        self._resumed.set()
        if self.a.is_alive():
            self.a.join(10)


class TestCallersSettleOnOneModel:
    """Callers that find or make the same model at once, or after a creator died, agree on it."""

    def test_a_caller_that_made_one_first_but_finalizes_last_gets_the_same_model(self, model_store):
        race = _Race(model_store)
        upload_b = race.client_b.log_model_artifacts
        uploaded_by_b: list = []

        def a_finalizes_while_b_finishes_its_model(model_id, local_dir):
            uploaded_by_b.append(model_id)
            if model_id == "m-1":
                race.resume_a()
            upload_b(model_id, local_dir)

        race.client_b.log_model_artifacts = a_finalizes_while_b_finishes_its_model
        try:
            race.returned["b"] = _find_or_create(model_store, race.client_b)
        finally:
            race.resume_a()
        assert race.returned == {"a": "m-1", "b": "m-1"}
        assert [(m.model_id, m.status) for m in model_store.models] == [
            ("m-1", "READY"),
            ("m-2", "READY"),
        ]
        assert uploaded_by_b == ["m-2", "m-1"]
        b, a = threading.get_ident(), race.a.ident
        assert [t for call, _, t in model_store.calls if call == "finalize_logged_model"] == [
            b,
            a,
            b,
        ]

    def test_a_caller_does_not_wait_for_a_creator_that_finishes_after_it_returned(
        self, model_store
    ):
        race = _Race(model_store)
        try:
            race.returned["b"] = _find_or_create(model_store, race.client_b)
            assert race.a_parked()
            first = model_store.models[0]
            assert (first.model_id, first.status) == ("m-1", "READY")
            assert "model_id: m-1" in model_store.uploads["m-1"]["MLmodel"]
        finally:
            race.resume_a()
        assert race.returned == {"a": "m-1", "b": "m-1"}
        b = threading.get_ident()
        assert [call for call, _, t in model_store.calls if t == b] == [
            "get_run",
            "search_logged_models",
            "create_logged_model",
            "get_logged_model",
            "log_model_artifacts",
            "finalize_logged_model",
            "search_logged_models",
            "get_logged_model",
            "log_model_artifacts",
            "finalize_logged_model",
        ]

    def test_a_creator_that_resumes_after_its_model_was_finished_writes_nothing_more(
        self, model_store
    ):
        client_a, client_b = model_store.client("uri-a"), model_store.client("uri-a")
        created, resumed = threading.Event(), threading.Event()
        create_a = client_a.create_logged_model

        def a_parks_after_creating(*args, **kwargs):
            model = create_a(*args, **kwargs)
            created.set()
            assert resumed.wait(10)
            return model

        client_a.create_logged_model = a_parks_after_creating
        returned: dict[str, str] = {}
        a = threading.Thread(
            target=lambda: returned.update(a=_find_or_create(model_store, client_a))
        )
        a.start()
        try:
            assert created.wait(10)
            returned["b"] = _find_or_create(model_store, client_b)
        finally:
            resumed.set()
            a.join(10)
        assert returned == {"a": "m-1", "b": "m-1"}
        a_calls = [call for call, _, t in model_store.calls if t == a.ident]
        assert "log_model_artifacts" not in a_calls and "finalize_logged_model" not in a_calls
        assert len(model_store.named("log_model_artifacts")) == 1

    def test_a_model_its_creator_left_pending_is_finished_by_the_next_caller(self, model_store):
        left = model_store.add("run-a-step_5", status="PENDING")
        assert _find_or_create(model_store) == left.model_id
        assert left.status == "READY" and len(model_store.models) == 1
        assert f"model_id: {left.model_id}" in model_store.uploads[left.model_id]["MLmodel"]
        assert [call for call, _, _ in model_store.calls] == [
            "get_run",
            "search_logged_models",
            "get_logged_model",
            "log_model_artifacts",
            "finalize_logged_model",
        ]

    def test_two_callers_that_each_made_one_both_return_the_one_made_first(self, model_store):
        clients = {"a": model_store.client("uri-a"), "b": model_store.client("uri-a")}
        both_searched = threading.Barrier(2, timeout=10)
        both_made = threading.Barrier(2, timeout=10)
        a_created = threading.Event()
        returned: dict[str, str] = {}

        def in_step(caller, client):
            search, create = client.search_logged_models, client.create_logged_model
            searches: list = []

            def search_together(*args, **kwargs):
                searches.append(kwargs)
                if len(searches) == 2:
                    both_made.wait()
                page = search(*args, **kwargs)
                if len(searches) == 1:
                    both_searched.wait()
                return page

            def a_creates_first(*args, **kwargs):
                assert caller == "a" or a_created.wait(10)
                model = create(*args, **kwargs)
                a_created.set()
                return model

            client.search_logged_models = search_together
            client.create_logged_model = a_creates_first

        def call(caller):
            returned[caller] = _find_or_create(model_store, clients[caller])

        for caller, client in clients.items():
            in_step(caller, client)
        threads = [threading.Thread(target=call, args=(caller,)) for caller in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert returned == {"a": "m-1", "b": "m-1"}
        assert [(m.model_id, m.status) for m in model_store.models] == [
            ("m-1", "READY"),
            ("m-2", "READY"),
        ]

    @pytest.mark.parametrize("status", ["FAILED", "UNSPECIFIED"])
    def test_a_failed_or_unspecified_model_is_never_chosen(self, model_store, status):
        skipped = model_store.add("run-a-step_5", status=status)
        pending = model_store.add("run-a-step_5", status="PENDING")
        assert _find_or_create(model_store) == pending.model_id
        assert (skipped.status, pending.status) == (status, "READY")
        assert list(model_store.uploads) == [pending.model_id]
        assert model_store.named("create_logged_model") == []

    def test_when_the_second_search_lists_nothing_the_caller_returns_its_own(self, model_store):
        client = model_store.client("uri-a")
        search = client.search_logged_models

        def lagging(*args, **kwargs):
            page = search(*args, **kwargs)
            first = len(model_store.named("search_logged_models")) == 1
            return page if first else type(page)([], None)

        client.search_logged_models = lagging
        model_id = _find_or_create(model_store, client)
        (model,) = model_store.models
        assert model_id == model.model_id and model.status == "READY"
        assert [call for call, _, _ in model_store.calls] == [
            "get_run",
            "search_logged_models",
            "create_logged_model",
            "get_logged_model",
            "log_model_artifacts",
            "finalize_logged_model",
            "search_logged_models",
        ]

    @pytest.mark.parametrize("found_pending", [False, True], ids=["its-own", "found-pending"])
    def test_a_finalize_refused_because_the_model_is_already_ready_is_tolerated(
        self, model_store, found_pending
    ):
        if found_pending:
            model_store.add("run-a-step_5", status="PENDING")
        client = model_store.client("uri-a")
        upload, finalize = client.log_model_artifacts, client.finalize_logged_model

        def another_caller_finalizes_it_meanwhile(model_id, local_dir):
            upload(model_id, local_dir)
            model_store.client("uri-a").finalize_logged_model(model_id, "READY")

        def refused_once_ready(model_id, status):
            (model,) = [m for m in model_store.models if m.model_id == model_id]
            if model.status == "READY":
                raise RuntimeError("INVALID_STATE: the logged model is already finalized")
            return finalize(model_id, status)

        client.log_model_artifacts = another_caller_finalizes_it_meanwhile
        client.finalize_logged_model = refused_once_ready
        model_id = _find_or_create(model_store, client)
        (model,) = model_store.models
        assert model_id == model.model_id and model.status == "READY"
        assert len(model_store.named("get_logged_model")) == 2

    @pytest.mark.parametrize("found_pending", [False, True], ids=["its-own", "found-pending"])
    @pytest.mark.parametrize("status", ["PENDING", "FAILED"])
    def test_any_other_finalize_error_propagates(self, model_store, found_pending, status):
        if found_pending:
            model_store.add("run-a-step_5", status="PENDING")
        client = model_store.client("uri-a")

        def refused(model_id, _):
            (model,) = [m for m in model_store.models if m.model_id == model_id]
            model.status = status
            raise RuntimeError("finalize refused")

        client.finalize_logged_model = refused
        with pytest.raises(RuntimeError, match="finalize refused"):
            _find_or_create(model_store, client)
        assert all(m.status != "READY" for m in model_store.models)
