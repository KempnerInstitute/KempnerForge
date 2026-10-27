# pyright: reportMissingImports=false
"""The MLflow logged model of a checkpoint: one external model per run and step.

The training backend registers each committed checkpoint through this module, and the eval
publisher finds or makes the model a checkpoint's ``metadata.json`` names. Both pass a client
bound to the tracking URI they mean, so the process-wide tracking URI is never read or changed.
mlflow is imported only when a model is created or finished.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

_NAME_UNSAFE = str.maketrans(dict.fromkeys("/:.%\"'", "_"))
_SEARCH_PAGE_SIZE = 50
_READY = "READY"
_PENDING = "PENDING"
_MODEL_TYPE = "checkpoint"


def checkpoint_model_name(run_label: str, step: int) -> str:
    """The logged model name of the checkpoint saved at ``step`` by the run named ``run_label``.

    ``run_label`` is the run's name, or its id when it has none. Characters MLflow rejects in a
    model name become ``_``.
    """
    return f"{run_label.translate(_NAME_UNSAFE)}-step_{step}"


def find_or_create_checkpoint_model(
    client: Any, run_id: str, name: str, step: int, checkpoint_dir: str
) -> str:
    """Id of the external model ``name`` of run ``run_id`` that every caller settles on.

    ``client`` is an ``mlflow.MlflowClient``; every call goes through it. That model is the
    earliest created (by creation time, then id) of the run's READY and PENDING models with that
    name, read from every page of the search; a FAILED or UNSPECIFIED one is never used. So a
    checkpoint registered again (saved twice, re-run after a resume, or published) keeps one
    model, and callers that race to make it settle on the one made first. Nothing waits: a
    PENDING model, still being made by another caller or left by one that died, is finished
    here by uploading the MLmodel descriptor built from that model and finalizing it READY,
    unless it already reads READY, so a caller that arrives after the model is finished does not
    write it again. A finalize that raises is ignored when the model reads READY afterwards.

    A model is made only when the search lists none to use, the way
    ``mlflow.create_external_model`` makes one: tagged external, typed ``checkpoint``, with
    params ``step`` and ``checkpoint_dir``, MLflow's descriptor of an external model uploaded,
    then finalized READY. The search is then run again, and the new model is returned only if
    it lists none to use. Server errors propagate.

    MLflow has no atomic create-if-absent, so this narrows the race rather than closing it: a
    model stamped with an earlier creation time that becomes visible to searches only after
    another caller has settled leaves the two on different models.
    """
    experiment_id = client.get_run(run_id).info.experiment_id
    model_id = _settled_model_id(client, experiment_id, run_id, name)
    if model_id is None:
        params = {"step": str(step), "checkpoint_dir": checkpoint_dir}
        created = _create_external_model(client, experiment_id, run_id, name, params)
        model_id = _settled_model_id(client, experiment_id, run_id, name) or created
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("the server returned no model id")
    return model_id


def _settled_model_id(client: Any, experiment_id: str, run_id: str, name: str) -> str | None:
    """The id of the model callers settle on, finished READY if PENDING; None if none is listed."""
    model = _earliest_live_model(client, experiment_id, run_id, name)
    if model is None:
        return None
    if model.status == _PENDING:
        _finish_external_model(client, model)
    return model.model_id


def _earliest_live_model(client: Any, experiment_id: str, run_id: str, name: str) -> Any | None:
    """The run's earliest created READY or PENDING model ``name``, read from every page."""
    live: list[Any] = []
    token = None
    while True:
        page = client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"name = '{name}'",
            max_results=_SEARCH_PAGE_SIZE,
            page_token=token,
        )
        live += [m for m in page if m.source_run_id == run_id and m.status in (_READY, _PENDING)]
        token = page.token
        if not token:
            break
    return min(live, key=lambda m: (m.creation_timestamp, m.model_id), default=None)


def _create_external_model(
    client: Any, experiment_id: str, run_id: str, name: str, params: dict[str, str]
) -> str:
    from mlflow.utils.mlflow_tags import MLFLOW_MODEL_IS_EXTERNAL

    model = client.create_logged_model(
        experiment_id=experiment_id,
        name=name,
        source_run_id=run_id,
        tags={MLFLOW_MODEL_IS_EXTERNAL: "true"},
        params=params,
        model_type=_MODEL_TYPE,
    )
    _finish_external_model(client, model)
    return model.model_id


def _finish_external_model(client: Any, model: Any) -> None:
    """Upload the external MLmodel descriptor built from ``model``, then finalize it READY.

    Does nothing when the model already reads READY, so a caller that arrives after another one
    finished it does not write its descriptor again; one that read PENDING just before can still
    rewrite it. A finalize that raises is ignored when the model then reads READY, and re-raised
    otherwise.
    """
    if client.get_logged_model(model.model_id).status == _READY:
        return
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.models.utils import get_external_mlflow_model_spec

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        get_external_mlflow_model_spec(model).save(os.path.join(directory, MLMODEL_FILE_NAME))
        client.log_model_artifacts(model.model_id, directory)
    try:
        client.finalize_logged_model(model.model_id, _READY)
    except Exception:
        if client.get_logged_model(model.model_id).status != _READY:
            raise
