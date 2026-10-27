# pyright: reportMissingImports=false
"""The MLflow logged model of a checkpoint: one external model per run and step.

The training backend registers each committed checkpoint through this module, and the eval
publisher finds or makes the model a checkpoint's ``metadata.json`` names. Both pass a client
bound to the tracking URI they mean, so the process-wide tracking URI is never read or changed.
mlflow is imported only when a model is created.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

_NAME_UNSAFE = str.maketrans(dict.fromkeys("/:.%\"'", "_"))
_SEARCH_PAGE_SIZE = 50
_READY = "READY"
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
    """Id of the READY external model ``name`` of run ``run_id``: an existing one, else a new one.

    ``client`` is an ``mlflow.MlflowClient``; every call goes through it. Of several READY models
    of the run with that name, the earliest is used, so a checkpoint registered again (saved
    twice, re-run after a resume, or published) keeps one model; a model an interrupted attempt
    left unfinished is never reused. After making a model it reads the earliest READY one again,
    so callers that make one at the same time settle on the same model. A new model is made the
    way ``mlflow.create_external_model`` makes one: tagged external, typed ``checkpoint``, with
    params ``step`` and ``checkpoint_dir``, MLflow's descriptor of an external model uploaded,
    then finalized READY. Server errors propagate.
    """
    experiment_id = client.get_run(run_id).info.experiment_id
    model_id = _earliest_ready_model(client, experiment_id, run_id, name)
    if model_id is None:
        params = {"step": str(step), "checkpoint_dir": checkpoint_dir}
        created = _create_external_model(client, experiment_id, run_id, name, params)
        model_id = _earliest_ready_model(client, experiment_id, run_id, name) or created
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("the server returned no model id")
    return model_id


def _earliest_ready_model(client: Any, experiment_id: str, run_id: str, name: str) -> str | None:
    """The id of the run's earliest READY model ``name``, read from every page of the search."""
    ready: list[Any] = []
    token = None
    while True:
        page = client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"name = '{name}'",
            max_results=_SEARCH_PAGE_SIZE,
            page_token=token,
        )
        ready += [m for m in page if m.source_run_id == run_id and m.status == _READY]
        token = page.token
        if not token:
            break
    if not ready:
        return None
    return min(ready, key=lambda m: (m.creation_timestamp, m.model_id)).model_id


def _create_external_model(
    client: Any, experiment_id: str, run_id: str, name: str, params: dict[str, str]
) -> str:
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.models.utils import get_external_mlflow_model_spec
    from mlflow.utils.mlflow_tags import MLFLOW_MODEL_IS_EXTERNAL

    model = client.create_logged_model(
        experiment_id=experiment_id,
        name=name,
        source_run_id=run_id,
        tags={MLFLOW_MODEL_IS_EXTERNAL: "true"},
        params=params,
        model_type=_MODEL_TYPE,
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        get_external_mlflow_model_spec(model).save(os.path.join(directory, MLMODEL_FILE_NAME))
        client.log_model_artifacts(model.model_id, directory)
    client.finalize_logged_model(model.model_id, _READY)
    return model.model_id
