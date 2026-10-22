"""Publish a checkpoint's lmms-eval scores to the MLflow run it was trained under.

``eval.py`` calls this after scoring. The run comes from the checkpoint's ``metadata.json``
and each score is appended through ``kempnerforge.metrics.publish`` as
``eval/<task>_<metric>`` at the checkpoint's step (``_<filter>`` follows when lmms-eval
reports a filter other than ``none``). A summary table and ``scores.json`` go to the same
run under ``eval/step_<N>/<UTC timestamp>/``: one directory per evaluation, so repeated or
concurrent evaluations never append to or overwrite each other. No run is started, ended
or edited.

Publishing never fails a scoring run: the scores are written to a local file first, and
every problem with MLflow is a warning that says how to publish the metrics later.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import numbers
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kempnerforge.metrics.publish import (
    MlflowTarget,
    PublishRefused,
    publish_metrics,
    read_mlflow_target,
)

logger = logging.getLogger(__name__)

_TABLE_COLUMNS = ["step", "benchmark", "metric", "filter", "score", "stderr", "n_samples", "name"]
# MLflow rejects names outside letters, digits, '_', '-', '.', ' ', ':' and '/', and lmms-eval
# has tasks such as 'refcoco+_bbox_val' and metrics such as 'avg@4'.
_NAME_SEPARATORS = re.compile(r"[^A-Za-z0-9.-]+")
# MLflow errors quote tracking URIs verbatim, and a URI can carry credentials anywhere in it.
_URL = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s'\"<>]+")
_DEPENDENCY_GROUPS = {"mlflow": "mlflow", "pandas": "eval"}


@dataclass(frozen=True)
class Score:
    """One lmms-eval metric of one task: a row of the summary table."""

    benchmark: str
    metric: str
    filter: str
    value: float
    stderr: float | None
    n_samples: int | None

    @property
    def name(self) -> str:
        """The published metric name, without the publisher's ``eval/`` prefix."""
        parts = [self.benchmark, self.metric]
        if self.filter != "none":
            parts.append(self.filter)
        return _NAME_SEPARATORS.sub("_", "_".join(parts)).strip("_")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _real(value: object) -> float | None:
    """``value`` as a float, or None when it is not a real number (``"N/A"``, a bool)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except OverflowError:
        return None


def _finite(value: object) -> float | None:
    number = _real(value)
    return number if number is not None and math.isfinite(number) else None


def _bypassed(results: Mapping[str, Any], task: str, seen: tuple[str, ...] = ()) -> set[str]:
    """The metrics of ``task`` (and of its subtasks, for a group) that lmms-eval aggregates
    with ``bypass``: their value is the placeholder 999 for a score computed elsewhere."""
    if task in seen:
        return set()
    entries = _mapping(_mapping(results.get("configs")).get(task)).get("metric_list")
    found = {
        str(entry.get("metric"))
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, Mapping) and entry.get("aggregation") == "bypass"
    }
    children = _mapping(results.get("group_subtasks")).get(task)
    for child in children if isinstance(children, list) else []:
        found |= _bypassed(results, str(child), (*seen, task))
    return found


def collect_scores(results: Mapping[str, Any]) -> list[Score]:
    """The scores in an lmms-eval results dict, ordered by task, metric and filter.

    A key ``"<metric>,<filter>"`` of ``results["results"][task]`` is a score when the task
    declares ``<metric>`` in ``higher_is_better`` and the value is a number: that leaves out
    ``alias``, the ``*_stderr`` and stability variants, ``"N/A"``, and ``bypass`` metrics. A
    NaN or infinite score is left out with a warning, since MLflow takes no such value.

    Raises:
        ValueError: Two scores would be published under one name.
    """
    declared = _mapping(results.get("higher_is_better"))
    counts = _mapping(results.get("n-samples"))
    scores: list[Score] = []
    names: dict[str, Score] = {}
    for task, entry in sorted(_mapping(results.get("results")).items()):
        values = _mapping(entry)
        metrics = _mapping(declared.get(task))
        effective = _mapping(counts.get(task)).get("effective")
        n_samples = effective if type(effective) is int else None
        placeholders = _bypassed(results, task)
        for key, value in sorted(values.items()):
            metric, comma, filter_name = key.partition(",")
            if not comma or metric not in metrics or metric in placeholders:
                continue
            if (number := _real(value)) is None:
                continue
            if not math.isfinite(number):
                logger.warning(f"Not publishing {task}/{key}: the score is {number}")
                continue
            stderr = _finite(values.get(f"{metric}_stderr,{filter_name}"))
            score = Score(task, metric, filter_name, number, stderr, n_samples)
            if (other := names.setdefault(score.name, score)) is not score:
                raise ValueError(
                    f"{other.benchmark}/{other.metric}/{other.filter} and "
                    f"{task}/{metric}/{filter_name} would both be published as {score.name!r}"
                )
            scores.append(score)
    return scores


def score_mapping(scores: Sequence[Score]) -> dict[str, float]:
    """Metric name to value, by name: what is published and what ``scores.json`` holds."""
    return {s.name: s.value for s in sorted(scores, key=lambda s: s.name)}


def scores_path_for(output: str | None, checkpoint: Path) -> Path:
    """Where the local scores file goes: beside ``output``, else in the working directory."""
    if output:
        return Path(output).with_suffix(".scores.json")
    return Path(f"{checkpoint.parent.name}_{checkpoint.name}.scores.json")


def _mask_urls(text: str) -> str:
    return _URL.sub(r"\1<redacted>", text)


def _failure(error: Exception) -> str:
    return f"{type(error).__name__}: {_mask_urls(str(error))}"


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module} could not be imported ({e}); "
            f"install it with: uv sync --group {_DEPENDENCY_GROUPS[module]}"
        ) from e


def preflight(checkpoint: Path, limit: int | float | None) -> None:
    """Before scoring: say whether the scores can be published, while that can be fixed."""
    problems: list[str] = []
    target: MlflowTarget | None = None
    try:
        target = read_mlflow_target(checkpoint)
    except PublishRefused as e:
        problems.append(str(e))
    for module in _DEPENDENCY_GROUPS:
        try:
            _import(module)
        except ImportError as e:
            problems.append(str(e))
    if target is None or problems:
        logger.warning(
            f"MLflow publishing will not work for this run: {'; '.join(problems)}. "
            "Scoring continues; pass --no-publish to silence this."
        )
        return
    logger.info(
        f"Scores will be published to MLflow run {target.run_id} at step {target.step}; "
        "pass --no-publish to keep them local"
    )
    if limit is not None:
        logger.warning(f"--limit {limit}: the published scores will cover a subset of each task")


def _save(path: Path, mapping: Mapping[str, float]) -> Path | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, indent=2) + "\n")
    except OSError as e:
        logger.warning(f"Could not save the scores to {path}: {e}")
        return None
    logger.info(f"Scores saved to {path}")
    return path


def _not_published(why: str, checkpoint: Path, saved: Path | None) -> None:
    logger.warning(f"The scores were NOT published to MLflow. {why}")
    if saved is not None:
        ckpt, scores = (shlex.quote(str(p.resolve())) for p in (checkpoint, saved))
        logger.warning(
            "To publish the metrics later (the summary table and scores.json are not part of "
            f"it): python -m kempnerforge.metrics.publish {ckpt} --metrics {scores}"
        )


def _upload_table(client: Any, target: MlflowTarget, scores: Sequence[Score], path: str) -> None:
    rows = [
        (target.step, s.benchmark, s.metric, s.filter, s.value, s.stderr, s.n_samples, s.name)
        for s in scores
    ]
    table = _import("pandas").DataFrame(rows, columns=_TABLE_COLUMNS)
    client.log_table(target.run_id, data=table, artifact_file=path)


def _upload_scores(client: Any, target: MlflowTarget, scores: Sequence[Score], path: str) -> None:
    client.log_dict(target.run_id, score_mapping(scores), path)


def _upload_artifacts(target: MlflowTarget, scores: Sequence[Score], directory: str) -> list[str]:
    """Upload the summary table and scores.json; one problem per upload that failed."""
    try:
        client = _import("mlflow").MlflowClient(tracking_uri=target.tracking_uri)
    except Exception as e:
        return [f"the upload of the summary table and scores.json failed: {_failure(e)}"]
    problems: list[str] = []
    for what, upload, file in (
        ("summary table", _upload_table, "summary.json"),
        ("scores.json", _upload_scores, "scores.json"),
    ):
        try:
            upload(client, target, scores, f"{directory}/{file}")
        except Exception as e:
            problems.append(f"the upload of the {what} failed: {_failure(e)}")
    return problems


def publish_results(checkpoint: Path, results: Mapping[str, Any], output: str | None) -> bool:
    """Save the scores of ``results`` locally, then publish them; True if all of it landed.

    ``checkpoint`` is the step directory the scored weights came from. A problem with the
    scores or with MLflow is a warning, never an exception.
    """
    try:
        scores = collect_scores(results)
    except ValueError as e:
        logger.warning(f"The scores were not saved or published: {e}")
        return False
    if not scores:
        logger.warning(
            "No scores to publish: no task in the results declares a numeric metric "
            "under 'higher_is_better'"
        )
        return False

    mapping = score_mapping(scores)
    saved = _save(scores_path_for(output, checkpoint), mapping)
    try:
        target = publish_metrics(checkpoint, mapping)
    except PublishRefused as e:
        _not_published(f"Refused: {e}", checkpoint, saved)
        return False
    except Exception as e:
        _not_published(f"Failed: {_failure(e)}", checkpoint, saved)
        return False

    directory = f"eval/step_{target.step}/{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}"
    problems = _upload_artifacts(target, scores, directory)
    for problem in problems:
        logger.warning(f"The metrics were published to MLflow, but {problem}")
    if not problems:
        logger.info(
            f"Published {len(mapping)} scores to MLflow run {target.run_id} at step "
            f"{target.step}, with the summary table and scores.json in {directory}"
        )
    return not problems
