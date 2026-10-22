# pyright: reportMissingImports=false
# ^ mlflow is an optional dependency group; CI type-checks without it,
#   so the lazy import below would otherwise raise reportMissingImports.
"""Append eval metrics to the MLflow run a checkpoint was trained under.

A checkpoint's ``metadata.json`` records the tracking backends that were active when
it was saved. ``publish_metrics`` reads the MLflow entry and appends the metrics to
that run at the checkpoint's step in a single ``log_batch`` request. Inputs are fully
validated before anything is sent, but MLflow's ``log_batch`` permits partial writes,
so a request that fails server side may still have stored some points. The publisher
never starts, ends, or edits a run and keeps no state of its own, so several publishers
can write to one run at once and the tracking server arbitrates. The call blocks on the
server under MLflow's own HTTP timeout and retry settings, and authentication is
whatever the MLflow client finds in the environment. It is an offline tool: call it from
one process, never from the training step.

Command line: ``python -m kempnerforge.metrics.publish <checkpoint_dir> --metrics <json>``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import numbers
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kempnerforge.metrics.logger import get_logger

logger = get_logger(__name__)

_SIDECAR_FILE = "metadata.json"
_MLFLOW_BACKEND = "mlflow"
# A larger batch is split by MLflow into requests that can succeed or fail separately.
_MAX_METRICS_PER_CALL = 1000
# MLflow errors quote tracking URIs verbatim, and a URI can carry credentials anywhere in it.
_URL = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s'\"<>]+")


class PublishRefused(ValueError):
    """Nothing was published: the sidecar, the metrics, or the environment is unusable."""


@dataclass(frozen=True)
class MlflowTarget:
    """The MLflow run a checkpoint was trained under, as its sidecar records it."""

    run_id: str
    experiment: str
    tracking_uri: str = field(repr=False)  # may embed credentials: keep it out of logs
    step: int
    model_id: str | None = None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate key {key!r}")
        parsed[key] = value
    return parsed


def _parse_json(raw: bytes) -> Any:
    """Parse JSON, refusing a repeated key rather than silently keeping the last one."""
    return json.loads(raw, object_pairs_hook=_no_duplicate_keys)


def _string_problem(entry: Mapping[str, Any], key: str) -> str | None:
    """Why ``entry[key]`` is not a non-empty string, or None when it is one."""
    if key not in entry:
        return f"'{key}' is missing"
    value = entry[key]
    if not isinstance(value, str):
        return f"'{key}' must be a string, got {type(value).__name__}"
    if not value.strip():
        return f"'{key}' is empty"
    return None


def _load_sidecar(directory: Path) -> tuple[Path, dict[str, Any]]:
    sidecar = directory / _SIDECAR_FILE
    if not directory.is_dir():
        raise PublishRefused(f"checkpoint directory {str(directory)!r} is not a directory")
    if not sidecar.is_file():
        raise PublishRefused(f"{sidecar} not found; pass the checkpoint step directory")
    try:
        meta = _parse_json(sidecar.read_bytes())
    except (OSError, ValueError) as e:
        raise PublishRefused(f"{sidecar} is unreadable or malformed JSON: {e}") from e
    if not isinstance(meta, dict):
        raise PublishRefused(f"{sidecar}: top level must be an object, got {type(meta).__name__}")
    return sidecar, meta


def _select_mlflow_entry(sidecar: Path, meta: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """The single ``backend == "mlflow"`` entry of the ``tracking`` list, with its index."""
    if "tracking" not in meta:
        raise PublishRefused(
            f"{sidecar}: no 'tracking' block; this checkpoint does not record a tracking backend"
        )
    tracking = meta["tracking"]
    if not isinstance(tracking, list):
        raise PublishRefused(
            f"{sidecar}: 'tracking' must be a list of backend entries, got "
            f"{type(tracking).__name__}"
        )
    backends: list[str] = []
    found: list[tuple[int, dict[str, Any]]] = []
    for i, entry in enumerate(tracking):
        if not isinstance(entry, dict):
            raise PublishRefused(
                f"{sidecar}: tracking[{i}] must be an object, got {type(entry).__name__}"
            )
        if problem := _string_problem(entry, "backend"):
            raise PublishRefused(f"{sidecar}: tracking[{i}]: {problem}")
        backends.append(entry["backend"])
        if entry["backend"] == _MLFLOW_BACKEND:
            found.append((i, entry))
    if not found:
        raise PublishRefused(
            f"{sidecar}: 'tracking' has no mlflow entry "
            f"(backends recorded: {', '.join(backends) or 'none'})"
        )
    if len(found) > 1:
        indexes = ", ".join(str(i) for i, _ in found)
        raise PublishRefused(
            f"{sidecar}: 'tracking' has {len(found)} mlflow entries (tracking[{indexes}]); "
            "cannot tell which run to use"
        )
    return found[0]


def read_mlflow_target(checkpoint_dir: str | os.PathLike[str]) -> MlflowTarget:
    """Read the MLflow run a checkpoint was trained under from its ``metadata.json``.

    The sidecar needs a non-negative integer ``step`` and a ``tracking`` list holding
    exactly one entry with ``backend == "mlflow"`` whose ``run_id``, ``experiment`` and
    ``tracking_uri`` are non-empty strings. ``model_id`` is optional. A key repeated
    anywhere in the file is refused, since the last value would silently win. A
    ``databricks://<profile>`` URI also selects that profile via
    DATABRICKS_CONFIG_PROFILE (setdefault), so artifact clients authenticate like the
    tracking client.

    Raises:
        PublishRefused: Naming each missing or malformed item; nothing is inferred.
    """
    sidecar, meta = _load_sidecar(Path(checkpoint_dir))
    index, entry = _select_mlflow_entry(sidecar, meta)

    problems: list[str] = []
    step = meta.get("step")
    if "step" not in meta:
        problems.append("'step' is missing")
    elif isinstance(step, bool) or not isinstance(step, int):
        problems.append(f"'step' must be an integer, got {type(step).__name__}")
    elif step < 0:
        problems.append(f"'step' must be non-negative, got {step}")

    fields = ["run_id", "experiment", "tracking_uri"]
    if entry.get("model_id") is not None:
        fields.append("model_id")
    for key in fields:
        if problem := _string_problem(entry, key):
            problems.append(f"tracking[{index}]: {problem}")
    if problems:
        raise PublishRefused(f"{sidecar}: " + "; ".join(problems))

    _export_profile(entry["tracking_uri"])
    return MlflowTarget(
        run_id=entry["run_id"],
        experiment=entry["experiment"],
        tracking_uri=entry["tracking_uri"],
        step=meta["step"],
        model_id=entry.get("model_id"),
    )


def _export_profile(tracking_uri: str) -> None:
    """Select the ``databricks://<profile>`` profile via DATABRICKS_CONFIG_PROFILE (setdefault).

    Artifact clients build their own SDK config from the environment and never see the
    tracking URI, so without this a profile-authenticated publish can upload metrics but
    not artifacts. ``databricks://`` and the ``databricks://scope:prefix`` secret-scope
    form select nothing; an operator-set value always wins.
    """
    rest = tracking_uri.removeprefix("databricks://")
    if rest != tracking_uri and rest and ":" not in rest:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", rest)


def _as_finite_float(value: object) -> float:
    """``value`` as a float; raises ValueError saying why it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("is too large for a float") from None
    if not math.isfinite(number):
        raise ValueError(f"must be finite, got {number}")
    return number


def _prefixed_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    """Validate ``metrics`` and return them keyed ``<prefix><name>`` as floats."""
    if not isinstance(prefix, str):
        raise PublishRefused(f"prefix must be a string, got {type(prefix).__name__}")
    if not isinstance(metrics, Mapping):
        raise PublishRefused(
            f"metrics must be a mapping of name to number, got {type(metrics).__name__}"
        )
    if not metrics:
        raise PublishRefused("no metrics to publish: the mapping is empty")
    if len(metrics) > _MAX_METRICS_PER_CALL:
        raise PublishRefused(
            f"{len(metrics)} metrics exceed the {_MAX_METRICS_PER_CALL} one request can carry; "
            "a larger batch would be split into requests that can succeed or fail separately"
        )
    named: dict[str, float] = {}
    problems: list[str] = []
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            problems.append(f"metric name {name!r} must be a non-empty string")
            continue
        try:
            named[prefix + name] = _as_finite_float(value)
        except ValueError as e:
            problems.append(f"metric {name!r} {e}")
    if problems:
        raise PublishRefused("; ".join(problems))
    return named


def _now_ms() -> int:
    return int(time.time() * 1000)


def publish_metrics(
    checkpoint_dir: str | os.PathLike[str],
    metrics: Mapping[str, float],
    prefix: str = "eval/",
) -> MlflowTarget:
    """Append ``metrics`` to the MLflow run a checkpoint was trained under.

    Every metric is logged as ``<prefix><name>`` at the checkpoint's step by one
    ``MlflowClient.log_batch`` request, on a client bound to the sidecar's tracking URI
    (the process-wide MLflow tracking URI is left alone). When the sidecar entry carries
    a ``model_id`` and the installed MLflow can link metrics to a logged model, the
    metrics are linked to it; otherwise the ``model_id`` is ignored.

    Args:
        checkpoint_dir: Checkpoint step directory holding ``metadata.json``.
        metrics: Metric name to finite number, at most 1000 of them (one request).
        prefix: Prepended verbatim to every name; may be empty.

    Returns:
        The target the metrics were appended to.

    Raises:
        PublishRefused: The sidecar or the metrics are missing or malformed, mlflow is
            not installed, or the sidecar has no model_id while this process has an
            active MLflow model (which ``log_batch`` would silently link); nothing was
            sent. Errors raised by MLflow itself (an invalid metric name, an
            unreachable server) propagate as raised; a request that fails server side
            may still have stored some of its points.
    """
    target = read_mlflow_target(checkpoint_dir)
    named = _prefixed_metrics(metrics, prefix)
    try:
        import mlflow
        from mlflow.entities import Metric
    except ImportError as e:
        raise PublishRefused(
            f"mlflow could not be imported ({e}); install it with: uv sync --group mlflow"
        ) from e

    link: dict[str, str] = {}
    if target.model_id is not None:
        if "model_id" in inspect.signature(Metric).parameters:
            link["model_id"] = target.model_id
        else:
            logger.debug("installed mlflow cannot link metrics to a model; ignoring model_id")
    else:
        # log_batch fills model_id-less metrics with the process's active model
        # (mlflow.get_active_model_id); refuse rather than mislink to an unrelated model.
        get_ambient = getattr(mlflow, "get_active_model_id", None)
        ambient = get_ambient() if get_ambient is not None else None
        if ambient:
            raise PublishRefused(
                "the sidecar entry has no model_id but this process has an active MLflow "
                f"model ({ambient}); log_batch would silently link the metrics to it. Clear "
                "the active model (MLFLOW_ACTIVE_MODEL_ID / mlflow.set_active_model) or "
                "record model_id in the sidecar"
            )

    timestamp = _now_ms()
    batch = [
        Metric(key=key, value=value, timestamp=timestamp, step=target.step, **link)
        for key, value in named.items()
    ]
    client = mlflow.MlflowClient(tracking_uri=target.tracking_uri)
    client.log_batch(target.run_id, metrics=batch, synchronous=True)
    return target


def _mask_urls(text: str) -> str:
    """``text`` with everything after each ``scheme://`` hidden."""
    return _URL.sub(r"\1<redacted>", text)


def _load_metrics_file(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return _parse_json(f.read())
    except (OSError, ValueError) as e:
        raise PublishRefused(f"cannot read metrics file {path!r}: {e}") from e


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code: 0 published, 2 refused, 1 failed."""
    parser = argparse.ArgumentParser(
        prog="python -m kempnerforge.metrics.publish",
        description="Append eval metrics to the MLflow run recorded in a checkpoint's "
        "metadata.json, at the checkpoint's step.",
    )
    parser.add_argument("checkpoint_dir", help="checkpoint step directory holding metadata.json")
    parser.add_argument(
        "--metrics",
        required=True,
        metavar="FILE",
        help="JSON file holding an object that maps metric name to number",
    )
    parser.add_argument(
        "--prefix", default="eval/", help="prepended to every metric name (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    try:
        metrics = _load_metrics_file(args.metrics)
        target = publish_metrics(args.checkpoint_dir, metrics, prefix=args.prefix)
    except PublishRefused as e:
        print(f"refused: {_mask_urls(str(e))}", file=sys.stderr)
        return 2
    except Exception as e:  # the MLflow client or server failed: one line, no traceback
        print(f"failed: {type(e).__name__}: {_mask_urls(str(e))}", file=sys.stderr)
        return 1
    print(f"published {len(metrics)} metric(s) to run {target.run_id} at step {target.step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
