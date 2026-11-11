"""Evaluates a saved checkpoint on evaluation tasks.

``evaluate_checkpoint`` decodes an answer for every row of each task's split
with the checkpoint's model, scores it with the task's scorer and writes the
task metrics and the per-row predictions. Under several processes each rank
decodes its own batches of a split and rank 0 merges them, so the results are
those of one process.

A row is scored as it is or counted under a reason, never changed to fit:

- ``unreadable_row``: the reader could not use the row and passed over it.
- ``prompt_too_long``: the prompt and the task's ``max_new_tokens`` do not fit
  ``vlm.max_text_len``, the text length training lays out.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from image_text.document import Document, Turn
from image_text.render import ChatRenderer, DocumentSource
from PIL import Image

from kempnerforge.config.distributed import DistributedConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.distributed.setup import destroy_distributed, get_world_info, init_distributed
from kempnerforge.distributed.utils import agree_across_ranks
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.mlflow.publish import PublishRefused, publish_metrics
from kempnerforge.model.vlm import VLMWrapper
from kempnerforge.training.runtime import apply_matmul_precision

from .checkpoint import check_supported, load_model, read_step, resolve_checkpoint
from .generation import greedy_decode, stop_token_ids
from .inputs import (
    ModelInputs,
    UnsupportedDocumentError,
    build_renderer,
    document_inputs,
    prompt_token_ids,
)
from .scorers import get_scorer
from .tasks import EvalTask, build_reader

logger = get_logger(__name__)

RESULTS_FILE = "results.json"
METRICS_FILE = "metrics.json"
PREDICTIONS_DIR = "predictions"

UNREADABLE_ROW = "unreadable_row"
PROMPT_TOO_LONG = "prompt_too_long"

_PROGRESS_LINES = 10
_URL = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s'\"<>]+")
_ONE_READ_AT_A_TIME = threading.Lock()

Record = dict[str, Any]


@dataclass(frozen=True)
class Decoder:
    """The checkpoint's model with what a row is rendered, budgeted and ended by.

    The renderer's ``max_text_len`` is the text length training lays out, which
    a prompt and its answer must fit, and ``stop_ids`` holds the tokens that end
    a reply. The rows of a batch are read and rendered by ``threads`` threads.
    """

    model: VLMWrapper
    renderer: ChatRenderer
    stop_ids: frozenset[int]
    threads: int = 1


def check_token_budgets(renderer: ChatRenderer, tasks: Sequence[EvalTask]) -> None:
    """Refuses a task whose answer cap leaves no room for a prompt in the text length.

    A row is decoded when its prompt and the task's ``max_new_tokens`` fit the
    renderer's ``max_text_len``, the text length training lays out. No prompt is
    shorter than an empty question's, so a task that does not fit that one can
    decode no row.
    """
    blank = Document(images=(Image.new("RGB", (1, 1)),), turns=(Turn("", ("x",)),))
    shortest = len(prompt_token_ids(renderer, blank))
    for task in tasks:
        if shortest + task.max_new_tokens > renderer.max_text_len:
            raise ValueError(
                f"task {task.name!r}: max_new_tokens={task.max_new_tokens} leaves no room for a "
                f"prompt in vlm.max_text_len={renderer.max_text_len}; the prompt of an empty "
                f"question is already {shortest} tokens"
            )


def clean_generation(text: str) -> str:
    """The decoded generation as it is scored: without surrounding whitespace."""
    return text.strip()


def read_row(reader: DocumentSource, index: int) -> Document | None:
    """The document of row ``index``, or ``None`` when ``reader`` passed over that row.

    A reader passes over a row it cannot use and returns a later one, which is
    not the answer to this row. A reader with ``read_with_index`` says which row
    it read. Any other is read by index, one read at a time across threads, and
    its ``skipped`` count, when it has one, moves if it went on to another row, or
    if it gave up after passing over a run of rows (a ``RuntimeError``).
    """
    read_with_index = getattr(reader, "read_with_index", None)
    if read_with_index is not None:
        actual, document = read_with_index(index)
        return document if actual == index else None
    with _ONE_READ_AT_A_TIME:
        skipped = getattr(reader, "skipped", None)
        try:
            document = reader[index]
        except RuntimeError:
            if getattr(reader, "skipped", None) == skipped:
                raise
            return None
        return document if getattr(reader, "skipped", None) == skipped else None


def shard_batches(rows: int, batch_size: int, rank: int, world_size: int) -> list[range]:
    """The batches of a ``rows``-row split that ``rank`` of ``world_size`` decodes.

    The split is cut into consecutive batches of ``batch_size`` rows and the
    ranks take them in turn, so every row is in one batch of one rank, and a
    batch holds the same rows for any number of ranks.
    """
    starts = range(rank * batch_size, rows, world_size * batch_size)
    return [range(start, min(start + batch_size, rows)) for start in starts]


def evaluate_batch(
    decoder: Decoder, task: EvalTask, reader: DocumentSource, batch: range
) -> list[Record]:
    """One record for each row of ``batch``: its scored prediction, or why it was skipped.

    The rows that can be decoded are decoded together. A scored row records its
    ``question``, reference ``answers``, the decoded ``generation``, the
    ``prediction`` that was scored, whether the decode ``stopped`` on a stop
    token, and its ``score``. A skipped row records the reason under ``skipped``.

    Raises:
        UnsupportedDocumentError: a row holds a document ``document_inputs`` refuses.
    """
    prepare = partial(_prepare_row, decoder.renderer, task, reader)
    if decoder.threads > 1:
        with ThreadPoolExecutor(decoder.threads) as pool:
            prepared = list(pool.map(prepare, batch))
    else:
        prepared = [prepare(row) for row in batch]
    records = [row for row in prepared if isinstance(row, dict)]
    ready = [row for row in prepared if not isinstance(row, dict)]
    if not ready:
        return records
    renderer, scorer = decoder.renderer, get_scorer(task.scorer)
    generations = greedy_decode(
        decoder.model,
        torch.stack([inputs.pixel_values for _, _, inputs in ready]),
        [inputs.prompt_ids for _, _, inputs in ready],
        max_new_tokens=task.max_new_tokens,
        stop_ids=decoder.stop_ids,
        pad_id=renderer.pad_id,
        max_text_len=renderer.max_text_len,
    )
    for (row, turn, _), generation in zip(ready, generations, strict=True):
        text = renderer.tokenizer.decode(list(generation.token_ids), skip_special_tokens=False)
        prediction = clean_generation(text)
        score = scorer.score(prediction, turn.answers)
        records.append(
            {
                "row": row,
                "question": turn.user,
                "answers": list(turn.answers),
                "generation": text,
                "prediction": prediction,
                "stopped": generation.stopped,
                "score": dict(score) if isinstance(score, Mapping) else score,
            }
        )
    return records


def merge_records(per_rank: Sequence[Sequence[Record]], rows: int) -> list[Record]:
    """Every rank's records in row order, refused unless each row has exactly one."""
    merged = sorted((record for records in per_rank for record in records), key=_row)
    if [_row(record) for record in merged] != list(range(rows)):
        raise RuntimeError(
            f"the ranks returned {len(merged)} records for a split of {rows} rows; every row "
            "must be decoded exactly once"
        )
    return merged


def summarize(task: EvalTask, records: Sequence[Record]) -> Record:
    """The report of ``task`` from its rows' records: counts, skips by reason and metrics.

    The metrics are the scorer's aggregate of the scored rows, in row order.

    Raises:
        ValueError: no row was scored.
    """
    scored = [record for record in records if "skipped" not in record]
    skipped = Counter(record["skipped"] for record in records if "skipped" in record)
    if not scored:
        raise ValueError(
            f"task {task.name!r}: none of its {len(records)} rows was scored; skipped "
            f"{dict(skipped)}"
        )
    return {
        "scorer": task.scorer,
        "max_new_tokens": task.max_new_tokens,
        "rows": len(records),
        "samples": len(scored),
        "skipped": dict(sorted(skipped.items())),
        "reached_max_new_tokens": sum(not record["stopped"] for record in scored),
        "metrics": get_scorer(task.scorer).aggregate([record["score"] for record in scored]),
    }


def evaluate_checkpoint(
    config: JobConfig,
    tasks: Sequence[EvalTask],
    checkpoint: str | Path,
    output: str | Path,
    *,
    batch_size: int = 32,
    publish: bool = True,
) -> Record | None:
    """Evaluate ``checkpoint`` on ``tasks`` and write the results under ``output``.

    ``config`` is the config the checkpoint was trained with and ``checkpoint``
    a run directory, whose latest checkpoint is evaluated, or a step directory.
    The config, the tasks' token budgets and their splits are checked before a
    process group or the model exists. Launched as several processes, this joins
    them into a process group, or uses the one already initialized; each rank
    loads the model and decodes the batches ``shard_batches`` gives it, reading
    their rows with ``data.num_workers`` threads.

    Rank 0 writes ``predictions/<task>.jsonl`` (one line per row, in row order),
    ``metrics.json`` (``<task>/<metric>`` to value) and last ``results.json``,
    which is removed when an evaluation starts and so exists only once one has
    finished. With ``publish``, rank 0 then appends the metrics as
    ``eval/<task>/<metric>`` to the MLflow run the checkpoint was trained under;
    a failure there is logged and never raised.

    Returns:
        What ``results.json`` holds on rank 0, ``None`` on the other ranks.

    Raises:
        ValueError: the config, a task or ``batch_size`` is refused, or a task
            scored no row.
        FileNotFoundError: ``checkpoint`` holds no saved weights, or a row's
            image file is missing.
        UnsupportedDocumentError: a row holds a document outside the limits.
        RuntimeError: on the other ranks, when a rank failed.
    """
    _clear_results(output)
    renderer, readers = _on_every_rank(
        partial(_preflight, config, tasks, batch_size), "Checking the tasks"
    )

    started_group, rank, world_size, device = _join_ranks(config)
    apply_matmul_precision(config.train)
    output = Path(output)
    resolved = _on_rank_zero(rank, partial(_start, output, checkpoint), "Starting the evaluation")
    step_dir = Path(_gather(resolved)[0])
    model = _on_every_rank(partial(load_model, config, step_dir, device), "Loading the checkpoint")
    decoder = Decoder(
        model, renderer, stop_token_ids(renderer), threads=max(1, config.data.num_workers)
    )
    logger.info(
        f"Evaluating {step_dir} on {len(tasks)} task(s) with {world_size} rank(s) and "
        f"batches of {batch_size}"
    )

    summaries: dict[str, Record] = {}
    for task in tasks:
        reader = readers[task.name]
        batches = shard_batches(len(reader), batch_size, rank, world_size)
        records = _on_every_rank(
            partial(_evaluate_batches, decoder, task, reader, batches),
            f"Evaluating task {task.name!r}",
        )
        summary = _on_rank_zero(
            rank,
            partial(_finish_task, task, _gather(records), len(reader), output),
            f"Writing the results of task {task.name!r}",
        )
        if summary is not None:
            summaries[task.name] = summary

    results: Record = {"checkpoint": str(step_dir), "step": read_step(step_dir), "tasks": summaries}
    metrics = {
        f"{name}/{metric}": value
        for name, summary in summaries.items()
        for metric, value in summary["metrics"].items()
    }
    _on_rank_zero(rank, partial(_finish, output, results, metrics), "Writing the results")
    if rank == 0 and publish:
        _publish(step_dir, metrics, output / METRICS_FILE)
    if started_group:
        destroy_distributed()
    return results if rank == 0 else None


def _row(record: Record) -> int:
    return record["row"]


def _prepare_row(
    renderer: ChatRenderer, task: EvalTask, reader: DocumentSource, row: int
) -> Record | tuple[int, Turn, ModelInputs]:
    """The row's turn and model inputs, or the record of why it is skipped."""
    document = read_row(reader, row)
    if document is None:
        return {"row": row, "skipped": UNREADABLE_ROW}
    try:
        inputs = document_inputs(renderer, document)
    except UnsupportedDocumentError as exc:
        raise UnsupportedDocumentError(f"task {task.name!r}, row {row}: {exc}") from exc
    turn = document.turns[0]
    if len(inputs.prompt_ids) + task.max_new_tokens > renderer.max_text_len:
        return {
            "row": row,
            "question": turn.user,
            "skipped": PROMPT_TOO_LONG,
            "prompt_tokens": len(inputs.prompt_ids),
        }
    return row, turn, inputs


def _preflight(
    config: JobConfig, tasks: Sequence[EvalTask], batch_size: int
) -> tuple[ChatRenderer, dict[str, DocumentSource]]:
    """The renderer and every task's reader, once the config, token budgets and splits check out."""
    check_supported(config)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not tasks:
        raise ValueError("no task to evaluate")
    renderer = build_renderer(config)
    check_token_budgets(renderer, tasks)
    return renderer, {task.name: _open_reader(task) for task in tasks}


def _clear_results(output: str | Path) -> None:
    """Remove the last evaluation's ``results.json``, so a failed run leaves no finished marker."""
    rank = dist.get_rank() if dist.is_initialized() else get_world_info()[0]
    if rank == 0:
        (Path(output) / RESULTS_FILE).unlink(missing_ok=True)


def _open_reader(task: EvalTask) -> DocumentSource:
    reader = build_reader(task)
    if len(reader) == 0:
        raise ValueError(f"task {task.name!r}: its split has no rows")
    return reader


def _join_ranks(config: JobConfig) -> tuple[bool, int, int, torch.device]:
    """Join the launch's ranks: whether this started the process group, rank, ranks, device.

    A process group that is already initialized is used as it is. Otherwise a
    launch of several processes gets one over all of them, each rank a whole
    replica of the model, whatever parallelism the config trains with.
    """
    rank, local_rank, world_size = get_world_info()
    started = False
    if not dist.is_initialized() and world_size > 1:
        replicas = DistributedConfig(
            backend=config.distributed.backend,
            nccl_timeout_sec=config.distributed.nccl_timeout_sec,
        )
        init_distributed(replicas, seed=config.train.seed)
        started = True
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return started, rank, world_size, device


def _gather(value: Any) -> list[Any]:
    """``value`` of every rank, in rank order."""
    if not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _on_every_rank[T](action: Callable[[], T], what: str) -> T:
    """``action()`` on this rank; if it fails on any rank, every rank raises."""
    value: T | None = None
    failure: Exception | None = None
    try:
        value = action()
    except Exception as error:
        failure = error
    agree_across_ranks(failure, what)
    return cast(T, value)


def _on_rank_zero[T](rank: int, action: Callable[[], T], what: str) -> T | None:
    """``action()`` on rank 0 and ``None`` elsewhere; if it fails, every rank raises."""
    return _on_every_rank(action if rank == 0 else lambda: None, what)


def _evaluate_batches(
    decoder: Decoder, task: EvalTask, reader: DocumentSource, batches: Sequence[range]
) -> list[Record]:
    records: list[Record] = []
    every = max(1, len(batches) // _PROGRESS_LINES)
    for done, batch in enumerate(batches, start=1):
        records.extend(evaluate_batch(decoder, task, reader, batch))
        if done % every == 0:
            logger.info(f"{task.name}: {done} of this rank's {len(batches)} batches decoded")
    return records


def _write_json(path: Path, value: Any) -> None:
    """Write ``value`` to ``path`` whole: to a sibling file first, then renamed over it."""
    partial_path = path.with_name(f"{path.name}.tmp")
    partial_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(partial_path, path)


def _start(output: Path, checkpoint: str | Path) -> str:
    """Create the output directories and name the step directory to evaluate."""
    (output / PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    return str(resolve_checkpoint(checkpoint))


def _finish_task(
    task: EvalTask, per_rank: Sequence[Sequence[Record]], rows: int, output: Path
) -> Record:
    """Merge the ranks' records of ``task``, write its predictions and return its report."""
    records = merge_records(per_rank, rows)
    summary = summarize(task, records)
    predictions = Path(PREDICTIONS_DIR) / f"{task.name}.jsonl"
    with open(output / predictions, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        f"{task.name}: {summary['metrics']} over {summary['samples']} of {summary['rows']} "
        f"rows, skipped {summary['skipped']}"
    )
    return {**summary, "predictions": str(predictions)}


def _finish(output: Path, results: Record, metrics: Mapping[str, float]) -> None:
    _write_json(output / METRICS_FILE, metrics)
    _write_json(output / RESULTS_FILE, results)
    logger.info(f"Results written to {output / RESULTS_FILE}")


def _publish(step_dir: Path, metrics: Mapping[str, float], saved: Path) -> None:
    """Append ``metrics`` to the checkpoint's MLflow run; a failure is a warning, never raised."""
    try:
        target = publish_metrics(step_dir, metrics)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, PublishRefused) else f"{type(exc).__name__}: {exc}"
        reason = _URL.sub(r"\1<redacted>", reason)
        logger.warning(
            f"The metrics were not published to MLflow: {reason}. They are in {saved}; to "
            f"publish them later: python -m kempnerforge.metrics.mlflow.publish {step_dir} "
            f"--metrics {saved}"
        )
        return
    logger.info(
        f"Published {len(metrics)} metric(s) to MLflow run {target.run_id} at step {target.step}"
    )
