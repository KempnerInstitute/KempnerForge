"""Benchmark manifest: per-benchmark metric knowledge for VLM eval.

Each entry records which metric in an lmms-eval results dict is a benchmark's
authoritative aggregate and how to map it into [0, 1]. Unregistered benchmarks
fall back to a metadata-driven guess with a loud warning.

``build_eval_metrics`` flattens a ``simple_evaluate`` results dict for logging:

    eval/benchmarks/agg/<benchmark>            normalized [0, 1] aggregate
    eval/benchmarks/raw/<task>/<metric>[/...]  every numeric metric, raw
    eval/benchmarks/throughput/overall/<key>   native lmms-eval throughput (per-invocation)
    eval/benchmarks/efficiency/<task>/<key>    only when the run logged samples
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_KEY_PREFIX = "eval/benchmarks"


@dataclass(frozen=True)
class MetricSpec:
    """How to read one benchmark's authoritative aggregate from an lmms-eval result.

    ``metric`` is the name before the ``,<filter>`` suffix; ``None`` = no local
    score (submission-only task). ``scale`` divides the raw value into [0, 1];
    ``None`` = infer by magnitude (raw > 1.0 -> /100, else /1). ``filter``
    disambiguates multi-filter metrics; ``subkey`` indexes a dict-valued one.
    """

    metric: str | None
    scale: float | None = 1.0
    filter: str | None = None
    subkey: str | None = None


# Authoritative aggregate per benchmark, keyed by the lmms-eval top-level task
# name. Neither the metric nor its scale can be inferred (``accuracy`` is 0-100
# for mmvu_val but 0-1 for perceptiontest_val_mc), so entries are explicit.
BENCHMARK_METRICS: dict[str, MetricSpec] = {
    # --- image ---
    "hallusion_bench_image": MetricSpec("aAcc", scale=100.0),
    "mmvu_val": MetricSpec("accuracy", scale=100.0),
    "mmstar": MetricSpec("average"),
    "blink": MetricSpec("blink_acc"),
    "realworldqa": MetricSpec("exact_match"),
    "mmmu_pro_standard": MetricSpec("mmmu_acc"),
    # --- video ---
    "mlvu_dev": MetricSpec("mlvu_percetion_score", scale=100.0),  # upstream typo; real key
    "tempcompass": MetricSpec("avg_accuracy", scale=100.0),  # main entry empty -> subtask mean
    "tvbench": MetricSpec("tvbench_acc"),  # main entry empty -> subtask mean
    "vsibench": MetricSpec("vsibench_overall"),
    "tomato": MetricSpec("tomato_score"),
    "nextqa_mc_test": MetricSpec("exact_match"),
    "perceptiontest_val_mc": MetricSpec("accuracy"),
    "videoevalpro": MetricSpec("videoevalpro_score", subkey="overall"),  # dict-valued aggregate
    "lvbench": MetricSpec("lvbench_score"),
    "videomme_v2": MetricSpec("videomme_v2_overall_acc", scale=100.0),
    "egoschema": MetricSpec(None),  # submission-only; no local score
    # --- vdc caption splits (upstream double-l typo) ---
    "camera_test": MetricSpec("llmms_eval_acc"),
    "background_test": MetricSpec("llmms_eval_acc"),
    "detailed_test": MetricSpec("llmms_eval_acc"),
    "main_object_test": MetricSpec("llmms_eval_acc"),
    "short_test": MetricSpec("llmms_eval_acc"),
    # --- text ---
    "gsm8k_cot_zeroshot": MetricSpec("exact_match", filter="flexible-extract"),
    "mmlu_flan_cot_zeroshot": MetricSpec("exact_match", filter="flexible-extract"),
    "gpqa_main_cot_zeroshot": MetricSpec("exact_match", filter="flexible-extract"),
}


def _lookup_metric(
    entry: dict, metric: str, filter: str | None = None, subkey: str | None = None
) -> float | None:
    """Value of ``metric`` in one result entry, matched on the name before the comma.

    A requested ``filter`` must match exactly (filter variants like strict-match
    vs flexible-extract are materially different numbers, so no other variant may
    stand in for a registered one); without a ``filter``, prefers ``"none"``, then
    the first match. Skips ``alias``/stderr columns. Returns ``None`` when absent
    or non-numeric.
    """
    matches: list[tuple[str, object]] = []
    for key, value in entry.items():
        if key == "alias" or "stderr" in key:
            continue
        base, _, filt = key.partition(",")
        if base == metric:
            matches.append((filt, value))
    if not matches:
        return None
    if filter is not None:
        chosen = next((v for f, v in matches if f == filter), None)
    else:
        chosen = next((v for f, v in matches if f == "none"), matches[0][1])
    if subkey is not None and isinstance(chosen, dict):
        chosen = chosen.get(subkey)
    if isinstance(chosen, bool) or not isinstance(chosen, (int, float)):
        return None
    return float(chosen)


def _available_filters(entry: dict, metric: str) -> list[str]:
    """Filter variants under which ``metric`` appears in one result entry."""
    filters: list[str] = []
    for key in entry:
        if key == "alias" or "stderr" in key:
            continue
        base, _, filt = key.partition(",")
        if base == metric:
            filters.append(filt)
    return filters


def _resolve_spec(task: str, higher_is_better: dict) -> MetricSpec | None:
    """Registry entry for ``task``, else a loud ``higher_is_better``-driven
    fallback, else ``None`` (skip).
    """
    spec = BENCHMARK_METRICS.get(task)
    if spec is not None:
        return spec  # registered (metric may be None -> caller skips silently)
    candidates = list((higher_is_better.get(task) or {}).keys())
    if len(candidates) == 1:
        logger.warning(
            "Benchmark %r is not registered in BENCHMARK_METRICS; falling back to its sole "
            "higher_is_better metric %r with a magnitude-inferred scale (raw > 1.0 is treated "
            "as 0-100, else 0-1). Register it in benchmark_manifest.py to make this explicit:\n"
            "    %r: MetricSpec(%r),  # add scale=100.0 if the metric is a 0-100 percentage",
            task,
            candidates[0],
            task,
            candidates[0],
        )
        return MetricSpec(candidates[0], scale=None)
    logger.warning(
        "Benchmark %r is not registered in BENCHMARK_METRICS and its higher_is_better names "
        "%d candidate metric(s) %s — cannot choose an aggregate, so it is skipped (raw "
        "metrics are still logged). Register it in benchmark_manifest.py:\n"
        '    %r: MetricSpec("<one of the candidates>"),',
        task,
        len(candidates),
        candidates,
        task,
    )
    return None


def benchmark_aggregates(results: dict, tasks: list[str]) -> dict[str, float]:
    """Map each requested benchmark to its aggregate as a [0, 1] fraction.

    Only the requested tasks are resolved (``group_subtasks`` also lists nested
    intermediate groups). A grouped task whose main entry carries no value is
    the mean of the metric over its subtasks.
    """
    task_results: dict[str, dict] = results.get("results", {})
    group_subtasks: dict[str, list] = results.get("group_subtasks", {})
    higher_is_better: dict = results.get("higher_is_better", {})
    scores: dict[str, float] = {}
    for task in tasks:
        spec = _resolve_spec(task, higher_is_better)
        if spec is None or spec.metric is None:
            continue  # unresolved (already warned) or submission-only
        raw = _lookup_metric(task_results.get(task) or {}, spec.metric, spec.filter, spec.subkey)
        if raw is None:
            subtask_values = [
                value
                for name in group_subtasks.get(task) or []
                for value in (
                    _lookup_metric(
                        task_results.get(name) or {}, spec.metric, spec.filter, spec.subkey
                    ),
                )
                if value is not None
            ]
            if subtask_values:
                raw = sum(subtask_values) / len(subtask_values)
        if raw is None:
            available = sorted(
                {
                    filt
                    for name in (task, *(group_subtasks.get(task) or []))
                    for filt in _available_filters(task_results.get(name) or {}, spec.metric)
                }
            )
            if spec.filter is not None and available:
                logger.warning(
                    "Benchmark %r: registered metric %r is present only under filter(s) %s, not "
                    "the registered filter %r; skipping its aggregate rather than silently "
                    "logging a different variant (raw metrics are still logged). If lmms-eval "
                    "renamed the filter, update benchmark_manifest.py:\n"
                    "    %r: MetricSpec(%r, filter=<one of %s>),",
                    task,
                    spec.metric,
                    available,
                    spec.filter,
                    task,
                    spec.metric,
                    available,
                )
            else:
                logger.warning(
                    "Benchmark %r: registered metric %r is absent from the results and no "
                    "subtask supplied it; skipping its aggregate (raw metrics are still logged).",
                    task,
                    spec.metric,
                )
            continue
        divisor = spec.scale if spec.scale is not None else (100.0 if raw > 1.0 else 1.0)
        scores[task] = raw / divisor
    return scores


def _raw_metrics(results: dict) -> dict[str, float]:
    """Flatten every numeric metric of every task and subtask, unnormalized.

    Keys are ``eval/benchmarks/raw/<task>/<metric>``, with non-``none`` filters
    and dict keys appended; ``alias``/stderr columns are skipped.
    """
    metrics: dict[str, float] = {}
    for task, entry in (results.get("results") or {}).items():
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if key == "alias" or "stderr" in key:
                continue
            base, _, filt = key.partition(",")
            parts = [_KEY_PREFIX, "raw", task, base]
            if filt and filt != "none":
                parts.append(filt)
            if isinstance(value, dict):
                for sub, sub_value in value.items():
                    if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                        metrics["/".join([*parts, str(sub)])] = float(sub_value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics["/".join(parts)] = float(value)
    return metrics


_THROUGHPUT_KEYS = ("avg_speed", "total_gen_tokens", "total_elapsed_time", "avg_latency")
_EFFICIENCY_KEYS = (
    "total_output_tokens",
    "total_input_tokens",
    "total_tokens",
    "avg_output_tokens_per_sample",
    "tokens_per_correct_answer",
)


def _perf_metrics(results: dict) -> dict[str, float]:
    """Native lmms-eval throughput/efficiency summaries as scalars.

    ``results["throughput"]`` is per-invocation (one ``generate_until`` call
    spans every task), so it is logged once under ``overall`` — comparable
    across runs only for the same task set. ``efficiency["by_task"]`` is
    genuinely per-task and exists only when samples were logged.
    """
    metrics: dict[str, float] = {}
    throughput = results.get("throughput") or {}
    for key in _THROUGHPUT_KEYS:
        value = throughput.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{_KEY_PREFIX}/throughput/overall/{key}"] = float(value)
    by_task = (results.get("efficiency") or {}).get("by_task") or {}
    for task, summary in by_task.items():
        if not isinstance(summary, dict):
            continue
        for key in _EFFICIENCY_KEYS:
            value = summary.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{_KEY_PREFIX}/efficiency/{task}/{key}"] = float(value)
    return metrics


def build_eval_metrics(results: dict, tasks: list[str]) -> dict[str, float]:
    """One flat ``{key: float}`` dict for logging: normalized aggregates, the
    full raw flatten, and native throughput/efficiency.
    """
    metrics = {
        f"{_KEY_PREFIX}/agg/{bench}": score
        for bench, score in benchmark_aggregates(results, tasks).items()
    }
    metrics.update(_raw_metrics(results))
    metrics.update(_perf_metrics(results))
    return metrics
