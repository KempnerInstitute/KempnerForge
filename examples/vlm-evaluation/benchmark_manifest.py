"""Benchmark manifest: the single place per-benchmark knowledge lives for VLM eval.

For each benchmark the registry records which metric in an lmms-eval results
dict is the authoritative aggregate, and how to map it into [0, 1] so scores
are comparable across benchmarks. Unregistered benchmarks fall back to a
metadata-driven guess with a loud warning telling you to register them.

``build_eval_metrics`` turns a ``simple_evaluate`` results dict into one flat
``{key: float}`` dict for logging through the framework's metrics backends:

    eval/benchmarks/agg/<benchmark>            normalized [0, 1] aggregate
    eval/benchmarks/raw/<task>/<metric>[/...]  every numeric metric, raw
    eval/benchmarks/throughput/<task>/<key>    native lmms-eval throughput
    eval/benchmarks/efficiency/<task>/<key>    only when the run logged samples

Deliberately not in the spec until a consumer lands: random-guess baselines
(today in ``local/scripts/plot_eval_results.py``; wanted for above-chance
normalization), display names, modality tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_KEY_PREFIX = "eval/benchmarks"


@dataclass(frozen=True)
class MetricSpec:
    """How to read one benchmark's authoritative aggregate out of an lmms-eval result.

    ``metric`` is the name *before* lmms-eval's ``,<filter>`` suffix (``"aAcc"``
    matches ``"aAcc,none"``); ``None`` marks a registered benchmark with no
    local score (a submission-only leaderboard task) — skipped without warning.
    ``scale`` is the divisor mapping the raw value into [0, 1] (``100.0`` for a
    0-100 percentage); ``None`` means "infer by magnitude" (raw > 1.0 -> /100,
    else /1) — used by the unregistered fallback, or when a scale is genuinely
    unknown. ``filter`` disambiguates a metric reported under several filters
    (gsm8k's strict vs flexible). ``subkey`` selects from a dict-valued metric
    (videoevalpro reports a dict keyed by task type plus ``"overall"``).
    """

    metric: str | None
    scale: float | None = 1.0
    filter: str | None = None
    subkey: str | None = None


# The authoritative aggregate for each benchmark we run, keyed by the lmms-eval
# top-level task name (the ``results`` key). Derived by crawling real lmms-eval
# outputs and task sources: the "overall" metric cannot be inferred from name
# substrings (a task reports several accuracies and only one is the aggregate),
# and scale cannot be inferred from the metric name (``accuracy`` is 0-100 for
# mmvu_val but 0-1 for perceptiontest_val_mc). Registering a benchmark = adding
# one line here.
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
    "egoschema": MetricSpec(None),  # submission-only leaderboard task; no local score
    # --- vdc caption splits (upstream metric name carries the double-l typo) ---
    "camera_test": MetricSpec("llmms_eval_acc"),
    "background_test": MetricSpec("llmms_eval_acc"),
    "detailed_test": MetricSpec("llmms_eval_acc"),
    "main_object_test": MetricSpec("llmms_eval_acc"),
    "short_test": MetricSpec("llmms_eval_acc"),
    # --- text (flexible-extract disambiguates the two exact_match filter columns) ---
    "gsm8k_cot_zeroshot": MetricSpec("exact_match", filter="flexible-extract"),
    "mmlu_flan_cot_zeroshot": MetricSpec("exact_match", filter="flexible-extract"),
    "gpqa_main_cot_zeroshot": MetricSpec("exact_match", filter="flexible-extract"),
}


def _lookup_metric(
    entry: dict, metric: str, filter: str | None = None, subkey: str | None = None
) -> float | None:
    """Value of ``metric`` in one result entry, matched on the name before the comma.

    Among keys whose base name (before ``,``) equals ``metric`` — skipping the
    ``alias`` and stderr columns — prefer the requested ``filter``, then
    ``"none"``, then the first match. When ``subkey`` is set and the chosen
    value is a dict, index into it. Returns ``None`` when absent or non-numeric.
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
    chosen: object | None = None
    if filter is not None:
        chosen = next((v for f, v in matches if f == filter), None)
    if chosen is None:
        chosen = next((v for f, v in matches if f == "none"), None)
    if chosen is None:
        chosen = matches[0][1]
    if subkey is not None and isinstance(chosen, dict):
        chosen = chosen.get(subkey)
    if isinstance(chosen, bool) or not isinstance(chosen, (int, float)):
        return None
    return float(chosen)


def _resolve_spec(task: str, higher_is_better: dict) -> MetricSpec | None:
    """Registry entry for ``task``, else a metadata-driven fallback (loud), else ``None``.

    The fallback uses the result's own ``higher_is_better`` metadata: exactly
    one named metric -> use it with a magnitude-inferred scale; zero or several
    -> no way to choose an aggregate, skip. Both paths warn with a paste-ready
    registry line so the benchmark gets registered.
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
    """Map each requested benchmark -> its authoritative aggregate as a [0, 1] fraction.

    Iterates the *requested* top-level tasks — never ``group_subtasks`` keys,
    which also list nested intermediate groups (mmlu's ``stem``/``humanities``)
    that would resolve as phantom benchmarks. A grouped task whose main entry
    carries no value for the metric (tempcompass, tvbench) is the mean of that
    metric over its subtasks.
    """
    task_results: dict[str, dict] = results.get("results", {})
    group_subtasks: dict[str, list] = results.get("group_subtasks", {})
    higher_is_better: dict = results.get("higher_is_better", {})
    scores: dict[str, float] = {}
    for task in tasks:
        spec = _resolve_spec(task, higher_is_better)
        if spec is None or spec.metric is None:
            continue  # unresolved (already warned) or submission-only (silent)
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
            logger.warning(
                "Benchmark %r: registered metric %r is absent from the results and no subtask "
                "supplied it; skipping its aggregate (raw metrics are still logged).",
                task,
                spec.metric,
            )
            continue
        divisor = spec.scale if spec.scale is not None else (100.0 if raw > 1.0 else 1.0)
        scores[task] = raw / divisor
    return scores


def _raw_metrics(results: dict) -> dict[str, float]:
    """Flatten every numeric metric of every task and subtask, unnormalized.

    Key shape: ``eval/benchmarks/raw/<task>/<metric>``, with the filter
    appended when it is not ``"none"`` and the dict key appended for
    dict-valued metrics. ``alias`` and stderr columns are skipped (they remain
    in the results JSON).
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


def _perf_metrics(results: dict, tasks: list[str]) -> dict[str, float]:
    """Native lmms-eval throughput/efficiency summaries as per-task scalars.

    ``results["throughput"]`` is per-invocation (one benchmark per job in the
    launch scripts), so it is attributed to each requested task;
    ``results["efficiency"]["by_task"]`` is already per-task and exists only
    when the run logged samples. Non-numeric entries are skipped.
    """
    metrics: dict[str, float] = {}
    throughput = results.get("throughput") or {}
    for task in tasks:
        for key in _THROUGHPUT_KEYS:
            value = throughput.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{_KEY_PREFIX}/throughput/{task}/{key}"] = float(value)
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
    """One flat ``{key: float}`` dict for logging.

    Normalized aggregates under ``eval/benchmarks/agg/<benchmark>``, the full
    raw flatten under ``eval/benchmarks/raw/...``, and native throughput /
    efficiency — the aggregate selection only adds to, never filters, what gets
    logged.
    """
    metrics = {
        f"{_KEY_PREFIX}/agg/{bench}": score
        for bench, score in benchmark_aggregates(results, tasks).items()
    }
    metrics.update(_raw_metrics(results))
    metrics.update(_perf_metrics(results, tasks))
    return metrics
