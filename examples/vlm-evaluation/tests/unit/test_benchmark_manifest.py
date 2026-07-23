"""Unit tests for the benchmark manifest. Fixtures mirror the result shapes
``simple_evaluate`` produces (leaf tasks, grouped tasks with empty main
entries, nested groups, dict-valued aggregates)."""

from __future__ import annotations

import logging

import pytest
from benchmark_manifest import (
    BENCHMARK_METRICS,
    MetricSpec,
    benchmark_aggregates,
    build_eval_metrics,
)

# --------------------------------------------------------------------------- #
# Aggregate resolution — registered benchmarks
# --------------------------------------------------------------------------- #


def test_registered_scale_normalizes_to_unit_interval():
    """A 0-100 metric (hallusion aAcc) is divided into [0, 1]."""
    results = {
        "results": {
            "hallusion_bench_image": {
                "alias": "hallusion_bench_image",
                "aAcc,none": 63.0,
                "aAcc_stderr,none": 1.2,
            }
        },
        "group_subtasks": {"hallusion_bench_image": []},
    }
    scores = benchmark_aggregates(results, ["hallusion_bench_image"])
    assert scores == pytest.approx({"hallusion_bench_image": 0.63})


def test_grouped_task_averages_subtasks_when_main_entry_empty():
    """tempcompass-shape: the group entry has no numeric metric -> mean over subtasks."""
    results = {
        "results": {
            "tempcompass": {" ": " ", "alias": "tempcompass"},
            "tempcompass_mc": {"alias": " - mc", "avg_accuracy,none": 40.0},
            "tempcompass_yn": {"alias": " - yn", "avg_accuracy,none": 60.0},
        },
        "group_subtasks": {"tempcompass": ["tempcompass_mc", "tempcompass_yn"]},
    }
    scores = benchmark_aggregates(results, ["tempcompass"])
    assert scores == pytest.approx({"tempcompass": 0.5})  # mean(40, 60) / 100


def test_filter_disambiguates_multi_filter_metric():
    """gsm8k reports exact_match under two filters; the registered one wins."""
    results = {
        "results": {
            "gsm8k_cot_zeroshot": {
                "alias": "gsm8k_cot_zeroshot",
                "exact_match,strict-match": 0.10,
                "exact_match,flexible-extract": 0.42,
            }
        },
        "group_subtasks": {"gsm8k_cot_zeroshot": []},
    }
    scores = benchmark_aggregates(results, ["gsm8k_cot_zeroshot"])
    assert scores == pytest.approx({"gsm8k_cot_zeroshot": 0.42})


def test_registered_filter_absent_warns_and_skips(caplog):
    """The registered filter must match exactly: another variant (strict-match)
    must never silently stand in for the registered one (flexible-extract)."""
    results = {
        "results": {
            "gsm8k_cot_zeroshot": {
                "alias": "gsm8k_cot_zeroshot",
                "exact_match,strict-match": 0.10,
            }
        },
        "group_subtasks": {"gsm8k_cot_zeroshot": []},
    }
    with caplog.at_level(logging.WARNING, logger="benchmark_manifest"):
        scores = benchmark_aggregates(results, ["gsm8k_cot_zeroshot"])
    assert scores == {}
    warnings = [r.getMessage() for r in caplog.records]
    assert any(
        "'flexible-extract'" in w and "strict-match" in w and "skipping its aggregate" in w
        for w in warnings
    )


def test_no_registered_filter_still_falls_back():
    """Specs without a filter keep the lenient chain: 'none' first, else the
    first available filter variant."""
    results = {
        "results": {"blink": {"alias": "blink", "blink_acc,custom-filter": 0.6}},
        "group_subtasks": {"blink": []},
    }
    scores = benchmark_aggregates(results, ["blink"])
    assert scores == pytest.approx({"blink": 0.6})


def test_subkey_selects_from_dict_valued_metric():
    """videoevalpro's aggregate is a dict keyed by task type plus 'overall'."""
    results = {
        "results": {
            "videoevalpro": {
                "alias": "videoevalpro",
                "videoevalpro_score,none": {"Local Perception": 0.30, "overall": 0.45},
            }
        },
        "group_subtasks": {"videoevalpro": []},
    }
    scores = benchmark_aggregates(results, ["videoevalpro"])
    assert scores == pytest.approx({"videoevalpro": 0.45})


def test_submission_only_benchmark_skipped_silently(caplog):
    """egoschema is registered with metric=None -> no aggregate, no warning."""
    results = {
        "results": {"egoschema": {"alias": "egoschema", "submission,none": None}},
        "group_subtasks": {"egoschema": []},
    }
    with caplog.at_level(logging.WARNING, logger="benchmark_manifest"):
        scores = benchmark_aggregates(results, ["egoschema"])
    assert scores == {}
    assert not caplog.records


def test_registered_metric_absent_warns_and_skips(caplog):
    """A registered metric missing from the results warns instead of guessing."""
    results = {
        "results": {"blink": {"alias": "blink", "some_other_metric,none": 0.5}},
        "group_subtasks": {"blink": []},
    }
    with caplog.at_level(logging.WARNING, logger="benchmark_manifest"):
        scores = benchmark_aggregates(results, ["blink"])
    assert scores == {}
    assert any("absent from the results" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Aggregate resolution — unregistered fallback
# --------------------------------------------------------------------------- #


def test_unregistered_sole_metric_fallback_infers_scale(caplog):
    """Sole higher_is_better metric is used; magnitude picks the divisor; warns loudly."""
    results = {
        "results": {
            "newbench_pct": {"alias": "newbench_pct", "acc,none": 73.2},
            "newbench_frac": {"alias": "newbench_frac", "acc,none": 0.73},
        },
        "group_subtasks": {"newbench_pct": [], "newbench_frac": []},
        "higher_is_better": {"newbench_pct": {"acc": True}, "newbench_frac": {"acc": True}},
    }
    with caplog.at_level(logging.WARNING, logger="benchmark_manifest"):
        scores = benchmark_aggregates(results, ["newbench_pct", "newbench_frac"])
    assert scores == pytest.approx({"newbench_pct": 0.732, "newbench_frac": 0.73})
    warnings = [r.getMessage() for r in caplog.records]
    assert sum("not registered in BENCHMARK_METRICS" in w for w in warnings) == 2
    # The warning embeds a paste-ready registry line for the offending task.
    assert any("'newbench_pct': MetricSpec('acc')" in w for w in warnings)


def test_unregistered_ambiguous_task_skipped_with_warning(caplog):
    """Several candidate metrics -> no way to choose an aggregate -> skip + warn."""
    results = {
        "results": {"newbench": {"alias": "newbench", "a,none": 0.1, "b,none": 0.2}},
        "group_subtasks": {"newbench": []},
        "higher_is_better": {"newbench": {"a": True, "b": True}},
    }
    with caplog.at_level(logging.WARNING, logger="benchmark_manifest"):
        scores = benchmark_aggregates(results, ["newbench"])
    assert scores == {}
    assert any("cannot choose an aggregate" in r.getMessage() for r in caplog.records)


def test_nested_group_names_never_resolved(caplog):
    """mmlu-shape: intermediate groups (stem, humanities) appear in group_subtasks and
    results but must not be resolved as benchmarks of their own."""
    results = {
        "results": {
            "mmlu_flan_cot_zeroshot": {
                "alias": "mmlu_flan_cot_zeroshot",
                "exact_match,flexible-extract": 0.31,
            },
            "stem": {"alias": " - stem", "exact_match,flexible-extract": 0.28},
            "humanities": {"alias": " - humanities", "exact_match,flexible-extract": 0.33},
            "mmlu_abstract_algebra": {"alias": "  - aa", "exact_match,flexible-extract": 0.2},
        },
        "group_subtasks": {
            "mmlu_flan_cot_zeroshot": ["stem", "humanities"],
            "stem": ["mmlu_abstract_algebra"],
            "humanities": [],
        },
        "higher_is_better": {"stem": {"exact_match": True}, "humanities": {"exact_match": True}},
    }
    with caplog.at_level(logging.WARNING, logger="benchmark_manifest"):
        scores = benchmark_aggregates(results, ["mmlu_flan_cot_zeroshot"])
    assert scores == pytest.approx({"mmlu_flan_cot_zeroshot": 0.31})
    assert not caplog.records  # stem/humanities never hit the fallback


# --------------------------------------------------------------------------- #
# build_eval_metrics — the flat log dict
# --------------------------------------------------------------------------- #


def test_build_eval_metrics_key_scheme():
    results = {
        "results": {
            "realworldqa": {
                "alias": "realworldqa",
                "exact_match,none": 0.44,
                "exact_match_stderr,none": 0.01,
            }
        },
        "group_subtasks": {"realworldqa": []},
        "throughput": {"avg_speed": 12.5, "total_gen_tokens": 300, "total_elapsed_time": 24.0},
    }
    metrics = build_eval_metrics(results, ["realworldqa"])
    assert metrics["eval/benchmarks/agg/realworldqa"] == pytest.approx(0.44)
    assert metrics["eval/benchmarks/raw/realworldqa/exact_match"] == pytest.approx(0.44)
    assert metrics["eval/benchmarks/throughput/overall/avg_speed"] == pytest.approx(12.5)
    assert metrics["eval/benchmarks/throughput/overall/total_gen_tokens"] == pytest.approx(300)
    assert not any("stderr" in k for k in metrics)


def test_throughput_is_run_level_never_per_task():
    """The invocation-wide throughput summary is logged once under ``overall``,
    never duplicated under each requested task (a two-task run would otherwise
    attribute the combined run's totals to both tasks)."""
    results = {
        "results": {
            "realworldqa": {"alias": "realworldqa", "exact_match,none": 0.44},
            "mmstar": {"alias": "mmstar", "average,none": 0.5},
        },
        "group_subtasks": {"realworldqa": [], "mmstar": []},
        "throughput": {"avg_speed": 12.5, "total_elapsed_time": 24.0},
    }
    metrics = build_eval_metrics(results, ["realworldqa", "mmstar"])
    assert metrics["eval/benchmarks/throughput/overall/total_elapsed_time"] == pytest.approx(24.0)
    per_task = [k for k in metrics if "/throughput/" in k and "/throughput/overall/" not in k]
    assert per_task == []


def test_build_eval_metrics_raw_includes_subtasks_and_filters():
    """Subtask rows and non-'none' filters are flattened; dicts flatten one level."""
    results = {
        "results": {
            "tvbench": {" ": " ", "alias": "tvbench"},
            "tvbench_action": {"alias": " - action", "tvbench_acc,none": 0.2},
            "gsm8k_cot_zeroshot": {
                "alias": "gsm8k",
                "exact_match,strict-match": 0.10,
                "exact_match,flexible-extract": 0.42,
            },
            "videoevalpro": {
                "alias": "videoevalpro",
                "videoevalpro_score,none": {"overall": 0.45},
            },
        },
        "group_subtasks": {"tvbench": ["tvbench_action"]},
    }
    metrics = build_eval_metrics(results, ["tvbench"])
    raw = {k.removeprefix("eval/benchmarks/raw/"): v for k, v in metrics.items() if "/raw/" in k}
    assert raw["tvbench_action/tvbench_acc"] == pytest.approx(0.2)
    assert raw["gsm8k_cot_zeroshot/exact_match/strict-match"] == pytest.approx(0.10)
    assert raw["gsm8k_cot_zeroshot/exact_match/flexible-extract"] == pytest.approx(0.42)
    assert raw["videoevalpro/videoevalpro_score/overall"] == pytest.approx(0.45)


def test_build_eval_metrics_efficiency_only_when_present():
    results = {
        "results": {"realworldqa": {"alias": "realworldqa", "exact_match,none": 0.4}},
        "group_subtasks": {"realworldqa": []},
        "efficiency": {
            "by_task": {
                "realworldqa": {
                    "total_output_tokens": 300.0,
                    "tokens_per_correct_answer": None,  # None must be skipped
                }
            },
            "overall": {},
        },
    }
    metrics = build_eval_metrics(results, ["realworldqa"])
    eff_prefix = "eval/benchmarks/efficiency/realworldqa"
    assert metrics[f"{eff_prefix}/total_output_tokens"] == pytest.approx(300.0)
    assert f"{eff_prefix}/tokens_per_correct_answer" not in metrics


# --------------------------------------------------------------------------- #
# Registry sanity
# --------------------------------------------------------------------------- #


def test_registry_specs_are_well_formed():
    """Every registered spec has a positive explicit scale (or None/submission-only)."""
    for task, spec in BENCHMARK_METRICS.items():
        assert isinstance(spec, MetricSpec)
        if spec.scale is not None:
            assert spec.scale > 0, f"{task}: non-positive scale"
        if spec.metric is None:
            assert spec.filter is None and spec.subkey is None, f"{task}: dead fields"
