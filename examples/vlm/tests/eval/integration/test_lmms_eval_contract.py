# pyright: reportMissingImports=false
# ^ lmms-eval is an optional, undeclared dependency; see lmms_adapter.py's directive.
"""Contract tests pinning the real ``lmms_eval`` API to what this example assumes.

The VLM-eval unit tests run against an in-repo fake ``lmms_eval``
(``../unit/_fake_lmms_eval.py``) so they execute without the optional, undeclared
``lmms-eval`` dependency. These tests are the fidelity net: they exercise the *real*
package and fail loudly if its API drifts from the fakes or from the call the harness
makes (in which case update them — and likely the adapter). They run wherever real
lmms-eval is installed and skip otherwise.
"""

from __future__ import annotations

import types

import pytest

lmms_eval = pytest.importorskip("lmms_eval")
if getattr(lmms_eval, "__file__", None) is None:
    # The unit fake (a bare module object) is active; contract tests need the real package.
    pytest.skip(
        "fake lmms_eval is active; skipping real-package contract tests", allow_module_level=True
    )

from eval_publish import collect_scores  # noqa: E402
from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.protocol import ChatMessages  # noqa: E402
from lmms_eval.utils import Collator  # noqa: E402


class TestChatMessagesContract:
    def test_extract_media_and_object_model(self):
        messages = ChatMessages(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image", "url": "IMG"},
                    ],
                }
            ]
        )
        images, videos, audios = messages.extract_media()
        assert images == ["IMG"]
        assert videos == [] and audios == []
        message = messages.messages[0]
        assert message.role == "user"
        assert message.content[0].type == "text" and message.content[0].text == "hello"
        assert message.content[1].type == "image" and message.content[1].url == "IMG"


class TestCollatorContract:
    def test_group_sort_batch_and_restore(self):
        # Mirrors the adapter: group_fn -> gen_kwargs dict, sort_fn -> -len(context).
        arr = [
            ("longest-context", None, {"max_new_tokens": 4}),
            ("c", None, {"max_new_tokens": 4}),
            ("mid-ctx", None, {"max_new_tokens": 8}),
        ]
        col = Collator(arr, lambda a: -len(a[0]), group_fn=lambda a: a[2], grouping=True)
        flat = [item for batch in col.get_batched(n=2, batch_fn=None) for item in batch]
        assert sorted(map(id, flat)) == sorted(map(id, arr))  # all items present
        assert col.get_original(flat) == arr  # original request order restored


class TestSimpleEvaluateContract:
    def test_harness_call_shape(self):
        import inspect

        # Unlike the api/protocol/utils imports above, importing the evaluator can
        # fail on the cluster with `GLIBCXX_... not found` — an environment problem,
        # not API drift; the README's libstdc++ note has the fix.
        from lmms_eval.evaluator import simple_evaluate

        sig = inspect.signature(simple_evaluate)
        # Exactly the harness call: a prebuilt adapter instance + a record-only
        # model_args string + tasks/device/batch_size/limit.
        sig.bind(
            model=object(),
            model_args="config=c.toml,checkpoint=ckpt",
            tasks=["mmmu_val"],
            device="cuda",
            batch_size=1,
            limit=8,
        )  # raises TypeError on drift
        required = [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty]
        assert required == ["model"]  # an lmms instance is accepted; everything else optional
        assert sig.parameters["model_args"].default is None
        assert sig.parameters["tasks"].default is None
        assert sig.parameters["limit"].default is None


class TestLmmsBaseContract:
    def test_single_process_defaults(self):
        class _Probe(lmms):
            is_simple = False

            def loglikelihood(self, requests):
                raise NotImplementedError

            def generate_until(self, requests):
                raise NotImplementedError

            def generate_until_multi_round(self, requests):
                raise NotImplementedError

        probe = _Probe()
        assert probe.rank == 0 and probe.world_size == 1
        assert probe.task_dict == {}
        probe.cache_hook.add_partial("generate_until", ("ctx", {}), "out")  # no-op, must not raise


class TestInstanceContract:
    def test_args_returns_arguments_tuple(self):
        inst = Instance(
            request_type="generate_until",
            arguments=("ctx", None, {}, "d0", "t", "test"),
            idx=0,
            metadata={"task": "t", "doc_id": "d0", "repeats": 1},
        )
        assert inst.args == ("ctx", None, {}, "d0", "t", "test")


class TestResultsContract:
    """What ``eval_publish`` reads out of ``simple_evaluate``'s results."""

    def test_consolidated_results_publish_as_scores(self):
        from lmms_eval.api.metrics import bypass_agg, mean
        from lmms_eval.evaluator_utils import TaskOutput, consolidate_results

        task_output = types.SimpleNamespace(
            task_name="t",
            task_config={
                "metric_list": [
                    {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                    {"metric": "extracted_answers", "aggregation": "bypass"},
                ]
            },
            group_name=None,
            group_alias=None,
            n_shot=0,
            version=1,
            logged_samples=[],
            task=types.SimpleNamespace(
                aggregation=lambda: {"acc": mean, "extracted_answers": bypass_agg},
                higher_is_better=lambda: {"acc": True, "extracted_answers": True},
            ),
            sample_metrics={
                ("acc", "none"): [1, 0, 1, 1],
                ("extracted_answers", "none"): list("abcd"),
            },
            agg_metrics={},
            sample_len=0,
        )
        TaskOutput.calculate_aggregate_metric(task_output, bootstrap_iters=100)
        results, _, configs, _, _, higher_is_better = consolidate_results([task_output])

        assert results["t"]["acc,none"] == 0.75 and results["t"]["extracted_answers,none"] == 999
        assert isinstance(results["t"]["acc_stderr,none"], float)
        scores = collect_scores(
            {"results": results, "higher_is_better": higher_is_better, "configs": configs}
        )
        assert [(s.name, s.value) for s in scores] == [("t_acc", 0.75)]
        assert scores[0].stderr == results["t"]["acc_stderr,none"]

    def test_evaluate_returns_the_keys_the_publisher_reads(self):
        import inspect

        # Importing the evaluator can fail with `GLIBCXX_... not found` on the cluster.
        from lmms_eval import evaluator

        source = inspect.getsource(evaluator.evaluate)
        keys = (
            "results",
            "higher_is_better",
            "n-samples",
            "effective",
            "configs",
            "group_subtasks",
        )
        assert [key for key in keys if f'"{key}"' not in source] == []
