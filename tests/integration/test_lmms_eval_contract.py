"""Contract tests pinning the real ``lmms_eval`` API to what the unit-test fakes assume.

The VLM-eval unit tests run against an in-repo fake ``lmms_eval``
(``tests/unit/eval/vlm/_fake_lmms_eval.py``) so they execute in CI without the optional,
undeclared ``lmms-eval`` dependency. These tests are the fidelity net: they exercise the
*real* package and fail loudly if its API drifts from the fakes (in which case update the
fakes — and likely the adapter). They run wherever real lmms-eval is installed (locally, the
manual ``gpu-tests`` CI job) and skip otherwise.

Also verifies the ``pyproject.toml`` ``lmms_eval.models`` entry point resolves to the adapter.
"""

from __future__ import annotations

import pytest

lmms_eval = pytest.importorskip("lmms_eval")
if getattr(lmms_eval, "__file__", None) is None:
    # The unit fake (a bare module object) is active; contract tests need the real package.
    pytest.skip(
        "fake lmms_eval is active; skipping real-package contract tests", allow_module_level=True
    )

from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.models import get_model  # noqa: E402
from lmms_eval.models.registry_v2 import ModelManifest  # noqa: E402
from lmms_eval.protocol import ChatMessages  # noqa: E402
from lmms_eval.utils import Collator  # noqa: E402

from kempnerforge.eval.vlm.adapter import KempnerForgeVLM  # noqa: E402


def test_entrypoint_resolves_to_adapter():
    """The pyproject ``lmms_eval.models`` entry point resolves ``kempnerforge_vlm``."""
    assert get_model("kempnerforge_vlm") is KempnerForgeVLM


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


class TestModelManifestContract:
    def test_fields_and_validation(self):
        manifest = ModelManifest(model_id="x", chat_class_path="a.b.C")
        assert manifest.model_id == "x"
        assert manifest.chat_class_path == "a.b.C"
        assert manifest.simple_class_path is None
        with pytest.raises(ValueError):
            ModelManifest(model_id="y")  # neither class path -> rejected


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
