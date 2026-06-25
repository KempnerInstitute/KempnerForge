"""Dependency-free fakes for the ``lmms_eval`` API surface the VLM adapter uses.

``lmms-eval`` is an optional, *undeclared* dependency, so
``kempnerforge.eval.vlm.adapter`` / ``registry`` import it at module top and cannot be
imported without it. ``conftest.py`` injects these fakes into ``sys.modules`` so the unit
tests run (and contribute coverage) in CI, where ``lmms-eval`` is absent. The fakes
reproduce ONLY the behavior the adapter relies on; their fidelity to the real package is
pinned by the gated contract test in ``tests/integration/`` (``test_lmms_eval_contract``).

Verified against the installed ``lmms_eval`` source:

- ``protocol.ChatMessages`` (pydantic): ``.messages[].role`` / ``.content[].type``; text
  blocks carry ``.text``, image/video/audio blocks carry ``.url``. ``extract_media()``
  returns ``(images, videos, audios)`` of the ``.url`` payloads.
- ``utils.Collator``: groups by ``group_fn(item)`` (a dict; key built from sorted items
  with list values tupled), sorts each group by ``sort_fn(item)``, ``get_batched(n)``
  yields lists of ``<= n`` while recording reorder indices, ``get_original`` inverts them.
- ``models.registry_v2.ModelManifest``: frozen dataclass requiring >= 1 class path.
- ``api.model.lmms``: base sets ``_rank=0/_world_size=1/cache_hook/task_dict`` and exposes
  ``rank``/``world_size`` properties.
- ``api.instance.Instance``: dataclass exposing ``.args`` (the arguments tuple).
"""

from __future__ import annotations

import collections
import dataclasses
import types
from collections.abc import Callable
from typing import Any

# --------------------------------------------------------------------------- #
# lmms_eval.protocol.ChatMessages
# --------------------------------------------------------------------------- #


class _Content:
    def __init__(self, block: dict) -> None:
        self.type = block["type"]
        if self.type == "text":
            self.text = block["text"]
        else:
            self.url = block["url"]


class _Message:
    def __init__(self, message: dict) -> None:
        self.role = message["role"]
        self.content = [_Content(block) for block in message["content"]]


class ChatMessages:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = [_Message(message) for message in messages]

    def extract_media(self) -> tuple[list[Any], list[Any], list[Any]]:
        images: list[Any] = []
        videos: list[Any] = []
        audios: list[Any] = []
        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)
        return images, videos, audios


# --------------------------------------------------------------------------- #
# lmms_eval.utils.Collator
# --------------------------------------------------------------------------- #


def _hashable_key(group: dict) -> tuple:
    # Mirror the real Collator: list/tuple values are tupled so the dict is hashable.
    return tuple(
        (key, tuple(value) if isinstance(value, (list, tuple)) else value)
        for key, value in sorted(group.items())
    )


class Collator:
    def __init__(
        self,
        arr: list,
        sort_fn: Callable,
        group_fn: Callable = lambda x: x[1],
        grouping: bool = False,
    ) -> None:
        self._sort_fn = sort_fn
        self.size = len(arr)
        self.reorder_indices: list[int] = []
        indexed = list(enumerate(arr))
        if grouping:
            groups: dict[Any, list] = collections.OrderedDict()
            for idx, item in indexed:
                groups.setdefault(_hashable_key(group_fn(item)), []).append((idx, item))
            self._groups = list(groups.values())
        else:
            self._groups = [indexed]

    def get_batched(self, n: int = 1, batch_fn: Callable | None = None):
        del batch_fn  # the adapter always passes None; size-by-n only
        for group in self._groups:
            ordered = sorted(group, key=lambda pair: self._sort_fn(pair[1]))
            self.reorder_indices.extend(idx for idx, _ in ordered)
            items = [item for _, item in ordered]
            for start in range(0, len(items), n):
                yield items[start : start + n]

    def get_original(self, newarr: list) -> list:
        res: list[Any] = [None] * self.size
        for ind, value in zip(self.reorder_indices, newarr, strict=True):
            res[ind] = value
        return res

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------- #
# lmms_eval.models.registry_v2.ModelManifest
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ModelManifest:
    model_id: str
    simple_class_path: str | None = None
    chat_class_path: str | None = None
    aliases: tuple = ()

    def __post_init__(self) -> None:
        if self.simple_class_path is None and self.chat_class_path is None:
            raise ValueError(f"ModelManifest('{self.model_id}') requires at least one class path")


# --------------------------------------------------------------------------- #
# lmms_eval.api.model.lmms
# --------------------------------------------------------------------------- #


class _CacheHook:
    def add_partial(self, attr: str, request: Any, result: Any) -> None:
        pass


class lmms:  # noqa: N801 — mirrors the real lowercase class name
    is_simple: bool = True

    def __init__(self) -> None:
        self._rank = 0
        self._world_size = 1
        self.cache_hook = _CacheHook()
        self.task_dict: dict = {}

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size


# --------------------------------------------------------------------------- #
# lmms_eval.api.instance.Instance
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Instance:
    request_type: str
    arguments: tuple
    idx: int
    metadata: Any = dataclasses.field(default_factory=dict)
    resps: list = dataclasses.field(default_factory=list)
    filtered_resps: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        meta = self.metadata or {}
        self.task_name = meta.get("task")
        self.doc_id = meta.get("doc_id")
        self.repeats = meta.get("repeats")

    @property
    def args(self) -> tuple:
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)


# --------------------------------------------------------------------------- #
# Module tree assembly
# --------------------------------------------------------------------------- #


def build_modules() -> dict[str, types.ModuleType]:
    """Build the fake ``lmms_eval`` module tree keyed by dotted import name.

    Submodules are also set as attributes on their parents so both ``import
    a.b.c`` and ``a.b.c`` attribute access resolve once these are in ``sys.modules``.
    """

    def _mod(name: str, **attrs: Any) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    root = _mod("lmms_eval")
    api = _mod("lmms_eval.api")
    api_model = _mod("lmms_eval.api.model", lmms=lmms, CacheHook=_CacheHook)
    api_instance = _mod("lmms_eval.api.instance", Instance=Instance)
    protocol = _mod("lmms_eval.protocol", ChatMessages=ChatMessages)
    utils = _mod("lmms_eval.utils", Collator=Collator)
    models = _mod("lmms_eval.models")
    registry_v2 = _mod("lmms_eval.models.registry_v2", ModelManifest=ModelManifest)

    root.api = api
    root.protocol = protocol
    root.utils = utils
    root.models = models
    api.model = api_model
    api.instance = api_instance
    models.registry_v2 = registry_v2

    return {
        "lmms_eval": root,
        "lmms_eval.api": api,
        "lmms_eval.api.model": api_model,
        "lmms_eval.api.instance": api_instance,
        "lmms_eval.protocol": protocol,
        "lmms_eval.utils": utils,
        "lmms_eval.models": models,
        "lmms_eval.models.registry_v2": registry_v2,
    }
