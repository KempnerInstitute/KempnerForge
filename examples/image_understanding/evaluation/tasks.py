"""Evaluation tasks read from a TOML file.

Each ``[[tasks]]`` table is one task: its name, the reader of its split with
that reader's arguments, the scorer of its answers and the cap on a decoded
answer::

    [[tasks]]
    name = "docvqa"
    reader = "docvqa"
    scorer = "anls"
    max_new_tokens = 50

    [tasks.reader_args]
    annotation_file = "path-to-docvqa-annotations"
    image_root = "path-to-docvqa-images"

The file is checked when it is loaded, before a reader opens anything.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from collections.abc import Callable, Collection, Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from image_text.testbed import chartqa, coco_captions, docvqa, textvqa, vqav2

from .scorers import SCORERS

if TYPE_CHECKING:
    from image_text.render import DocumentSource

READERS: dict[str, Callable[..., DocumentSource]] = {
    "chartqa": chartqa,
    "coco_captions": coco_captions,
    "docvqa": docvqa,
    "textvqa": textvqa,
    "vqav2": vqav2,
}

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TOML_VALUES: dict[Any, tuple[str, tuple[type, ...]]] = {
    str: ("a string", (str,)),
    Path: ("a string", (str,)),
    bool: ("a boolean", (bool,)),
    int: ("an integer", (int,)),
    float: ("a number", (int, float)),
}


def _judged(hint: Any) -> list[Any] | None:
    """The types of ``hint`` a TOML value can be checked against, or ``None`` when one cannot be."""
    members = get_args(hint) if get_origin(hint) in (Union, UnionType) else (hint,)
    judged = [member for member in members if member is not type(None)]
    return judged if judged and all(member in _TOML_VALUES for member in judged) else None


def _check_choice(task: str, key: str, value: Any, options: Collection[str]) -> None:
    if not isinstance(value, str) or value not in options:
        raise ValueError(
            f"task {task!r}: unknown {key} {value!r}; expected one of {sorted(options)}"
        )


def _check_reader_args(task: str, reader: str, args: Mapping[str, Any]) -> None:
    """Refuses ``args`` the factory of ``reader`` does not take, lacks or cannot use."""
    factory = READERS[reader]
    parameters = {
        name: parameter
        for name, parameter in inspect.signature(factory).parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(args) - set(parameters))
    if unknown:
        raise ValueError(
            f"task {task!r}: reader {reader!r} takes no {unknown}; "
            f"expected some of {sorted(parameters)}"
        )
    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is parameter.empty and name not in args
    ]
    if missing:
        raise ValueError(
            f"task {task!r}: reader {reader!r} needs {missing} in reader_args; "
            f"it takes {sorted(parameters)}"
        )
    hints = get_type_hints(factory.__init__ if inspect.isclass(factory) else factory)
    for key, value in args.items():
        judged = _judged(hints[key]) if key in hints else None
        if judged is not None and not any(type(value) in _TOML_VALUES[m][1] for m in judged):
            expected = " or ".join(dict.fromkeys(_TOML_VALUES[m][0] for m in judged))
            raise ValueError(f"task {task!r}: reader_args.{key} must be {expected}, got {value!r}")


@dataclass(frozen=True)
class EvalTask:
    """One evaluation task.

    ``reader`` names a factory of ``READERS`` and ``reader_args`` holds its keyword
    arguments; ``scorer`` names an entry of ``SCORERS``; ``max_new_tokens`` caps a
    decoded answer. A task is refused with ``ValueError`` when its name is not
    letters, digits, ``.``, ``_`` and ``-`` after a leading letter or digit, when
    it names an unknown reader or scorer, when ``reader_args`` holds an argument
    the reader does not take, lacks one it needs or gives one the wrong type, or
    when the cap is not an integer of at least 1.
    """

    name: str
    reader: str
    scorer: str
    max_new_tokens: int
    reader_args: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(
                "a task name starts with a letter or digit and continues with letters, "
                f"digits, '.', '_' or '-'; got {self.name!r}"
            )
        _check_choice(self.name, "reader", self.reader, READERS)
        _check_choice(self.name, "scorer", self.scorer, SCORERS)
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError(
                f"task {self.name!r}: max_new_tokens must be an integer of at least 1, "
                f"got {self.max_new_tokens!r}"
            )
        if not isinstance(self.reader_args, Mapping):
            raise ValueError(
                f"task {self.name!r}: reader_args must be a table of the reader's arguments, "
                f"got {self.reader_args!r}"
            )
        _check_reader_args(self.name, self.reader, self.reader_args)
        object.__setattr__(self, "reader_args", dict(self.reader_args))


_TASK_KEYS = tuple(f.name for f in fields(EvalTask))
_REQUIRED_TASK_KEYS = tuple(
    f.name for f in fields(EvalTask) if f.default is MISSING and f.default_factory is MISSING
)


def _check_keys(
    where: str, table: Mapping[str, Any], allowed: Collection[str], required: Collection[str]
) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; expected some of {sorted(allowed)}")
    missing = sorted(set(required) - set(table))
    if missing:
        raise ValueError(f"{where}: missing keys {missing}; expected {sorted(allowed)}")


def _tasks(document: Mapping[str, Any]) -> tuple[EvalTask, ...]:
    _check_keys("the task file", document, ("tasks",), ("tasks",))
    tables = document["tasks"]
    if (
        not isinstance(tables, list)
        or not tables
        or not all(isinstance(table, dict) for table in tables)
    ):
        raise ValueError("tasks must be one or more [[tasks]] tables")
    tasks = []
    for index, table in enumerate(tables):
        _check_keys(f"tasks[{index}]", table, _TASK_KEYS, _REQUIRED_TASK_KEYS)
        tasks.append(EvalTask(**table))
    names = [task.name for task in tasks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"task names must be unique; repeated: {repeated}")
    return tuple(tasks)


def load_tasks(path: str | Path) -> tuple[EvalTask, ...]:
    """The tasks of the TOML file at ``path``, in file order.

    The file holds only ``[[tasks]]`` tables, each with the keys ``name``,
    ``reader``, ``scorer`` and ``max_new_tokens`` and, for the reader's arguments,
    ``reader_args``. Invalid TOML, an unknown or missing key, a task ``EvalTask``
    refuses or a name used twice raises ``ValueError`` that names the file and
    what is accepted.
    """
    try:
        with open(path, "rb") as f:
            return _tasks(tomllib.load(f))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def build_reader(task: EvalTask) -> DocumentSource:
    """The reader of ``task``'s split, built by its factory from ``reader_args``."""
    return READERS[task.reader](**task.reader_args)
