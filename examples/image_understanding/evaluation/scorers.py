"""Answer scorers for evaluation tasks, selected by name.

A scorer rates one sample: the decoded prediction against that sample's
reference, which for the scorers here is its reference answers. The text scorers
take the prediction as a ``str`` and the references as a non-empty sequence of
``str``; anything else raises before any scoring (``TypeError``, or
``ValueError`` for no reference). A sample's score
is one value in [0, 1], or a mapping of named components for a scorer that
reports several. A task's metrics are the scorer's aggregate of its samples'
scores, by default the mean of each component.

``extract_mcq_answer`` and the code it calls (``_Choice``, ``_word_character``,
``_starts_with_alternative``, ``_read_choice``, ``_has_alternative``,
``_declared_answer``, ``_closing_brace``, ``_last_container``,
``_implicit_answer`` and the constants from ``_DEFAULT_CHOICES`` to
``_EMPHASIS_CLOSERS``) come from lmms-eval 0.7.4, file
``lmms_eval/tasks/_task_utils/mcq_extract.py``, Copyright (c) 2025 LMMs-Lab.
lmms-eval's LICENSE applies the Apache License, Version 2.0, to its code under
``lmms_eval/tasks``, so that license covers this code.

Adopted from lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval. Below
is the original copyright:

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

The changes: the comments are removed, the type hints use ``list``, ``set`` and
``X | None`` in place of ``typing.List``, ``Set`` and ``Optional``, the code is
laid out as ruff formats it, and the two nested ``if`` statements at the end of
``_read_choice`` are joined into one, as ruff's SIM102 asks. The logic is
unchanged.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .vqa_normalization import normalize_vqa_answer

SampleScore = float | Mapping[str, float]

VQA_FULL_CREDIT_MATCHES = 3
ANLS_THRESHOLD = 0.5
RELAXED_TOLERANCE = 0.05
ZERO_TARGET_GUARD = 1e-10
REALWORLDQA_CHOICES: tuple[str, ...] = ("A", "B", "C", "D")


def _references(prediction: str, answers: Sequence[str]) -> Sequence[str]:
    """``answers`` once ``prediction`` and every reference are known to be strings.

    Raises ``TypeError`` naming the argument when ``prediction`` is not a ``str``,
    when ``answers`` is not a sequence of strings (a lone string, ``bytes``, a set
    and an iterator are not) or when one reference is not a ``str``, and
    ``ValueError`` when there is no reference.
    """
    if not isinstance(prediction, str):
        raise TypeError(f"prediction must be a str, got {prediction!r}")
    if isinstance(answers, str):
        raise TypeError("answers must be a sequence of str, not one string")
    if not isinstance(answers, Sequence) or isinstance(answers, bytes):
        raise TypeError(f"answers must be a sequence of str, got {answers!r}")
    if not answers:
        raise ValueError("a prediction needs at least one reference answer to be scored")
    for position, answer in enumerate(answers):
        if not isinstance(answer, str):
            raise TypeError(f"answers[{position}] must be a str, got {answer!r}")
    return answers


def vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """VQA accuracy of ``prediction`` against the annotators' ``answers``.

    Each annotator is left out in turn. In a round, the prediction scores
    ``min(1, m / 3)``, where ``m`` is the number of the remaining annotators who
    gave the same answer, and the result is the mean over the rounds (Antol et
    al., "VQA: Visual Question Answering", ICCV 2015; VQA v2 keeps the metric).
    With ten references, three matching annotators score 0.9 (the rounds that
    leave one of them out give 2/3) and four score 1. Answers are compared after
    ``normalize_vqa_answer``, applied to the prediction and to every reference
    whether or not the references agree, as https://visualqa.org/evaluation.html
    describes; the official code skips it when every reference is the same.

    The rounds follow the number of references given. With ``n`` references the
    best score is ``min(1, (n - 1) / 3)``: four or more can reach 1, and a single
    reference, whose only round leaves nothing to match, scores 0, as in the
    official evaluation. Data with one answer per question needs another scorer.
    """
    references = [normalize_vqa_answer(answer) for answer in _references(prediction, answers)]
    predicted = normalize_vqa_answer(prediction)
    matches = sum(answer == predicted for answer in references)
    rounds = [
        min(1, (matches - (answer == predicted)) / VQA_FULL_CREDIT_MATCHES) for answer in references
    ]
    return sum(rounds) / len(rounds)


def _levenshtein(a: str, b: str) -> int:
    """The fewest one-character insertions, deletions and substitutions turning ``a`` into ``b``."""
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other))
            )
        previous = current
    return previous[-1]


def _similarity(prediction: str, answer: str) -> float:
    """``1 - NL`` of the two strings, or 0 once ``NL`` exceeds ``ANLS_THRESHOLD``."""
    longest = max(len(prediction), len(answer))
    distance = _levenshtein(prediction, answer) / longest if longest else 0.0
    return 1 - distance if distance <= ANLS_THRESHOLD else 0.0


def _collapsed(text: str) -> str:
    """``text`` lowercased, with no whitespace at either end and each inner run one space."""
    return " ".join(text.lower().split())


def anls(prediction: str, answers: Sequence[str]) -> float:
    """Normalized Levenshtein similarity of ``prediction`` to its closest reference.

    Both strings are lowercased and their whitespace collapsed: none is left at
    either end, and each run inside becomes one space, so ``"123\\nMain  St"``
    equals ``"123 main st"``. The similarity to a reference is ``1 - NL``, where
    ``NL`` is the Levenshtein distance between the collapsed strings divided by
    the length of the longer one, and it counts while ``NL`` is at most
    ``ANLS_THRESHOLD``. The score is the best over the references, and its mean
    over a task is ANLS (Biten et al., "Scene Text Visual Question Answering",
    ICCV 2019, equation 1; the DocVQA metric). Equation 1 leaves ``NL`` equal to
    the threshold open: here a similarity of exactly one half scores 0.5, as in
    the evaluation script distributed with the DocVQA datasets (docvqa.org) and
    in lmms-eval. This departs from the recipe's ``DOCVQAScorer``, which scores
    it 0 (``nl < tau``). Two empty strings are identical.
    """
    references = _references(prediction, answers)
    predicted = _collapsed(prediction)
    return max(_similarity(predicted, _collapsed(answer)) for answer in references)


def _trimmed(text: str) -> str:
    """``text`` stripped, lowercased, then without trailing ``?``, ``.`` and ``!``.

    Whitespace is removed first, so any left before a removed mark stays.
    """
    return text.strip().lower().rstrip("?.!")


def _number(text: str) -> float | None:
    """``text`` as a number, a trailing ``%`` read as hundredths, or ``None`` when it is not one."""
    try:
        return float(text.rstrip("%")) / 100 if text.endswith("%") else float(text)
    except ValueError:
        return None


def _matches_relaxed(prediction: str, answer: str) -> bool:
    predicted, target = _trimmed(prediction), _trimmed(answer)
    predicted_value, target_value = _number(predicted), _number(target)
    if predicted_value is None or target_value is None:
        return predicted == target
    error = abs(predicted_value - target_value) / (abs(target_value) + ZERO_TARGET_GUARD)
    return error <= RELAXED_TOLERANCE


def relaxed_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """1 when ``prediction`` matches a reference, numbers within 5 percent, and 0 otherwise.

    Each side is stripped of surrounding whitespace, lowercased, and then stripped
    of trailing ``?``, ``.`` and ``!``, in that order, so whitespace before a
    trailing mark stays (``"yes !"`` becomes ``"yes "``). A side is then a number
    when ``float`` parses it, or parses it once a trailing ``%`` is removed, which
    makes it hundredths: ``"45%"`` is 0.45, so it matches ``"45.1%"`` and
    ``"0.45"`` but not ``"45"``, as in the relaxed correctness of Pix2Struct (Lee
    et al., "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language
    Understanding", ICML 2023). When both sides are numbers, the prediction is
    correct if ``|prediction - target| / (|target| + ZERO_TARGET_GUARD)`` is at
    most ``RELAXED_TOLERANCE``; the guard keeps a zero target defined, and only a
    prediction within 5e-12 of zero matches one. Any other pair must be the same
    string. A comma or a unit makes an answer text, while ``float`` reads an
    underscore between digits as a separator (``"1_000"`` is 1000), and ``nan``
    and ``inf`` parse but match nothing. With several references any match counts
    (Masry et al., "ChartQA: A Benchmark for Question Answering about Charts with
    Visual and Logical Reasoning", Findings of ACL 2022, section 5.1, after
    Methani et al., "PlotQA: Reasoning over Scientific Plots", WACV 2020).
    """
    return float(
        any(_matches_relaxed(prediction, answer) for answer in _references(prediction, answers))
    )


_DEFAULT_CHOICES = list("ABCDEFGH")

_CONTAINER_START = re.compile(r"<answer\b[^<>]*>|(?<![\w</])answer>|\\boxed\s*\{", re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer\b", re.IGNORECASE)
_ANSWER_MARKUP = re.compile(r"<answer\b[^<>]*>|(?<![\w</])answer>|</answer\s*>", re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer\s*>", re.IGNORECASE)
_DECLARATION = re.compile(
    r"(?P<prefix>"
    r"(?<!\w)(?:(?:the|my)\s+)?(?:(?:correct|best|final)\s+)?(?:answer|option|choice)\s*(?:is\b|[:：=])"
    r"|(?<!\w)i\s+(?:choose|select|pick)\b"
    r"|(?:最终)?答案\s*(?:是|为|[:：])|答\s*[:：]|选项|选|옵션|정답은|답은|답\s*[:：]|答えは"
    r")"
    r"|(?P<postfix>(?<!\w)option\s+(?P<option>[a-z])(?!\w)\s+is\s+(?:the\s+)?(?:correct|right|best)\b)",
    re.IGNORECASE,
)
_OPTION_PREFIX = re.compile(r"(?:option|choice)\b\s*[:：]?\s*", re.IGNORECASE)
_LATEX_WRAPPER = re.compile(r"\\(?:text|mathrm|mathbf|textbf)\s*\{")
_PAREN_CHOICE = re.compile(r"(?<!\w)(?:\(\s*([A-Za-z])\s*\)|（\s*([A-Za-z])\s*）)(?!\w)")
_OPTION_LABEL = re.compile(r"(?<!\w)([A-Za-z])[.):](?=\s|$)")
_ASSERTION = re.compile(
    r"(?:because\b|(?:is|seems|looks|appears)\s+(?:(?:the|a)\s+)?(?:correct|right|best|answer|option|choice)\b)",
    re.IGNORECASE,
)
_ALTERNATIVE = re.compile(
    r"(?:and\s*/\s*or\b|and\b|or\b|[/&,+|]|或(?:者)?|和|또는|か|と)\s*", re.IGNORECASE
)
_IMPLICIT_ALTERNATIVES = re.compile(r"(?<!\w)[A-Z]\s*(?:[,/&+|]|and\b|or\b)\s*[A-Z](?!\w)")
_URL = re.compile(r"(?<!\w)(?:[a-z][a-z0-9+.-]*://|www\.)\S*", re.IGNORECASE)
_TRAILING_SEGMENT = re.compile(r"(?:^|[\n,;，；])\s*([^\n,;，；]+)\s*$")
_WRAPPERS = (
    ("**", "**"),
    ("__", "__"),
    ("`", "`"),
    ("*", "*"),
    ("_", "_"),
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("(", ")"),
    ("（", "）"),
    ("[", "]"),
    ("{", "}"),
    ("$", "$"),
)
_PUNCTUATION = " \t\r\n.!?,;:。！？，；："
_EMPHASIS_CLOSERS = ("**", "__", "*", "_", "`")


class _Choice(NamedTuple):
    letter: str
    tail: str
    wrapped: bool
    uppercase: bool


def _word_character(character: str) -> bool:
    """Include Unicode combining marks and joins in word/identifier boundaries."""
    return (
        character.isalnum()
        or character == "_"
        or unicodedata.category(character) in {"Mn", "Mc", "Me", "Cf"}
    )


def _starts_with_alternative(text: str) -> bool:
    """Return whether ``text`` opens with a connector to another choice."""
    body = text.lstrip(" \t")
    connector = _ALTERNATIVE.match(body)
    if connector is None:
        return False
    return body.startswith("/") or _read_choice(body[connector.end() :]) is not None


def _read_choice(fragment: str, grammatical_suffix: str = "") -> _Choice | None:
    """Read one adjacent ASCII choice, with paired presentation wrappers."""
    text = fragment.lstrip().lstrip(":：=,，;；").lstrip()
    closers = []
    option_prefix_used = False
    while text:
        wrapper = _LATEX_WRAPPER.match(text)
        if wrapper:
            closers.append("}")
            text = text[wrapper.end() :].lstrip()
            continue
        prefix = _OPTION_PREFIX.match(text) if not option_prefix_used else None
        if prefix:
            option_prefix_used = True
            text = text[prefix.end() :].lstrip()
            continue
        pair = next((pair for pair in _WRAPPERS if text.startswith(pair[0])), None)
        if pair is None:
            break
        closers.append(pair[1])
        text = text[len(pair[0]) :].lstrip()

    if not text or not text[0].isascii() or not text[0].isalpha():
        return None
    letter, tail = text[0], text[1:]
    if grammatical_suffix and tail.startswith(grammatical_suffix):
        tail = tail[len(grammatical_suffix) :]
    if tail and _word_character(tail[0]) and not (closers and tail.startswith(closers[-1])):
        return None
    punctuation = ""
    labelled = False
    for closer in reversed(closers):
        separated = bool(tail[:1].isspace())
        tail = tail.lstrip()
        inner_punctuation = ".:：!?。！？" + (")" if closer != ")" else "")
        punctuation_end = len(tail) - len(tail.lstrip(inner_punctuation))
        punctuation += tail[:punctuation_end]
        labelled = labelled or any(mark in tail[:punctuation_end] for mark in ".:：)")
        remainder = tail[punctuation_end:]
        separated = separated or bool(remainder[:1].isspace())
        tail = remainder.lstrip()
        if not tail.startswith(closer):
            line, separator, rest = tail.partition("\n")
            closer_at = line.find(closer)
            skipped = line[:closer_at]
            if (
                closer not in _EMPHASIS_CLOSERS
                or not labelled
                or not separated
                or closer_at <= 0
                or _starts_with_alternative(skipped)
            ):
                return None
            tail = line[closer_at:] + separator + rest
        tail = tail[len(closer) :]
        labelled = labelled or closer == ")"
    tail = punctuation + tail
    if tail and _word_character(tail[0]):
        return None
    if (
        tail
        and tail[0] in ".:-/'’"
        and len(tail) > 1
        and (tail[1].isalnum() or tail[1] in "/_")
        and tail[0] != "/"
    ):
        return None
    return _Choice(letter.upper(), tail, bool(closers), letter.isupper())


def _has_alternative(choice: _Choice) -> bool:
    """Reject an adjacent second selection, not letters in an explanation."""
    tail = choice.tail.lstrip(" \t")
    label_punctuation = tail[:1] in (".", ":", ")", "：")
    if label_punctuation:
        tail = tail[1:].lstrip(" \t")
    connector = _ALTERNATIVE.match(tail)
    if connector:
        return tail.startswith("/") or _read_choice(tail[connector.end() :]) is not None
    if label_punctuation or tail.startswith(("\n", "\r")):
        return False
    other = _read_choice(tail)
    return other is not None and not other.tail.strip(_PUNCTUATION)


def _declared_answer(text: str, allowed: set[str]) -> str | None:
    """Return the last declaration, preserving an invalid declaration as empty."""
    answer = None
    for declaration in _DECLARATION.finditer(text):
        fragment = (
            declaration.group("option") + text[declaration.end() :]
            if declaration.group("postfix")
            else text[declaration.end() :]
        )
        marker = declaration.group("prefix") or ""
        suffix = (
            "입니다"
            if marker.startswith(("옵션", "정답은", "답은", "답"))
            else "です"
            if marker == "答えは"
            else ""
        )
        choice = _read_choice(fragment, grammatical_suffix=suffix)
        answer = (
            choice.letter
            if choice and choice.letter in allowed and not _has_alternative(choice)
            else ""
        )
    return answer


def _closing_brace(text: str, start: int) -> int | None:
    """Find a box's closing brace, allowing balanced presentation wrappers."""
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _last_container(text: str) -> str | None:
    """Scope to the last container; malformed final evidence never falls back."""
    tag_open = False
    for tag in _ANSWER_MARKUP.finditer(text):
        if tag.group().startswith("</"):
            if not tag_open:
                return ""
            tag_open = False
        else:
            if tag_open:
                return ""
            tag_open = True
    starts = list(_CONTAINER_START.finditer(text))
    raw_tags = list(_ANSWER_OPEN.finditer(text))
    tag_starts = {opening.start() for opening in starts if opening.group().startswith("<")}
    if any(tag.start() not in tag_starts for tag in raw_tags):
        return ""
    if not starts:
        return None
    box_end = -1
    for opening in starts:
        if not opening.group().startswith("\\"):
            continue
        if opening.start() < box_end:
            return ""
        closing_brace = _closing_brace(text, opening.end())
        box_end = closing_brace if closing_brace is not None else len(text)
    start = starts[-1]
    if not start.group().startswith("\\"):
        closing = _ANSWER_CLOSE.search(text, start.end())
        return text[start.end() : closing.start() if closing else len(text)]
    return text[start.end() : box_end] if box_end < len(text) else ""


def _implicit_answer(text: str, allowed: set[str]) -> str:
    """Accept structural labels while abstaining on conflicting option lists."""
    urls = [match.span() for match in _URL.finditer(text)]
    matches = [
        match
        for pattern in (_PAREN_CHOICE, _OPTION_LABEL)
        for match in pattern.finditer(text)
        if (match.start() == 0 or not _word_character(text[match.start() - 1]))
        and (match.end() == len(text) or not _word_character(text[match.end()]))
        and not any(start <= match.start() < end for start, end in urls)
    ]
    labels = {next(group for group in match.groups() if group).upper() for match in matches}
    for line in text.splitlines():
        choice = _read_choice(line)
        if choice and choice.uppercase and not choice.tail.strip(_PUNCTUATION):
            labels.add(choice.letter)
    candidates = set()
    for match in matches:
        if match.re is not _PAREN_CHOICE:
            continue
        choice = _read_choice(text[match.start() :])
        if choice and _has_alternative(choice):
            return ""
        candidates.add(next(group for group in match.groups() if group).upper())

    leading = _read_choice(text)
    if leading and _has_alternative(leading):
        return ""
    if leading:
        tail = leading.tail.lstrip()
        if (
            leading.wrapped
            or not tail.strip(_PUNCTUATION)
            or tail.startswith((".", ":", ")", "："))
            or _ASSERTION.match(tail)
        ):
            candidates.add(leading.letter)
    if not (leading and leading.letter in candidates) and any(
        not any(start <= match.start() < end for start, end in urls)
        for match in _IMPLICIT_ALTERNATIVES.finditer(text)
    ):
        return ""

    trailing = _TRAILING_SEGMENT.search(text)
    if trailing and not any(start <= trailing.start(1) < end for start, end in urls):
        choice = _read_choice(trailing.group(1))
        if choice and choice.uppercase and not choice.tail.strip(_PUNCTUATION):
            candidates.add(choice.letter)

    if len(labels | candidates) != 1 or not candidates:
        return ""
    return next(iter(candidates)) if candidates <= allowed else ""


def extract_mcq_answer(response: str, choices: list[str] | None = None) -> str:
    r"""Extract a single answer choice, or abstain with ``""``.

    The last ``<answer>``/``\boxed{}`` container scopes parsing to its content.
    Otherwise the last explicit declaration wins, including invalid final
    declarations: ``Answer: B. Final answer: Z`` does not fall back to B when Z
    is unavailable. Adjacent alternatives such as ``A or B`` return empty.

    Without explicit evidence, accept bare letters, leading punctuated labels,
    standalone parenthesized choices and uppercase final lines/comma-separated
    conclusions. Conflicting implicit labels abstain. Ordinary prose and
    embedded letters are not searched for fallback candidates. Markdown and
    paired presentation wrappers preserve choices, including a punctuated
    label wrapped with its option text (``**B. dog**``, ``**(C) Red**``).
    Lowercase choices require an answer format rather than a mention in prose.

    This deliberately conservative format policy cannot resolve arbitrary
    author intent. Callers must strip reasoning blocks before extraction and
    retain any benchmark-specific official scorer.

    Args:
        response: Model output, with reasoning already stripped upstream.
        choices: Single ASCII letters, case insensitive. None or an empty list
            defaults to A through H. Custom alphabets such as I through N work.

    Returns:
        An uppercase offered choice, or an empty string for absent, invalid,
        or ambiguous answer evidence.
    """
    if not response or not response.strip():
        return ""
    allowed = {
        choice.upper()
        for choice in choices or _DEFAULT_CHOICES
        if len(choice) == 1 and choice.isascii() and choice.isalpha()
    }
    text = response.strip()
    scope = _last_container(text)
    if scope is not None:
        if re.search(r"</?\w", scope):
            return ""
        declared = _declared_answer(scope, allowed)
        if declared is not None:
            return declared
        choice = _read_choice(scope)
        standalone = set()
        for line in scope.splitlines():
            line_choice = _read_choice(line)
            if line_choice and not line_choice.tail.strip(_PUNCTUATION):
                standalone.add(line_choice.letter)
        if choice and len(standalone | {choice.letter}) > 1:
            return ""
        return (
            choice.letter
            if choice and choice.letter in allowed and not _has_alternative(choice)
            else ""
        )
    declared = _declared_answer(text, allowed)
    return declared if declared is not None else _implicit_answer(text, allowed)


def _realworldqa_match(prediction: str, answer: str) -> bool:
    reference = answer.strip()
    if reference.upper() in REALWORLDQA_CHOICES:
        choices = list(REALWORLDQA_CHOICES)
        return extract_mcq_answer(prediction, choices=choices) == reference.upper()
    return prediction.lower().strip().rstrip(".") == reference.lower()


def realworldqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """1 when ``prediction`` matches a reference as lmms-eval 0.7.4 scores RealWorldQA, else 0.

    This is lmms-eval's ``realworldqa_process_results``. A reference is first
    stripped of surrounding whitespace. One that is a letter from A to D, in
    either case, names an option: the prediction matches it when
    ``extract_mcq_answer`` over the choices ``REALWORLDQA_CHOICES`` returns that
    letter, so ``"B. Left"``, ``"(B)"``, ``"**B**"`` and ``"The answer is B"``
    match ``"B"``, while a reply that names no option or more than one, such as
    ``"A or B"``, matches none. Any other reference must equal the prediction
    once the prediction is lowercased, stripped of surrounding whitespace and
    then of trailing periods, and the reference lowercased: ``"Downhill."``
    matches ``"Downhill"`` but ``"Downhill!"`` does not, and a number is
    compared as text, so ``"3.0"`` does not match ``"3"``. lmms-eval strips the
    prediction before this rule, which changes no score. With several references
    any match counts.
    """
    references = _references(prediction, answers)
    return float(any(_realworldqa_match(prediction, answer) for answer in references))


def mean_scores(scores: Iterable[SampleScore]) -> dict[str, float]:
    """The mean of each component over a task's samples.

    A plain value is the component ``"score"``. Every sample must report the same
    components; a task without samples, or samples that disagree, raise
    ``ValueError``.
    """
    rows = [score if isinstance(score, Mapping) else {"score": score} for score in scores]
    if not rows:
        raise ValueError("a task needs at least one scored sample")
    components = rows[0].keys()
    if any(row.keys() != components for row in rows):
        raise ValueError(
            f"samples report different score components; the first has {sorted(components)}"
        )
    return {name: statistics.fmean(row[name] for row in rows) for name in components}


@dataclass(frozen=True)
class Scorer:
    """How a task is scored.

    ``score(prediction, reference)`` rates one sample from its decoded text and
    its reference, and returns one value or a mapping of named components.
    ``aggregate`` reduces the sequence of a task's sample scores to its metrics by
    name, by default the mean of each component.
    """

    score: Callable[[str, Any], SampleScore]
    aggregate: Callable[[Sequence[Any]], dict[str, float]] = field(default=mean_scores)


SCORERS: dict[str, Scorer] = {
    "anls": Scorer(anls),
    "relaxed_accuracy": Scorer(relaxed_accuracy),
    "vqa_accuracy": Scorer(vqa_accuracy),
    "realworldqa_accuracy": Scorer(realworldqa_accuracy),
}


def get_scorer(name: str) -> Scorer:
    """The scorer registered as ``name``; an unknown name lists the valid ones."""
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; expected one of {sorted(SCORERS)}")
    return SCORERS[name]
