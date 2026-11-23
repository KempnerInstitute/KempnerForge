"""Answer scorers for evaluation tasks, selected by name.

A scorer rates one sample: the decoded prediction against that sample's
reference, which is its reference answers for every scorer here but
``points_in_masks``, whose reference is the objects' masks, and
``mathvista_accuracy``, whose reference is the problem (``MathVistaReference``).
The text scorers take the prediction as a ``str`` and the references as a non-empty sequence of
``str``; anything else raises before any scoring (``TypeError``, or
``ValueError`` for no reference). A sample's score
is one value in [0, 1], or a mapping of named components for a scorer that
reports several (an error component may exceed 1). A task's metrics are the
scorer's aggregate of its samples' scores, by default the mean of each component.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .vqa_normalization import normalize_vqa_answer

SampleScore = float | Mapping[str, float]

VQA_FULL_CREDIT_MATCHES = 3
ANLS_THRESHOLD = 0.5
RELAXED_TOLERANCE = 0.05
ZERO_TARGET_GUARD = 1e-10


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


def exact_vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """``min(1, m / 3)`` for the ``m`` references ``prediction`` equals exactly, as A-OKVQA scores.

    Nothing is normalized and no annotator is left out, which is how the board
    scores A-OKVQA's direct answers.
    """
    references = _references(prediction, answers)
    matches = sum(prediction == answer for answer in references)
    return min(1.0, matches / VQA_FULL_CREDIT_MATCHES)


_COUNT_PHRASE = re.compile(r"\bi\s+count(?:ed)?\s+(\d+)", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
_INTEGER = re.compile(r"\d+")


def stated_count(text: str) -> int | None:
    """The count a counting answer states: the ``N`` of "I count N", else its last integer.

    Point tags are dropped first, as their coordinates are numbers too.
    """
    prose = _TAG.sub(" ", text)
    phrase = _COUNT_PHRASE.search(prose)
    if phrase:
        return int(phrase.group(1))
    integers = _INTEGER.findall(prose)
    return int(integers[-1]) if integers else None


def count_accuracy(prediction: str, answers: Sequence[str]) -> dict[str, float]:
    """Whether the stated count is the reference count, and how far off it is.

    ``accuracy`` is 1 for the right count, and ``mae`` its absolute error; an
    answer that states no count is wrong by the whole reference count.
    """
    reference = int(_references(prediction, answers)[0])
    stated = stated_count(prediction)
    if stated is None:
        return {"accuracy": 0.0, "mae": float(reference)}
    return {"accuracy": float(stated == reference), "mae": float(abs(stated - reference))}


_POINT_TAG = re.compile(r"<points?\b([^>]*)(?:/?>|$)", re.IGNORECASE)
_COORDINATE = re.compile(r"\b([xy])(\d*)\s*=\s*[\"']?\s*([-+]?\d*\.?\d+)", re.IGNORECASE)


def parse_points(text: str) -> list[tuple[float, float]]:
    """The ``(x, y)`` percent points of every ``<point>`` or ``<points>`` tag in ``text``.

    A tag's ``x``/``y`` (or ``x1``/``y1``, ...) attributes pair by index, and the
    parse is lenient: loose quotes and a missing tag end still yield points.
    """
    points: list[tuple[float, float]] = []
    for tag in _POINT_TAG.finditer(text):
        xs: dict[str, float] = {}
        ys: dict[str, float] = {}
        for coordinate in _COORDINATE.finditer(tag.group(1)):
            axis, index, value = (
                coordinate.group(1).lower(),
                coordinate.group(2),
                coordinate.group(3),
            )
            (xs if axis == "x" else ys)[index] = float(value)
        for index in sorted(xs, key=lambda key: int(key) if key else 0):
            if index in ys:
                points.append((xs[index], ys[index]))
    return points


def points_in_masks(prediction: str, masks: Any) -> dict[str, float]:
    """How the predicted points fall on the objects' masks, ``(objects, H, W)`` booleans.

    ``precision`` is the share of points inside any object, ``recall`` the share
    of objects with a point inside, and ``count_mae`` the gap between the counts
    of points and objects. A point in percent lands on the nearest mask pixel.
    Without an object, precision and recall are 0.
    """
    points = parse_points(prediction)
    objects = int(masks.shape[0]) if masks.size else 0
    count_mae = float(abs(len(points) - objects))
    if objects == 0:
        return {"precision": 0.0, "recall": 0.0, "count_mae": count_mae}
    _, height, width = masks.shape
    union = masks.any(axis=0)
    inside = 0
    found: set[int] = set()
    for x, y in points:
        column = min(width - 1, max(0, round(x / 100.0 * (width - 1))))
        row = min(height - 1, max(0, round(y / 100.0 * (height - 1))))
        inside += bool(union[row, column])
        found.update(int(i) for i in np.flatnonzero(masks[:, row, column]))
    return {
        "precision": inside / len(points) if points else 0.0,
        "recall": len(found) / objects,
        "count_mae": count_mae,
    }


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(prediction: str, answers: Sequence[str]) -> float:
    """1 when the last number in ``prediction`` equals the reference number, as GSM8K scores."""
    reference = _references(prediction, answers)[0]
    numbers = _NUMBER.findall(prediction)
    if not numbers:
        return 0.0
    stated = numbers[-1].replace(",", "").rstrip(".")
    try:
        return float(abs(float(stated) - float(reference)) < 1e-6)
    except ValueError:
        return float(stated == reference)


_LETTER = re.compile(r"\b([ABCD])\b")


def choice_letter(prediction: str, answers: Sequence[str]) -> float:
    """1 when the first standalone capital A-D in ``prediction`` is the reference letter."""
    reference = _references(prediction, answers)[0]
    letter = _LETTER.search(prediction)
    return float(letter is not None and letter.group(1) == reference)


MATHVISTA_QUESTION_TYPES = ("multi_choice", "free_form")
MATHVISTA_ANSWER_TYPES = ("text", "integer", "float", "list")
_LETTER_IN_PARENTHESES = re.compile(r"\(([a-zA-Z])\)")
_NUMBER_TYPES: dict[str, Callable[[str], int | float]] = {"integer": int, "float": float}


def _non_blank(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string, got {value!r}")


def _check_choice_of(name: str, value: Any, options: Sequence[str]) -> None:
    if value not in options:
        raise ValueError(f"{name} must be one of {list(options)}, got {value!r}")


@dataclass(frozen=True)
class MathVistaReference:
    """What a reply to a MathVista problem is scored against.

    ``answer`` is the reference answer and ``query`` the prompt the model was
    asked. ``question_type`` is ``multi_choice``, whose options are ``choices``
    and lettered from ``A``, or ``free_form``. ``answer_type`` is ``text``,
    ``integer``, ``float``, rounded to ``precision`` decimal places, or ``list``.
    ``extract(question, reply)`` reads the answer to a question out of a free
    reply. A problem that could never be scored is refused with ``ValueError``:
    a blank answer or query, an unknown type, a multiple-choice problem whose
    answer is not one of its options, a free-form ``text`` answer, or a
    ``float`` answer without a precision.
    """

    answer: str
    query: str
    question_type: str
    answer_type: str
    extract: Callable[[str, str], str]
    choices: tuple[str, ...] = ()
    precision: int | None = None

    def __post_init__(self) -> None:
        _non_blank("answer", self.answer)
        _non_blank("query", self.query)
        _check_choice_of("question_type", self.question_type, MATHVISTA_QUESTION_TYPES)
        _check_choice_of("answer_type", self.answer_type, MATHVISTA_ANSWER_TYPES)
        if self.question_type == "multi_choice":
            if not all(isinstance(choice, str) for choice in self.choices):
                raise ValueError(f"the options must be strings, got {self.choices!r}")
            if self.answer not in self.choices:
                raise ValueError(f"the answer {self.answer!r} is not one of {list(self.choices)}")
        elif self.answer_type == "text":
            raise ValueError("a free-form problem has an integer, float or list answer, not text")
        if self.answer_type == "float" and self.precision is None:
            raise ValueError("a float answer needs a precision")


def mathvista_extraction(reply: str, reference: MathVistaReference) -> str:
    """The answer ``reply`` gives before it is normalized, as the recipe extracts it.

    An empty reply gives ``""``. A multiple-choice reply that is exactly one of
    the options is taken as it is, and so is the number of an ``integer`` or
    ``float`` answer's reply that ``int`` or ``float`` parses whole, written back
    by ``str``. Any other reply goes to ``reference.extract``, asked the
    problem's query.
    """
    if reply == "":
        return ""
    if reference.question_type == "multi_choice" and reply in reference.choices:
        return reply
    parse = _NUMBER_TYPES.get(reference.answer_type)
    if parse is not None:
        try:
            return str(parse(reply))
        except ValueError:
            pass
    return reference.extract(reference.query, reply)


def _closest_option(text: str, choices: Sequence[str]) -> str:
    """The option of ``choices`` fewest edits from ``text``; the first of equally close ones."""
    distances = [_levenshtein(text, choice) for choice in choices]
    return choices[distances.index(min(distances))]


def normalize_mathvista_answer(extraction: str, reference: MathVistaReference) -> str | None:
    """``extraction`` in the form of the reference answer, as MathVista's evaluation normalizes it.

    A multiple-choice extraction loses its surrounding whitespace, and the first
    letter in parentheses, ``(b)``, becomes that capital. A capital that letters
    an option (``A`` the first) gives that option; anything else gives the option
    the fewest Levenshtein edits away. An ``integer`` answer becomes
    ``int(float(extraction))`` and a ``float`` answer ``round(float(extraction),
    precision)``, each written by ``str``, or ``None`` when ``float`` cannot read
    the extraction; a ``list`` answer is the extraction as it is.
    """
    if reference.question_type == "multi_choice":
        text = extraction.strip()
        letters = _LETTER_IN_PARENTHESES.findall(text)
        if letters:
            text = letters[0].upper()
        labels = [chr(ord("A") + index) for index in range(len(reference.choices))]
        if text in labels:
            return reference.choices[labels.index(text)]
        return _closest_option(text, reference.choices)
    try:
        if reference.answer_type == "integer":
            return str(int(float(extraction)))
        if reference.answer_type == "float" and reference.precision is not None:
            return str(round(float(extraction), reference.precision))
    except (ValueError, OverflowError):
        return None
    return extraction


def mathvista_accuracy(prediction: str, reference: MathVistaReference) -> float:
    """1 when the answer ``prediction`` gives is the reference answer, as the recipe scores it.

    The answer is extracted by ``mathvista_extraction`` and normalized by
    ``normalize_mathvista_answer``, after MathVista's ``extract_answer.py`` and
    ``calculate_score.py``, and compared with the reference answer as text
    (``str``), as the recipe compares them.

    Raises:
        TypeError: ``prediction`` is not a ``str`` or ``reference`` is not a
            ``MathVistaReference``.
    """
    if not isinstance(prediction, str):
        raise TypeError(f"prediction must be a str, got {prediction!r}")
    if not isinstance(reference, MathVistaReference):
        raise TypeError(f"reference must be a MathVistaReference, got {reference!r}")
    normalized = normalize_mathvista_answer(mathvista_extraction(prediction, reference), reference)
    return float(str(normalized) == reference.answer)


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
    "choice_letter": Scorer(choice_letter),
    "count_accuracy": Scorer(count_accuracy),
    "exact_vqa_accuracy": Scorer(exact_vqa_accuracy),
    "final_number": Scorer(final_number),
    "mathvista_accuracy": Scorer(mathvista_accuracy),
    "points_in_masks": Scorer(points_in_masks),
    "relaxed_accuracy": Scorer(relaxed_accuracy),
    "vqa_accuracy": Scorer(vqa_accuracy),
}


def get_scorer(name: str) -> Scorer:
    """The scorer registered as ``name``; an unknown name lists the valid ones."""
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; expected one of {sorted(SCORERS)}")
    return SCORERS[name]
