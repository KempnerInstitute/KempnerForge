"""Answer scorers for evaluation tasks, selected by name.

A scorer rates one sample: the decoded prediction against that sample's
reference, which for the scorers here is its reference answers. The text scorers
take the prediction as a ``str`` and the references as a non-empty sequence of
``str``; anything else raises before any scoring (``TypeError``, or
``ValueError`` for no reference). A sample's score
is one value in [0, 1], or a mapping of named components for a scorer that
reports several. A task's metrics are the scorer's aggregate of its samples'
scores, by default the mean of each component.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    """``1 - NL`` of the two strings, or 0 once ``NL`` reaches ``ANLS_THRESHOLD``."""
    longest = max(len(prediction), len(answer))
    distance = _levenshtein(prediction, answer) / longest if longest else 0.0
    return 1 - distance if distance < ANLS_THRESHOLD else 0.0


def _collapsed(text: str) -> str:
    """``text`` lowercased, with no whitespace at either end and each inner run one space."""
    return " ".join(text.lower().split())


def anls(prediction: str, answers: Sequence[str]) -> float:
    """Normalized Levenshtein similarity of ``prediction`` to its closest reference.

    Both strings are lowercased and their whitespace collapsed: none is left at
    either end, and each run inside becomes one space, so ``"123\\nMain  St"``
    equals ``"123 main st"``. The similarity to a reference is ``1 - NL``, where
    ``NL`` is the Levenshtein distance between the collapsed strings divided by
    the length of the longer one, and it counts only while ``NL`` is below
    ``ANLS_THRESHOLD``. The score is the best over the references, and its mean
    over a task is ANLS (Biten et al., "Scene Text Visual Question Answering",
    ICCV 2019, equation 1; the DocVQA metric). Equation 1 leaves ``NL`` equal to
    the threshold open: here a similarity of exactly one half scores 0, as in
    Pix2Struct, while the evaluation script distributed with the DocVQA datasets
    (docvqa.org) scores it 0.5. Two empty strings are identical.
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
}


def get_scorer(name: str) -> Scorer:
    """The scorer registered as ``name``; an unknown name lists the valid ones."""
    if name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; expected one of {sorted(SCORERS)}")
    return SCORERS[name]
