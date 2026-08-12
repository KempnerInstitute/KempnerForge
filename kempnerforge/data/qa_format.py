"""Prompt/target rendering for video question-answering corpora.

A QA corpus gives a question, some options, and which option is right. How that
becomes supervised text is a research choice, not a property of the corpus, so
it lives behind the ``qa_format`` registry (mirroring ``sampling_policy``) and
is selected per source from TOML:

- ``mcq_letter`` (default): target is the bare option letter — one or two
  supervised tokens, matching the usual multiple-choice eval protocol.
- ``mcq_letter_text``: target is ``"C. lay on floor"`` — still letter-parseable
  at eval, but supervises the answer wording too.
- ``mcq_text``: target is the answer text alone, for a generative setup.
- ``mcq_plain``: question and lettered options with no added instruction,
  matching how published multiple-choice benchmarks render their prompts.
- ``mcq_varied``: draws the marker, label alphabet, layout, wording and target
  shape per sample from ``seed``. The fixed formats above teach one cue, which a
  model then depends on; see ``_mcq_varied``.

Every format takes a ``seed``; the fixed ones ignore it.

The prompt (question plus options) is masked out of the loss by
``_tokenize_and_mask``; only the target is supervised.

Prompt and target are tokenized *independently* and then concatenated, so the
target carries a leading space rather than the prompt carrying a trailing one —
a trailing prompt space would otherwise tokenize as its own token.
"""

from __future__ import annotations

import random
from typing import NamedTuple

from kempnerforge.config.registry import registry

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class QAText(NamedTuple):
    """Rendered example: ``prompt`` is masked from the loss, ``target`` is not."""

    prompt: str
    target: str


def _option_letter(index: int, n_options: int) -> str:
    if not 0 <= index < n_options:
        raise ValueError(f"answer index {index} out of range for {n_options} options")
    if n_options > len(_LETTERS):
        raise ValueError(f"at most {len(_LETTERS)} options are supported (got {n_options})")
    return _LETTERS[index]


def _mcq_prompt(question: str, options: list[str], instruction: str, hint: str) -> str:
    """``[instruction] Question / lettered options / hint / "Answer:"``."""
    lines = [instruction.strip()] if instruction.strip() else []
    lines.append(f"Question: {question.strip()}")
    lines.extend(
        f"{_option_letter(i, len(options))}. {opt.strip()}" for i, opt in enumerate(options)
    )
    lines.append(hint)
    lines.append("Answer:")
    return "\n".join(lines)


@registry.register_qa_format("mcq_letter")
def _mcq_letter(
    question: str, options: list[str], answer_index: int, instruction: str,
    seed: int | None = None,
) -> QAText:
    """Supervise the option letter alone (``" C"``)."""
    prompt = _mcq_prompt(question, options, instruction, "Answer with the option's letter.")
    return QAText(prompt, f" {_option_letter(answer_index, len(options))}")


@registry.register_qa_format("mcq_letter_text")
def _mcq_letter_text(
    question: str, options: list[str], answer_index: int, instruction: str,
    seed: int | None = None,
) -> QAText:
    """Supervise letter and answer text (``" C. lay on floor"``)."""
    prompt = _mcq_prompt(question, options, instruction, "Answer with the option's letter.")
    letter = _option_letter(answer_index, len(options))
    return QAText(prompt, f" {letter}. {options[answer_index].strip()}")


@registry.register_qa_format("mcq_plain")
def _mcq_plain(
    question: str, options: list[str], answer_index: int, instruction: str,
    seed: int | None = None,
) -> QAText:
    """Question and lettered options, nothing else -- the published eval rendering.

    The other formats add ``Question: ``, a hint line and an ``Answer:`` cue,
    which is a training choice. Scoring against a published benchmark number
    requires the prompt that benchmark actually sends, so this format renders
    only what the harness renders.
    """
    lines = [instruction.strip()] if instruction.strip() else []
    lines.append(question.strip())
    lines.extend(
        f"{_option_letter(i, len(options))}. {opt.strip()}" for i, opt in enumerate(options)
    )
    return QAText("\n".join(lines), f" {_option_letter(answer_index, len(options))}")


@registry.register_qa_format("mcq_text")
def _mcq_text(
    question: str, options: list[str], answer_index: int, instruction: str,
    seed: int | None = None,
) -> QAText:
    """Supervise the answer text alone (``" lay on floor"``)."""
    prompt = _mcq_prompt(question, options, instruction, "Answer with the correct option.")
    _option_letter(answer_index, len(options))  # bounds check
    return QAText(prompt, f" {options[answer_index].strip()}")


# Rendering axes for mcq_varied. Markers are format strings taking one label.
_MARKERS_COMMON = ("{}. ", "{}) ", "{}: ", "({}) ")
_MARKERS_RARE = ("{}; ", "{} ", "{}=", "[{}] ", "<{}> ", "{{{}}} ", "{} -> ")

_NUMERALS = (
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
)
# (name used in the instruction, symbols). The name is interpolated so the
# instruction agrees with the labels actually rendered.
_ALPHABETS = (
    ("letter", tuple(_LETTERS)),
    ("letter", tuple(_LETTERS.lower())),
    ("number", tuple(str(i) for i in range(1, len(_LETTERS) + 1))),
    ("numeral", _NUMERALS),
)

# Instructions that ask for the label alone. ``{kind}`` is interpolated with the
# alphabet actually rendered, so the wording never asks for a "letter" while the
# options are numbered -- that mismatch is what makes a model answer "B" when the
# labels are 1/2.
_LABEL_INSTRUCTIONS = (
    "Answer with the option's {kind}.",
    "Answer with the option's {kind} from the given choices directly.",
    "Give the {kind} of the correct option and nothing else.",
    "Respond with a single option {kind}.",
    "Which {kind} is correct? Answer with that {kind} only.",
    "Identify the right option and return only its {kind}.",
    "Output just the option {kind}.",
    "Your answer should be one option {kind}, with no other text.",
    "Name the {kind} of the best option.",
    "Just the {kind}, please.",
    "Answer using only the {kind} labelling the correct option.",
    "Pick an option and reply with its {kind} alone.",
    "State the {kind} of the option you choose.",
    "Reply with one {kind} and stop.",
    "Return the correct option's {kind} on its own.",
)

# Instructions that ask for the option itself. The first three are the exact
# wordings the held-out benchmarks send (TempCompass, MLVU, PerceptionTest);
# training on them is what keeps those scores from measuring format mismatch
# instead of understanding.
_OPTION_INSTRUCTIONS = (
    "Please directly give the best option.",
    "Only give the best option.",
    "Answer with one of the options.",
    "Reply with the option that fits the video.",
    "Choose the option that best matches what you saw.",
    "Return the correct option exactly as written.",
    "Which of these describes the video?",
    "Select the best answer from the list.",
    "Respond with the matching option and nothing more.",
    "Answer the question by quoting one option.",
    "Give the option you believe is right.",
)

# Layouts, including plain conversational forms no benchmark uses, so the model
# handles an ordinary request as readily as a harness-formatted one.
_LAYOUTS = (
    "{q}\n{opts}\n{instr}",
    "Question: {q}\n{opts}\n{instr}",
    "Question: {q}\n\nOptions:\n{opts}\n\n{instr}",
    "{instr}\n{q}\n{opts}",
    "{instr}\n\n{q}\n\n{opts}",
    "Q: {q}\n\n{opts}\n\n{instr}",
    "{q}\n\n{opts}\n\n{instr}",
    "{q}\nOptions:\n{opts}\n{instr}",
    "{instr}\nQuestion: {q}\nChoices:\n{opts}",
    "Watch the video and answer:\n{q}\n{opts}\n{instr}",
    "Here is a question about the video.\n{q}\n{opts}\n{instr}",
    "Consider the video, then answer.\n{q}\n{opts}\n{instr}",
    "{q}\nChoices: {opts}\n{instr}",
    "### Question\n{q}\n\n### Options\n{opts}\n\n### Instruction\n{instr}",
)

# Half the renderings end on an explicit cue, half on the instruction, so the
# model does not learn that a trailing cue is what licenses an answer.
_CUES = ("Answer:", "Answer", "Best option:", "My answer:", "", "", "", "")

# Wording, layout and marker variety cost nothing: they change the wrapper, not
# what is supervised. The target shape does change supervision, so it stays
# deliberately conservative -- a bare label is what the multiple-choice eval
# protocol scores, so it remains the majority case. Yields label-only 0.60,
# label+text 0.36, unlabelled 0.04.
_P_LABEL_ONLY = 0.6
_P_LABELLED = 0.9
_P_RARE_MARKER = 0.2


@registry.register_qa_format("mcq_varied")
def _mcq_varied(
    question: str, options: list[str], answer_index: int, instruction: str,
    seed: int | None = None,
) -> QAText:
    """Randomise marker, label alphabet, layout, wording and target per sample.

    A single fixed rendering teaches the cue rather than the task: a model
    trained only on ``"Answer with the option's letter." / "Answer:"`` emits
    end-of-turn when that cue is absent, and answers with a letter when the
    options are numbered. Drawing every axis from ``seed`` keeps all of those
    renderings in distribution, and keeps the draw reproducible across ranks,
    dataloader workers and resumes.

    ``seed=None`` yields one arbitrary but fixed rendering, so an unseeded
    caller still gets deterministic output.
    """
    _option_letter(answer_index, len(options))  # bounds check (also caps at 26)
    # Not random.Random(None): that seeds from OS entropy, so a caller who
    # forgot the seed would silently produce different text on every process and
    # every resume. Fall back to a fixed seed instead.
    rng = random.Random(0 if seed is None else seed)

    label_only = rng.random() < _P_LABEL_ONLY
    labelled = label_only or rng.random() < _P_LABELLED

    if labelled:
        pool = _MARKERS_RARE if rng.random() < _P_RARE_MARKER else _MARKERS_COMMON
        marker = rng.choice(pool)
        kind, symbols = rng.choice([a for a in _ALPHABETS if len(a[1]) >= len(options)])
        labels = [marker.format(s) for s in symbols[: len(options)]]
        opts = "\n".join(f"{lab}{opt.strip()}" for lab, opt in zip(labels, options))
        if label_only:
            instr = rng.choice(_LABEL_INSTRUCTIONS).format(kind=kind)
            target = symbols[answer_index]
        else:
            instr = rng.choice(_OPTION_INSTRUCTIONS)
            target = f"{labels[answer_index]}{options[answer_index].strip()}"
    else:
        # Unlabelled options: the answer can only be the option text itself.
        opts = "\n".join(opt.strip() for opt in options)
        instr = rng.choice(_OPTION_INSTRUCTIONS)
        target = options[answer_index].strip()

    prompt = rng.choice(_LAYOUTS).format(q=question.strip(), opts=opts, instr=instr)
    if instruction.strip():
        prompt = f"{instruction.strip()}\n{prompt}"
    cue = rng.choice(_CUES)
    if cue:
        prompt = f"{prompt}\n{cue}"
    return QAText(prompt, f" {target}")


def format_multiple_choice(
    qa_format: str,
    *,
    question: str,
    options: list[str],
    answer_index: int,
    instruction: str = "",
    seed: int | None = None,
) -> QAText:
    """Render a multiple-choice question with the registered ``qa_format``.

    ``seed`` drives per-sample randomisation for ``mcq_varied``; the fixed
    formats ignore it, so callers can pass it unconditionally.
    """
    if not options:
        raise ValueError("a multiple-choice question needs at least one option")
    return registry.get_qa_format(qa_format)(question, options, answer_index, instruction, seed)


def format_open_ended(*, question: str, answer: str, instruction: str = "") -> QAText:
    """Render a free-text question. Has no options, so no format policy applies."""
    lines = [instruction.strip()] if instruction.strip() else []
    lines.append(f"Question: {question.strip()}")
    lines.append("Answer:")
    return QAText("\n".join(lines), f" {answer.strip()}")
