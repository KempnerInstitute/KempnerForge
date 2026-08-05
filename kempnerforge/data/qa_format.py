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

The prompt (question plus options) is masked out of the loss by
``_tokenize_and_mask``; only the target is supervised.

Prompt and target are tokenized *independently* and then concatenated, so the
target carries a leading space rather than the prompt carrying a trailing one —
a trailing prompt space would otherwise tokenize as its own token.
"""

from __future__ import annotations

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
def _mcq_letter(question: str, options: list[str], answer_index: int, instruction: str) -> QAText:
    """Supervise the option letter alone (``" C"``)."""
    prompt = _mcq_prompt(question, options, instruction, "Answer with the option's letter.")
    return QAText(prompt, f" {_option_letter(answer_index, len(options))}")


@registry.register_qa_format("mcq_letter_text")
def _mcq_letter_text(
    question: str, options: list[str], answer_index: int, instruction: str
) -> QAText:
    """Supervise letter and answer text (``" C. lay on floor"``)."""
    prompt = _mcq_prompt(question, options, instruction, "Answer with the option's letter.")
    letter = _option_letter(answer_index, len(options))
    return QAText(prompt, f" {letter}. {options[answer_index].strip()}")


@registry.register_qa_format("mcq_text")
def _mcq_text(question: str, options: list[str], answer_index: int, instruction: str) -> QAText:
    """Supervise the answer text alone (``" lay on floor"``)."""
    prompt = _mcq_prompt(question, options, instruction, "Answer with the correct option.")
    _option_letter(answer_index, len(options))  # bounds check
    return QAText(prompt, f" {options[answer_index].strip()}")


def format_multiple_choice(
    qa_format: str,
    *,
    question: str,
    options: list[str],
    answer_index: int,
    instruction: str = "",
) -> QAText:
    """Render a multiple-choice question with the registered ``qa_format``."""
    if not options:
        raise ValueError("a multiple-choice question needs at least one option")
    return registry.get_qa_format(qa_format)(question, options, answer_index, instruction)


def format_open_ended(*, question: str, answer: str, instruction: str = "") -> QAText:
    """Render a free-text question. Has no options, so no format policy applies."""
    lines = [instruction.strip()] if instruction.strip() else []
    lines.append(f"Question: {question.strip()}")
    lines.append("Answer:")
    return QAText("\n".join(lines), f" {answer.strip()}")
