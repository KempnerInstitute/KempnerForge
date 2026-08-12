"""Unit tests for the ``qa_format`` registry policies.

Pure functions — no tokenizer, no video, no HF download.
"""

from __future__ import annotations

import re

import pytest

from kempnerforge.config.registry import registry
from kempnerforge.data.qa_format import format_multiple_choice, format_open_ended

OPTIONS = ["clap proudly", "the lady sitting down", "lay on floor"]
QUESTION = "what did the baby do"


class TestRegistry:
    def test_all_formats_registered(self):
        assert set(registry.list_qa_formats()) >= {"mcq_letter", "mcq_letter_text", "mcq_text"}

    def test_unknown_format_raises(self):
        with pytest.raises(KeyError, match="qa_format"):
            format_multiple_choice("bogus", question=QUESTION, options=OPTIONS, answer_index=0)


class TestMCQLetter:
    def test_target_is_the_bare_letter(self):
        out = format_multiple_choice(
            "mcq_letter", question=QUESTION, options=OPTIONS, answer_index=2
        )
        assert out.target == " C"

    def test_prompt_lists_every_option_with_a_letter(self):
        out = format_multiple_choice(
            "mcq_letter", question=QUESTION, options=OPTIONS, answer_index=0
        )
        assert f"Question: {QUESTION}" in out.prompt
        for letter, opt in zip("ABC", OPTIONS, strict=True):
            assert f"{letter}. {opt}" in out.prompt
        assert out.prompt.endswith("Answer:")

    def test_instruction_leads_the_prompt(self):
        out = format_multiple_choice(
            "mcq_letter",
            question=QUESTION,
            options=OPTIONS,
            answer_index=1,
            instruction="Watch the clip.",
        )
        assert out.prompt.startswith("Watch the clip.\nQuestion:")

    def test_target_leads_with_a_space_not_the_prompt(self):
        # Prompt and target are tokenized independently, so the separating space
        # must live on the target or it becomes a lone token.
        out = format_multiple_choice(
            "mcq_letter", question=QUESTION, options=OPTIONS, answer_index=0
        )
        assert not out.prompt.endswith(" ")
        assert out.target.startswith(" ")


class TestOtherFormats:
    def test_letter_text_supervises_both(self):
        out = format_multiple_choice(
            "mcq_letter_text", question=QUESTION, options=OPTIONS, answer_index=2
        )
        assert out.target == " C. lay on floor"

    def test_text_only_supervises_the_answer(self):
        out = format_multiple_choice("mcq_text", question=QUESTION, options=OPTIONS, answer_index=2)
        assert out.target == " lay on floor"

    def test_open_ended_has_no_options(self):
        out = format_open_ended(question="is the man proficient", answer="yes")
        assert out.target == " yes"
        assert "A." not in out.prompt
        assert out.prompt.endswith("Answer:")


class TestValidation:
    @pytest.mark.parametrize("bad_index", [-1, 3, 99])
    def test_out_of_range_answer_rejected(self, bad_index):
        with pytest.raises(ValueError, match="out of range"):
            format_multiple_choice(
                "mcq_letter", question=QUESTION, options=OPTIONS, answer_index=bad_index
            )

    def test_no_options_rejected(self):
        with pytest.raises(ValueError, match="at least one option"):
            format_multiple_choice("mcq_letter", question=QUESTION, options=[], answer_index=0)

    def test_more_options_than_letters_rejected(self):
        with pytest.raises(ValueError, match="at most 26"):
            format_multiple_choice(
                "mcq_letter",
                question=QUESTION,
                options=[f"o{i}" for i in range(27)],
                answer_index=0,
            )


class TestMCQVaried:
    """``mcq_varied`` must be reproducible, honest about the answer, and varied.

    Reproducible because a resume re-renders the same sample and must get the
    same text; honest because the target has to name the option at
    ``answer_index`` under every rendering; varied because a single fixed
    rendering is what makes a model depend on one cue.
    """

    def _render(self, seed, options=None, answer_index=1):
        return format_multiple_choice(
            "mcq_varied",
            question=QUESTION,
            options=options or OPTIONS,
            answer_index=answer_index,
            seed=seed,
        )

    def test_registered(self):
        assert "mcq_varied" in registry.list_qa_formats()

    def test_same_seed_is_identical(self):
        assert self._render(11) == self._render(11)

    def test_unseeded_is_still_deterministic(self):
        a = format_multiple_choice(
            "mcq_varied", question=QUESTION, options=OPTIONS, answer_index=0
        )
        b = format_multiple_choice(
            "mcq_varied", question=QUESTION, options=OPTIONS, answer_index=0
        )
        assert a == b

    def test_different_seeds_give_different_renderings(self):
        prompts = {self._render(s).prompt for s in range(60)}
        assert len(prompts) > 10, f"only {len(prompts)} distinct prompts in 60 draws"

    def test_target_always_identifies_the_right_option(self):
        # Whatever the label alphabet or target shape, the target must resolve to
        # the option at answer_index -- either its text, or its label as rendered.
        for seed in range(120):
            for ai in range(len(OPTIONS)):
                out = self._render(seed, answer_index=ai)
                target = out.target.strip()
                answer = OPTIONS[ai].strip()
                if target.endswith(answer):
                    continue  # option text, with or without a label prefix
                # label-only target: the label must sit against the right option
                line = next(
                    ln for ln in out.prompt.split("\n") if ln.strip().endswith(answer)
                )
                assert target in line, f"seed={seed} ai={ai}: {target!r} not in {line!r}"

    def test_never_leaks_another_option_as_the_target(self):
        for seed in range(120):
            out = self._render(seed, answer_index=0)
            for other in OPTIONS[1:]:
                assert not out.target.strip().endswith(other.strip())

    def test_all_label_alphabets_appear(self):
        kinds = set()
        for seed in range(300):
            p = self._render(seed).prompt
            if any(f"{m}clap proudly" in p for m in ("A. ", "A) ", "A: ", "(A) ")):
                kinds.add("upper")
            if any(f"{m}clap proudly" in p for m in ("a. ", "a) ", "a: ", "(a) ")):
                kinds.add("lower")
            if any(f"{m}clap proudly" in p for m in ("1. ", "1) ", "1: ", "(1) ")):
                kinds.add("number")
            if any(f"{m}clap proudly" in p for m in ("i. ", "i) ", "i: ", "(i) ")):
                kinds.add("numeral")
        assert kinds == {"upper", "lower", "number", "numeral"}, kinds

    def test_instruction_never_names_the_wrong_alphabet(self):
        # "Answer with the option's letter" over options labelled 1/2 is the exact
        # mismatch that makes a model answer 'B' when the labels are numbers, so an
        # instruction must never name an alphabet other than the rendered one.
        # Read the alphabet off the label-only target, which is the bare symbol:
        # at answer_index=1 the four alphabets give 'B'/'b'/'2'/'ii', all distinct.
        kinds = {"B": "letter", "b": "letter", "2": "number", "ii": "numeral"}
        seen = set()
        for seed in range(400):
            out = self._render(seed, answer_index=1)
            actual = kinds.get(out.target.strip())
            if actual is None:
                continue  # target is option text, not a bare label
            seen.add(actual)
            for wrong in set(kinds.values()) - {actual}:
                assert f"option {wrong}" not in out.prompt, (
                    f"seed={seed}: label is {actual} but prompt names {wrong}\n{out.prompt}"
                )
                assert f"option's {wrong}" not in out.prompt, (
                    f"seed={seed}: label is {actual} but prompt names {wrong}\n{out.prompt}"
                )
        assert seen == {"letter", "number", "numeral"}, seen

    def test_both_cued_and_uncued_prompts_occur(self):
        cued = sum(1 for s in range(200) if self._render(s).prompt.endswith("Answer:"))
        assert 0 < cued < 200, f"cued={cued}/200; need a mix so neither is required"

    def test_label_only_and_full_text_targets_both_occur(self):
        short = sum(1 for s in range(200) if len(self._render(s).target.strip()) <= 4)
        assert 0 < short < 200, f"label-only={short}/200; need both shapes"

    def test_two_options_supported(self):
        out = self._render(3, options=["yes it is", "no it is not"], answer_index=0)
        assert "yes it is" in out.prompt and "no it is not" in out.prompt

    def test_answer_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="answer index"):
            self._render(1, answer_index=99)
