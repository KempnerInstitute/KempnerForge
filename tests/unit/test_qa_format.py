"""Unit tests for the ``qa_format`` registry policies.

Pure functions — no tokenizer, no video, no HF download.
"""

from __future__ import annotations

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
