"""Tests for the answer normalization of VQA accuracy."""

from __future__ import annotations

import pytest
from evaluation.vqa_normalization import (
    ARTICLES,
    CONTRACTIONS,
    MAX_PERIODS_REMOVED,
    NUMBER_WORDS,
    PUNCTUATION,
    normalize_vqa_answer,
)


class TestWhitespaceAndCase:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("  Yes\n", "yes"),
            ("red\tcar", "red car"),
            ("red\ncar", "red car"),
            ("RED   Car", "red car"),
            ("", ""),
            (" \n\t ", ""),
        ],
    )
    def test_whitespace_is_collapsed_and_case_dropped(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected


class TestWords:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("Two", "2"),
            ("none", "0"),
            ("zero", "0"),
            ("ten", "10"),
            ("two dogs", "2 dogs"),
            ("eleven", "eleven"),
            ("twenty one", "twenty 1"),
        ],
    )
    def test_number_words_up_to_ten_become_digits(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("the red car", "red car"),
            ("A dog", "dog"),
            ("an apple", "apple"),
            ("The", ""),
            ("another", "another"),
        ],
    )
    def test_articles_are_dropped(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("dont", "don't"),
            ("Dont know", "don't know"),
            ("thats it", "that's it"),
            ("don't", "don't"),
            ("im", "im"),
        ],
    )
    def test_contractions_get_their_apostrophe(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected

    def test_the_irregular_entry_of_the_official_table_is_kept(self) -> None:
        assert normalize_vqa_answer("somebody'd") == "somebodyd"

    def test_the_tables_hold_only_entries_that_can_apply(self) -> None:
        assert all(key == key.lower() and key != value for key, value in CONTRACTIONS.items())
        assert all(key == key.lower() for key in NUMBER_WORDS)
        assert set(ARTICLES) == {"a", "an", "the"}


class TestPunctuation:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("yes, it is", "yes it is"),
            ("x-y - z", "xy z"),
            ("red (dark)", "red dark"),
        ],
    )
    def test_a_mark_touching_a_space_is_deleted_everywhere(
        self, answer: str, expected: str
    ) -> None:
        assert normalize_vqa_answer(answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("black/white", "black white"),
            ("t-shirt", "t shirt"),
            ("x_y", "x y"),
            ("what?", "what"),
            ("1000/2", "1000 2"),
        ],
    )
    def test_a_mark_touching_no_space_becomes_a_space(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("1,000", "1000"),
            ("100,978", "100978"),
            ("1,000/2", "10002"),
            ("3,4-5", "345"),
        ],
    )
    def test_a_comma_between_digits_deletes_every_mark(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected

    @pytest.mark.parametrize("answer", ["girl's", "2:50 pm", "#1", "50%", "$5"])
    def test_other_characters_are_kept(self, answer: str) -> None:
        assert normalize_vqa_answer(answer) == answer

    def test_every_listed_mark_is_removed(self) -> None:
        for mark in PUNCTUATION:
            assert normalize_vqa_answer(f"x{mark}y") == "x y"
            assert normalize_vqa_answer(f"x {mark}y") == "x y"


class TestPeriods:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("3.5", "3.5"),
            (".5", ".5"),
            ("no.5", "no.5"),
            ("5.", "5"),
            ("st. louis", "st louis"),
            ("u.s.a.", "usa"),
            ("e.g. 3.5.", "eg 3.5"),
        ],
    )
    def test_a_period_stays_only_before_a_digit(self, answer: str, expected: str) -> None:
        assert normalize_vqa_answer(answer) == expected

    def test_periods_past_the_cap_stay(self) -> None:
        assert normalize_vqa_answer("x" + "." * MAX_PERIODS_REMOVED) == "x"
        assert normalize_vqa_answer("x" + "." * (MAX_PERIODS_REMOVED + 1)) == "x."
