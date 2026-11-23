"""Tests for the answer scorers and their lookup."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pytest
from evaluation.scorers import (
    SCORERS,
    Scorer,
    anls,
    choice_letter,
    count_accuracy,
    exact_vqa_accuracy,
    final_number,
    get_scorer,
    mean_scores,
    parse_points,
    points_in_masks,
    relaxed_accuracy,
    stated_count,
    vqa_accuracy,
)

TEXT_SCORERS = {
    "anls": anls,
    "choice_letter": choice_letter,
    "exact_vqa_accuracy": exact_vqa_accuracy,
    "final_number": final_number,
    "relaxed_accuracy": relaxed_accuracy,
    "vqa_accuracy": vqa_accuracy,
}


def _votes(matching: int, total: int) -> list[str]:
    return ["yes"] * matching + ["no"] * (total - matching)


class TestVQAAccuracy:
    @pytest.mark.parametrize(
        ("matching", "expected"),
        [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)],
    )
    def test_ten_annotators(self, matching: int, expected: float) -> None:
        assert vqa_accuracy("yes", _votes(matching, 10)) == pytest.approx(expected)

    def test_the_order_of_the_references_does_not_matter(self) -> None:
        assert vqa_accuracy("yes", ["no", "yes", "no", "no", "yes"]) == pytest.approx(
            vqa_accuracy("yes", _votes(2, 5))
        )

    @pytest.mark.parametrize("count", range(1, 13))
    def test_every_reference_count_has_a_best_score(self, count: int) -> None:
        best = min(1.0, (count - 1) / 3)
        assert vqa_accuracy("yes", ["yes"] * count) == pytest.approx(best)

    @pytest.mark.parametrize("answers", [["yes"], ["no"], ("yes",)])
    def test_a_single_reference_scores_zero(self, answers: Sequence[str]) -> None:
        assert vqa_accuracy("yes", answers) == 0.0

    @pytest.mark.parametrize(
        ("matching", "total", "expected"),
        [(1, 2, 1 / 6), (2, 2, 1 / 3), (2, 3, 4 / 9), (2, 5, 8 / 15), (3, 9, 8 / 9)],
    )
    def test_each_reference_is_left_out_in_turn(
        self, matching: int, total: int, expected: float
    ) -> None:
        assert vqa_accuracy("yes", _votes(matching, total)) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("prediction", "answers"),
        [
            ("yes", ["Yes."] * 10),
            ("Two", ["2"] * 10),
            ("dog", ["the dog"] * 4 + ["a dog"] * 6),
            (" DONT\n", ["don't"] * 10),
        ],
    )
    def test_both_sides_are_normalized_even_when_the_references_agree(
        self, prediction: str, answers: list[str]
    ) -> None:
        assert vqa_accuracy(prediction, answers) == 1.0

    @pytest.mark.parametrize("prediction", ["", "maybe", "yes no"])
    def test_an_answer_no_annotator_gave_scores_zero(self, prediction: str) -> None:
        assert vqa_accuracy(prediction, _votes(6, 10)) == 0.0

    def test_a_tuple_of_references_is_scored(self) -> None:
        assert vqa_accuracy("yes", tuple(_votes(4, 10))) == 1.0


class TestANLS:
    @pytest.mark.parametrize(
        ("prediction", "answer", "expected"),
        [
            ("hello", "hello", 1.0),
            (" Hello World\n", "hello world", 1.0),
            ("kitten", "sitten", 1 - 1 / 6),
            ("sunday", "saturday", 1 - 3 / 8),
            ("abcd", "abcdef", 1 - 2 / 6),
            ("abcdef", "abcd", 1 - 2 / 6),
            ("newyork", "new york", 1 - 1 / 8),
        ],
    )
    def test_similarity_is_one_minus_the_normalized_distance(
        self, prediction: str, answer: str, expected: float
    ) -> None:
        assert anls(prediction, [answer]) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("prediction", "answer"),
        [
            ("123\nMain St", "123 main st"),
            ("new  york", "new york"),
            ("new york", "new \t york"),
            ("\n new\r\nyork ", "New York"),
        ],
    )
    def test_each_run_of_whitespace_counts_as_one_space(self, prediction: str, answer: str) -> None:
        assert anls(prediction, [answer]) == 1.0

    def test_whitespace_does_not_lengthen_the_compared_strings(self) -> None:
        assert anls("abd" + " " * 20, ["abc"]) == pytest.approx(2 / 3)
        assert anls("a   b   d", ["a b c"]) == pytest.approx(1 - 1 / 5)

    @pytest.mark.parametrize(
        ("prediction", "answer"),
        [("ab", "ax"), ("abcd", "axyd"), ("abc", "abcdef"), ("flaw", "lawn")],
    )
    def test_exactly_half_of_the_longer_string_wrong_scores_one_half(
        self, prediction: str, answer: str
    ) -> None:
        assert anls(prediction, [answer]) == 0.5

    @pytest.mark.parametrize(
        ("prediction", "answer"),
        [("abc", "axy"), ("abcde", "abxyz"), ("abc", "abcdefg"), ("a", "b"), ("abcd", "wxyz")],
    )
    def test_more_than_half_of_the_longer_string_wrong_scores_zero(
        self, prediction: str, answer: str
    ) -> None:
        assert anls(prediction, [answer]) == 0.0

    def test_just_under_half_wrong_still_counts(self) -> None:
        assert anls("abc", ["abx"]) == pytest.approx(2 / 3)
        assert anls("abcde", ["abcxy"]) == pytest.approx(3 / 5)

    def test_the_closest_reference_counts(self) -> None:
        assert anls("hello", ["zzzzz", "hallo", "hello"]) == 1.0
        assert anls("hallo", ["zzzzz", "hello"]) == pytest.approx(0.8)

    @pytest.mark.parametrize("prediction", ["", "  ", "\n"])
    def test_an_empty_prediction_matches_only_an_empty_reference(self, prediction: str) -> None:
        assert anls(prediction, ["abc"]) == 0.0
        assert anls(prediction, ["abc", " "]) == 1.0


class TestRelaxedAccuracy:
    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [("100", 1.0), ("95", 1.0), ("105", 1.0), ("94.9", 0.0), ("105.1", 0.0), ("0", 0.0)],
    )
    def test_a_number_within_five_percent_is_correct(
        self, prediction: str, expected: float
    ) -> None:
        assert relaxed_accuracy(prediction, ["100"]) == expected

    @pytest.mark.parametrize(
        ("prediction", "answer", "expected"),
        [
            ("1.05", "1", 1.0),
            ("0.95", "1", 1.0),
            ("1.06", "1", 0.0),
            ("-95", "-100", 1.0),
            ("-105", "-100", 1.0),
            ("95", "-100", 0.0),
        ],
    )
    def test_the_error_is_relative_to_the_size_of_the_target(
        self, prediction: str, answer: str, expected: float
    ) -> None:
        assert relaxed_accuracy(prediction, [answer]) == expected

    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [("0", 1.0), ("0.0", 1.0), ("-0", 1.0), ("1e-12", 1.0), ("1e-11", 0.0), ("0.001", 0.0)],
    )
    def test_a_zero_target_needs_a_zero_prediction(self, prediction: str, expected: float) -> None:
        assert relaxed_accuracy(prediction, ["0"]) == expected

    @pytest.mark.parametrize(
        ("prediction", "answer"),
        [("1e2", "100"), (".5", "0.5"), ("+5", "5"), ("5.", "5"), (" 5 \n", "5"), ("5.0", "5")],
    )
    def test_numbers_are_read_in_any_form_float_parses(self, prediction: str, answer: str) -> None:
        assert relaxed_accuracy(prediction, [answer]) == 1.0

    @pytest.mark.parametrize(
        ("prediction", "answer", "expected"),
        [
            ("45.1%", "45%", 1.0),
            ("4.9%", "5%", 1.0),
            ("4.7%", "5%", 0.0),
            ("0.45", "45%", 1.0),
            ("45%", "0.45", 1.0),
            ("45 %", "0.45", 1.0),
            ("45%.", "45%", 1.0),
            ("-5%", "-0.05", 1.0),
            ("0%", "0", 1.0),
            ("45%", "45", 0.0),
            ("45", "45%", 0.0),
        ],
    )
    def test_a_trailing_percent_sign_makes_hundredths(
        self, prediction: str, answer: str, expected: float
    ) -> None:
        assert relaxed_accuracy(prediction, [answer]) == expected

    @pytest.mark.parametrize(
        ("prediction", "answer", "expected"),
        [
            ("%", "%", 1.0),
            ("%5", "5", 0.0),
            ("5%x", "5%x", 1.0),
            ("1,000", "1000", 0.0),
            ("1,000", "1,000", 1.0),
            ("$5", "5", 0.0),
            ("about 5", "5", 0.0),
            ("five", "5", 0.0),
        ],
    )
    def test_a_number_with_a_sign_separator_or_unit_is_text(
        self, prediction: str, answer: str, expected: float
    ) -> None:
        assert relaxed_accuracy(prediction, [answer]) == expected

    @pytest.mark.parametrize("answer", ["nan", "inf", "-inf", "nan%", "inf%"])
    def test_values_that_are_not_finite_match_nothing(self, answer: str) -> None:
        assert relaxed_accuracy(answer, [answer]) == 0.0

    @pytest.mark.parametrize(
        ("prediction", "answer", "expected"),
        [
            ("Yes", "yes", 1.0),
            (" YES. ", "yes", 1.0),
            ("yes!", "Yes?", 1.0),
            ("no", "yes", 0.0),
            ("u.s.", "us", 0.0),
            ("new  york", "new york", 0.0),
            ("", "yes", 0.0),
            ("", "0", 0.0),
        ],
    )
    def test_text_must_match_but_for_case_edges_and_final_marks(
        self, prediction: str, answer: str, expected: float
    ) -> None:
        assert relaxed_accuracy(prediction, [answer]) == expected

    def test_any_matching_reference_counts(self) -> None:
        assert relaxed_accuracy("12", ["10", "12.5"]) == 1.0
        assert relaxed_accuracy("12", ["10", "14"]) == 0.0
        assert relaxed_accuracy("b", ("a", "B")) == 1.0


class TestReferences:
    @pytest.mark.parametrize("name", sorted(TEXT_SCORERS))
    def test_a_scorer_needs_a_reference(self, name: str) -> None:
        with pytest.raises(ValueError, match="at least one reference answer"):
            TEXT_SCORERS[name]("yes", [])

    @pytest.mark.parametrize("name", sorted(TEXT_SCORERS))
    def test_one_string_is_not_a_list_of_references(self, name: str) -> None:
        with pytest.raises(TypeError, match="not one string"):
            TEXT_SCORERS[name]("yes", "yes")

    @pytest.mark.parametrize("name", sorted(TEXT_SCORERS))
    @pytest.mark.parametrize("prediction", [None, 0, False, b"yes", ["yes"]])
    def test_the_prediction_must_be_a_string(self, name: str, prediction: object) -> None:
        with pytest.raises(TypeError, match="prediction must be a str"):
            TEXT_SCORERS[name](prediction, ["yes"])

    @pytest.mark.parametrize("name", sorted(TEXT_SCORERS))
    @pytest.mark.parametrize("answers", [None, 3, b"yes", {"yes"}, iter(["yes"])])
    def test_the_references_must_be_a_sequence(self, name: str, answers: object) -> None:
        with pytest.raises(TypeError, match="answers must be a sequence of str"):
            TEXT_SCORERS[name]("yes", answers)

    @pytest.mark.parametrize("name", sorted(TEXT_SCORERS))
    @pytest.mark.parametrize("bad", [None, 0, b"yes", ["yes"]])
    @pytest.mark.parametrize("position", [0, 1])
    def test_every_reference_must_be_a_string(self, name: str, bad: object, position: int) -> None:
        answers = ["yes", "yes"]
        answers[position] = bad
        with pytest.raises(TypeError, match=rf"answers\[{position}\] must be a str"):
            TEXT_SCORERS[name]("yes", answers)

    @pytest.mark.parametrize("name", sorted(TEXT_SCORERS))
    @pytest.mark.parametrize("prediction", ["yes", "no", ""])
    def test_a_score_is_a_float_between_zero_and_one(self, name: str, prediction: str) -> None:
        score = TEXT_SCORERS[name](prediction, _votes(7, 10))
        assert type(score) is float and 0.0 <= score <= 1.0


class TestExactVQAAccuracy:
    @pytest.mark.parametrize(("matching", "expected"), [(0, 0.0), (1, 1 / 3), (3, 1.0), (5, 1.0)])
    def test_a_third_per_exact_match(self, matching: int, expected: float) -> None:
        assert exact_vqa_accuracy("yes", _votes(matching, 10)) == pytest.approx(expected)

    @pytest.mark.parametrize("prediction", ["Yes", "yes.", " yes", "a yes"])
    def test_nothing_is_normalized(self, prediction: str) -> None:
        assert exact_vqa_accuracy(prediction, _votes(10, 10)) == 0.0


class TestCountAccuracy:
    @pytest.mark.parametrize(
        ("text", "count"),
        [
            ("I count 5 people in the image.", 5),
            ('<points x1="12.5" y1="40" x2="3" y2="4">2 cats</points> i counted 2', 2),
            ("There are 3 cats and 4 dogs.", 4),
            ('<point x="10" y="20" alt="dog">dog</point>', None),
            ("none", None),
        ],
    )
    def test_the_stated_count_prefers_the_counting_phrase(self, text: str, count: int) -> None:
        assert stated_count(text) == count

    def test_the_count_scores_accuracy_and_error(self) -> None:
        assert count_accuracy("I count 5.", ["5"]) == {"accuracy": 1.0, "mae": 0.0}
        assert count_accuracy("I count 2.", ["5"]) == {"accuracy": 0.0, "mae": 3.0}
        assert count_accuracy("Many.", ["5"]) == {"accuracy": 0.0, "mae": 5.0}


class TestPointsInMasks:
    def test_points_are_read_from_every_tag(self) -> None:
        text = '<point x="10.0" y="20.0" alt="a">a</point> then <points x1=1 y1=2 x2="3" y2=\'4\''
        assert parse_points(text) == [(10.0, 20.0), (1.0, 2.0), (3.0, 4.0)]
        assert parse_points("no points here") == []

    def test_precision_recall_and_count_error(self) -> None:
        masks = np.zeros((2, 3, 5), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 2, 4] = True
        prediction = '<points x1="0" y1="0" x2="100" y2="100" x3="50" y3="50">'
        assert points_in_masks(prediction, masks) == {
            "precision": 2 / 3,
            "recall": 1.0,
            "count_mae": 1.0,
        }
        assert points_in_masks('<point x="0" y="0">', masks) == {
            "precision": 1.0,
            "recall": 0.5,
            "count_mae": 1.0,
        }
        assert points_in_masks("none", masks) == {"precision": 0.0, "recall": 0.0, "count_mae": 2.0}

    def test_without_an_object_only_the_count_error_moves(self) -> None:
        empty = np.zeros((0, 0, 0), dtype=bool)
        assert points_in_masks('<point x="1" y="1">', empty) == {
            "precision": 0.0,
            "recall": 0.0,
            "count_mae": 1.0,
        }


class TestFinalNumber:
    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [
            ("So the answer is 1,234.", 1.0),
            ("First 3, then 1234", 1.0),
            ("1234.0", 1.0),
            ("1234 apples, then 12", 0.0),
            ("no number", 0.0),
        ],
    )
    def test_the_last_number_is_compared(self, prediction: str, expected: float) -> None:
        assert final_number(prediction, ["1234"]) == expected


class TestChoiceLetter:
    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [("C", 1.0), ("The answer is C.", 1.0), ("a C", 1.0), ("B or C", 0.0), ("c", 0.0)],
    )
    def test_the_first_standalone_capital_letter_is_the_choice(
        self, prediction: str, expected: float
    ) -> None:
        assert choice_letter(prediction, ["C"]) == expected


class TestMeanScores:
    def test_plain_scores_average_into_one_component(self) -> None:
        assert mean_scores([1.0, 0.0, 0.5]) == {"score": 0.5}

    def test_components_are_averaged_by_name(self) -> None:
        scores = [{"correct": 1.0, "error": 0.0}, {"correct": 0.0, "error": 3.0}]
        assert mean_scores(scores) == {"correct": 0.5, "error": 1.5}

    def test_the_mean_does_not_depend_on_the_sample_order(self) -> None:
        scores = [0.1, 0.2, 0.3, 1 / 3, 1.0, 2 / 3]
        assert mean_scores(iter(scores)) == mean_scores(reversed(scores))

    def test_a_task_without_samples_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one scored sample"):
            mean_scores([])

    @pytest.mark.parametrize("other", [{"correct": 1.0, "error": 0.0}, {"error": 0.0}, 1.0])
    def test_samples_must_report_the_same_components(self, other: float | dict[str, float]) -> None:
        with pytest.raises(ValueError, match=re.escape("the first has ['correct']")):
            mean_scores([{"correct": 1.0}, other])


class TestLookup:
    def test_the_scorers_are_registered_by_name(self) -> None:
        for name, score in TEXT_SCORERS.items():
            assert get_scorer(name) is SCORERS[name]
            assert get_scorer(name).score is score
            assert get_scorer(name).aggregate is mean_scores

    def test_an_unknown_name_lists_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match=re.escape(f"expected one of {sorted(SCORERS)}")):
            get_scorer("bleu")

    def test_a_task_score_is_the_mean_over_its_samples(self) -> None:
        scorer = get_scorer("relaxed_accuracy")
        scores = [scorer.score(prediction, ["10"]) for prediction in ("10", "10.4", "12", "ten")]
        assert scorer.aggregate(scores) == {"score": 0.5}

    def test_a_scorer_may_report_components_and_aggregate_them_itself(self) -> None:
        def count_error(prediction: str, count: int) -> dict[str, float]:
            error = abs(int(prediction) - count)
            return {"correct": float(error == 0), "error": float(error)}

        def worst(scores: Sequence[dict[str, float]]) -> dict[str, float]:
            return {"worst_error": max(score["error"] for score in scores)}

        scores = [count_error("3", 3), count_error("5", 2)]
        assert Scorer(count_error).aggregate(scores) == {"correct": 0.5, "error": 1.5}
        assert Scorer(count_error, aggregate=worst).aggregate(scores) == {"worst_error": 3.0}
