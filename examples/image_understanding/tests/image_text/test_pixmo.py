"""Tests for the PixMo readers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from image_text.document import Document, Turn
from image_text.pixmo import (
    PixmoRows,
    explanation_answer,
    pixmo_cap_qa,
    pixmo_captions,
    pixmo_counting,
    pixmo_point_explanations,
    pixmo_pointing,
)
from image_text.render import ChatRenderer
from PIL import Image


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _write_stub(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("404 Client Error: Not Found for url")


class TestCaptionsAndConversations:
    def test_captions_are_asked_with_the_prompt(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/aa")
        ann = _write_jsonl(
            tmp_path / "cap.jsonl",
            [{"image": "pixmo_images/aa", "caption": "A long caption.", "transcripts": ["t"]}],
        )
        doc = pixmo_captions(ann, tmp_path, prompt="Describe.")[0]
        assert doc.kind == "qa"
        assert doc.turns[0].user == "Describe."
        assert doc.turns[0].answers == ("A long caption.",)

    def test_cap_qa_takes_the_last_pair_of_each_message(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/bb")
        messages = [["Q1", "A1"], ["Q1", "A1", "Q2", "A2"], ["Q3", ""], "junk", ["only"]]
        ann = _write_jsonl(
            tmp_path / "capqa.jsonl", [{"image": "pixmo_images/bb", "messages": messages}]
        )
        doc = pixmo_cap_qa(ann, tmp_path)[0]
        assert doc.kind == "conversation"
        assert [(t.user, t.answers) for t in doc.turns] == [("Q1", ("A1",)), ("Q2", ("A2",))]

    def test_rows_without_turns_are_dropped(self, tmp_path: Path) -> None:
        ann = _write_jsonl(
            tmp_path / "capqa.jsonl",
            [{"image": "pixmo_images/cc", "messages": [["", "A"]]}, {"image": "x", "messages": []}],
        )
        assert len(pixmo_cap_qa(ann, tmp_path)) == 0

    def test_point_explanations_are_rebuilt_from_the_parsed_response(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/dd")
        ann = _write_jsonl(
            tmp_path / "expl.jsonl",
            [
                {
                    "image": "pixmo_images/dd",
                    "question": ["Where?", "How many?", "Why?", " "],
                    "response": ["raw", "raw", "raw", "raw"],
                    "parsed_response": [
                        "See <|POINT|> here.",
                        "I see <|POINT|> there.",
                        "Because.",
                        "x",
                    ],
                    "points": [[[[1.0, 2.0]]], [[[3.0, 4.0], [5.0, 6.0]]], [], []],
                    "inline_text": [["the sign"], ["2 clouds"], [], []],
                    "alt_text": [["A sign"], ["Clouds in the sky"], [], []],
                }
            ],
        )
        doc = pixmo_point_explanations(ann, tmp_path)[0]
        assert [t.user for t in doc.turns] == ["Where?", "How many?", "Why?"]
        assert doc.turns[0].answers == (
            'See <point x="1.0" y="2.0" alt="A sign">the sign</point> here.',
        )
        assert doc.turns[1].answers == (
            'I see <points x1="3.0" y1="4.0" x2="5.0" y2="6.0" alt="Clouds in the sky">'
            "2 clouds</points> there.",
        )
        assert doc.turns[2].answers == ("Because.",)

    def test_explanation_turns_with_bad_points_are_dropped(self) -> None:
        assert explanation_answer("A <|POINT|>.", [[[430.0, 12.5]]], ["x"], ["y"]) is None
        assert explanation_answer("A <|POINT|>.", [[[-0.2, 81.6]]], ["x"], ["y"]) is None
        assert explanation_answer("A <|POINT|> <|POINT|>.", [[[1.0, 1.0]]], ["x"], ["y"]) is None
        assert explanation_answer("A <|POINT|>.", [[]], ["x"], ["y"]) is None
        assert explanation_answer("Plain.", [], [], []) == "Plain."


class TestPointingAndCounting:
    def test_pointing_unrolls_one_example_per_label(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/ee")
        ann = _write_jsonl(
            tmp_path / "pointing.jsonl",
            [
                {
                    "image": "pixmo_images/ee",
                    "label": ["cat", "dog"],
                    "points": [[{"x": 10.0, "y": 20.0}], []],
                }
            ],
        )
        reader = pixmo_pointing(ann, tmp_path)
        assert len(reader) == 2
        assert reader[0].turns[0].user == "Point to the cat."
        assert reader[0].turns[0].answers == ('<point x="10.0" y="20.0" alt="cat">cat</point>',)
        assert reader[1].turns[0].answers == ("There are no dog in the image.",)

    def test_multi_label_counting_keeps_percent_points(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/ff", size=(400, 200))
        ann = _write_jsonl(
            tmp_path / "points_counting.jsonl",
            [
                {
                    "image": "pixmo_images/ff",
                    "label": ["cup", "pen"],
                    "count": [2, 0],
                    "points": [[{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}], []],
                }
            ],
        )
        reader = pixmo_counting(ann, tmp_path, multi_label=True)
        assert reader.coordinates == "percent"
        assert reader[0].turns[0].user == "How many cup are in the image?"
        assert reader[0].turns[0].answers == (
            '<points x1="10.0" y1="20.0" x2="30.0" y2="40.0" alt="cup">cup</points>\n'
            "Counting the cup, I count 2.",
        )
        assert reader[1].turns[0].answers == ("I count 0 pen in the image.",)

    def test_pixel_points_are_converted_to_percent(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/gg", size=(400, 200))
        ann = _write_jsonl(
            tmp_path / "count.jsonl",
            [
                {
                    "image": "pixmo_images/gg",
                    "label": "bird",
                    "count": 2,
                    "points": {"x": [200.0, 400.0], "y": [50.0, 200.0]},
                }
            ],
        )
        reader = pixmo_counting(ann, tmp_path, multi_label=False)
        assert reader.coordinates == "pixels"
        assert reader[0].turns[0].answers == (
            '<points x1="50.0" y1="25.0" x2="100.0" y2="100.0" alt="bird">bird</points>\n'
            "Counting the bird, I count 2.",
        )

    def test_counting_keeps_every_point_unless_capped(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/aa", size=(100, 100))
        ann = _write_jsonl(
            tmp_path / "count.jsonl",
            [
                {
                    "image": "pixmo_images/aa",
                    "label": "dot",
                    "count": 40,
                    "points": {"x": list(range(40)), "y": list(range(40))},
                }
            ],
        )
        answer = pixmo_counting(ann, tmp_path, multi_label=False)[0]
        assert 'x40="39.0"' in answer.turns[0].answers[0]
        capped = pixmo_counting(ann, tmp_path, multi_label=False, max_points=30)[0]
        assert 'x30="29.0"' in capped.turns[0].answers[0]
        assert 'x31="' not in capped.turns[0].answers[0]
        assert capped.turns[0].answers[0].endswith("I count 40.")

    @pytest.mark.parametrize("max_points", [0, -1])
    def test_a_cap_below_one_point_is_refused(self, tmp_path: Path, max_points: int) -> None:
        ann = _write_jsonl(tmp_path / "count.jsonl", [])
        with pytest.raises(ValueError, match="at least 1"):
            pixmo_counting(ann, tmp_path, multi_label=False, max_points=max_points)
        with pytest.raises(ValueError, match="at least 1"):
            pixmo_pointing(ann, tmp_path, max_points=max_points)

    def test_a_long_counting_answer_trains_its_points_before_the_count(self) -> None:
        """At a small budget the count is cut away; a cap on the points keeps it."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        points = [(float(i), float(i)) for i in range(40)]
        image = Image.new("RGB", (10, 10))

        def answer(max_points: int | None) -> str:
            from image_text.points import format_point_answer

            tag = format_point_answer(points, "dot", absent_text="", max_points=max_points)
            return f"{tag}\nCounting the dot, I count 40."

        def trained(reply: str) -> str:
            doc = Document(
                images=(image,), turns=(Turn("How many dot are in the image?", (reply,)),)
            )
            renderer = ChatRenderer(tokenizer, max_text_len=96, image_size=8, answer_only=True)
            sample = renderer.render(doc, [reply])
            labels = sample["labels"]
            return tokenizer.decode(labels[labels != -100])

        every = trained(answer(None))
        assert every.startswith('<points x1="0.0"') and "I count" not in every
        capped = trained(answer(3))
        assert capped.endswith("Counting the dot, I count 40.<|im_end|>")

    def test_a_count_file_is_read_in_pixels(self, tmp_path: Path, write_image: Callable) -> None:
        """The unit follows the file layout, so PixMo-Count's pixels cannot be read as percent."""
        write_image("pixmo_images/ii", size=(200, 200))
        ann = _write_jsonl(
            tmp_path / "count.jsonl",
            [
                {
                    "image": "pixmo_images/ii",
                    "label": "x",
                    "count": 1,
                    "points": {"x": [150], "y": [150]},
                }
            ],
        )
        assert pixmo_counting(ann, tmp_path, multi_label=False)[0].turns[0].answers == (
            '<point x="75.0" y="75.0" alt="x">x</point>\nCounting the x, I count 1.',
        )

    def test_missing_count_reads_as_zero(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("pixmo_images/hh")
        ann = _write_jsonl(
            tmp_path / "count.jsonl",
            [
                {
                    "image": "pixmo_images/hh",
                    "label": "ant",
                    "count": None,
                    "points": {"x": [], "y": []},
                }
            ],
        )
        doc = pixmo_counting(ann, tmp_path, multi_label=False)[0]
        assert doc.turns[0].answers == ("I count 0 ant in the image.",)

    def test_unknown_coordinate_unit_is_refused(self, tmp_path: Path) -> None:
        from image_text.pixmo import PixmoCounting

        with pytest.raises(ValueError, match="'percent' or 'pixels'"):
            PixmoCounting(tmp_path, {"image": []}, coordinates="inches")  # type: ignore[arg-type]


class TestUnreadableImages:
    def _captions(self, tmp_path: Path, names: list[str]) -> Path:
        return _write_jsonl(
            tmp_path / "cap.jsonl",
            [{"image": f"pixmo_images/{n}", "caption": f"caption {n}"} for n in names],
        )

    def test_a_stub_steps_to_the_next_row(self, tmp_path: Path, write_image: Callable) -> None:
        _write_stub(tmp_path, "pixmo_images/s1")
        write_image("pixmo_images/ok")
        reader = pixmo_captions(self._captions(tmp_path, ["s1", "ok"]), tmp_path, prompt="P")
        assert reader[0].turns[0].answers == ("caption ok",)
        assert reader.skipped == 1

    def test_the_step_wraps_to_the_first_row(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("pixmo_images/ok")
        _write_stub(tmp_path, "pixmo_images/s1")
        reader = pixmo_captions(self._captions(tmp_path, ["ok", "s1"]), tmp_path, prompt="P")
        assert reader[1].turns[0].answers == ("caption ok",)

    def test_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        for name in ("s1", "s2", "s3"):
            _write_stub(tmp_path, f"pixmo_images/{name}")
        ann = self._captions(tmp_path, ["s1", "s2", "s3"])
        reader = pixmo_captions(ann, tmp_path, prompt="P", max_attempts=3)
        with pytest.raises(RuntimeError, match="3 consecutive rows"):
            reader[0]

    def test_every_reader_takes_max_attempts(self, tmp_path: Path) -> None:
        ann = _write_jsonl(tmp_path / "empty.jsonl", [])
        readers = [
            pixmo_captions(ann, tmp_path, prompt="P", max_attempts=2),
            pixmo_cap_qa(ann, tmp_path, max_attempts=2),
            pixmo_point_explanations(ann, tmp_path, max_attempts=2),
            pixmo_pointing(ann, tmp_path, max_attempts=2),
            pixmo_counting(ann, tmp_path, multi_label=False, max_attempts=2),
        ]
        assert [reader.max_attempts for reader in readers] == [2] * 5

    def test_the_rows_of_an_unreadable_image_cost_one_attempt(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        """An image with as many labels as attempts is skipped as one, to the next image."""
        _write_stub(tmp_path, "pixmo_images/bad")
        write_image("pixmo_images/good")
        ann = _write_jsonl(
            tmp_path / "pointing.jsonl",
            [
                {
                    "image": "pixmo_images/bad",
                    "label": [f"thing {i}" for i in range(3)],
                    "points": [[{"x": 1.0, "y": 1.0}]] * 3,
                },
                {"image": "pixmo_images/good", "label": ["cat"], "points": [[]]},
            ],
        )
        reader = pixmo_pointing(ann, tmp_path, max_attempts=3)
        assert len(reader) == 4
        assert reader[0].turns[0].user == "Point to the cat."
        assert reader.skipped == 1

    def test_rows_need_an_image_column(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'image' column"):
            PixmoRows(tmp_path, {"caption": ["x"]})

    def test_out_of_range_index_raises(self, tmp_path: Path) -> None:
        reader = pixmo_captions(self._captions(tmp_path, ["a"]), tmp_path, prompt="P")
        with pytest.raises(IndexError):
            reader[1]
