"""The document type every image-text reader returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image

DocumentKind = Literal["qa", "conversation"]


@dataclass(frozen=True)
class Turn:
    """A user message and the reference replies, any one of which trains the response."""

    user: str
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError("a turn needs at least one answer")


@dataclass(frozen=True)
class Document:
    """The images of one source row and the conversation about them.

    ``kind`` selects the rendering: ``"qa"`` is one prompt and its answer,
    ``"conversation"`` is the full chat template over every turn.
    """

    images: tuple[Image.Image, ...]
    turns: tuple[Turn, ...]
    kind: DocumentKind = "qa"

    def __post_init__(self) -> None:
        if self.kind not in ("qa", "conversation"):
            raise ValueError(f"unknown document kind {self.kind!r}")
        if not self.turns:
            raise ValueError("a document needs at least one turn")
        if self.kind == "qa" and len(self.turns) != 1:
            raise ValueError(f"a qa document has exactly one turn, got {len(self.turns)}")
