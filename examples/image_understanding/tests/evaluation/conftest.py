"""Puts the example root on ``sys.path`` and provides a tiny run for the evaluator tests.

The run is a word-level chat tokenizer saved to disk, a Joint-Decoder config
with the random image encoder, a checkpoint of it and task files, so those
tests need neither the hub nor a GPU.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QWEN = "Qwen/Qwen3-0.6B"
SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>")
WORDS = (
    "user assistant what is shown in the image ? how many which color a an one two three "
    "four cat dog bird red green blue square circle yes no left right big small of and it"
)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
CONFIG = """\
[model]
dim = 32
n_layers = 2
n_heads = 2
n_kv_heads = 2
vocab_size = {vocab_size}
max_seq_len = 48
init_std = 0.2

[train]
mixed_precision = "fp32"
compile_model = false

[data]
tokenizer_path = "{tokenizer_path}"
hf_image_size = 16

[vision_encoder]
type = "random"
feature_dim = 24
num_tokens = 4

[vlm]
arch = "joint_decoder"
max_text_len = 32

[checkpoint]
dir = "{checkpoint_dir}"
"""
QUESTIONS = (
    "what is shown in the image ?",
    "how many ?",
    "which color is it ?",
    "is it a cat ?",
    "what is left of the big red square and right of the small blue circle ?",
)


@pytest.fixture(scope="session")
def qwen_tokenizer() -> Any:
    """The Qwen3 tokenizer, loaded once for the tests of the recipe's prompt."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


@pytest.fixture(scope="session")
def tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A word-level chat tokenizer saved where ``AutoTokenizer`` loads it from disk."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate((*SPECIAL_TOKENS, *WORDS.split()))}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
        chat_template=CHAT_TEMPLATE,
    )
    path = tmp_path_factory.mktemp("tokenizer")
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def write_config(tmp_path: Path, tokenizer_path: Path) -> Callable[..., Path]:
    """Writes the tiny run's TOML config, with ``extra`` TOML appended, and returns its path."""

    def write(extra: str = "", name: str = "config.toml") -> Path:
        path = tmp_path / name
        path.write_text(
            CONFIG.format(
                vocab_size=len(SPECIAL_TOKENS) + len(WORDS.split()),
                tokenizer_path=tokenizer_path,
                checkpoint_dir=tmp_path / "run",
            )
            + extra
        )
        return path

    return write


@pytest.fixture
def save_checkpoint() -> Callable[..., Path]:
    """Saves a checkpoint of a config's freshly initialized model and returns its step directory."""

    def save(config: Any, step: int = 3, seed: int = 0) -> Path:
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.model.vlm import build_vlm_wrapper

        torch.manual_seed(seed)
        model = build_vlm_wrapper(config.model, config.vision_encoder, config.adapter, config.vlm)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        CheckpointManager(config.checkpoint, model, optimizer).save(step=step, tokens_seen=step)
        return Path(config.checkpoint.dir) / f"step_{step}"

    return save


@pytest.fixture
def write_task(tmp_path: Path) -> Callable[..., Path]:
    """Writes a DocVQA-layout split with its images and a task file for it; returns the file."""

    def write(
        rows: int = 7,
        name: str = "docvqa",
        scorer: str = "anls",
        max_new_tokens: int = 4,
        answers: Callable[[int], list[str]] = lambda row: ["cat", "a dog"],
    ) -> Path:
        root = tmp_path / name
        root.mkdir()
        data = []
        for row in range(rows):
            color = (30 * row % 256, 255 - 20 * row % 256, 7 * row % 256)
            Image.new("RGB", (8, 6), color).save(root / f"{row}.png")
            question = QUESTIONS[row % len(QUESTIONS)]
            data.append({"image": f"{row}.png", "question": question, "answers": answers(row)})
        (root / "annotations.json").write_text(json.dumps({"data": data}))
        path = tmp_path / f"{name}.toml"
        path.write_text(
            f'[[tasks]]\nname = "{name}"\nreader = "docvqa"\nscorer = "{scorer}"\n'
            f"max_new_tokens = {max_new_tokens}\n\n[tasks.reader_args]\n"
            f'annotation_file = "{root / "annotations.json"}"\nimage_root = "{root}"\n'
        )
        return path

    return write
