"""Tokenizer integration for KempnerForge.

Thin wrapper around HuggingFace tokenizers providing a minimal interface
for encoding text to token IDs and decoding back.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizerFast
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Tokenizer:
    """Unified tokenizer interface wrapping HuggingFace tokenizers.

    Supports loading from:
      - A local directory containing tokenizer files
      - A HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        # Cache key properties
        self._vocab_size = tokenizer.vocab_size
        self._eos_id = tokenizer.eos_token_id
        self._bos_id = tokenizer.bos_token_id

    @classmethod
    def from_pretrained(cls, path_or_name: str) -> Tokenizer:
        """Load tokenizer from a local path or HuggingFace model name.

        Args:
            path_or_name: Local directory, file path, or HF model identifier.

        Returns:
            Tokenizer instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
        logger.info(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}, source={path_or_name}")
        return cls(tokenizer)

    @classmethod
    def from_file(cls, path: str) -> Tokenizer:
        """Load a standalone tokenizer.json file (HuggingFace fast tokenizer format).

        Args:
            path: Path to a tokenizer.json file.

        Returns:
            Tokenizer instance.
        """
        fast_tok = HFTokenizerFast.from_file(path)
        # Wrap in AutoTokenizer for a consistent interface
        tokenizer = AutoTokenizer.from_pretrained(str(Path(path).parent))
        logger.info(f"Loaded tokenizer from file: vocab_size={fast_tok.get_vocab_size()}")
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_id

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of integer token IDs.
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: Whether to remove special tokens from output.

        Returns:
            Decoded string.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        """Encode a batch of texts to token IDs.

        Args:
            texts: List of input strings.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of token ID lists.
        """
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
