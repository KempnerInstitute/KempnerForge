"""Unit tests for scripts/prepare_data.py: the .bin dtype it reports and the flags it prints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prepare_data  # noqa: E402


def _bin_shard(directory, dtype, n_tokens: int, *, metadata: str = "", trim: int = 0) -> None:
    raw = np.arange(n_tokens, dtype=dtype).tobytes()
    (directory / "tokenized_0.bin").write_bytes(raw[: len(raw) - trim])
    if metadata:
        (directory / "metadata.yaml").write_text(metadata)


class TestBinDtypeReport:
    def test_recorded_uint16_is_reported_with_its_flag(self, tmp_path, capsys):
        """The dtype that metadata.yaml records is the one training reads, not uint32."""
        _bin_shard(
            tmp_path,
            np.uint16,
            4096,
            metadata="tokenized_info:\n  dtype: uint16\n  vocab_size: 32100\n",
        )
        prepare_data.validate(tmp_path)
        out = capsys.readouterr().out
        assert "Tokens:     4,096\n" in out
        assert "Dtype:      uint16\n" in out
        assert "Compatible with KempnerForge MemoryMappedDataset." in out
        assert "--data.token_dtype=uint16\n" in out
        assert "--model.vocab_size=32100\n" in out

    def test_undeclared_dtype_is_not_assumed(self, tmp_path, capsys):
        _bin_shard(tmp_path, np.uint32, 4096)
        prepare_data.validate(tmp_path)
        out = capsys.readouterr().out
        assert "Dtype:      not declared" in out
        assert "Tokens:     4,096 if uint32, 8,192 if uint16" in out
        assert "Compatible with" not in out
        assert "--data.token_dtype=<uint16 or uint32>" in out

    def test_token_dtype_option_declares_it(self, tmp_path, capsys):
        _bin_shard(tmp_path, np.uint16, 4097)
        prepare_data.validate(tmp_path, "uint16")
        out = capsys.readouterr().out
        assert "Tokens:     4,097\n" in out
        assert "Compatible with KempnerForge MemoryMappedDataset." in out
        assert "--data.token_dtype=uint16\n" in out

    def test_truncated_shard_fails_validation(self, tmp_path, capsys):
        _bin_shard(tmp_path, np.uint32, 4096, metadata="tokenized_info:\n  dtype: uint32\n", trim=2)
        with pytest.raises(SystemExit) as excinfo:
            prepare_data.validate(tmp_path)
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "NOT compatible: 1 shard(s) are not a whole number of uint32 tokens" in out
        assert "Train with:" not in out

    def test_undeclared_shard_that_is_not_whole_uint32_fails_validation(self, tmp_path, capsys):
        _bin_shard(tmp_path, np.uint16, 4097)
        with pytest.raises(SystemExit) as excinfo:
            prepare_data.validate(tmp_path)
        assert excinfo.value.code == 1
        assert "Rerun with --token-dtype uint16 or uint32" in capsys.readouterr().out

    def test_token_dtype_option_must_agree_with_metadata(self, tmp_path, capsys):
        _bin_shard(tmp_path, np.uint32, 4096, metadata="tokenized_info:\n  dtype: uint32\n")
        with pytest.raises(SystemExit) as excinfo:
            prepare_data.validate(tmp_path, "uint16")
        assert excinfo.value.code == 1
        assert "records uint32" in capsys.readouterr().err

    def test_unreadable_metadata_fails_validation(self, tmp_path, capsys):
        _bin_shard(tmp_path, np.uint32, 4096, metadata="tokenized_info: [dtype: uint32\n")
        with pytest.raises(SystemExit) as excinfo:
            prepare_data.validate(tmp_path)
        assert excinfo.value.code == 1
        assert "could not be read as YAML" in capsys.readouterr().err
