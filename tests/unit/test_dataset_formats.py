"""Unit tests for how MemoryMappedDataset reads .bin and .npy token files.

A raw ``.bin`` shard carries no dtype, so the dataset takes it from
``token_dtype`` or from the ``metadata.yaml`` next to the shard, and rejects a
shard that is not a whole number of tokens. Each matched file is read by its
own suffix. Every test checks the exact tokens a sample holds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kempnerforge.data.dataset import MemoryMappedDataset

SEQ_LEN = 8


def _write_bin(path, tokens) -> None:
    np.asarray(tokens).tofile(path)


def _write_metadata(directory, dtype: str) -> None:
    (directory / "metadata.yaml").write_text(f"tokenized_info:\n  dtype: {dtype}\n")


def _samples(ds) -> list[list[int]]:
    """Every sample's ``seq_len`` tokens, rebuilt from its inputs and labels."""
    out = []
    for i in range(len(ds)):
        sample = ds[i]
        out.append(torch.cat([sample["input_ids"], sample["labels"][-1:]]).tolist())
    return out


def _chunks(tokens) -> list[list[int]]:
    """The samples a flat token stream splits into, partial tail dropped."""
    tokens = [int(t) for t in tokens]
    n = len(tokens) // SEQ_LEN
    return [tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n)]


class TestDeclaredBinDtype:
    @pytest.mark.parametrize("n_tokens", [64, 67], ids=["even_count", "odd_count"])
    @pytest.mark.parametrize("declared_by", ["metadata", "token_dtype"])
    def test_uint16_shard_reads_exact_tokens(self, tmp_path, n_tokens, declared_by):
        tokens = np.arange(1000, 1000 + n_tokens, dtype=np.uint16)
        _write_bin(tmp_path / "tokens_000.bin", tokens)
        kwargs = {}
        if declared_by == "metadata":
            _write_metadata(tmp_path, "uint16")
        else:
            kwargs["token_dtype"] = "uint16"
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", **kwargs)
        assert _samples(ds) == _chunks(tokens)

    @pytest.mark.parametrize("declared_by", ["metadata", "token_dtype"])
    def test_uint32_shard_keeps_ids_above_uint16(self, tmp_path, declared_by):
        tokens = np.arange(128_000, 128_000 + 64, dtype=np.uint32)
        _write_bin(tmp_path / "tokens_000.bin", tokens)
        kwargs = {}
        if declared_by == "metadata":
            _write_metadata(tmp_path, "uint32")
        else:
            kwargs["token_dtype"] = "uint32"
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", **kwargs)
        assert _samples(ds) == _chunks(tokens)

    def test_shards_of_both_parities_in_one_uint16_corpus(self, tmp_path):
        """The size guess read the even shard as uint32 and the odd one as uint16."""
        even = np.arange(0, 64, dtype=np.uint16)
        odd = np.arange(500, 567, dtype=np.uint16)
        _write_bin(tmp_path / "tokens_000.bin", even)
        _write_bin(tmp_path / "tokens_001.bin", odd)
        _write_metadata(tmp_path, "uint16")
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")
        assert _samples(ds) == _chunks(even) + _chunks(odd)

    def test_explicit_dtype_that_matches_metadata(self, tmp_path):
        tokens = np.arange(64, dtype=np.uint16)
        _write_bin(tmp_path / "tokens_000.bin", tokens)
        _write_metadata(tmp_path, "uint16")
        ds = MemoryMappedDataset(
            str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", token_dtype="uint16"
        )
        assert _samples(ds) == _chunks(tokens)

    def test_explicit_dtype_that_contradicts_metadata_is_rejected(self, tmp_path):
        _write_bin(tmp_path / "tokens_000.bin", np.arange(64, dtype=np.uint16))
        _write_metadata(tmp_path, "uint16")
        with pytest.raises(ValueError, match="records tokenized_info.dtype 'uint16'"):
            MemoryMappedDataset(
                str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", token_dtype="uint32"
            )

    def test_metadata_next_to_each_shard_decides_its_dtype(self, tmp_path):
        """A pattern that reaches into subdirectories reads each one's own metadata.yaml."""
        u16 = np.arange(64, dtype=np.uint16)
        u32 = np.arange(100_000, 100_064, dtype=np.uint32)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        _write_bin(tmp_path / "a" / "tokens_000.bin", u16)
        _write_metadata(tmp_path / "a", "uint16")
        _write_bin(tmp_path / "b" / "tokens_000.bin", u32)
        _write_metadata(tmp_path / "b", "uint32")
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*/*.bin")
        assert _samples(ds) == _chunks(u16) + _chunks(u32)

    def test_invalid_token_dtype_is_rejected(self, tmp_path):
        _write_bin(tmp_path / "tokens_000.bin", np.arange(64, dtype=np.uint32))
        with pytest.raises(ValueError, match="token_dtype must be 'uint16' or 'uint32'"):
            MemoryMappedDataset(
                str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", token_dtype="int32"
            )

    def test_token_dtype_leaves_npy_files_to_their_header(self, tmp_path):
        tokens = np.arange(64, dtype=np.uint16)
        np.save(tmp_path / "shard_000.npy", tokens)
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, token_dtype="uint32")
        assert _samples(ds) == _chunks(tokens)


class TestPartialBinShards:
    @pytest.mark.parametrize("missing_bytes", [1, 2, 3])
    def test_truncated_uint32_shard_is_rejected(self, tmp_path, missing_bytes):
        """The size guess read a uint32 shard missing 1 to 3 bytes as uint16 and trained on it."""
        raw = np.arange(64, dtype=np.uint32).tobytes()[:-missing_bytes]
        (tmp_path / "tokens_000.bin").write_bytes(raw)
        _write_metadata(tmp_path, "uint32")
        with pytest.raises(ValueError, match="not a whole number of uint32 tokens"):
            MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")

    def test_truncated_uint16_shard_is_rejected(self, tmp_path):
        raw = np.arange(64, dtype=np.uint16).tobytes()[:-1]
        (tmp_path / "tokens_000.bin").write_bytes(raw)
        with pytest.raises(ValueError, match="not a whole number of uint16 tokens"):
            MemoryMappedDataset(
                str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", token_dtype="uint16"
            )

    @pytest.mark.parametrize("missing_bytes", [1, 2, 3])
    def test_undeclared_shard_that_is_not_whole_uint32_is_rejected(self, tmp_path, missing_bytes):
        raw = np.arange(64, dtype=np.uint32).tobytes()[:-missing_bytes]
        (tmp_path / "tokens_000.bin").write_bytes(raw)
        with pytest.raises(ValueError, match="holds uint16 or uint32 tokens") as excinfo:
            MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")
        assert "token_dtype" in str(excinfo.value)
        assert str(tmp_path / "metadata.yaml") in str(excinfo.value)

    def test_undeclared_odd_count_uint16_shard_is_rejected(self, tmp_path):
        """Its size fits a uint16 shard and a truncated uint32 shard alike."""
        _write_bin(tmp_path / "tokens_000.bin", np.arange(67, dtype=np.uint16))
        with pytest.raises(ValueError, match="holds uint16 or uint32 tokens"):
            MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")
