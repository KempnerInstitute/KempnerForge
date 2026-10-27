"""Unit tests for how MemoryMappedDataset reads .bin and .npy token files.

A raw ``.bin`` shard carries no dtype, so the dataset takes it from
``token_dtype`` or from the ``metadata.yaml`` next to the shard, and rejects a
shard that is not a whole number of tokens. Each matched file is read by its
own suffix. Every test checks the exact tokens a sample holds.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from kempnerforge.data.dataset import MemoryMappedDataset

SEQ_LEN = 8


def _write_bin(path, tokens) -> None:
    np.asarray(tokens).tofile(path)


def _write_metadata(directory, dtype: str) -> None:
    (directory / "metadata.yaml").write_text(f"tokenized_info:\n  dtype: {dtype}\n")


@pytest.fixture
def dataset_warnings(caplog, monkeypatch):
    """Warnings from the dataset module.

    Once anything configures KempnerForge logging, the ``kempnerforge`` logger
    stops propagating to the root logger where caplog listens.
    """
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="kempnerforge.data.dataset")

    def _messages() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    return _messages


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


class TestUndeclaredBinDtype:
    def test_whole_uint32_shards_read_as_uint32_with_a_warning(self, tmp_path, dataset_warnings):
        tokens = np.arange(100_000, 100_064, dtype=np.uint32)
        _write_bin(tmp_path / "tokens_000.bin", tokens)
        _write_bin(tmp_path / "tokens_001.bin", tokens + 64)
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")
        assert _samples(ds) == _chunks(tokens) + _chunks(tokens + 64)
        warnings = dataset_warnings()
        assert len(warnings) == 1
        assert f"No token dtype is declared for the .bin shards in {tmp_path}" in warnings[0]
        assert "token_dtype" in warnings[0]
        assert str(tmp_path / "metadata.yaml") in warnings[0]

    def test_metadata_without_a_dtype_counts_as_undeclared(self, tmp_path, dataset_warnings):
        tokens = np.arange(64, dtype=np.uint32)
        _write_bin(tmp_path / "tokens_000.bin", tokens)
        (tmp_path / "metadata.yaml").write_text("name: raw corpus\ntokenized_info: {}\n")
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")
        assert _samples(ds) == _chunks(tokens)
        assert any("No token dtype is declared" in m for m in dataset_warnings())

    @pytest.mark.parametrize("declared_by", ["metadata", "token_dtype"])
    def test_declared_shards_do_not_warn(self, tmp_path, dataset_warnings, declared_by):
        _write_bin(tmp_path / "tokens_000.bin", np.arange(64, dtype=np.uint32))
        kwargs = {}
        if declared_by == "metadata":
            _write_metadata(tmp_path, "uint32")
        else:
            kwargs["token_dtype"] = "uint32"
        MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", **kwargs)
        assert dataset_warnings() == []


class TestTokenMetadataFile:
    def test_unparseable_metadata_is_rejected(self, tmp_path):
        """It must not fall through to reading the shards with no declared dtype."""
        _write_bin(tmp_path / "tokens_000.bin", np.arange(64, dtype=np.uint16))
        (tmp_path / "metadata.yaml").write_text("tokenized_info: [dtype: uint16\n")
        with pytest.raises(ValueError, match="could not be read as YAML"):
            MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")

    def test_unparseable_metadata_is_rejected_even_with_token_dtype(self, tmp_path):
        _write_bin(tmp_path / "tokens_000.bin", np.arange(64, dtype=np.uint16))
        (tmp_path / "metadata.yaml").write_text("tokenized_info: [dtype: uint16\n")
        with pytest.raises(ValueError, match="could not be read as YAML"):
            MemoryMappedDataset(
                str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin", token_dtype="uint16"
            )

    @pytest.mark.parametrize("recorded", ["int32", "uint8", "float16"])
    def test_metadata_with_an_unsupported_dtype_is_rejected(self, tmp_path, recorded):
        _write_bin(tmp_path / "tokens_000.bin", np.arange(64, dtype=np.uint32))
        _write_metadata(tmp_path, recorded)
        with pytest.raises(ValueError, match=f"records tokenized_info.dtype '{recorded}'"):
            MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*.bin")

    def test_npy_only_directories_never_read_metadata(self, tmp_path):
        tokens = np.arange(64, dtype=np.uint16)
        np.save(tmp_path / "shard_000.npy", tokens)
        (tmp_path / "metadata.yaml").write_text("tokenized_info: [dtype: uint16\n")
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN)
        assert _samples(ds) == _chunks(tokens)


class TestMixedFileFormats:
    @pytest.mark.parametrize("bin_sorts_first", [True, False], ids=["bin_first", "npy_first"])
    def test_each_file_is_read_by_its_own_suffix(self, tmp_path, bin_sorts_first):
        """The format of the first sorted file used to decide how every file was read."""
        bin_tokens = np.arange(1000, 1067, dtype=np.uint16)
        npy_tokens = np.arange(100_000, 100_064, dtype=np.uint32)
        bin_name, npy_name = ("a", "b") if bin_sorts_first else ("b", "a")
        _write_bin(tmp_path / f"shard_{bin_name}.bin", bin_tokens)
        np.save(tmp_path / f"shard_{npy_name}.npy", npy_tokens)
        _write_metadata(tmp_path, "uint16")
        ds = MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="shard_*")
        if bin_sorts_first:
            assert _samples(ds) == _chunks(bin_tokens) + _chunks(npy_tokens)
        else:
            assert _samples(ds) == _chunks(npy_tokens) + _chunks(bin_tokens)

    def test_a_matched_file_that_is_neither_bin_nor_npy_is_rejected(self, tmp_path):
        """A broad pattern over a tatm directory also matches its metadata and tokenizer."""
        _write_bin(tmp_path / "tokenized_0.bin", np.arange(64, dtype=np.uint32))
        _write_metadata(tmp_path, "uint32")
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(ValueError, match="neither a .bin nor a .npy token file") as excinfo:
            MemoryMappedDataset(str(tmp_path), seq_len=SEQ_LEN, file_pattern="*")
        assert str(tmp_path / "metadata.yaml") in str(excinfo.value)
        assert "2 such file(s)" in str(excinfo.value)
