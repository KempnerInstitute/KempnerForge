"""Unit tests for multi-dataset mixing (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from kempnerforge.config.schema import DataConfig, DatasetSource
from kempnerforge.data.dataset import MixtureDataset
from kempnerforge.data.sampler import MixtureSampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_mmap_dirs(tmp_path):
    """Create two temp directories with .npy token files of different sizes."""
    dir_a = tmp_path / "dataset_a"
    dir_a.mkdir()
    # 160 tokens → 10 samples at seq_len=16
    np.save(dir_a / "shard.npy", np.arange(160, dtype=np.uint16))

    dir_b = tmp_path / "dataset_b"
    dir_b.mkdir()
    # 320 tokens → 20 samples at seq_len=16
    np.save(dir_b / "shard.npy", np.arange(320, dtype=np.uint16))

    return dir_a, dir_b


def _make_mmap_dataset(data_dir, seq_len=16):
    from kempnerforge.data.dataset import MemoryMappedDataset

    return MemoryMappedDataset(str(data_dir), seq_len=seq_len)


# ---------------------------------------------------------------------------
# DatasetSource config
# ---------------------------------------------------------------------------


class TestDatasetSource:
    def test_valid_path_source(self):
        src = DatasetSource(path="/data/code", weight=0.7, name="code")
        assert src.weight == 0.7

    def test_valid_hf_source(self):
        src = DatasetSource(hf_name="allenai/c4", weight=1.0, name="c4")
        assert src.hf_name == "allenai/c4"

    def test_weight_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            DatasetSource(path="/data", weight=0.0)
        with pytest.raises(ValueError, match="positive"):
            DatasetSource(path="/data", weight=-1.0)

    def test_must_have_path_or_hf_name(self):
        """Validated at DataConfig level (not DatasetSource) to allow TOML loader defaults."""
        with pytest.raises(ValueError, match="path or hf_name"):
            DataConfig(datasets=[DatasetSource(weight=1.0)])

    def test_name_auto_empty(self):
        src = DatasetSource(path="/data/code")
        assert src.name == ""


class TestDataConfigMixing:
    def test_default_datasets_empty(self):
        config = DataConfig()
        assert config.datasets == []

    def test_default_temperature(self):
        config = DataConfig()
        assert config.mix_temperature == 1.0

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValueError, match="mix_temperature"):
            DataConfig(mix_temperature=0.0)

    def test_datasets_set(self):
        config = DataConfig(
            datasets=[
                DatasetSource(path="/a", weight=0.7, name="a"),
                DatasetSource(path="/b", weight=0.3, name="b"),
            ]
        )
        assert len(config.datasets) == 2
        assert config.datasets[0].weight == 0.7


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


class TestMixingTomlLoading:
    def test_load_datasets_from_toml(self, tmp_path):
        """TOML [[data.datasets]] table arrays parse into DatasetSource list."""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[train]
max_steps = 10

[[data.datasets]]
path = "/data/code"
weight = 0.7
name = "code"

[[data.datasets]]
path = "/data/text"
weight = 0.3
name = "text"
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert len(config.data.datasets) == 2
        assert isinstance(config.data.datasets[0], DatasetSource)
        assert config.data.datasets[0].path == "/data/code"
        assert config.data.datasets[0].weight == 0.7
        assert config.data.datasets[1].name == "text"

    def test_load_hf_dataset_source(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[[data.datasets]]
hf_name = "allenai/c4"
weight = 1.0
name = "c4"
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert config.data.datasets[0].hf_name == "allenai/c4"

    def test_unknown_key_in_dataset_source_rejected(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[[data.datasets]]
path = "/data/code"
typo_field = "oops"
"""
        )
        from kempnerforge.config.loader import load_config

        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(str(toml_path), cli_args=[])


# ---------------------------------------------------------------------------
# MixtureDataset
# ---------------------------------------------------------------------------


class TestMixtureDataset:
    def test_len_is_sum(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        assert len(mix) == len(ds_a) + len(ds_b)

    def test_cumulative_sizes(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        assert mix.cumulative_sizes == [0, len(ds_a), len(ds_a) + len(ds_b)]

    def test_getitem_first_dataset(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        sample = mix[0]
        assert sample["dataset_idx"] == 0
        assert "input_ids" in sample
        assert "labels" in sample

    def test_getitem_second_dataset(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        # Index into second dataset
        sample = mix[len(ds_a)]
        assert sample["dataset_idx"] == 1

    def test_getitem_last_index(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        sample = mix[len(mix) - 1]
        assert sample["dataset_idx"] == 1

    def test_index_out_of_range(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        with pytest.raises(IndexError):
            mix[len(mix)]

    def test_dataset_names(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["alpha", "beta"])
        assert mix.dataset_names == ["alpha", "beta"]

    def test_state_dict_round_trip(self, two_mmap_dirs):
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])
        state = mix.state_dict()
        assert "dataset_0" in state
        assert "dataset_1" in state

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MixtureDataset([], [])

    def test_mismatched_lengths_raises(self, two_mmap_dirs):
        dir_a, _ = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        with pytest.raises(ValueError, match="same length"):
            MixtureDataset([ds_a], ["a", "b"])


# ---------------------------------------------------------------------------
# MixtureSampler
# ---------------------------------------------------------------------------


class TestMixtureSampler:
    def test_total_samples_matches_datasets(self):
        """Total samples = sum of per-rank available across datasets."""
        cumulative = [0, 100, 300]  # ds0=100, ds1=200
        sampler = MixtureSampler(
            cumulative, weights=[1.0, 1.0], num_replicas=1, rank=0, shuffle=False
        )
        assert len(sampler) == 300

    def test_weight_proportionality(self):
        """Higher weight → more samples from that dataset."""
        cumulative = [0, 1000, 2000]  # ds0=1000, ds1=1000
        sampler = MixtureSampler(
            cumulative, weights=[0.8, 0.2], num_replicas=1, rank=0, shuffle=False
        )
        indices = list(sampler)
        # Count samples from each dataset
        ds0_count = sum(1 for i in indices if i < 1000)
        # ds0 should get ~80% of samples
        ratio = ds0_count / len(indices)
        assert 0.7 < ratio < 0.9

    def test_distributed_no_overlap(self):
        """Indices across ranks should not overlap within the same dataset."""
        cumulative = [0, 100]
        all_indices = []
        for rank in range(4):
            sampler = MixtureSampler(
                cumulative, weights=[1.0], num_replicas=4, rank=rank, shuffle=False
            )
            all_indices.extend(list(sampler))
        # No duplicates
        assert len(all_indices) == len(set(all_indices))

    def test_distributed_covers_data(self):
        """Union of all ranks' samples covers the dataset (with drop_last)."""
        cumulative = [0, 100]
        all_indices = set()
        for rank in range(4):
            sampler = MixtureSampler(
                cumulative, weights=[1.0], num_replicas=4, rank=rank, shuffle=False
            )
            all_indices.update(list(sampler))
        # drop_last: 100 // 4 = 25 per rank → 100 covered
        assert len(all_indices) == 100

    def test_deterministic(self):
        """Same seed and epoch produce identical sequences."""
        cumulative = [0, 50, 100]
        s1 = MixtureSampler(cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, seed=42)
        s2 = MixtureSampler(cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, seed=42)
        assert list(s1) == list(s2)

    def test_different_epoch_different_order(self):
        """Different epochs produce different orderings."""
        cumulative = [0, 100]
        sampler = MixtureSampler(cumulative, weights=[1.0], num_replicas=1, rank=0, seed=42)
        order_0 = list(sampler)
        sampler.set_epoch(1)
        order_1 = list(sampler)
        assert order_0 != order_1

    def test_skip_ahead(self):
        """Skip-ahead produces tail of full iteration."""
        cumulative = [0, 100]
        sampler = MixtureSampler(cumulative, weights=[1.0], num_replicas=1, rank=0, shuffle=False)
        full = list(sampler)
        sampler.set_skip(10)
        skipped = list(sampler)
        assert skipped == full[10:]

    def test_state_dict_round_trip(self):
        cumulative = [0, 50, 100]
        sampler = MixtureSampler(cumulative, weights=[0.5, 0.5], num_replicas=2, rank=1, seed=99)
        sampler.set_epoch(5)
        state = sampler.state_dict()
        assert state["epoch"] == 5
        assert state["seed"] == 99
        assert state["rank"] == 1

        sampler2 = MixtureSampler(cumulative, weights=[0.5, 0.5], num_replicas=2, rank=1, seed=99)
        sampler2.load_state_dict(state)
        assert sampler2._epoch == 5

    def test_oversampling_with_high_weight(self):
        """Small dataset with high weight should be oversampled (repeats)."""
        # ds0 has 10 samples, ds1 has 990. Weight ds0 at 90%.
        cumulative = [0, 10, 1000]
        sampler = MixtureSampler(
            cumulative, weights=[0.9, 0.1], num_replicas=1, rank=0, shuffle=False
        )
        indices = list(sampler)
        ds0_indices = [i for i in indices if i < 10]
        # ds0 should have ~900 samples (oversampled from 10 unique)
        assert len(ds0_indices) > 100  # Well over the 10 unique samples


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    def test_temperature_1_unchanged(self):
        """Temperature=1.0 should produce same proportions as raw weights."""
        cumulative = [0, 500, 1000]
        s_t1 = MixtureSampler(
            cumulative,
            weights=[0.7, 0.3],
            num_replicas=1,
            rank=0,
            shuffle=False,
            temperature=1.0,
        )
        s_raw = MixtureSampler(
            cumulative,
            weights=[0.7, 0.3],
            num_replicas=1,
            rank=0,
            shuffle=False,
        )
        assert list(s_t1) == list(s_raw)

    def test_high_temperature_more_uniform(self):
        """High temperature should make distribution more uniform."""
        cumulative = [0, 500, 1000]
        # With T=1, 0.9/0.1 → ds0 gets 90%
        s_hot = MixtureSampler(
            cumulative,
            weights=[0.9, 0.1],
            num_replicas=1,
            rank=0,
            shuffle=False,
            temperature=5.0,
        )
        indices = list(s_hot)
        ds0_count = sum(1 for i in indices if i < 500)
        ratio = ds0_count / len(indices)
        # With high temperature, ratio should be closer to 0.5 than 0.9
        assert 0.4 < ratio < 0.75

    def test_low_temperature_more_peaked(self):
        """Low temperature should concentrate on the highest-weight dataset."""
        cumulative = [0, 500, 1000]
        s_cold = MixtureSampler(
            cumulative,
            weights=[0.6, 0.4],
            num_replicas=1,
            rank=0,
            shuffle=False,
            temperature=0.1,
        )
        indices = list(s_cold)
        ds0_count = sum(1 for i in indices if i < 500)
        ratio = ds0_count / len(indices)
        # Low temp concentrates on ds0: ratio should be very high
        assert ratio > 0.9


# ---------------------------------------------------------------------------
# Integration with MixtureDataset
# ---------------------------------------------------------------------------


class TestMixtureSamplerWithDataset:
    def test_sampler_indices_valid(self, two_mmap_dirs):
        """All indices from sampler are valid MixtureDataset indices."""
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])

        sampler = MixtureSampler(
            mix.cumulative_sizes,
            weights=[0.5, 0.5],
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
        )
        for idx in sampler:
            assert 0 <= idx < len(mix)
            sample = mix[idx]
            assert "input_ids" in sample
            assert "dataset_idx" in sample

    def test_dataset_idx_matches_sampler_intent(self, two_mmap_dirs):
        """Samples from each dataset range return correct dataset_idx."""
        dir_a, dir_b = two_mmap_dirs
        ds_a = _make_mmap_dataset(dir_a)
        ds_b = _make_mmap_dataset(dir_b)
        mix = MixtureDataset([ds_a, ds_b], ["a", "b"])

        # All indices in ds_a range → dataset_idx=0
        for i in range(len(ds_a)):
            assert mix[i]["dataset_idx"] == 0
        # All indices in ds_b range → dataset_idx=1
        for i in range(len(ds_a), len(mix)):
            assert mix[i]["dataset_idx"] == 1
