"""Unit tests for TimeEmbeddingConfig (the [time_embedding] section)."""

from __future__ import annotations

import pytest

from kempnerforge.config.time_embedding import TimeEmbeddingConfig


class TestTimeEmbeddingConfig:
    def test_defaults(self):
        c = TimeEmbeddingConfig()
        assert c.type == "sinusoidal"
        assert c.num_bands == 16
        assert c.enabled is True

    def test_none_is_disabled(self):
        c = TimeEmbeddingConfig(type="none")
        assert c.enabled is False

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown time_embedding.type"):
            TimeEmbeddingConfig(type="bogus")

    def test_non_positive_num_bands_rejected(self):
        with pytest.raises(ValueError, match="num_bands must be positive"):
            TimeEmbeddingConfig(num_bands=0)

    def test_bad_periods_rejected(self):
        with pytest.raises(ValueError, match="min_period < max_period"):
            TimeEmbeddingConfig(min_period=10.0, max_period=5.0)

    def test_extra_kwargs(self):
        c = TimeEmbeddingConfig(num_bands=8, min_period=1.0, max_period=100.0)
        assert c.extra_kwargs() == {"num_bands": 8, "min_period": 1.0, "max_period": 100.0}
