"""Unit tests for the KempnerForge configuration system."""

from __future__ import annotations

import pytest

from kempnerforge.config import JobConfig, ModelConfig, load_config
from kempnerforge.config.loader import _parse_cli_overrides
from kempnerforge.config.registry import Registry
from kempnerforge.config.schema import (
    ActivationCheckpointing,
    AsyncCheckpointMode,
    CheckpointConfig,
    DistributedConfig,
    OptimizerConfig,
    ProfilingConfig,
    SchedulerType,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig()
        assert m.dim == 4096
        assert m.n_layers == 32
        assert m.n_heads == 32
        assert m.n_kv_heads == 32  # resolved from None
        assert m.head_dim == 128

    def test_n_kv_heads_defaults_to_n_heads(self):
        m = ModelConfig(n_heads=16, n_kv_heads=None)
        assert m.n_kv_heads == 16

    def test_gqa_config(self):
        m = ModelConfig(n_heads=32, n_kv_heads=8)
        assert m.n_kv_heads == 8

    def test_head_dim(self):
        m = ModelConfig(dim=1024, n_heads=16)
        assert m.head_dim == 64

    def test_ffn_hidden_dim_computed(self):
        m = ModelConfig(dim=4096, ffn_dim_multiplier=1.0)
        # 4 * 4096 * 2/3 ≈ 10922.67, ceil to multiple of 256 = 11008
        assert m.computed_ffn_hidden_dim == 11008

    def test_ffn_hidden_dim_override(self):
        m = ModelConfig(ffn_hidden_dim=8192)
        assert m.computed_ffn_hidden_dim == 8192

    def test_param_estimate_reasonable(self):
        # Llama 7B-like config
        m = ModelConfig(dim=4096, n_layers=32, n_heads=32, vocab_size=32000)
        params = m.num_params_estimate
        assert 6e9 < params < 8e9, f"Expected ~6.7B params, got {params / 1e9:.2f}B"

    def test_rejects_dim_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            ModelConfig(dim=1000, n_heads=32)

    def test_rejects_heads_not_divisible_by_kv_heads(self):
        with pytest.raises(ValueError, match="divisible by n_kv_heads"):
            ModelConfig(n_heads=32, n_kv_heads=5)

    def test_rejects_negative_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(dim=-1)

    def test_rejects_zero_heads(self):
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(n_heads=0)

    def test_rejects_zero_kv_heads(self):
        with pytest.raises(ValueError, match="n_kv_heads must be positive"):
            ModelConfig(n_kv_heads=0)

    def test_rejects_zero_vocab(self):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ModelConfig(vocab_size=0)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults(self):
        t = TrainConfig()
        assert t.batch_size == 8
        assert t.compile_model is True

    def test_rejects_zero_batch(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainConfig(batch_size=0)

    def test_rejects_zero_grad_accum(self):
        with pytest.raises(ValueError, match="grad_accum_steps must be positive"):
            TrainConfig(grad_accum_steps=0)


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_defaults(self):
        o = OptimizerConfig()
        assert o.lr == 3e-4
        assert o.betas == (0.9, 0.95)

    def test_rejects_negative_lr(self):
        with pytest.raises(ValueError, match="lr must be positive"):
            OptimizerConfig(lr=-1)

    def test_rejects_bad_betas(self):
        with pytest.raises(ValueError, match="betas"):
            OptimizerConfig(betas=(1.0, 0.95))


# ---------------------------------------------------------------------------
# DistributedConfig
# ---------------------------------------------------------------------------


class TestDistributedConfig:
    def test_auto_dp_shard(self):
        d = DistributedConfig(dp_shard=-1, tp=2)
        resolved = d.resolve(world_size=8)
        assert resolved.dp_shard == 4

    def test_validate_world_size_matches(self):
        d = DistributedConfig(dp_shard=4, tp=2)
        d.validate_world_size(world_size=8)  # Should not raise

    def test_validate_world_size_mismatch(self):
        d = DistributedConfig(dp_shard=4, tp=2)
        with pytest.raises(ValueError, match="do not match world_size"):
            d.validate_world_size(world_size=16)

    def test_rejects_zero_tp(self):
        with pytest.raises(ValueError, match="tp must be >= 1"):
            DistributedConfig(tp=0)

    def test_rejects_bad_dp_shard(self):
        with pytest.raises(ValueError, match="dp_shard must be"):
            DistributedConfig(dp_shard=-2)

    def test_auto_dp_not_divisible(self):
        d = DistributedConfig(dp_shard=-1, tp=3)
        with pytest.raises(ValueError, match="not divisible"):
            d.resolve(world_size=8)


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------


class TestCheckpointConfig:
    def test_defaults(self):
        c = CheckpointConfig()
        assert c.async_mode == AsyncCheckpointMode.disabled
        assert c.keep_last_n == 3

    def test_rejects_zero_interval(self):
        with pytest.raises(ValueError, match="interval must be positive"):
            CheckpointConfig(interval=0)


# ---------------------------------------------------------------------------
# ProfilingConfig
# ---------------------------------------------------------------------------


class TestProfilingConfig:
    def test_rejects_end_before_start(self):
        with pytest.raises(ValueError, match="end_step must be greater"):
            ProfilingConfig(start_step=10, end_step=5)


# ---------------------------------------------------------------------------
# JobConfig cross-validation
# ---------------------------------------------------------------------------


class TestJobConfig:
    def test_validate_seq_len_exceeds_max(self):
        config = JobConfig(
            model=ModelConfig(max_seq_len=512),
            train=TrainConfig(seq_len=1024),
        )
        with pytest.raises(ValueError, match="exceeds"):
            config.validate(world_size=1)

    def test_validate_tied_embeddings_with_pp(self):
        config = JobConfig(
            model=ModelConfig(tie_embeddings=True),
            distributed=DistributedConfig(pp=2, dp_shard=1),
        )
        with pytest.raises(ValueError, match="Tied embeddings"):
            config.validate(world_size=2)

    def test_validate_tp_not_divisible_heads(self):
        config = JobConfig(
            model=ModelConfig(n_heads=32),
            distributed=DistributedConfig(tp=3, dp_shard=1),
        )
        with pytest.raises(ValueError, match="n_heads.*divisible by.*tp"):
            config.validate(world_size=3)

    def test_validate_passes_for_valid_config(self):
        config = JobConfig()
        config.validate(world_size=1)  # Should not raise


# ---------------------------------------------------------------------------
# TOML Loading
# ---------------------------------------------------------------------------


class TestTomlLoading:
    def test_load_debug_toml(self):
        config = load_config("configs/train/debug.toml", cli_args=[])
        assert config.model.dim == 256
        assert config.model.n_layers == 4
        assert config.train.max_steps == 100
        assert config.train.compile_model is False

    def test_load_default_toml(self):
        config = load_config("configs/train/default.toml", cli_args=[])
        assert config.model.dim == 4096
        assert config.model.n_kv_heads == 8
        assert config.train.compile_model is True
        assert config.train.activation_checkpointing == ActivationCheckpointing.full
        assert config.checkpoint.async_mode == AsyncCheckpointMode.async_pinned

    def test_enum_from_toml(self):
        config = load_config("configs/train/default.toml", cli_args=[])
        assert config.scheduler.name == SchedulerType.cosine
        assert isinstance(config.scheduler.name, SchedulerType)

    def test_missing_toml_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.toml", cli_args=[])

    def test_toml_with_typo_raises(self, tmp_path):
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("[modell]\ndim = 512\n")
        with pytest.raises(ValueError, match="Unknown config keys.*modell"):
            load_config(str(bad_toml), cli_args=[])

    def test_toml_field_typo_raises(self, tmp_path):
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("[model]\ndimm = 512\n")
        with pytest.raises(ValueError, match="Unknown config keys.*dimm"):
            load_config(str(bad_toml), cli_args=[])


# ---------------------------------------------------------------------------
# CLI Overrides
# ---------------------------------------------------------------------------


class TestCliOverrides:
    def test_int_override(self):
        config = load_config(cli_args=["--model.dim=512"])
        assert config.model.dim == 512

    def test_float_override(self):
        config = load_config(cli_args=["--optimizer.lr=1e-4"])
        assert config.optimizer.lr == 1e-4

    def test_bool_override(self):
        config = load_config(cli_args=["--train.compile_model=false"])
        assert config.train.compile_model is False

    def test_string_override(self):
        config = load_config(cli_args=["--checkpoint.load_path=/my/path"])
        assert config.checkpoint.load_path == "/my/path"

    def test_list_override(self):
        config = load_config(cli_args=["--optimizer.betas=[0.8,0.99]"])
        assert config.optimizer.betas == (0.8, 0.99)

    def test_enum_override(self):
        config = load_config(cli_args=["--checkpoint.async_mode=async"])
        assert config.checkpoint.async_mode == AsyncCheckpointMode.async_

    def test_none_default_field_override(self):
        config = load_config(cli_args=["--model.n_kv_heads=4"])
        assert config.model.n_kv_heads == 4

    def test_literal_field_override(self):
        config = load_config(cli_args=["--checkpoint.export_dtype=float32"])
        assert config.checkpoint.export_dtype == "float32"

    def test_literal_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="not in allowed Literal"):
            load_config(cli_args=["--checkpoint.export_dtype=float16"])

    def test_typo_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(cli_args=["--model.dimm=512"])

    def test_toml_then_cli_override(self):
        config = load_config(
            "configs/train/debug.toml",
            cli_args=["--model.dim=512", "--train.max_steps=50"],
        )
        assert config.model.dim == 512
        assert config.train.max_steps == 50
        # Non-overridden values preserved from TOML
        assert config.model.n_layers == 4


# ---------------------------------------------------------------------------
# CLI Parsing
# ---------------------------------------------------------------------------


class TestCliParsing:
    def test_parse_int(self):
        result = _parse_cli_overrides(["--model.dim=512"])
        assert result == {"model": {"dim": 512}}

    def test_parse_float(self):
        result = _parse_cli_overrides(["--optimizer.lr=3e-4"])
        assert result == {"optimizer": {"lr": 3e-4}}

    def test_parse_bool_flag(self):
        # Raw parse produces string "true"; coercion to bool happens in _apply_dict_to_dataclass
        result = _parse_cli_overrides(["--train.compile_model"])
        assert result == {"train": {"compile_model": "true"}}

    def test_parse_string(self):
        result = _parse_cli_overrides(["--data.dataset_path=/my/data"])
        assert result == {"data": {"dataset_path": "/my/data"}}

    def test_ignores_non_flag_args(self):
        result = _parse_cli_overrides(["config.toml", "--model.dim=512"])
        assert result == {"model": {"dim": 512}}

    def test_multiple_overrides(self):
        result = _parse_cli_overrides(["--model.dim=512", "--model.n_layers=8"])
        assert result == {"model": {"dim": 512, "n_layers": 8}}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry()
        reg.register("model", "test", lambda: "hello")
        assert reg.get("model", "test")() == "hello"

    def test_duplicate_raises(self):
        reg = Registry()
        reg.register("model", "test", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("model", "test", lambda: None)

    def test_unknown_raises(self):
        reg = Registry()
        with pytest.raises(KeyError, match="Unknown model"):
            reg.get("model", "nonexistent")

    def test_list(self):
        reg = Registry()
        reg.register("model", "a", None)
        reg.register("model", "b", None)
        assert reg.list("model") == ["a", "b"]

    def test_decorator_register_model(self):
        reg = Registry()

        @reg.register_model("my_model")
        def build_model(cfg):
            return "built"

        assert reg.get_model("my_model")(None) == "built"

    def test_separate_categories(self):
        reg = Registry()
        reg.register("model", "x", "model_x")
        reg.register("optimizer", "x", "opt_x")
        assert reg.get("model", "x") == "model_x"
        assert reg.get("optimizer", "x") == "opt_x"
