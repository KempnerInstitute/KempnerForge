"""Tests for resolving a saved checkpoint and loading its model."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from evaluation.checkpoint import (
    SUPPORTED_ARCHES,
    check_supported,
    load_model,
    read_step,
    resolve_checkpoint,
    step_dir_complete,
)

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper

CPU = torch.device("cpu")


@pytest.fixture
def load(write_config: Callable[..., Path]) -> Callable[..., JobConfig]:
    """Loads the tiny run's config with ``--section.key=value`` overrides."""
    return lambda *overrides: load_config(write_config(), cli_args=list(overrides))


def _fresh_model(config: JobConfig) -> VLMWrapper:
    vision, adapter, vlm = config.vision_encoder, config.adapter, config.vlm
    assert vision is not None and adapter is not None and vlm is not None
    return build_vlm_wrapper(config.model, vision, adapter, vlm)


def _refused(config: JobConfig, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        check_supported(config)


class TestCheckSupported:
    def test_the_joint_decoder_recipe_is_accepted(self, load: Callable[..., JobConfig]) -> None:
        check_supported(load())
        assert SUPPORTED_ARCHES == ("joint_decoder",)

    def test_a_text_config_is_refused(self) -> None:
        _refused(JobConfig(), "no [vlm] section")

    @pytest.mark.parametrize("arch", ["cross_attention", "mot", "moma"])
    def test_another_arch_is_refused_with_the_supported_ones(
        self, write_config: Callable[..., Path], tmp_path: Path, arch: str
    ) -> None:
        text = write_config().read_text().replace('"joint_decoder"', f'"{arch}"')
        path = tmp_path / f"{arch}.toml"
        path.write_text(text)
        config = load_config(path, cli_args=[])
        assert config.vlm is not None and type(config.vlm).arch == arch
        _refused(
            config,
            f"vlm.arch={arch!r} is not supported: the evaluator decodes ['joint_decoder']",
        )

    def test_an_arch_set_by_an_override_is_refused(self, load: Callable[..., JobConfig]) -> None:
        _refused(load("--vlm.arch=mot"), "vlm.arch='mot' is not supported")

    def test_a_video_config_is_refused(self, write_config: Callable[..., Path]) -> None:
        video = "\n[video]\nmax_frames = 2\nmin_frames = 1\nframe_size = 16\n"
        config = load_config(write_config(video), cli_args=[])
        _refused(config, "a [video] config is not supported")

    def test_a_tiled_config_is_refused(self, load: Callable[..., JobConfig]) -> None:
        config = load("--vlm.tile_grid=2", "--model.max_seq_len=64")
        _refused(
            config,
            "vlm.tile_grid=2 is not supported: the renderer makes one view of an image, and "
            "this model takes 5",
        )

    def test_capped_expert_capacity_is_refused(self, load: Callable[..., JobConfig]) -> None:
        """A capped expert drops tokens by batch content, so a row would depend on its batch."""
        experts = ("--model.num_experts=4", "--model.moe_top_k=2")
        check_supported(load(*experts))
        _refused(
            load(*experts, "--model.moe_capacity_factor=1.5"),
            "model.moe_capacity_factor=1.5 is not supported",
        )

    def test_pipeline_parallelism_is_refused(self, load: Callable[..., JobConfig]) -> None:
        check_supported(load("--model.tie_embeddings=false"))
        _refused(
            load("--model.tie_embeddings=false", "--distributed.pp=2"),
            "distributed.pp=2 is not supported: the evaluator loads the whole model in each "
            "process from one set of DCP shards, while a pipeline-parallel save holds one set "
            "per stage",
        )


class TestResolveCheckpoint:
    def test_a_run_directory_resolves_to_its_latest_step(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        config = load()
        save_checkpoint(config, step=3)
        latest = save_checkpoint(config, step=5)
        assert resolve_checkpoint(config.checkpoint.dir) == latest.resolve()

    def test_a_step_directory_resolves_to_itself(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        config = load()
        first = save_checkpoint(config, step=3)
        save_checkpoint(config, step=5)
        assert resolve_checkpoint(first) == first.resolve()
        assert resolve_checkpoint(str(first)) == first.resolve()

    def test_symlinks_are_followed(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path], tmp_path: Path
    ) -> None:
        config = load()
        step_dir = save_checkpoint(config)
        link = tmp_path / "link"
        link.symlink_to(step_dir)
        assert resolve_checkpoint(link) == step_dir.resolve()
        assert resolve_checkpoint(Path(config.checkpoint.dir) / "latest") == step_dir.resolve()

    def test_a_step_without_train_state_still_resolves(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        config = load()
        step_dir = save_checkpoint(config)
        (step_dir / "train_state.pt").unlink()
        (Path(config.checkpoint.dir) / "latest").unlink()
        assert resolve_checkpoint(config.checkpoint.dir) == step_dir.resolve()

    def test_a_path_that_is_not_a_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="is not a directory"):
            resolve_checkpoint(tmp_path / "missing")

    def test_a_directory_without_weights_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "run" / "step_9").mkdir(parents=True)
        for path in (tmp_path / "run", tmp_path / "run" / "step_9"):
            with pytest.raises(FileNotFoundError, match="holds no saved weights"):
                resolve_checkpoint(path)


class TestReadStep:
    def test_the_saved_step_is_read(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        assert read_step(save_checkpoint(load(), step=7)) == 7

    @pytest.mark.parametrize("text", [None, "{", "[1]", '{"step": "7"}', '{"step": true}', "{}"])
    def test_a_checkpoint_without_a_readable_step_has_none(
        self, tmp_path: Path, text: str | None
    ) -> None:
        if text is not None:
            (tmp_path / "metadata.json").write_text(text)
        assert read_step(tmp_path) is None


class TestStepDirComplete:
    @pytest.mark.parametrize(
        ("present", "expected"),
        [
            ((".metadata", "train_state.pt", "metadata.json"), True),
            ((".metadata", "train_state.pt"), False),
            (("train_state.pt", "metadata.json"), False),
            ((".metadata", "metadata.json"), False),
            ((), False),
        ],
    )
    def test_a_step_directory_is_complete_only_with_every_file(
        self, tmp_path: Path, present: tuple[str, ...], expected: bool
    ) -> None:
        step_dir = tmp_path / "step_1"
        step_dir.mkdir()
        for name in present:
            (step_dir / name).write_text("")
        assert step_dir_complete(step_dir) is expected
        assert step_dir_complete(tmp_path / "missing") is False


class TestLoadModel:
    def test_the_saved_weights_are_loaded_in_eval_mode(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        config = load()
        step_dir = save_checkpoint(config, seed=5)
        torch.manual_seed(5)
        saved = _fresh_model(config)
        torch.manual_seed(6)
        model = load_model(config, step_dir, CPU)
        loaded = model.state_dict()
        assert loaded.keys() == saved.state_dict().keys()
        assert all(torch.equal(loaded[name], value) for name, value in saved.state_dict().items())
        assert not model.training

    def test_the_loaded_state_is_not_copied_a_second_time(
        self,
        load: Callable[..., JobConfig],
        save_checkpoint: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = load()
        step_dir = save_checkpoint(config, seed=5)

        def refuse(self: VLMWrapper, *args: object, **kwargs: object) -> None:
            pytest.fail("load_state_dict copies what dcp.load already wrote in place")

        monkeypatch.setattr(VLMWrapper, "load_state_dict", refuse)
        torch.manual_seed(5)
        saved = _fresh_model(config)
        torch.manual_seed(6)
        loaded = load_model(config, step_dir, CPU).state_dict()
        assert all(torch.equal(loaded[name], value) for name, value in saved.state_dict().items())

    def test_the_model_computes_in_the_training_dtype(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        """fp32 weights load into the dtype of ``train.mixed_precision``; the encoder keeps its."""
        config = load("--train.mixed_precision=bf16")
        step_dir = save_checkpoint(config, seed=5)
        torch.manual_seed(5)
        saved = _fresh_model(config)
        model = load_model(config, step_dir, CPU)
        for part in ("transformer", "adapter"):
            for name, value in getattr(model, part).state_dict().items():
                assert value.dtype == torch.bfloat16
                assert torch.equal(value, getattr(saved, part).state_dict()[name].bfloat16())
        assert model.vision_encoder.get_buffer("_anchor").dtype == torch.float32

    def test_the_encoder_computes_in_its_own_dtype_when_one_is_set(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        config = load("--vision_encoder.mixed_precision=bf16")
        model = load_model(config, save_checkpoint(config), CPU)
        assert model.vision_encoder.get_buffer("_anchor").dtype == torch.bfloat16
        assert next(model.transformer.parameters()).dtype == torch.float32

    def test_a_checkpoint_of_another_model_is_refused(
        self, load: Callable[..., JobConfig], save_checkpoint: Callable[..., Path]
    ) -> None:
        step_dir = save_checkpoint(load())
        with pytest.raises(RuntimeError, match="does not load into the model.*Size mismatch"):
            load_model(load("--model.dim=16"), step_dir, CPU)
