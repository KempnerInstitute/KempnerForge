"""Unit tests for data annealing & phase scheduling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import (
    DataConfig,
    DatasetSource,
    JobConfig,
    ModelConfig,
    TrainConfig,
    TrainingPhase,
)
from kempnerforge.data.sampler import MixtureSampler
from kempnerforge.training.data_pipeline import (
    DataPipeline,
    advance_phases,
    build_data_pipeline,
    build_phase_state,
)
from kempnerforge.training.runtime import RuntimeContext


class TestTrainingPhase:
    def test_valid_phase(self):
        p = TrainingPhase(start_step=100, dataset_weights={"code": 0.3, "text": 0.7})
        assert p.start_step == 100
        assert p.dataset_weights == {"code": 0.3, "text": 0.7}
        assert p.lr_scale == 1.0

    def test_lr_scale(self):
        p = TrainingPhase(start_step=50, lr_scale=0.5)
        assert p.lr_scale == 0.5

    def test_negative_start_step_rejected(self):
        with pytest.raises(ValueError, match="start_step"):
            TrainingPhase(start_step=-1)

    def test_zero_lr_scale_rejected(self):
        with pytest.raises(ValueError, match="lr_scale"):
            TrainingPhase(lr_scale=0.0)

    def test_negative_lr_scale_rejected(self):
        with pytest.raises(ValueError, match="lr_scale"):
            TrainingPhase(lr_scale=-0.5)

    def test_negative_dataset_weight_rejected(self):
        with pytest.raises(ValueError, match="dataset_weights"):
            TrainingPhase(dataset_weights={"bad": -1.0})

    def test_zero_dataset_weight_allowed(self):
        """Zero weight is valid — it drops a dataset from the mix."""
        p = TrainingPhase(dataset_weights={"drop_me": 0.0, "keep": 1.0})
        assert p.dataset_weights["drop_me"] == 0.0

    def test_empty_weights_allowed(self):
        """Empty weights = no override (use original)."""
        p = TrainingPhase(start_step=10)
        assert p.dataset_weights == {}


class TestDataConfigPhases:
    def test_default_no_phases(self):
        config = DataConfig()
        assert config.phases == []
        assert config.anneal_start_step == 0
        assert config.anneal_weights == {}

    def test_phases_set(self):
        config = DataConfig(
            phases=[
                TrainingPhase(start_step=100, dataset_weights={"a": 0.5}),
                TrainingPhase(start_step=200, dataset_weights={"b": 0.5}),
            ]
        )
        assert len(config.phases) == 2

    def test_phases_must_be_monotonic(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            DataConfig(
                phases=[
                    TrainingPhase(start_step=200),
                    TrainingPhase(start_step=100),
                ]
            )

    def test_phases_duplicate_start_step_rejected(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            DataConfig(
                phases=[
                    TrainingPhase(start_step=100),
                    TrainingPhase(start_step=100),
                ]
            )

    def test_phases_and_anneal_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot use both"):
            DataConfig(
                phases=[TrainingPhase(start_step=100)],
                anneal_start_step=200,
            )

    def test_anneal_shortcut_fields(self):
        config = DataConfig(
            anneal_start_step=500,
            anneal_weights={"code": 0.3, "text": 0.7},
        )
        assert config.anneal_start_step == 500
        assert config.anneal_weights["code"] == 0.3

    def test_negative_anneal_start_step_rejected(self):
        with pytest.raises(ValueError, match="anneal_start_step"):
            DataConfig(anneal_start_step=-1)


class TestAnnealingTomlLoading:
    def test_load_phases_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[train]
max_steps = 1000

[[data.phases]]
start_step = 100
lr_scale = 0.5

[data.phases.dataset_weights]
code = 0.3
text = 0.7

[[data.phases]]
start_step = 500
lr_scale = 0.1

[data.phases.dataset_weights]
code = 0.1
text = 0.9
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert len(config.data.phases) == 2
        assert isinstance(config.data.phases[0], TrainingPhase)
        assert config.data.phases[0].start_step == 100
        assert config.data.phases[0].lr_scale == 0.5
        assert config.data.phases[0].dataset_weights == {"code": 0.3, "text": 0.7}
        assert config.data.phases[1].start_step == 500

    def test_load_anneal_shortcut_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[data]
anneal_start_step = 800

[data.anneal_weights]
code = 0.2
math = 0.8
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert config.data.anneal_start_step == 800
        assert config.data.anneal_weights == {"code": 0.2, "math": 0.8}

    def test_load_phases_with_inline_weights(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[[data.phases]]
start_step = 50
dataset_weights = {a = 0.6, b = 0.4}
"""
        )
        from kempnerforge.config.loader import load_config

        config = load_config(str(toml_path), cli_args=[])
        assert config.data.phases[0].dataset_weights == {"a": 0.6, "b": 0.4}

    def test_unknown_key_in_phase_rejected(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            """
[[data.phases]]
start_step = 100
typo_field = "oops"
"""
        )
        from kempnerforge.config.loader import load_config

        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(str(toml_path), cli_args=[])


class TestUpdateWeights:
    def test_update_changes_proportions(self):
        """Updating weights shifts sampling proportions."""
        cumulative = [0, 500, 1000]
        sampler = MixtureSampler(
            cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, shuffle=False
        )
        indices = list(sampler)
        ds0_before = sum(1 for i in indices if i < 500)
        ratio_before = ds0_before / len(indices)
        assert 0.4 < ratio_before < 0.6

        sampler.update_weights([0.9, 0.1])
        indices = list(sampler)
        ds0_after = sum(1 for i in indices if i < 500)
        ratio_after = ds0_after / len(indices)
        assert 0.8 < ratio_after < 1.0

    def test_update_preserves_total(self):
        """Total samples stays the same after weight update."""
        cumulative = [0, 200, 500]
        sampler = MixtureSampler(
            cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, shuffle=False
        )
        total_before = len(sampler)
        sampler.update_weights([0.8, 0.2])
        assert len(sampler) == total_before

    def test_update_wrong_count_raises(self):
        cumulative = [0, 100, 200]
        sampler = MixtureSampler(cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0)
        with pytest.raises(ValueError, match="Expected 2"):
            sampler.update_weights([1.0])

    def test_update_with_temperature(self):
        """Temperature scaling applies to updated weights."""
        cumulative = [0, 500, 1000]
        sampler = MixtureSampler(
            cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, shuffle=False
        )
        sampler.update_weights([0.9, 0.1], temperature=5.0)
        indices = list(sampler)
        ds0_count = sum(1 for i in indices if i < 500)
        ratio = ds0_count / len(indices)
        assert 0.4 < ratio < 0.75


class TestPhaseTransitionLogic:
    def test_phase_activates_at_correct_step(self):
        """Phase transition fires when step >= start_step."""
        phases = [
            TrainingPhase(start_step=5, dataset_weights={"a": 0.9, "b": 0.1}),
        ]
        cumulative = [0, 100, 200]
        sampler = MixtureSampler(
            cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, shuffle=False
        )
        names = ["a", "b"]
        original_weights = {"a": 0.5, "b": 0.5}

        phase_idx = 0
        for step in range(1, 11):
            while phase_idx < len(phases) and step >= phases[phase_idx].start_step:
                phase = phases[phase_idx]
                new_w = [phase.dataset_weights.get(n, original_weights[n]) for n in names]
                sampler.update_weights(new_w)
                phase_idx += 1

        assert phase_idx == 1
        indices = list(sampler)
        ds0_count = sum(1 for i in indices if i < 100)
        ratio = ds0_count / len(indices)
        assert ratio > 0.7

    def test_multiple_phases_activate_sequentially(self):
        """Multiple phases activate in order as steps progress."""
        phases = [
            TrainingPhase(start_step=5, dataset_weights={"a": 0.8, "b": 0.2}),
            TrainingPhase(start_step=10, dataset_weights={"a": 0.2, "b": 0.8}),
        ]
        cumulative = [0, 500, 1000]
        sampler = MixtureSampler(
            cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, shuffle=False
        )
        names = ["a", "b"]
        original = {"a": 0.5, "b": 0.5}

        phase_idx = 0
        for step in range(1, 15):
            while phase_idx < len(phases) and step >= phases[phase_idx].start_step:
                phase = phases[phase_idx]
                new_w = [phase.dataset_weights.get(n, original[n]) for n in names]
                sampler.update_weights(new_w)
                phase_idx += 1

        assert phase_idx == 2
        indices = list(sampler)
        ds1_count = sum(1 for i in indices if i >= 500)
        ratio = ds1_count / len(indices)
        assert ratio > 0.7

    def test_phase_partial_override(self):
        """Phase that only specifies some datasets inherits original weights for others."""
        phase = TrainingPhase(start_step=1, dataset_weights={"a": 0.9})
        names = ["a", "b"]
        original = {"a": 0.5, "b": 0.5}

        new_w = [phase.dataset_weights.get(n, original[n]) for n in names]
        assert new_w == [0.9, 0.5]

    def test_lr_scale_from_phase(self):
        """LR scale is correctly read from phase config."""
        phases = [
            TrainingPhase(start_step=10, lr_scale=0.5),
            TrainingPhase(start_step=20, lr_scale=0.1),
        ]
        lr_scale = 1.0
        phase_idx = 0
        for step in range(1, 25):
            while phase_idx < len(phases) and step >= phases[phase_idx].start_step:
                lr_scale = phases[phase_idx].lr_scale
                phase_idx += 1

        assert lr_scale == 0.1

    def test_resume_derives_phase_from_step(self):
        """On resume, correct phase is derived from current step."""
        phases = [
            TrainingPhase(start_step=10, dataset_weights={"a": 0.8, "b": 0.2}, lr_scale=0.5),
            TrainingPhase(start_step=50, dataset_weights={"a": 0.2, "b": 0.8}, lr_scale=0.1),
        ]

        resume_step = 30
        phase_idx = 0
        lr_scale = 1.0
        for i, phase in enumerate(phases):
            if resume_step >= phase.start_step:
                lr_scale = phase.lr_scale
                phase_idx = i + 1

        assert phase_idx == 1
        assert lr_scale == 0.5

        resume_step = 60
        phase_idx = 0
        lr_scale = 1.0
        for i, phase in enumerate(phases):
            if resume_step >= phase.start_step:
                lr_scale = phase.lr_scale
                phase_idx = i + 1

        assert phase_idx == 2
        assert lr_scale == 0.1


class TestAnnealingShortcut:
    def test_shortcut_produces_single_phase(self):
        """anneal_start_step + anneal_weights is equivalent to a single TrainingPhase."""
        config = DataConfig(
            anneal_start_step=500,
            anneal_weights={"code": 0.3, "text": 0.7},
        )
        active_phases = []
        if config.phases:
            active_phases = config.phases
        elif config.anneal_start_step > 0 and config.anneal_weights:
            active_phases = [
                TrainingPhase(
                    start_step=config.anneal_start_step,
                    dataset_weights=dict(config.anneal_weights),
                )
            ]

        assert len(active_phases) == 1
        assert active_phases[0].start_step == 500
        assert active_phases[0].dataset_weights == {"code": 0.3, "text": 0.7}
        assert active_phases[0].lr_scale == 1.0

    def test_shortcut_disabled_when_zero(self):
        """anneal_start_step=0 means no annealing."""
        config = DataConfig(anneal_start_step=0, anneal_weights={"code": 0.5})
        active_phases = []
        if config.anneal_start_step > 0 and config.anneal_weights:
            active_phases = [
                TrainingPhase(
                    start_step=config.anneal_start_step,
                    dataset_weights=dict(config.anneal_weights),
                )
            ]
        assert active_phases == []

    def test_shortcut_disabled_when_empty_weights(self):
        """Empty anneal_weights means no annealing even with start_step > 0."""
        config = DataConfig(anneal_start_step=100)
        active_phases = []
        if config.anneal_start_step > 0 and config.anneal_weights:
            active_phases = [
                TrainingPhase(
                    start_step=config.anneal_start_step,
                    dataset_weights=dict(config.anneal_weights),
                )
            ]
        assert active_phases == []

    def test_shortcut_same_result_as_explicit_phase(self):
        """Annealing shortcut produces identical sampling as explicit phase config."""
        cumulative = [0, 500, 1000]
        names = ["code", "text"]

        shortcut_phase = TrainingPhase(start_step=100, dataset_weights={"code": 0.3, "text": 0.7})
        s1 = MixtureSampler(
            cumulative,
            weights=[0.5, 0.5],
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        new_w = [shortcut_phase.dataset_weights.get(n, 0.5) for n in names]
        s1.update_weights(new_w)

        s2 = MixtureSampler(
            cumulative,
            weights=[0.5, 0.5],
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        s2.update_weights([0.3, 0.7])

        assert list(s1) == list(s2)


class TestMixtureWeightsNeedAMixture:
    """``JobConfig.validate`` refuses mixture weights on a run that builds no mixture: a phase's
    ``dataset_weights`` and the annealing shortcut, on a run without ``[[data.datasets]]`` and on
    a VLM run, which never builds one. A phase's ``lr_scale`` needs no mixture."""

    @staticmethod
    def _vlm(tiny_vlm_configs, data: DataConfig) -> JobConfig:
        model, vision, adapter, vlm = tiny_vlm_configs
        return JobConfig(
            model=model,
            train=TrainConfig(batch_size=1, seq_len=64),
            data=data,
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
        )

    def test_phase_weights_on_a_single_source_run_are_refused(self):
        data = DataConfig(
            dataset_path="/data/tokens",
            phases=[
                TrainingPhase(start_step=5, dataset_weights={"a": 2.0}),
                TrainingPhase(start_step=9, lr_scale=0.5),
                TrainingPhase(start_step=12, dataset_weights={"b": 1.0}),
            ],
        )
        refusal = (
            r"^data\.phases\[0\]\.dataset_weights, data\.phases\[2\]\.dataset_weights would "
            r"reweight a \[\[data\.datasets\]\] mixture, but this run has none \(data\.datasets "
            r"is empty\), so the weights would be ignored\. List the sources as "
            r"\[\[data\.datasets\]\] tables or remove the weights; a data phase's lr_scale "
            r"applies without a mixture\.$"
        )
        with pytest.raises(ValueError, match=refusal):
            JobConfig(data=data).validate(world_size=1)

    def test_the_annealing_shortcut_without_a_mixture_is_refused(self):
        data = DataConfig(
            hf_dataset_name="org/corpus", anneal_start_step=7, anneal_weights={"a": 2.0}
        )
        refusal = r"^data\.anneal_start_step and data\.anneal_weights would reweight"
        with pytest.raises(ValueError, match=refusal):
            JobConfig(data=data).validate(world_size=1)

    def test_a_vlm_run_refuses_phase_weights_even_with_datasets(self, tiny_vlm_configs):
        data = DataConfig(
            datasets=[DatasetSource(path="/data/a", name="a")],
            phases=[TrainingPhase(start_step=5, dataset_weights={"a": 2.0})],
        )
        refusal = (
            r"^A VLM run never builds a \[\[data\.datasets\]\] mixture \(it reads its data from "
            r"data\.hf_dataset_name and the other data\.hf_\* fields\), so these \[data\] fields "
            r"would be ignored: data\.datasets, data\.phases\[0\]\.dataset_weights\. Remove them; "
            r"a data phase's lr_scale applies without a mixture\.$"
        )
        with pytest.raises(ValueError, match=refusal):
            self._vlm(tiny_vlm_configs, data).validate(world_size=1)

    def test_a_vlm_run_refuses_the_annealing_shortcut(self, tiny_vlm_configs):
        data = DataConfig(anneal_start_step=7, anneal_weights={"a": 2.0})
        refusal = (
            r"would be ignored: data\.anneal_start_step and data\.anneal_weights\. Remove them; "
            r"a data phase's lr_scale applies without a mixture\.$"
        )
        with pytest.raises(ValueError, match=refusal):
            self._vlm(tiny_vlm_configs, data).validate(world_size=1)

    def test_lr_scale_phases_need_no_mixture(self, tiny_vlm_configs):
        data = DataConfig(
            phases=[
                TrainingPhase(start_step=0, lr_scale=0.5),
                TrainingPhase(start_step=5, dataset_weights={}, lr_scale=0.1),
            ]
        )
        JobConfig(data=data).validate(world_size=1)
        self._vlm(tiny_vlm_configs, data).validate(world_size=1)

    def test_a_mixture_takes_phase_weights_and_the_annealing_shortcut(self):
        sources = [DatasetSource(path="/data/a", name="a"), DatasetSource(path="/data/b", name="b")]
        phases = [TrainingPhase(start_step=5, dataset_weights={"a": 2.0})]
        JobConfig(data=DataConfig(datasets=sources, phases=phases)).validate(world_size=1)
        shortcut = DataConfig(datasets=sources, anneal_start_step=7, anneal_weights={"b": 3.0})
        JobConfig(data=shortcut).validate(world_size=1)

    def test_the_settings_that_set_mixture_weights(self):
        """Each phase with ``dataset_weights``, and the annealing shortcut only when its two
        fields make a phase."""
        phases = [
            TrainingPhase(start_step=0, lr_scale=0.5),
            TrainingPhase(start_step=5, dataset_weights={"a": 2.0}),
            TrainingPhase(start_step=9, dataset_weights={}),
        ]
        assert DataConfig(phases=phases).mixture_weight_fields() == [
            "data.phases[1].dataset_weights"
        ]
        shortcut = DataConfig(anneal_start_step=7, anneal_weights={"a": 2.0})
        assert shortcut.mixture_weight_fields() == [
            "data.anneal_start_step and data.anneal_weights"
        ]
        assert DataConfig(anneal_weights={"a": 2.0}).mixture_weight_fields() == []

    def test_the_mixture_may_come_from_the_command_line(self, tmp_path):
        """The check reads the config after the overrides: phase weights in the TOML pass with
        the sources given on the command line, and are refused without them."""
        toml = tmp_path / "phases.toml"
        toml.write_text("[[data.phases]]\nstart_step = 5\ndataset_weights = { a = 2.0 }\n")
        sources = "--data.datasets=[{'path': '/data/a', 'name': 'a'}]"
        load_config(str(toml), cli_args=[sources]).validate(world_size=1)
        with pytest.raises(ValueError, match=r"^data\.phases\[0\]\.dataset_weights would"):
            load_config(str(toml), cli_args=[]).validate(world_size=1)


def _cpu_runtime() -> RuntimeContext:
    return RuntimeContext(
        rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), device_mesh=None
    )


def _share_after_the_phase(config: JobConfig, data: DataPipeline, source: int) -> float:
    """The share of the draws from source ``source`` once the step-1 phase is active."""
    assert data.mixture_dataset is not None and data.mixture_sampler is not None
    state = build_phase_state(config, data, step=0)
    advance_phases(state, data, step=1)
    drawn = list(data.mixture_sampler)
    first, end = data.mixture_dataset.cumulative_sizes[source : source + 2]
    return sum(first <= index < end for index in drawn) / len(drawn)


class TestMixtureWeightsNameSources:
    """``JobConfig.validate`` refuses a phase's ``dataset_weights`` key, or an ``anneal_weights``
    key, that names no ``[[data.datasets]]`` source, naming each field with its unknown keys and
    listing the sources. A source goes by its ``name``, else its ``path``, else its ``hf_name``
    (``DatasetSource.mixture_name``), the name the mixture builder gives it."""

    @staticmethod
    def _config(sources: list[DatasetSource], **schedule) -> JobConfig:
        for source in sources:
            Path(source.path).mkdir(parents=True, exist_ok=True)
            np.save(Path(source.path, "tokens.npy"), np.arange(500, dtype=np.uint16) % 16)
        return JobConfig(
            model=ModelConfig(dim=8, n_layers=1, n_heads=1, vocab_size=16, max_seq_len=64),
            train=TrainConfig(batch_size=2, seq_len=4, max_steps=4, compile_model=False),
            data=DataConfig(num_workers=0, pin_memory=False, datasets=sources, **schedule),
        )

    def test_the_issue_reproduction(self, tmp_path):
        """Two sources of one size at equal weights, and a phase at step 1 that sets ``code`` to
        9.0: spelled right, the config validates and ``code`` takes 0.90 of the draws after the
        phase; misspelled ``cod``, in a phase or in the annealing shortcut, it is refused."""
        sources = [DatasetSource(name=name, path=str(tmp_path / name)) for name in ("web", "code")]
        right = [TrainingPhase(start_step=1, dataset_weights={"code": 9.0})]
        config = self._config(sources, phases=right)
        config.validate(world_size=1)
        data = build_data_pipeline(config, _cpu_runtime())
        assert _share_after_the_phase(config, data, 1) == pytest.approx(0.9, abs=0.01)

        misspelled = [
            (
                {"phases": [TrainingPhase(start_step=1, dataset_weights={"cod": 9.0})]},
                r"data\.phases\[0\]\.dataset_weights",
            ),
            ({"anneal_start_step": 1, "anneal_weights": {"cod": 9.0}}, r"data\.anneal_weights"),
        ]
        for schedule, field in misspelled:
            refusal = (
                rf"^{field} names unknown sources \['cod'\]; the \[\[data\.datasets\]\] sources "
                r"are \['web', 'code'\] \(a source without a name goes by its path or hf_name\), "
                r"and a weight for an unknown source would be ignored\.$"
            )
            with pytest.raises(ValueError, match=refusal):
                self._config(sources, **schedule).validate(world_size=1)

    def test_each_field_is_named_with_its_unknown_keys(self):
        sources = [DatasetSource(name="web", path="/data/web"), DatasetSource(path="/data/code")]
        phases = [
            TrainingPhase(start_step=1, dataset_weights={"web": 1.0, "wbe": 2.0, "code": 1.0}),
            TrainingPhase(start_step=2, dataset_weights={"/data/code": 3.0}),
            TrainingPhase(start_step=3, dataset_weights={"/data/web": 1.0}),
        ]
        refusal = (
            r"^data\.phases\[0\]\.dataset_weights names unknown sources \['code', 'wbe'\], "
            r"data\.phases\[2\]\.dataset_weights names unknown sources \['/data/web'\]; the "
            r"\[\[data\.datasets\]\] sources are \['web', '/data/code'\] "
        )
        with pytest.raises(ValueError, match=refusal):
            JobConfig(data=DataConfig(datasets=sources, phases=phases)).validate(world_size=1)

    def test_a_key_names_the_source_the_phase_reweights(self, tmp_path):
        """The builder names each source as ``validate`` does: a source without a ``name`` goes
        by its ``path``, and a phase keyed by that path takes that source's share."""
        sources = [
            DatasetSource(name="web", path=str(tmp_path / "web")),
            DatasetSource(path=str(tmp_path / "code")),
        ]
        phases = [TrainingPhase(start_step=1, dataset_weights={str(tmp_path / "code"): 9.0})]
        config = self._config(sources, phases=phases)
        config.validate(world_size=1)
        data = build_data_pipeline(config, _cpu_runtime())
        assert data.mixture_dataset is not None
        assert data.mixture_dataset.dataset_names == [s.mixture_name for s in sources]
        assert _share_after_the_phase(config, data, 1) == pytest.approx(0.9, abs=0.01)

    def test_a_source_goes_by_its_name_else_its_path_else_its_hf_name(self):
        assert DatasetSource(name="web", path="/data/web").mixture_name == "web"
        assert DatasetSource(path="/data/web", hf_name="org/web").mixture_name == "/data/web"
        assert DatasetSource(hf_name="org/web").mixture_name == "org/web"

    def test_the_unknown_keys_of_each_setting_that_reweights_the_mixture(self):
        """The settings ``mixture_weight_fields`` lists: each phase's ``dataset_weights``, and
        ``anneal_weights`` when ``anneal_start_step`` makes it a phase."""
        sources = [DatasetSource(name="web", path="/data/web"), DatasetSource(hf_name="org/code")]
        phases = [
            TrainingPhase(start_step=1, dataset_weights={"web": 1.0, "cod": 2.0, "/data/web": 1.0}),
            TrainingPhase(start_step=2, dataset_weights={"org/code": 1.0}),
            TrainingPhase(start_step=3, lr_scale=0.5),
        ]
        weighted = DataConfig(datasets=sources, phases=phases)
        assert weighted.unknown_mixture_weight_keys() == {
            "data.phases[0].dataset_weights": ["/data/web", "cod"]
        }
        shortcut = DataConfig(datasets=sources, anneal_start_step=7, anneal_weights={"code": 2.0})
        assert shortcut.unknown_mixture_weight_keys() == {"data.anneal_weights": ["code"]}
        assert DataConfig(datasets=sources).unknown_mixture_weight_keys() == {}

    def test_the_sources_and_the_weights_may_come_from_the_command_line(self, tmp_path):
        """The check reads the config after the overrides: TOML phase weights against sources
        given on the command line, and command-line anneal weights against TOML sources."""
        phases = tmp_path / "phases.toml"
        phases.write_text("[[data.phases]]\nstart_step = 5\ndataset_weights = { web = 2.0 }\n")
        named = "--data.datasets=[{'path': '/data/web', 'name': 'web'}]"
        load_config(str(phases), cli_args=[named]).validate(world_size=1)
        renamed = "--data.datasets=[{'path': '/data/web', 'name': 'webtext'}]"
        refusal = (
            r"^data\.phases\[0\]\.dataset_weights names unknown sources \['web'\]; the "
            r"\[\[data\.datasets\]\] sources are \['webtext'\] "
        )
        with pytest.raises(ValueError, match=refusal):
            load_config(str(phases), cli_args=[renamed]).validate(world_size=1)

        sources = tmp_path / "sources.toml"
        sources.write_text('[[data.datasets]]\nname = "web"\npath = "/data/web"\n')
        anneal = ["--data.anneal_start_step=5", '--data.anneal_weights={"wbe": 2.0}']
        refusal = r"^data\.anneal_weights names unknown sources \['wbe'\]; the .* \['web'\] "
        with pytest.raises(ValueError, match=refusal):
            load_config(str(sources), cli_args=anneal).validate(world_size=1)


class TestVlmRunsBuildNoMixture:
    """A VLM run reads its data from ``data.hf_dataset_name`` and the other ``data.hf_*`` fields,
    or from ``[video]``, and never builds a ``[[data.datasets]]`` mixture, so ``JobConfig.validate``
    refuses ``[[data.datasets]]`` there, in one error with any mixture weights (see
    ``TestMixtureWeightsNeedAMixture``) that names each field and where the run reads its data."""

    @staticmethod
    def _vlm(tiny_vlm_configs, data: DataConfig) -> JobConfig:
        model, vision, adapter, vlm = tiny_vlm_configs
        return JobConfig(
            model=model,
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=2, compile_model=False),
            data=data,
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
        )

    IMAGE_REFUSAL = (
        r"^A VLM run never builds a \[\[data\.datasets\]\] mixture \(it reads its data from "
        r"data\.hf_dataset_name and the other data\.hf_\* fields\), so these \[data\] fields "
        r"would be ignored: data\.datasets\. Remove them\.$"
    )

    def test_the_issue_reproduction(self, tiny_vlm_configs):
        """One ``[[data.datasets]]`` text source beside an image-text ``data.hf_dataset_name``,
        then the same config without ``data.hf_dataset_name``: both validated, and the run
        trained on the image-text dataset alone or failed naming only ``data.hf_dataset_name``;
        both are refused, naming ``data.datasets``."""
        data = DataConfig(
            num_workers=0,
            pin_memory=False,
            datasets=[DatasetSource(name="text", path="/data/text")],
            hf_dataset_name="/data/captions",
            tokenizer_path="/data/tokenizer",
        )
        config = self._vlm(tiny_vlm_configs, data)
        with pytest.raises(ValueError, match=self.IMAGE_REFUSAL):
            config.validate(world_size=1)
        config.data.hf_dataset_name = None
        with pytest.raises(ValueError, match=self.IMAGE_REFUSAL):
            config.validate(world_size=1)

    def test_lr_scale_phases_leave_the_sources_refused(self, tiny_vlm_configs):
        data = DataConfig(
            datasets=[DatasetSource(path="/data/text")],
            phases=[TrainingPhase(start_step=5, lr_scale=0.5)],
        )
        with pytest.raises(ValueError, match=self.IMAGE_REFUSAL):
            self._vlm(tiny_vlm_configs, data).validate(world_size=1)

    def test_a_video_run_reads_its_data_from_its_video_section(self, tiny_video_configs):
        model, vision, adapter, vlm, video = tiny_video_configs
        config = JobConfig(
            model=model,
            train=TrainConfig(batch_size=1, seq_len=64),
            data=DataConfig(datasets=[DatasetSource(path="/data/text")]),
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
            video=video,
        )
        refusal = (
            r"^A VLM run never builds a \[\[data\.datasets\]\] mixture \(it reads its data from "
            r"\[video\]\), so these \[data\] fields would be ignored: data\.datasets\. Remove "
            r"them\.$"
        )
        with pytest.raises(ValueError, match=refusal):
            config.validate(world_size=1)

    def test_the_sources_may_come_from_the_command_line(self, tmp_path):
        """The check reads the config after the overrides: a VLM TOML validates alone and is
        refused with sources given on the command line."""
        toml = tmp_path / "vlm.toml"
        toml.write_text(
            "[model]\ndim = 64\nn_layers = 2\nn_heads = 4\nn_kv_heads = 4\nvocab_size = 256\n"
            "max_seq_len = 64\n"
            "[train]\nbatch_size = 1\nseq_len = 64\n"
            '[vision_encoder]\ntype = "random"\nfeature_dim = 96\nnum_tokens = 8\n'
            "[vlm]\nmax_text_len = 32\n"
        )
        load_config(str(toml), cli_args=[]).validate(world_size=1)
        sources = "--data.datasets=[{'path': '/data/text', 'name': 'text'}]"
        with pytest.raises(ValueError, match=self.IMAGE_REFUSAL):
            load_config(str(toml), cli_args=[sources]).validate(world_size=1)
