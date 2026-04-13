"""Unit tests for data annealing & phase scheduling."""

from __future__ import annotations

import pytest

from kempnerforge.config.schema import DataConfig, TrainingPhase
from kempnerforge.data.sampler import MixtureSampler

# ---------------------------------------------------------------------------
# TrainingPhase config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DataConfig phase validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MixtureSampler.update_weights
# ---------------------------------------------------------------------------


class TestUpdateWeights:
    def test_update_changes_proportions(self):
        """Updating weights shifts sampling proportions."""
        cumulative = [0, 500, 1000]
        sampler = MixtureSampler(
            cumulative, weights=[0.5, 0.5], num_replicas=1, rank=0, shuffle=False
        )
        # Initially ~50/50
        indices = list(sampler)
        ds0_before = sum(1 for i in indices if i < 500)
        ratio_before = ds0_before / len(indices)
        assert 0.4 < ratio_before < 0.6

        # Update to 90/10
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
        # With high temperature, even 0.9/0.1 should be more uniform
        sampler.update_weights([0.9, 0.1], temperature=5.0)
        indices = list(sampler)
        ds0_count = sum(1 for i in indices if i < 500)
        ratio = ds0_count / len(indices)
        assert 0.4 < ratio < 0.75


# ---------------------------------------------------------------------------
# Phase transition logic (simulated)
# ---------------------------------------------------------------------------


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

        # After step 5, phase should be active
        assert phase_idx == 1
        # Verify new weights took effect
        indices = list(sampler)
        ds0_count = sum(1 for i in indices if i < 100)
        ratio = ds0_count / len(indices)
        assert ratio > 0.7  # ds0 should dominate with weight 0.9

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
        # After phase 2 (step 10), ds1 should dominate
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
        assert new_w == [0.9, 0.5]  # 'b' falls back to original

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

        assert lr_scale == 0.1  # Last active phase

    def test_resume_derives_phase_from_step(self):
        """On resume, correct phase is derived from current step."""
        phases = [
            TrainingPhase(start_step=10, dataset_weights={"a": 0.8, "b": 0.2}, lr_scale=0.5),
            TrainingPhase(start_step=50, dataset_weights={"a": 0.2, "b": 0.8}, lr_scale=0.1),
        ]

        # Simulate resume at step 30 (should be in phase 0, before phase 1)
        resume_step = 30
        phase_idx = 0
        lr_scale = 1.0
        for i, phase in enumerate(phases):
            if resume_step >= phase.start_step:
                lr_scale = phase.lr_scale
                phase_idx = i + 1

        assert phase_idx == 1  # Past first phase
        assert lr_scale == 0.5

        # Simulate resume at step 60 (both phases past)
        resume_step = 60
        phase_idx = 0
        lr_scale = 1.0
        for i, phase in enumerate(phases):
            if resume_step >= phase.start_step:
                lr_scale = phase.lr_scale
                phase_idx = i + 1

        assert phase_idx == 2
        assert lr_scale == 0.1


# ---------------------------------------------------------------------------
# Annealing shortcut equivalence
# ---------------------------------------------------------------------------


class TestAnnealingShortcut:
    def test_shortcut_produces_single_phase(self):
        """anneal_start_step + anneal_weights is equivalent to a single TrainingPhase."""
        config = DataConfig(
            anneal_start_step=500,
            anneal_weights={"code": 0.3, "text": 0.7},
        )
        # Simulate the resolution logic from train.py
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

        # Via shortcut
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

        # Via explicit phase
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
