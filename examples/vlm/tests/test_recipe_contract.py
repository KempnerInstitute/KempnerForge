"""Exact-behavior tests for the image-understanding recipe contract.

The expected stage-2 source list is enumerated here independently of the
shipped manifest, transcribed from the pinned stage-2 resolved config:
ten cycles of the six base sources, 167 visual FineVision subsets once,
the PixMo block with its multiplicities, then 13 text-only FineVision
subsets once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import recipe

from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import FreezeSpec

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXAMPLE_ROOT / "configs"
STAGE1_TOML = CONFIG_DIR / "image_understanding_stage1.toml"
STAGE2_TOML = CONFIG_DIR / "image_understanding_stage2.toml"
MANIFEST = CONFIG_DIR / "image_understanding_manifest.json"

# Content hashes of the two pinned source artifacts (read-only audit values).
STAGE1_YML_SHA256 = "d3e8e7546be8ae9342633aab2ab37892717c87704405a60484d84d75ca24dce1"
STAGE2_YML_SHA256 = "ad51bd30c335b3f559b9dc95fd56ab7c1fc3d38bfc40e2ca7f3f17d3eb58165c"

BASE_CYCLE = (
    "coco-captions",
    "coco-vqa",
    "chartqa-human",
    "chartqa-machine",
    "docvqa",
    "textvqa",
)

FINEVISION_VISUAL = (
    "finevision-CoSyn_400k_chart",
    "finevision-CoSyn_400k_chemical",
    "finevision-CoSyn_400k_circuit",
    "finevision-CoSyn_400k_diagram",
    "finevision-CoSyn_400k_document",
    "finevision-CoSyn_400k_graphic",
    "finevision-CoSyn_400k_math",
    "finevision-CoSyn_400k_music",
    "finevision-CoSyn_400k_nutrition",
    "finevision-CoSyn_400k_table",
    "finevision-DoclingMatix",
    "finevision-LLaVA_Instruct_150K",
    "finevision-SynthChartNet",
    "finevision-SynthCodeNet",
    "finevision-SynthFormulaNet",
    "finevision-Unichart",
    "finevision-aguvis-stage-1",
    "finevision-ai2d_merged",
    "finevision-alfworldgpt",
    "finevision-allava_laion",
    "finevision-allava_vflan",
    "finevision-art",
    "finevision-arxivqa",
    "finevision-bentham",
    "finevision-blockdiagramcomputerized",
    "finevision-blockdiagramhandwritten",
    "finevision-cambrian(filtered)_processed",
    "finevision-captcha",
    "finevision-chart2text",
    "finevision-chartqa",
    "finevision-chinesememe",
    "finevision-chrome_writting",
    "finevision-clevr",
    "finevision-clevr_math",
    "finevision-clevr_math(mathv360k)",
    "finevision-coco_colors",
    "finevision-cocoqa",
    "finevision-cocotext",
    "finevision-ctw",
    "finevision-datik",
    "finevision-datikz",
    "finevision-densefusion_1m",
    "finevision-diagram_image_to_text",
    "finevision-docvqa",
    "finevision-drivelm",
    "finevision-dvqa",
    "finevision-est_vqa",
    "finevision-face_emotion",
    "finevision-figureqa",
    "finevision-figureqa(mathv360k)",
    "finevision-finqa",
    "finevision-funsd",
    "finevision-geo170k(align)",
    "finevision-geo170k(qa)",
    "finevision-geo3k",
    "finevision-geometry3k(mathv360k)",
    "finevision-geomverse",
    "finevision-geoqa+(mathv360k)",
    "finevision-geos(mathv360k)",
    "finevision-google_landmarks",
    "finevision-groundui",
    "finevision-handwriting_forms",
    "finevision-hateful_memes",
    "finevision-hitab",
    "finevision-hme100k",
    "finevision-hw_squad",
    "finevision-iam",
    "finevision-iconqa",
    "finevision-iconqa(mathv360k)",
    "finevision-idk",
    "finevision-iiit5k",
    "finevision-image_textualization(filtered)",
    "finevision-imgur5k",
    "finevision-indoor_qa",
    "finevision-infographic(gpt4v)",
    "finevision-infographic_vqa",
    "finevision-infographic_vqa_llava_format",
    "finevision-intergps",
    "finevision-invoices_receipts",
    "finevision-k12_printing",
    "finevision-laion_gpt4v",
    "finevision-latex_handwritten",
    "finevision-latexformulas",
    "finevision-llavar_gpt4_20k",
    "finevision-lnqa",
    "finevision-localized_narratives",
    "finevision-lrv_chart",
    "finevision-lrv_normal(filtered)",
    "finevision-lvis_instruct4v",
    "finevision-mapqa",
    "finevision-mapqa(mathv360k)",
    "finevision-maptext",
    "finevision-mathwriting-google",
    "finevision-mavis_math_metagen",
    "finevision-mavis_math_rule_geo",
    "finevision-memotion",
    "finevision-mimic_cgd",
    "finevision-mmc_instruct",
    "finevision-mmevol",
    "finevision-mmra",
    "finevision-mmsoc_memotion",
    "finevision-multihiertt",
    "finevision-nlvr2",
    "finevision-objects365_qa",
    "finevision-ocrvqa",
    "finevision-olmOCR-mix-0225-books",
    "finevision-olmOCR-mix-0225-documents",
    "finevision-oodvqa",
    "finevision-orand_car_a",
    "finevision-pathvqa",
    "finevision-pdfvqa",
    "finevision-plotqa",
    "finevision-pmc_vqa(mathv360k)",
    "finevision-raven",
    "finevision-rendered_text",
    "finevision-robut_sqa",
    "finevision-robut_wikisql",
    "finevision-robut_wtq",
    "finevision-scienceqa",
    "finevision-scienceqa(nona_context)",
    "finevision-screen2words",
    "finevision-screenqa",
    "finevision-sharegpt4o",
    "finevision-sharegpt4v(coco)",
    "finevision-sharegpt4v(knowledge)",
    "finevision-sharegpt4v(llava)",
    "finevision-sharegpt4v(sam)",
    "finevision-sketchyvqa",
    "finevision-slidevqa",
    "finevision-spark",
    "finevision-spatialsense",
    "finevision-spot_the_diff",
    "finevision-sroie",
    "finevision-st_vqa",
    "finevision-sujet_finance",
    "finevision-super_clevr(mathv360k)",
    "finevision-svrd",
    "finevision-synthdog",
    "finevision-tabmwp",
    "finevision-tabmwp(mathv360k)",
    "finevision-tal_ocr_eng",
    "finevision-tallyqa",
    "finevision-tat_dqa",
    "finevision-tat_qa",
    "finevision-textcaps",
    "finevision-textocr(gpt4v)",
    "finevision-textvqa",
    "finevision-tqa",
    "finevision-unigeo(mathv360k)",
    "finevision-ureader_cap",
    "finevision-ureader_ie",
    "finevision-ureader_kg_processed",
    "finevision-ureader_qa_processed",
    "finevision-vision_flan(filtered)",
    "finevision-vistext",
    "finevision-visual7w",
    "finevision-visualmrc",
    "finevision-visualwebinstruct(filtered)",
    "finevision-vizwiz(mathv360k)",
    "finevision-vqaonbd",
    "finevision-vqarad",
    "finevision-vqav2",
    "finevision-vsr",
    "finevision-websight",
    "finevision-wildvision",
    "finevision-wordart",
    "finevision-yesbut",
)

PIXMO_BLOCK = (
    ("pixmo-captions",)
    + ("pixmo-cap-qa",) * 2
    + ("pixmo-pointing",) * 2
    + ("pixmo-points-counting",) * 3
    + ("pixmo-point-explanations",) * 5
    + ("pixmo-count",) * 20
)

FINEVISION_TEXT = (
    "finevision-text_openhermes_2_5",
    "finevision-text_OpenMathInstruct-2",
    "finevision-text_mathinstruct",
    "finevision-text_orcamath",
    "finevision-text_codefeedback_filtered_instruction",
    "finevision-text_infinitymath",
    "finevision-text_wizardlm_evol",
    "finevision-text_code_feedback",
    "finevision-text_pythoncode25k",
    "finevision-text_pythoncodealpaca",
    "finevision-text_mathstepdpo10k",
    "finevision-text_ruozhiba",
    "finevision-text_theoremqa",
)

EXPECTED_SOURCES = BASE_CYCLE * 10 + FINEVISION_VISUAL + PIXMO_BLOCK + FINEVISION_TEXT

EXPECTED_MULTIPLICITY = (
    {name: 10 for name in BASE_CYCLE}
    | dict.fromkeys(FINEVISION_VISUAL, 1)
    | {
        "pixmo-captions": 1,
        "pixmo-cap-qa": 2,
        "pixmo-pointing": 2,
        "pixmo-points-counting": 3,
        "pixmo-point-explanations": 5,
        "pixmo-count": 20,
    }
    | dict.fromkeys(FINEVISION_TEXT, 1)
)


def test_expected_enumeration_is_consistent() -> None:
    """Internal consistency of the independently enumerated expectation."""
    assert len(FINEVISION_VISUAL) == 167
    assert len(set(FINEVISION_VISUAL)) == 167
    assert len(FINEVISION_TEXT) == 13
    assert len(set(FINEVISION_TEXT)) == 13
    assert len(PIXMO_BLOCK) == 33
    assert len(EXPECTED_SOURCES) == 273
    assert len(set(EXPECTED_SOURCES)) == 192
    assert sum(EXPECTED_MULTIPLICITY.values()) == 273


@pytest.fixture(scope="module")
def stage1_vlm() -> recipe.ImageUnderstandingConfig:
    config = load_config(str(STAGE1_TOML), cli_args=[])
    config.validate(world_size=4)
    assert isinstance(config.vlm, recipe.ImageUnderstandingConfig)
    return config.vlm


@pytest.fixture(scope="module")
def stage2_vlm() -> recipe.ImageUnderstandingConfig:
    config = load_config(str(STAGE2_TOML), cli_args=[])
    config.validate(world_size=4)
    assert isinstance(config.vlm, recipe.ImageUnderstandingConfig)
    return config.vlm


class TestStageContracts:
    def test_arch_dispatches_to_the_registered_subclass(self) -> None:
        assert registry.get_vlm_config(recipe.ARCH_NAME) is recipe.ImageUnderstandingConfig
        assert recipe.ARCH_NAME in registry.list_vlm_configs()

    def test_stage1_pins_the_recipe(self, stage1_vlm: recipe.ImageUnderstandingConfig) -> None:
        vlm = stage1_vlm
        assert vlm.lm_backbone_id == "Qwen/Qwen3-0.6B"
        assert vlm.tokenizer_id == "Qwen/Qwen3-0.6B"
        assert vlm.vision_tower_id == "google/siglip2-so400m-patch16-512"
        assert vlm.connector_type == "avgpool"
        assert vlm.connector_hidden_dim == 1152
        assert vlm.use_chat_template is True
        assert vlm.caption_prompt == "Provide a short caption for the image."
        assert vlm.vision_border is False
        assert vlm.loss_reduction == "sample_mean"
        assert vlm.freeze == [FreezeSpec("transformer", True)]
        stage = vlm.stage
        assert stage is not None
        assert (stage.stage, stage.max_steps, stage.epochs, stage.seed) == (1, 50000, 1, 1)
        assert (stage.accumulation_steps, stage.data_parallel_ranks) == (1, 16)
        assert stage.train_lm is False
        assert stage.answer_only_loss is False
        assert (stage.image_attention, stage.bidirectional_scope) == ("causal", "")
        assert stage.vision_tower_precision == "float32"
        assert stage.vision_attn_implementation == ""
        assert stage.resume_reset_data_order is False
        assert stage.grad_clip_norm == 0.0
        geo = stage.geometry
        assert (geo.image_size, geo.patch_size, geo.pool_window) == (512, 16, 2)
        assert (geo.tile_grid, geo.include_global_view) == (1, False)
        assert geo.views_per_image == 1
        assert geo.tokens_per_view == 256
        assert geo.tokens_per_image == 256
        assert geo.pack_budget == 12288
        assert stage.optim_vision == recipe.StageOptimGroup(6e-6, 5000, 45000, 6e-7, 5e-4)
        assert stage.optim_language == recipe.StageOptimGroup(2e-5, 5000, 45000, 2e-6, 5e-4)
        assert stage.optim_mixer == recipe.StageOptimGroup(2e-4, 5000, 45000, 2e-5, 5e-4)

    def test_stage2_pins_the_recipe(self, stage2_vlm: recipe.ImageUnderstandingConfig) -> None:
        vlm = stage2_vlm
        assert vlm.freeze == []
        stage = vlm.stage
        assert stage is not None
        assert (stage.stage, stage.max_steps, stage.epochs, stage.seed) == (2, 90000, 40, 1)
        assert (stage.accumulation_steps, stage.data_parallel_ranks) == (2, 16)
        assert stage.train_lm is True
        assert stage.answer_only_loss is True
        assert (stage.image_attention, stage.bidirectional_scope) == ("bidirectional", "view")
        assert stage.vision_tower_precision == "mixed_bfloat16"
        assert stage.vision_attn_implementation == "sdpa"
        assert stage.resume_reset_data_order is True
        assert stage.grad_clip_norm == 0.0
        geo = stage.geometry
        assert (geo.tile_grid, geo.include_global_view) == (3, True)
        assert geo.views_per_image == 10
        assert geo.tokens_per_view == 256
        assert geo.tokens_per_image == 2560
        assert geo.pack_budget == 16384
        assert stage.optim_vision == recipe.StageOptimGroup(1e-5, 6000, 84000, 1e-6, 5e-4)
        assert stage.optim_language == recipe.StageOptimGroup(5e-5, 6000, 84000, 5e-6, 5e-4)
        assert stage.optim_mixer == recipe.StageOptimGroup(1e-5, 6000, 84000, 1e-6, 5e-4)

    def test_tie_metadata_is_distinct_from_runtime_aliasing(self) -> None:
        """The published LM ties embeddings (metadata); the runtime unties.

        Both facts are carried, on different fields, and flipping the
        metadata bit does not change runtime validation.
        """
        for path in (STAGE1_TOML, STAGE2_TOML):
            config = load_config(str(path), cli_args=[])
            assert isinstance(config.vlm, recipe.ImageUnderstandingConfig)
            assert config.vlm.lm_tie_embeddings_metadata is True
            assert config.vlm.lm_untie_at_runtime is True
            assert config.model.tie_embeddings is False
            flipped = dataclasses.replace(config.vlm, lm_tie_embeddings_metadata=False)
            assert flipped.lm_untie_at_runtime is True

    def test_nominal_text_context_is_not_a_pack_budget(
        self, stage1_vlm: recipe.ImageUnderstandingConfig
    ) -> None:
        assert stage1_vlm.nominal_text_context == 256
        assert stage1_vlm.stage is not None
        assert stage1_vlm.stage.geometry.pack_budget == 12288
        changed = dataclasses.replace(stage1_vlm, nominal_text_context=4096)
        assert changed.stage is not None
        assert changed.stage.geometry.pack_budget == 12288


class TestStrictParsing:
    @pytest.mark.parametrize(
        ("anchor", "injected"),
        [
            ('arch = "image_understanding"', "bogus_knob = 1"),
            ("stage = 1", "bogus_stage_knob = 1"),
            ("image_size = 512", "bogus_geometry_knob = 1"),
        ],
    )
    def test_unknown_fields_are_rejected(self, tmp_path: Path, anchor: str, injected: str) -> None:
        text = STAGE1_TOML.read_text()
        assert text.count(anchor) == 1
        bad = tmp_path / "bad.toml"
        bad.write_text(text.replace(anchor, f"{anchor}\n{injected}", 1))
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(str(bad), cli_args=[])

    def test_missing_stage_table_is_rejected(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match=r"\[vlm.stage\]"):
            dataclasses.replace(stage1_vlm, stage=None)

    def test_schedule_must_sum_to_max_steps(self, stage1_vlm) -> None:
        assert stage1_vlm.stage is not None
        short = dataclasses.replace(stage1_vlm.stage.optim_language, warmup_steps=4999)
        with pytest.raises(ValueError, match="must equal stage.max_steps"):
            dataclasses.replace(stage1_vlm.stage, optim_language=short)

    def test_bidirectional_scope_coupling(self, stage2_vlm) -> None:
        assert stage2_vlm.stage is not None
        with pytest.raises(ValueError, match="requires a scope"):
            dataclasses.replace(stage2_vlm.stage, bidirectional_scope="")
        with pytest.raises(ValueError, match="requires image_attention"):
            dataclasses.replace(
                stage2_vlm.stage, image_attention="causal", bidirectional_scope="view"
            )

    def test_unknown_connector_is_rejected(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="Unknown vlm.connector_type"):
            dataclasses.replace(stage1_vlm, connector_type="not_a_registered_adapter")


class TestTokenBudgets:
    def test_declared_budget_must_match_the_connector_derivation(self, stage1_vlm) -> None:
        assert stage1_vlm.stage is not None
        geo = stage1_vlm.stage.geometry
        with pytest.raises(ValueError, match="connector-derived count"):
            dataclasses.replace(geo, tokens_per_view=300)

    def test_budget_follows_the_geometry_not_a_constant(self) -> None:
        """A different connector geometry yields a different validated budget."""
        geo = recipe.PackGeometry(
            image_size=512,
            patch_size=16,
            pool_window=4,
            tile_grid=1,
            include_global_view=False,
            tokens_per_view=64,
            pack_budget=12288,
        )
        assert geo.tokens_per_image == 64

    def test_indivisible_geometry_is_rejected(self, stage1_vlm) -> None:
        assert stage1_vlm.stage is not None
        geo = stage1_vlm.stage.geometry
        with pytest.raises(ValueError, match="divisible by"):
            dataclasses.replace(geo, image_size=500)
        with pytest.raises(ValueError, match="divisible by"):
            dataclasses.replace(geo, pool_window=3)

    def test_pack_budget_must_fit_an_image_and_text(self, stage1_vlm) -> None:
        assert stage1_vlm.stage is not None
        with pytest.raises(ValueError, match="must exceed tokens_per_image"):
            dataclasses.replace(stage1_vlm.stage.geometry, pack_budget=256)

    def test_stage1_is_single_view(self, stage1_vlm, stage2_vlm) -> None:
        assert stage1_vlm.stage is not None and stage2_vlm.stage is not None
        with pytest.raises(ValueError, match="single-view"):
            dataclasses.replace(stage1_vlm.stage, geometry=stage2_vlm.stage.geometry)
        with pytest.raises(ValueError, match="requires tiling"):
            dataclasses.replace(stage1_vlm.stage.geometry, include_global_view=True)


class TestFreezePolicy:
    def test_stage1_requires_a_frozen_lm(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="requires freeze to include"):
            dataclasses.replace(stage1_vlm, freeze=[])

    def test_stage2_requires_a_trainable_lm(self, stage2_vlm) -> None:
        with pytest.raises(ValueError, match="conflicts with a frozen"):
            dataclasses.replace(stage2_vlm, freeze=[FreezeSpec("transformer", True)])

    def test_tower_and_connector_always_train(self, stage1_vlm) -> None:
        base = list(stage1_vlm.freeze)
        for module in ("vision_encoder", "adapter"):
            with pytest.raises(ValueError, match="must not include"):
                dataclasses.replace(stage1_vlm, freeze=[*base, FreezeSpec(module, True)])


class TestLaunchGating:
    def test_stage1_launch_refused_while_provenance_is_open(self) -> None:
        with pytest.raises(ValueError, match="production_launch=true refused") as err:
            load_config(str(STAGE1_TOML), cli_args=["--vlm.production_launch=true"])
        for name in (
            "stage1_shard_listing",
            "stage1_shard_hashes",
            "stage1_loader_workers",
            "source_dependency_versions",
            "source_runtime_commit",
        ):
            assert name in str(err.value)

    def test_stage2_launch_refused_while_provenance_is_open(self) -> None:
        with pytest.raises(ValueError, match="production_launch=true refused") as err:
            load_config(str(STAGE2_TOML), cli_args=["--vlm.production_launch=true"])
        message = str(err.value)
        for name in (
            "source_dependency_versions",
            "source_runtime_commit",
            "endpoint_config_hash_resolution",
            "vision_mixed_precision_impl",
            "resume_reset_impl",
            "init_checkpoint",
        ):
            assert name in message
        assert "stage1_shard_listing" not in message

    def test_stage1_launch_allowed_once_resolved(self, stage1_vlm) -> None:
        resolved = dataclasses.replace(
            stage1_vlm,
            production_launch=True,
            stage1_shard_listing="shard-listing-v1",
            stage1_shard_hashes="shard-hashes-v1",
            stage1_loader_workers="4",
            source_dependency_versions="lock-v1",
            source_runtime_commit="0123abc",
        )
        assert resolved.unresolved_launch_fields() == []

    def test_stage2_launch_allowed_once_resolved(self, stage2_vlm) -> None:
        resolved = dataclasses.replace(
            stage2_vlm,
            production_launch=True,
            source_dependency_versions="lock-v1",
            source_runtime_commit="0123abc",
            endpoint_config_hash_resolution="recorded-hash-explained-v1",
            vision_mixed_precision_impl="autocast-policy-v1",
            resume_reset_impl="reset-policy-v1",
            init_checkpoint="checkpoints/image_understanding_stage1/step_50000",
        )
        assert resolved.unresolved_launch_fields() == []

    def test_stage1_rejects_an_init_checkpoint(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="init_checkpoint must be empty"):
            dataclasses.replace(stage1_vlm, init_checkpoint="somewhere")

    def test_open_fields_may_not_be_blanked(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="not empty"):
            dataclasses.replace(stage1_vlm, source_runtime_commit="")


class TestManifest:
    def test_order_and_multiplicity_are_preserved_exactly(self, stage1_vlm) -> None:
        manifest = recipe.load_source_manifest(stage1_vlm)
        assert manifest.occurrence_count() == 273
        assert manifest.unique_count() == 192
        assert manifest.stage2_sources == EXPECTED_SOURCES
        assert manifest.multiplicity() == EXPECTED_MULTIPLICITY

    def test_both_configs_reference_the_same_pinned_manifest(self, stage1_vlm, stage2_vlm) -> None:
        assert stage1_vlm.source_manifest_sha256 == stage2_vlm.source_manifest_sha256
        assert recipe.manifest_path(stage1_vlm) == MANIFEST
        digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        assert digest == stage1_vlm.source_manifest_sha256

    def test_provenance_records_the_source_artifact_hashes(self, stage1_vlm) -> None:
        manifest = recipe.load_source_manifest(stage1_vlm)
        by_label = {a.label: a for a in manifest.artifacts}
        assert set(by_label) == {"stage1_resolved_config", "stage2_resolved_config"}
        assert by_label["stage1_resolved_config"].sha256 == STAGE1_YML_SHA256
        assert by_label["stage1_resolved_config"].byte_size == 2577
        assert by_label["stage2_resolved_config"].sha256 == STAGE2_YML_SHA256
        assert by_label["stage2_resolved_config"].byte_size == 13700

    def test_open_checklist_mirrors_the_config_fields(self, stage1_vlm) -> None:
        manifest = recipe.load_source_manifest(stage1_vlm)
        prefixes = {item.split(":", 1)[0] for item in manifest.open_provenance}
        assert prefixes == {
            "stage1_shard_listing",
            "stage1_shard_hashes",
            "stage1_loader_workers",
            "source_dependency_versions",
            "source_runtime_commit",
            "endpoint_config_hash_resolution",
            "vision_mixed_precision_impl",
            "resume_reset_impl",
        }

    def test_stage1_source_identity(self, stage1_vlm) -> None:
        manifest = recipe.load_source_manifest(stage1_vlm)
        assert manifest.stage1_source == recipe.Stage1Source(
            dataset_class="PackedWebdataset", name="datacomp-1b", epoch_size=700000000
        )

    def test_edited_manifest_is_refused_by_the_hash_pin(self, tmp_path: Path, stage1_vlm) -> None:
        raw = json.loads(MANIFEST.read_bytes())
        raw["stage2_sources"] = raw["stage2_sources"][:-1]
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(raw, indent=2))
        config = dataclasses.replace(stage1_vlm, source_manifest=str(tampered))
        with pytest.raises(ValueError, match="content hash mismatch"):
            recipe.load_source_manifest(config)

    def test_truncated_manifest_is_refused_by_the_count_guard(
        self, tmp_path: Path, stage1_vlm
    ) -> None:
        """Even with a matching hash, a list that disagrees with its own
        recorded counts is refused rather than silently accepted."""
        raw = json.loads(MANIFEST.read_bytes())
        raw["stage2_sources"] = raw["stage2_sources"][:-1]
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(raw, indent=2))
        digest = hashlib.sha256(tampered.read_bytes()).hexdigest()
        config = dataclasses.replace(
            stage1_vlm, source_manifest=str(tampered), source_manifest_sha256=digest
        )
        with pytest.raises(ValueError, match="refusing a truncated"):
            recipe.load_source_manifest(config)

    def test_malformed_provenance_hash_is_refused(self, tmp_path: Path, stage1_vlm) -> None:
        raw = json.loads(MANIFEST.read_bytes())
        raw["provenance"]["artifacts"][0]["sha256"] = "OPEN"
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(raw, indent=2))
        digest = hashlib.sha256(tampered.read_bytes()).hexdigest()
        config = dataclasses.replace(
            stage1_vlm, source_manifest=str(tampered), source_manifest_sha256=digest
        )
        with pytest.raises(ValueError, match="malformed sha256"):
            recipe.load_source_manifest(config)

    def test_manifest_with_unknown_block_is_refused(self, tmp_path: Path, stage1_vlm) -> None:
        raw = json.loads(MANIFEST.read_bytes())
        raw["extra_block"] = {}
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(raw, indent=2))
        digest = hashlib.sha256(tampered.read_bytes()).hexdigest()
        config = dataclasses.replace(
            stage1_vlm, source_manifest=str(tampered), source_manifest_sha256=digest
        )
        with pytest.raises(ValueError, match="unknown keys"):
            recipe.load_source_manifest(config)

    def test_bad_manifest_reference_hash_format_is_refused(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex"):
            dataclasses.replace(stage1_vlm, source_manifest_sha256="abc123")
