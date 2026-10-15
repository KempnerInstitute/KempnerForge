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
import torch

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import FreezeSpec, FreezeStage
from kempnerforge.model.adapter import build_adapter

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXAMPLE_ROOT / "configs"
STAGE1_TOML = CONFIG_DIR / "image_understanding_stage1.toml"
STAGE2_TOML = CONFIG_DIR / "image_understanding_stage2.toml"
MANIFEST = CONFIG_DIR / "image_understanding_manifest.json"

# Content hashes of the two pinned source artifacts (read-only audit values).
STAGE1_YML_SHA256 = "d3e8e7546be8ae9342633aab2ab37892717c87704405a60484d84d75ca24dce1"
STAGE2_YML_SHA256 = "ad51bd30c335b3f559b9dc95fd56ab7c1fc3d38bfc40e2ca7f3f17d3eb58165c"

# Registered connectors by how they treat the patch grid.
POOLING_CONNECTORS = ("avgpool", "attentional_pool")
PROJECTION_CONNECTORS = ("linear", "mlp_2layer")

STAGE1_RESOLVED = {
    "stage1_shard_listing": "shard-listing-v1",
    "stage1_shard_hashes": "shard-hashes-v1",
    "stage1_loader_workers": "4",
    "source_dependency_versions": "lock-v1",
    "source_runtime_commit": "0123abc",
}

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


def _with_geometry(
    vlm: recipe.ImageUnderstandingConfig, *, connector_type: str | None = None, **changes: object
) -> recipe.ImageUnderstandingConfig:
    """``vlm`` with geometry fields (and optionally the connector) changed, revalidated."""
    assert vlm.stage is not None
    geometry = dataclasses.replace(vlm.stage.geometry, **changes)
    extra = {} if connector_type is None else {"connector_type": connector_type}
    return dataclasses.replace(
        vlm, stage=dataclasses.replace(vlm.stage, geometry=geometry), **extra
    )


def _raw_manifest() -> dict:
    return json.loads(MANIFEST.read_bytes())


def _config_with_manifest(
    vlm: recipe.ImageUnderstandingConfig, directory: Path, raw: dict
) -> recipe.ImageUnderstandingConfig:
    """``vlm`` pinned to a manifest written from ``raw``, so the pin matches its bytes."""
    path = directory / "manifest.json"
    path.write_text(json.dumps(raw, indent=2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dataclasses.replace(vlm, source_manifest=str(path), source_manifest_sha256=digest)


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
        with pytest.raises(ValueError, match="connector-derived count"):
            _with_geometry(stage1_vlm, tokens_per_view=300)

    def test_declared_budget_must_be_positive(self, stage1_vlm) -> None:
        assert stage1_vlm.stage is not None
        for tokens in (0, -256):
            with pytest.raises(ValueError, match="tokens_per_view must be positive"):
                dataclasses.replace(stage1_vlm.stage.geometry, tokens_per_view=tokens)

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


class TestConnectorDerivedBudget:
    """A 512 px view with 16 px patches is a 32 x 32 grid: 1024 patch tokens."""

    @pytest.mark.parametrize("connector", POOLING_CONNECTORS)
    @pytest.mark.parametrize(("window", "tokens"), [(2, 256), (4, 64)])
    def test_pooling_connector_reduces_by_its_window(
        self, stage1_vlm, connector: str, window: int, tokens: int
    ) -> None:
        vlm = _with_geometry(
            stage1_vlm, connector_type=connector, pool_window=window, tokens_per_view=tokens
        )
        assert vlm.stage is not None
        assert vlm.stage.geometry.tokens_per_image == tokens
        with pytest.raises(ValueError, match="connector-derived count"):
            _with_geometry(
                stage1_vlm,
                connector_type=connector,
                pool_window=window,
                tokens_per_view=tokens + 1,
            )

    @pytest.mark.parametrize("connector", PROJECTION_CONNECTORS)
    def test_projection_connector_keeps_every_patch(
        self, stage1_vlm, stage2_vlm, connector: str
    ) -> None:
        one_view = _with_geometry(
            stage1_vlm, connector_type=connector, pool_window=1, tokens_per_view=1024
        )
        assert one_view.stage is not None
        assert one_view.stage.geometry.tokens_per_image == 1024
        # Stage 2 is a 3 x 3 tiling plus a global view: ten views.
        ten_views = _with_geometry(
            stage2_vlm, connector_type=connector, pool_window=1, tokens_per_view=1024
        )
        assert ten_views.stage is not None
        assert ten_views.stage.geometry.tokens_per_image == 10240

    @pytest.mark.parametrize("connector", PROJECTION_CONNECTORS)
    def test_projection_connector_refuses_the_pooled_count(
        self, stage1_vlm, connector: str
    ) -> None:
        """Switching only the connector leaves 256 declared; the connector yields 1024."""
        with pytest.raises(ValueError, match=r"pool_window \(2\) must be 1"):
            dataclasses.replace(stage1_vlm, connector_type=connector)
        with pytest.raises(ValueError, match="connector-derived count"):
            _with_geometry(stage1_vlm, connector_type=connector, pool_window=1)

    @pytest.mark.parametrize("connector", PROJECTION_CONNECTORS)
    def test_projection_connector_carries_no_pool_window(self, stage1_vlm, connector: str) -> None:
        """The right count with a leftover window still contradicts the connector."""
        with pytest.raises(ValueError, match=r"pool_window \(2\) must be 1"):
            _with_geometry(stage1_vlm, connector_type=connector, tokens_per_view=1024)

    @pytest.mark.parametrize(
        ("connector", "window", "tokens"),
        [
            ("avgpool", 2, 256),
            ("avgpool", 4, 64),
            ("attentional_pool", 2, 256),
            ("linear", 1, 1024),
            ("mlp_2layer", 1, 1024),
        ],
    )
    def test_derived_count_is_what_the_built_adapter_emits(
        self, stage1_vlm, connector: str, window: int, tokens: int
    ) -> None:
        assert stage1_vlm.stage is not None
        geometry = dataclasses.replace(stage1_vlm.stage.geometry, pool_window=window)
        adapter = build_adapter(AdapterConfig(type=connector, pool_window=window), 32, 32)
        emitted = adapter(torch.zeros(1, 1024, 32)).shape[1]
        assert emitted == tokens
        assert geometry.connector_tokens_per_view(connector) == tokens

    def test_loader_refuses_a_connector_switch(self) -> None:
        with pytest.raises(ValueError, match=r"pool_window \(2\) must be 1"):
            load_config(str(STAGE1_TOML), cli_args=["--vlm.connector_type=linear"])


# The one directive per (stage, module) that breaks that stage's contract. Stage 1 freezes
# the LM and trains the tower and connector; stage 2 trains all three.
CONTRACT_BREAKS = [
    pytest.param(1, FreezeSpec("transformer", False), id="stage1-lm-unfrozen"),
    pytest.param(1, FreezeSpec("vision_encoder", True), id="stage1-tower-frozen"),
    pytest.param(1, FreezeSpec("adapter", True), id="stage1-connector-frozen"),
    pytest.param(2, FreezeSpec("transformer", True), id="stage2-lm-frozen"),
    pytest.param(2, FreezeSpec("vision_encoder", True), id="stage2-tower-frozen"),
    pytest.param(2, FreezeSpec("adapter", True), id="stage2-connector-frozen"),
]


class TestFreezePolicy:
    def test_stage1_requires_a_frozen_lm(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="requires freeze to include"):
            dataclasses.replace(stage1_vlm, freeze=[])

    def test_a_declared_lm_flag_must_match_the_stage(self, stage1_vlm, stage2_vlm) -> None:
        assert stage1_vlm.stage is not None and stage2_vlm.stage is not None
        with pytest.raises(ValueError, match="stage 1 freezes the LM; train_lm must be false"):
            dataclasses.replace(stage1_vlm.stage, train_lm=True)
        with pytest.raises(ValueError, match="stage 2 trains the LM; train_lm must be true"):
            dataclasses.replace(stage2_vlm.stage, train_lm=False)

    def test_stage2_with_a_consistent_frozen_lm_is_refused(self, tmp_path: Path) -> None:
        """train_lm=false and a frozen transformer agree with each other, not with stage 2."""
        text = STAGE2_TOML.read_text()
        for old, new in {
            "train_lm = true": "train_lm = false",
            "freeze = []": 'freeze = [{ module = "transformer", frozen = true }]',
        }.items():
            assert text.count(old) == 1
            text = text.replace(old, new)
        bad = tmp_path / "bad.toml"
        bad.write_text(text)
        with pytest.raises(ValueError, match="stage 2 trains the LM"):
            load_config(str(bad), cli_args=[])

    @pytest.mark.parametrize(
        ("stage", "flag", "freeze"),
        [(1, True, []), (2, False, [FreezeSpec("transformer", True)])],
        ids=["stage1-lm-trainable", "stage2-lm-frozen"],
    )
    def test_the_freeze_state_is_checked_whatever_the_declared_flag(
        self, stage1_vlm, stage2_vlm, stage: int, flag: bool, freeze: list[FreezeSpec]
    ) -> None:
        """The flag is forged to agree with the directives, so only the contract can refuse."""
        vlm = {1: stage1_vlm, 2: stage2_vlm}[stage]
        assert vlm.stage is not None
        forged = dataclasses.replace(vlm.stage)
        object.__setattr__(forged, "train_lm", flag)
        with pytest.raises(ValueError, match=rf"stage {stage}.*'transformer'.*build time"):
            dataclasses.replace(vlm, stage=forged, freeze=freeze)

    @pytest.mark.parametrize(("stage", "breaker"), CONTRACT_BREAKS)
    def test_build_time_freeze_must_match_the_stage_contract(
        self, stage1_vlm, stage2_vlm, stage: int, breaker: FreezeSpec
    ) -> None:
        vlm = {1: stage1_vlm, 2: stage2_vlm}[stage]
        kept = [spec for spec in vlm.freeze if spec.module != breaker.module]
        with pytest.raises(ValueError, match=rf"stage {stage}.*'{breaker.module}'.*build time"):
            dataclasses.replace(vlm, freeze=[*kept, breaker])

    @pytest.mark.parametrize(("stage", "breaker"), CONTRACT_BREAKS)
    def test_a_schedule_must_keep_the_stage_contract(
        self, stage1_vlm, stage2_vlm, stage: int, breaker: FreezeSpec
    ) -> None:
        vlm = {1: stage1_vlm, 2: stage2_vlm}[stage]
        schedule = [FreezeStage(1000, (breaker,))]
        with pytest.raises(ValueError, match=rf"stage {stage}.*'{breaker.module}'.*step 1000"):
            dataclasses.replace(vlm, freeze_schedule=schedule)

    def test_the_last_directive_per_module_decides(self, stage1_vlm, stage2_vlm) -> None:
        """The runtime keeps one directive per module and the last one wins."""
        unfrozen_last = [FreezeSpec("transformer", True), FreezeSpec("transformer", False)]
        with pytest.raises(ValueError, match="requires freeze to include"):
            dataclasses.replace(stage1_vlm, freeze=unfrozen_last)
        frozen_last = unfrozen_last[::-1]
        assert dataclasses.replace(stage1_vlm, freeze=frozen_last).freeze == frozen_last
        with pytest.raises(ValueError, match="must not include"):
            dataclasses.replace(stage2_vlm, freeze=frozen_last)

        tower_unfrozen_last = [
            FreezeSpec("vision_encoder", True),
            FreezeSpec("vision_encoder", False),
        ]
        assert (
            dataclasses.replace(stage2_vlm, freeze=tower_unfrozen_last).freeze
            == tower_unfrozen_last
        )
        with pytest.raises(ValueError, match="must not include"):
            dataclasses.replace(stage2_vlm, freeze=tower_unfrozen_last[::-1])

    def test_a_later_transition_is_checked_after_an_earlier_harmless_one(self, stage1_vlm) -> None:
        schedule = [
            FreezeStage(100, (FreezeSpec("adapter", False),)),
            FreezeStage(200, (FreezeSpec("transformer", False),)),
        ]
        with pytest.raises(ValueError, match="requires freeze to include") as err:
            dataclasses.replace(stage1_vlm, freeze_schedule=schedule)
        assert "step 200" in str(err.value)

    def test_every_state_the_run_passes_through_is_checked(self, stage2_vlm) -> None:
        """A transient violation counts even when a later transition repairs it."""
        transient = [
            FreezeStage(100, (FreezeSpec("vision_encoder", True),)),
            FreezeStage(200, (FreezeSpec("vision_encoder", False),)),
        ]
        with pytest.raises(ValueError, match="must not include") as err:
            dataclasses.replace(stage2_vlm, freeze_schedule=transient)
        assert "step 100" in str(err.value)

        with pytest.raises(ValueError, match="must not include") as err:
            dataclasses.replace(
                stage2_vlm,
                freeze=[FreezeSpec("transformer", True)],
                freeze_schedule=[FreezeStage(100, (FreezeSpec("transformer", False),))],
            )
        assert "build time" in str(err.value)

    def test_a_schedule_that_keeps_the_policy_is_accepted(self, stage1_vlm, stage2_vlm) -> None:
        stage1_schedule = [
            FreezeStage(100, (FreezeSpec("transformer", True),)),
            FreezeStage(200, (FreezeSpec("adapter", False),)),
        ]
        vlm1 = dataclasses.replace(stage1_vlm, freeze_schedule=stage1_schedule)
        assert vlm1.freeze_schedule == stage1_schedule
        stage2_schedule = [
            FreezeStage(
                100, (FreezeSpec("vision_encoder", False), FreezeSpec("transformer", False))
            )
        ]
        vlm2 = dataclasses.replace(stage2_vlm, freeze_schedule=stage2_schedule)
        assert vlm2.freeze_schedule == stage2_schedule

    def test_directives_must_name_a_recipe_module(self, stage2_vlm) -> None:
        """A raw pattern could freeze part of a module the policy treats as whole."""
        for module in ("transformer.layers.*", "vision_encoder.*", "mot"):
            with pytest.raises(ValueError, match="not in known module aliases"):
                dataclasses.replace(stage2_vlm, freeze=[FreezeSpec(module, True)])
            with pytest.raises(ValueError, match="not in known module aliases"):
                dataclasses.replace(
                    stage2_vlm, freeze_schedule=[FreezeStage(10, (FreezeSpec(module, True),))]
                )

    @pytest.mark.parametrize("alias", ["transformer", "vision_encoder", "adapter"])
    def test_recipe_aliases_must_cover_whole_modules(self, stage1_vlm, alias: str) -> None:
        """Redefining an alias narrows what a freeze directive on it touches."""
        narrowed = {**stage1_vlm.module_patterns, alias: [f"{alias}.layers.0.*"]}
        with pytest.raises(ValueError, match=rf"module_patterns\['{alias}'\]"):
            dataclasses.replace(stage1_vlm, module_patterns=narrowed)
        missing = {k: v for k, v in stage1_vlm.module_patterns.items() if k != alias}
        with pytest.raises(ValueError, match=rf"module_patterns\['{alias}'\]"):
            dataclasses.replace(stage1_vlm, module_patterns=missing)


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
        resolved = dataclasses.replace(stage1_vlm, production_launch=True, **STAGE1_RESOLVED)
        assert resolved.unresolved_launch_fields() == []

    def test_production_launch_with_a_wrong_manifest_pin_is_refused(self, stage1_vlm) -> None:
        """Every other launch requirement is met; only the pin is wrong."""
        with pytest.raises(ValueError, match="content hash mismatch"):
            dataclasses.replace(
                stage1_vlm,
                production_launch=True,
                source_manifest_sha256="0" * 64,
                **STAGE1_RESOLVED,
            )

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

    def test_wrong_pin_is_refused_when_the_config_is_built(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="content hash mismatch"):
            dataclasses.replace(stage1_vlm, source_manifest_sha256="0" * 64)

    @pytest.mark.parametrize("path", [STAGE1_TOML, STAGE2_TOML], ids=lambda p: p.name)
    def test_wrong_pin_is_refused_while_loading_a_config(self, path: Path) -> None:
        with pytest.raises(ValueError, match="content hash mismatch"):
            load_config(str(path), cli_args=["--vlm.source_manifest_sha256=" + "f" * 64])

    def test_missing_manifest_is_refused_when_the_config_is_built(
        self, tmp_path: Path, stage1_vlm
    ) -> None:
        with pytest.raises(ValueError, match="cannot read"):
            dataclasses.replace(stage1_vlm, source_manifest=str(tmp_path / "absent.json"))

    def test_edited_manifest_is_refused_by_the_loader(self, tmp_path: Path, stage1_vlm) -> None:
        """The pin is checked again at load, since the file can change after the
        config was built."""
        raw = _raw_manifest()
        raw["stage2_sources"] = raw["stage2_sources"][:-1]
        (tmp_path / stage1_vlm.source_manifest).write_text(json.dumps(raw, indent=2))
        with pytest.raises(ValueError, match="content hash mismatch"):
            recipe.load_source_manifest(stage1_vlm, base_dir=tmp_path)

    def test_truncated_manifest_is_refused_by_the_count_guard(
        self, tmp_path: Path, stage1_vlm
    ) -> None:
        """Even with a matching hash, a list that disagrees with its own
        recorded counts is refused rather than silently accepted."""
        raw = _raw_manifest()
        raw["stage2_sources"] = raw["stage2_sources"][:-1]
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match="refusing a truncated"):
            recipe.load_source_manifest(config)

    def test_malformed_provenance_hash_is_refused(self, tmp_path: Path, stage1_vlm) -> None:
        raw = _raw_manifest()
        raw["provenance"]["artifacts"][0]["sha256"] = "OPEN"
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match="malformed sha256"):
            recipe.load_source_manifest(config)

    def test_manifest_with_unknown_block_is_refused(self, tmp_path: Path, stage1_vlm) -> None:
        raw = _raw_manifest()
        raw["extra_block"] = {}
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match="unknown keys"):
            recipe.load_source_manifest(config)

    def test_bad_manifest_reference_hash_format_is_refused(self, stage1_vlm) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex"):
            dataclasses.replace(stage1_vlm, source_manifest_sha256="abc123")


class TestManifestContainers:
    """Iterating a JSON string or object also yields strings, so a source list
    given as either would pass any check that only looks at its items."""

    @pytest.mark.parametrize("bad", ["abc", {"a": 1, "b": 2}], ids=["string", "object"])
    def test_sources_must_be_a_list(self, tmp_path: Path, stage1_vlm, bad) -> None:
        raw = _raw_manifest()
        raw["stage2_sources"] = bad
        # Counts that agree with the iteration, so only the container check can refuse it.
        raw["source_counts"] = {"occurrences": len(bad), "unique": len(set(bad))}
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match="stage2_sources must be a list"):
            recipe.load_source_manifest(config)

    @pytest.mark.parametrize("field", ["open", "notes"])
    @pytest.mark.parametrize("bad", ["abc", {"a": 1}], ids=["string", "object"])
    def test_provenance_lists_must_be_lists(
        self, tmp_path: Path, stage1_vlm, field: str, bad
    ) -> None:
        raw = _raw_manifest()
        raw["provenance"][field] = bad
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match=rf"provenance\.{field} must be a list"):
            recipe.load_source_manifest(config)

    @pytest.mark.parametrize("bad", [[], [1, 2], ["ok", ""], ["ok", "  "], ["ok", None]])
    def test_source_entries_must_be_non_empty_strings(
        self, tmp_path: Path, stage1_vlm, bad: list
    ) -> None:
        raw = _raw_manifest()
        raw["stage2_sources"] = bad
        raw["source_counts"] = {"occurrences": len(bad), "unique": len(set(bad))}
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match="non-empty"):
            recipe.load_source_manifest(config)

    @pytest.mark.parametrize("field", ["open", "notes"])
    @pytest.mark.parametrize("bad", [[1], [""], ["ok", None]])
    def test_provenance_entries_must_be_non_empty_strings(
        self, tmp_path: Path, stage1_vlm, field: str, bad: list
    ) -> None:
        raw = _raw_manifest()
        raw["provenance"][field] = bad
        config = _config_with_manifest(stage1_vlm, tmp_path, raw)
        with pytest.raises(ValueError, match="non-empty"):
            recipe.load_source_manifest(config)
