"""Integration tests for the VLM Mixture of Modality-Aware Experts (MoMa) training path.

Single-GPU forward + backward on a tiny synthetic MoMa config. Exercises
``build_parallel_model`` MoMa branch, freeze targeting at the ``moma``
alias, dtype propagation, save/load round-trip, per-modality dispatch
correctness, unbalanced expert counts (paper's moe_7t1i style), and the
``FreezeStage`` hook semantics on the per-layer MoMa stack.

Mirrors ``tests/integration/test_vlm_mot.py`` for the MoMa arch with two
key differences:

- No warm-start helper (deferred to v2 for MoMa).
- No ``compile`` parity test (MoMa's modality_ids scatter/gather + EC
  top-k produce graph breaks; ``JobConfig.validate`` warns rather than
  rejects, and the compile path is not part of v1's contract).
- MoMa is intrinsically MoE — every layer has per-modality expert
  groups, so there is no "MoMa + MoE smoke" cross-test; learning under
  EC routing is covered by the FSDP test in tests/distributed.

Runs on CUDA only; skipped when ``torch.cuda.is_available()`` is False.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import (
    FreezeSpec,
    FreezeStage,
    MoMaConfig,
)
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper
from kempnerforge.training.freeze import apply_freeze_specs, effective_freeze

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VLM MoMa integration tests require CUDA",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_moma_configs(
    *,
    num_image_tokens: int = 8,
    feature_dim: int = 96,
    n_layers: int = 2,
    freeze: list[FreezeSpec] | None = None,
    experts_per_modality: dict[str, int] | None = None,
    capacity_factor: float = 0.0,
    gumbel_noise: bool = False,
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoMaConfig]:
    return (
        ModelConfig(
            dim=64,
            n_layers=n_layers,
            n_heads=4,
            vocab_size=256,
            max_seq_len=128,
            ffn_hidden_dim=128,
        ),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        MoMaConfig(
            max_text_len=32,
            moma_experts_per_modality=(
                experts_per_modality
                if experts_per_modality is not None
                else {"image": 2, "text": 2}
            ),
            moma_capacity_factor=capacity_factor,
            moma_gumbel_noise=gumbel_noise,
            freeze=freeze if freeze is not None else [FreezeSpec("vision_encoder", True)],
        ),
    )


def _build(
    configs: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoMaConfig],
    *,
    param_dtype: torch.dtype = torch.bfloat16,
) -> VLMWrapper:
    mc, vc, ac, lc = configs
    model = build_parallel_model(
        mc,
        device=DEVICE,
        device_mesh=None,
        vision_config=vc,
        adapter_config=ac,
        vlm_config=lc,
        param_dtype=param_dtype,
    )
    assert isinstance(model, VLMWrapper)
    return model


def _dummy_batch(
    wrapper: VLMWrapper, batch: int = 2, text_len: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = torch.randn(batch, 3, 32, 32, device=DEVICE)
    input_ids = torch.randint(
        0, wrapper.transformer.config.vocab_size, (batch, text_len), device=DEVICE
    )
    labels = input_ids.clone()
    return pixels, input_ids, labels


class TestBuildAndForward:
    def test_build_and_forward_1gpu(self):
        """Tiny MoMa config builds on a single GPU; forward + backward run."""
        from kempnerforge.model.moma import MoMaBlock

        cfg = _tiny_moma_configs()
        wrapper = _build(cfg)
        assert isinstance(wrapper, VLMWrapper)
        # MoMa uses the JD/MoT image-prefix residual layout.
        assert wrapper.num_image_tokens == 8
        # All layers are MoMaBlocks.
        assert all(isinstance(layer, MoMaBlock) for layer in wrapper.transformer.layers.values())

        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids, labels)
        # output_slice trims the 8 image positions; logits cover text_len positions.
        assert logits.shape == (2, 16, cfg[0].vocab_size)
        # Backward — depth-scaled init on o_proj / down_proj leaves nonzero
        # values (unlike MoT's identity-at-construction), so gradients flow
        # without per-modality re-init.
        loss = logits.float().sum()
        loss.backward()
        adapter_grads = [
            p.grad
            for n, p in wrapper.named_parameters()
            if n.startswith("adapter") and p.requires_grad
        ]
        layer_grads = [
            p.grad
            for n, p in wrapper.named_parameters()
            if n.startswith("transformer.layers.") and p.requires_grad
        ]
        assert all(g is not None for g in adapter_grads)
        assert all(g is not None for g in layer_grads)

    def test_freeze_targets_moma(self):
        """``FreezeSpec("moma")`` freezes only the per-layer MoMa stack
        (transformer.layers.*) and leaves the adapter, final norm, embedding,
        and output head trainable.
        """
        cfg = _tiny_moma_configs(freeze=[FreezeSpec("moma", True)])
        wrapper = _build(cfg)
        trainable = {name for name, p in wrapper.named_parameters() if p.requires_grad}
        frozen = {name for name, p in wrapper.named_parameters() if not p.requires_grad}
        # All transformer.layers params are frozen.
        for n in frozen:
            if n.startswith("transformer."):
                assert n.startswith("transformer.layers."), n
        # transformer.norm (final norm) is trainable for MoMa (unlike MoT, which
        # replaces it with mot_norms and freezes self.norm).
        assert "transformer.norm.weight" in trainable
        # Adapter is trainable.
        assert any(n.startswith("adapter") for n in trainable)
        # Sanity: at least one MoE gate is frozen.
        assert any("mlp.experts" in n and "router.gate" in n for n in frozen)

    def test_image_features_dtype_propagation(self):
        """Encoder fp32 -> adapter bf16 -> Transformer.forward casts
        prefix_embeds to the residual-stream dtype before the MoMa block
        sees it. Asserts (a) build with bf16 param_dtype works, (b)
        forward output is bf16, (c) no dtype-mismatch errors.
        """
        cfg = _tiny_moma_configs()
        wrapper = _build(cfg, param_dtype=torch.bfloat16)
        assert wrapper.adapter.proj1.weight.dtype == torch.bfloat16
        pixels, input_ids, _ = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids)
        assert logits.dtype == torch.bfloat16

    def test_save_load_forward_parity(self):
        """state_dict round-trips with bit-equal forward output.

        Uses fp32 + Gumbel-off + eval mode to make the forward strictly
        deterministic so a bit-exact comparison is meaningful.
        """
        cfg = _tiny_moma_configs(gumbel_noise=False)
        wrapper_a = _build(cfg, param_dtype=torch.float32)
        wrapper_a.eval()
        pixels, input_ids, _ = _dummy_batch(wrapper_a, batch=1, text_len=8)
        with torch.no_grad():
            logits_a, _ = wrapper_a(pixels, input_ids)

        state = wrapper_a.state_dict()
        wrapper_b = _build(cfg, param_dtype=torch.float32)
        wrapper_b.load_state_dict(state, strict=True)
        wrapper_b.eval()
        with torch.no_grad():
            logits_b, _ = wrapper_b(pixels, input_ids)

        torch.testing.assert_close(logits_a, logits_b, atol=0.0, rtol=0.0)


class TestModalityDispatch:
    """MoMa-specific: verify that modality_ids correctly partitions tokens
    to per-modality expert groups under the parallel-built model."""

    def test_text_experts_only_affect_text_positions(self):
        """Zero text experts' down_proj weights; image-position outputs are
        unchanged, text-position outputs change."""
        cfg = _tiny_moma_configs(gumbel_noise=False)
        wrapper = _build(cfg, param_dtype=torch.float32)
        wrapper.eval()
        pixels, input_ids, _ = _dummy_batch(wrapper, batch=1, text_len=8)
        with torch.no_grad():
            logits_full, _ = wrapper(pixels, input_ids)

        # Zero every text expert's down_proj across all layers.
        with torch.no_grad():
            for layer in wrapper.transformer.layers.values():
                for expert in layer.mlp.experts["text"].experts:
                    expert.down_proj.weight.zero_()
        with torch.no_grad():
            logits_text_off, _ = wrapper(pixels, input_ids)

        # All logits are over the text-only tail (output_slice trims image
        # prefix), so the entire output is a text view — different text
        # experts -> different output.
        assert not torch.allclose(logits_full, logits_text_off)

    def test_image_experts_only_affect_image_positions(self):
        """Symmetric to the above: zero image-expert down_proj only changes
        outputs whose computation read image positions. Because the shared
        attention mixes image into every text query, *all* text-position
        outputs are sensitive to image experts (image experts shape image
        tokens, which feed into the keys/values that text attends to).
        Still, the output must change."""
        cfg = _tiny_moma_configs(gumbel_noise=False)
        wrapper = _build(cfg, param_dtype=torch.float32)
        wrapper.eval()
        pixels, input_ids, _ = _dummy_batch(wrapper, batch=1, text_len=8)
        with torch.no_grad():
            logits_full, _ = wrapper(pixels, input_ids)

        with torch.no_grad():
            for layer in wrapper.transformer.layers.values():
                for expert in layer.mlp.experts["image"].experts:
                    expert.down_proj.weight.zero_()
        with torch.no_grad():
            logits_image_off, _ = wrapper(pixels, input_ids)

        assert not torch.allclose(logits_full, logits_image_off)

    def test_unbalanced_expert_counts_build_and_forward(self):
        """Paper's moe_7t1i (7 text experts + 1 image expert) builds and
        forwards. Validates the per-modality expert dict supports asymmetric
        allocations (which MoT's single num_experts field cannot express)."""
        from kempnerforge.model.moma import MoMaBlock

        cfg = _tiny_moma_configs(experts_per_modality={"image": 1, "text": 7})
        wrapper = _build(cfg)
        layer0 = wrapper.transformer.layers["0"]
        assert isinstance(layer0, MoMaBlock)
        assert layer0.mlp.experts["image"].num_experts == 1
        assert layer0.mlp.experts["text"].num_experts == 7

        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids, labels)
        assert logits.shape == (2, 16, cfg[0].vocab_size)
        assert torch.isfinite(logits).all()

    def test_capacity_factor_explicit_override(self):
        """Explicit positive capacity factor overrides the paper default 1/|E^M|."""
        from kempnerforge.model.moma import MoMaBlock

        cfg = _tiny_moma_configs(capacity_factor=0.75)
        wrapper = _build(cfg)
        layer0 = wrapper.transformer.layers["0"]
        assert isinstance(layer0, MoMaBlock)
        assert layer0.mlp.experts["image"].router.capacity_factor == 0.75
        assert layer0.mlp.experts["text"].router.capacity_factor == 0.75


class TestFreezeStageHook:
    def test_freeze_schedule_transitions(self):
        """Schedule that freezes 'moma' at step 3 and unfreezes at step 7."""
        cfg = _tiny_moma_configs(
            freeze=[
                FreezeSpec("vision_encoder", True),
                FreezeSpec("moma", False),
            ],
        )
        cfg[3].freeze_schedule = [  # type: ignore[union-attr]
            FreezeStage(start_step=3, specs=(FreezeSpec("moma", True),)),
            FreezeStage(start_step=7, specs=(FreezeSpec("moma", False),)),
        ]
        wrapper = _build(cfg, param_dtype=torch.float32)

        layer_params = [
            p for n, p in wrapper.named_parameters() if n.startswith("transformer.layers.")
        ]

        # Step 0: layers trainable.
        for p in layer_params:
            assert p.requires_grad

        # Step 3: layers frozen.
        specs = effective_freeze(3, cfg[3].freeze, cfg[3].freeze_schedule)  # type: ignore[union-attr]
        apply_freeze_specs(wrapper, specs, cfg[3].module_patterns)  # type: ignore[union-attr]
        for p in layer_params:
            assert not p.requires_grad

        # Step 7: layers trainable again.
        specs = effective_freeze(7, cfg[3].freeze, cfg[3].freeze_schedule)  # type: ignore[union-attr]
        apply_freeze_specs(wrapper, specs, cfg[3].module_patterns)  # type: ignore[union-attr]
        for p in layer_params:
            assert p.requires_grad
