"""Integration tests for the VLM Mixture-of-Transformers (MoT) training path.

Single-GPU forward + backward on a tiny synthetic MoT config. Exercises
``build_parallel_model`` MoT branch, freeze targeting at the ``mot``
alias, dtype propagation, ``torch.compile`` parity, the FreezeStage
hook semantics, save/load round-trip, and an early MoT+MoE smoke
(num_experts=4, moe_frequency=2).

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
    MoTConfig,
)
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper
from kempnerforge.training.freeze import apply_freeze_specs, effective_freeze

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VLM MoT integration tests require CUDA",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_mot_configs(
    *,
    num_image_tokens: int = 8,
    feature_dim: int = 96,
    n_layers: int = 2,
    freeze: list[FreezeSpec] | None = None,
    moe: bool = False,
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoTConfig]:
    return (
        ModelConfig(
            dim=64,
            n_layers=n_layers,
            n_heads=4,
            vocab_size=256,
            max_seq_len=128,
            ffn_hidden_dim=128,
            num_experts=4 if moe else 0,
            moe_top_k=2 if moe else 2,
            moe_frequency=2 if moe else 1,
        ),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        MoTConfig(
            max_text_len=32,
            freeze=freeze if freeze is not None else [FreezeSpec("vision_encoder", True)],
        ),
    )


def _build(
    configs: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoTConfig],
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
        """Tiny MoT config builds on a single GPU; forward + backward run."""
        from kempnerforge.model.mot import MoTBlock

        cfg = _tiny_mot_configs()
        wrapper = _build(cfg)
        assert isinstance(wrapper, VLMWrapper)
        # MoT extends the residual stream with num_image_tokens.
        assert wrapper.num_image_tokens == 8
        # All layers are MoTBlocks.
        assert all(isinstance(layer, MoTBlock) for layer in wrapper.transformer.layers.values())

        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids, labels)
        # output_slice trims the 8 image positions; logits cover text_len positions.
        assert logits.shape == (2, 16, cfg[0].vocab_size)
        # Re-init per-modality o_proj / down_proj so backward exercises Q/K/V grads.
        with torch.no_grad():
            for layer in wrapper.transformer.layers.values():
                for m in layer.modalities:
                    torch.nn.init.normal_(layer.attn.o_proj[m].weight, std=0.01)
                    torch.nn.init.normal_(layer.mlp[m].down_proj.weight, std=0.01)
        logits, _ = wrapper(pixels, input_ids, labels)
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

    def test_freeze_targets_mot(self):
        """`FreezeSpec("mot")` freezes only the per-modality main stack
        (transformer.layers.*) and leaves the adapter and final norms
        trainable.
        """
        cfg = _tiny_mot_configs(freeze=[FreezeSpec("mot", True)])
        wrapper = _build(cfg)
        trainable = {name for name, p in wrapper.named_parameters() if p.requires_grad}
        frozen = {name for name, p in wrapper.named_parameters() if not p.requires_grad}
        # All transformer.layers params are frozen; transformer.norm is
        # frozen because MoT does not use it (replaced by mot_norms);
        # the rest (mot_norms, embedding, output head, adapter) is
        # trainable.
        for n in frozen:
            if n.startswith("transformer."):
                assert n.startswith("transformer.layers.") or n == "transformer.norm.weight", n
        assert any(n.startswith("transformer.mot_norms") for n in trainable)
        assert any(n.startswith("adapter") for n in trainable)
        # Sanity: at least one per-modality projection is frozen.
        assert any("attn.q_proj." in n for n in frozen)

    def test_image_features_dtype_propagation(self):
        """Encoder fp32 -> adapter bf16 -> Transformer.forward casts
        prefix_embeds to the residual-stream dtype before the MoT
        block sees it. Asserts (a) build with bf16 param_dtype works,
        (b) forward output is bf16, (c) no dtype-mismatch errors.
        """
        cfg = _tiny_mot_configs()
        wrapper = _build(cfg, param_dtype=torch.bfloat16)
        assert wrapper.adapter.proj1.weight.dtype == torch.bfloat16
        pixels, input_ids, _ = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids)
        assert logits.dtype == torch.bfloat16

    def test_mot_compile_eager_parity(self):
        """torch.compile(wrapper) output matches eager output within a
        small tolerance. Catches compile-graph divergence on the MoT
        per-modality forward path."""
        cfg = _tiny_mot_configs()
        wrapper = _build(cfg, param_dtype=torch.float32)
        wrapper.eval()
        pixels, input_ids, _ = _dummy_batch(wrapper, batch=1, text_len=8)

        with torch.no_grad():
            eager_logits, _ = wrapper(pixels, input_ids)

        compiled = torch.compile(wrapper)
        with torch.no_grad():
            compiled_logits, _ = compiled(pixels, input_ids)

        torch.testing.assert_close(eager_logits, compiled_logits, atol=1e-5, rtol=1e-5)

    def test_save_load_forward_parity(self):
        """state_dict round-trips with bit-equal forward output."""
        cfg = _tiny_mot_configs()
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


class TestMoTPlusMoESmoke:
    """Early MoT + MoE smoke (num_experts=4, moe_frequency=2). Single
    GPU, no expert parallelism. If this fails, gate the combination at
    JobConfig.validate before Step 8."""

    def test_mot_plus_moe_build_smoke_1gpu(self):
        from kempnerforge.model.moe import MoEMLP
        from kempnerforge.model.mot import MoTBlock

        cfg = _tiny_mot_configs(moe=True, n_layers=2)
        wrapper = _build(cfg)
        # Layer 1 (i=1, (i+1) % moe_frequency == 0) has MoE FFNs per modality.
        layer1 = wrapper.transformer.layers["1"]
        assert isinstance(layer1, MoTBlock)
        for m in layer1.modalities:
            assert isinstance(layer1.mlp[m], MoEMLP)
        # Layer 0 keeps dense per-modality FFNs.
        layer0 = wrapper.transformer.layers["0"]
        for m in layer0.modalities:
            assert not isinstance(layer0.mlp[m], MoEMLP)

        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids, labels)
        assert logits.shape == (2, 16, cfg[0].vocab_size)
        assert torch.isfinite(logits).all()

    def test_mot_plus_moe_backward_1gpu(self):
        cfg = _tiny_mot_configs(moe=True, n_layers=2)
        wrapper = _build(cfg, param_dtype=torch.float32)
        with torch.no_grad():
            for layer in wrapper.transformer.layers.values():
                for m in layer.modalities:
                    torch.nn.init.normal_(layer.attn.o_proj[m].weight, std=0.01)
        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids, labels)
        loss = logits.float().sum()
        loss.backward()
        # At least one per-modality MoE expert receives a grad on layer 1.
        layer1 = wrapper.transformer.layers["1"]
        any_grad = False
        for m in layer1.modalities:
            mlp_m = layer1.mlp[m]
            for p in mlp_m.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    any_grad = True
                    break
        assert any_grad


class TestWarmStartFromJD:
    """Step 7 hook: copy a JD checkpoint's dense block weights into every
    per-modality copy in every MoTBlock. The hook lives in scripts/train.py
    next to ckpt_mgr.load; here we exercise it in isolation to confirm
    the helper translates correctly under the parallel-built MoT model."""

    def test_warm_start_from_jd_after_load_before_freeze(self, tmp_path):
        """JD state -> torch.save -> mot_warm_start_from_text_stack ->
        per-modality copies bit-equal to the source dense block weights.
        """
        from kempnerforge.config.vlm import JointDecoderConfig
        from kempnerforge.model.mot import mot_warm_start_from_text_stack
        from kempnerforge.model.vlm import inner_transformer

        # Build a JD model with the same backbone shape and dump its state.
        jd_cfg = (
            ModelConfig(
                dim=64,
                n_layers=2,
                n_heads=4,
                vocab_size=256,
                max_seq_len=128,
                ffn_hidden_dim=128,
            ),
            VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            AdapterConfig(),
            JointDecoderConfig(max_text_len=32),
        )
        jd_wrapper = _build(jd_cfg, param_dtype=torch.float32)
        jd_state = inner_transformer(jd_wrapper).state_dict()
        ckpt_path = tmp_path / "jd_ckpt.pt"
        torch.save(jd_state, ckpt_path)

        # Build a MoT model and run the warm-start helper.
        mot_cfg = _tiny_mot_configs(num_image_tokens=8, n_layers=2)
        mot_wrapper = _build(mot_cfg, param_dtype=torch.float32)
        source = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mot_warm_start_from_text_stack(inner_transformer(mot_wrapper), source)

        # Per-modality copies equal the source dense weights. (Compare via
        # the on-disk CPU state to avoid cross-device tensor compares.)
        mot_t = inner_transformer(mot_wrapper)
        for i in range(mot_cfg[0].n_layers):
            for m in mot_t.layers[str(i)].modalities:  # type: ignore[union-attr]
                assert torch.equal(
                    mot_t.layers[str(i)].attn.q_proj[m].weight.cpu(),  # type: ignore[union-attr]
                    source[f"layers.{i}.attention.q_proj.weight"],
                )
                assert torch.equal(
                    mot_t.layers[str(i)].mlp[m].down_proj.weight.cpu(),  # type: ignore[union-attr]
                    source[f"layers.{i}.mlp.down_proj.weight"],
                )

    def test_warm_start_idempotent_under_parallel_build(self, tmp_path):
        """Running the helper twice produces no further state change."""
        from kempnerforge.config.vlm import JointDecoderConfig
        from kempnerforge.model.mot import mot_warm_start_from_text_stack
        from kempnerforge.model.vlm import inner_transformer

        jd_cfg = (
            ModelConfig(
                dim=64,
                n_layers=2,
                n_heads=4,
                vocab_size=256,
                max_seq_len=128,
                ffn_hidden_dim=128,
            ),
            VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            AdapterConfig(),
            JointDecoderConfig(max_text_len=32),
        )
        jd_wrapper = _build(jd_cfg, param_dtype=torch.float32)
        jd_state = inner_transformer(jd_wrapper).state_dict()
        ckpt_path = tmp_path / "jd_ckpt.pt"
        torch.save(jd_state, ckpt_path)

        mot_cfg = _tiny_mot_configs(num_image_tokens=8, n_layers=2)
        mot_wrapper = _build(mot_cfg, param_dtype=torch.float32)
        source = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mot_warm_start_from_text_stack(inner_transformer(mot_wrapper), source)
        snap = {k: v.detach().clone() for k, v in mot_wrapper.state_dict().items()}
        mot_warm_start_from_text_stack(inner_transformer(mot_wrapper), source)
        for k, v in mot_wrapper.state_dict().items():
            assert torch.equal(v, snap[k]), f"key {k} changed on second call"


class TestFreezeStageHook:
    def test_freeze_schedule_transitions(self):
        """Schedule that freezes 'mot' at step 3 and unfreezes at step 7."""
        cfg = _tiny_mot_configs(
            freeze=[
                FreezeSpec("vision_encoder", True),
                FreezeSpec("mot", False),
            ],
        )
        cfg[3].freeze_schedule = [  # type: ignore[union-attr]
            FreezeStage(start_step=3, specs=(FreezeSpec("mot", True),)),
            FreezeStage(start_step=7, specs=(FreezeSpec("mot", False),)),
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
