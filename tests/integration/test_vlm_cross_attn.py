"""Integration tests for the VLM Cross-Attention training path.

Single-GPU forward + backward on a tiny synthetic CA config.
Exercises ``build_parallel_model`` Cross-Attention branch, freeze
targeting at the ``cross_attention`` alias, dtype propagation
through the residual stream + image-features side channel,
``torch.compile`` parity, and the FreezeStage hook semantics
(transition flips ``requires_grad``, AdamW skips frozen params, and
saved/loaded state dicts produce bit-equal forward output).

Runs on CUDA only; skipped when ``torch.cuda.is_available()`` is
False.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vlm import (
    CrossAttentionConfig,
    FreezeSpec,
    FreezeStage,
    JointDecoderConfig,
)
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper
from kempnerforge.training.freeze import apply_freeze_specs, effective_freeze

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VLM cross-attention integration tests require CUDA",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_ca_config(
    *,
    num_image_tokens: int = 8,
    feature_dim: int = 96,
    cadence: int = 2,
    freeze: list[FreezeSpec] | None = None,
) -> ModelConfig:
    return ModelConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        vocab_size=256,
        max_seq_len=128,
        vlm=CrossAttentionConfig(
            vision_encoder="random",
            feature_dim=feature_dim,
            num_tokens=num_image_tokens,
            max_text_len=32,
            cross_attention_every_n_layers=cadence,
            freeze=freeze if freeze is not None else [FreezeSpec("vision_encoder", True)],
        ),
    )


def _build(cfg: ModelConfig, *, param_dtype: torch.dtype = torch.bfloat16) -> VLMWrapper:
    model = build_parallel_model(cfg, device=DEVICE, device_mesh=None, param_dtype=param_dtype)
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
        """Tiny CA config builds on a single GPU; forward + backward run."""
        cfg = _tiny_ca_config()
        wrapper = _build(cfg)
        assert isinstance(wrapper, VLMWrapper)
        # CA arch: residual stream is text-only.
        assert wrapper.num_image_tokens == 0
        # 4 layers + cadence 2 -> 2 CA blocks.
        assert len(wrapper.transformer.cross_attention_layers) == 2

        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids, labels)
        assert logits.shape == (2, 16, cfg.vocab_size)
        loss = logits.float().sum()
        loss.backward()
        # Adapter and CA layers receive gradients (encoder is frozen).
        adapter_grads = [
            p.grad
            for n, p in wrapper.named_parameters()
            if n.startswith("adapter") and p.requires_grad
        ]
        ca_grads = [
            p.grad
            for n, p in wrapper.named_parameters()
            if n.startswith("transformer.cross_attention_layers") and p.requires_grad
        ]
        assert all(g is not None for g in adapter_grads)
        assert all(g is not None for g in ca_grads)

    def test_freeze_targets_cross_attention_layers(self):
        """`FreezeSpec("cross_attention")` freezes only CA params and
        leaves the rest of the transformer trainable.
        """
        cfg = _tiny_ca_config(freeze=[FreezeSpec("cross_attention", True)])
        wrapper = _build(cfg)
        trainable = {name for name, p in wrapper.named_parameters() if p.requires_grad}
        frozen = {name for name, p in wrapper.named_parameters() if not p.requires_grad}
        # CA params must be frozen.
        assert all(
            n.startswith("transformer.cross_attention_layers")
            for n in frozen
            if n.startswith("transformer.")
        )
        # Regular transformer.layers params must be trainable.
        assert any(n.startswith("transformer.layers") for n in trainable)
        # Adapter is trainable too.
        assert any(n.startswith("adapter") for n in trainable)

    def test_image_features_dtype_propagation(self):
        """Encoder fp32 -> adapter bf16 -> Transformer.forward casts
        image_features to the residual-stream dtype before the CA block
        sees it. The test asserts (a) build with bf16 param_dtype works,
        (b) forward output is bf16, (c) no dtype-mismatch errors.
        """
        cfg = _tiny_ca_config()
        wrapper = _build(cfg, param_dtype=torch.bfloat16)
        # Adapter params are bf16; encoder is in HF default (fp32).
        assert wrapper.adapter.proj1.weight.dtype == torch.bfloat16
        pixels, input_ids, _ = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, input_ids)
        # Output dtype matches transformer (bf16).
        assert logits.dtype == torch.bfloat16

    def test_ca_compile_eager_parity(self):
        """torch.compile(wrapper) output matches eager output within
        a small tolerance. Catches compile-graph divergence in the
        CA-interleaved forward path."""
        cfg = _tiny_ca_config()
        wrapper = _build(cfg, param_dtype=torch.float32)  # fp32 for tighter comparison
        wrapper.eval()
        pixels, input_ids, _ = _dummy_batch(wrapper, batch=1, text_len=8)

        with torch.no_grad():
            eager_logits, _ = wrapper(pixels, input_ids)

        compiled = torch.compile(wrapper)
        with torch.no_grad():
            compiled_logits, _ = compiled(pixels, input_ids)

        torch.testing.assert_close(eager_logits, compiled_logits, atol=1e-5, rtol=1e-5)


class TestFreezeStageHook:
    def test_freeze_schedule_transitions(self):
        """A two-stage freeze schedule flips requires_grad at the
        configured step boundaries. The transition is deterministic and
        idempotent: running effective_freeze + apply_freeze_specs at the
        boundary matches the on-the-fly hook in scripts/train.py.
        """
        cfg = _tiny_ca_config(
            freeze=[
                FreezeSpec("vision_encoder", True),
                FreezeSpec("adapter", False),
            ],
        )
        # Schedule: at step 3, freeze adapter. At step 7, unfreeze adapter.
        cfg.vlm.freeze_schedule = [
            FreezeStage(start_step=3, specs=(FreezeSpec("adapter", True),)),
            FreezeStage(start_step=7, specs=(FreezeSpec("adapter", False),)),
        ]
        wrapper = _build(cfg, param_dtype=torch.float32)

        adapter_params = [p for n, p in wrapper.named_parameters() if n.startswith("adapter")]

        # Step 0: adapter trainable (base spec).
        for p in adapter_params:
            assert p.requires_grad

        # At step 3: apply effective_freeze and confirm adapter is frozen.
        specs = effective_freeze(3, cfg.vlm.freeze, cfg.vlm.freeze_schedule)
        apply_freeze_specs(wrapper, specs, cfg.vlm.module_patterns)
        for p in adapter_params:
            assert not p.requires_grad

        # At step 7: unfreeze.
        specs = effective_freeze(7, cfg.vlm.freeze, cfg.vlm.freeze_schedule)
        apply_freeze_specs(wrapper, specs, cfg.vlm.module_patterns)
        for p in adapter_params:
            assert p.requires_grad

    def test_adamw_no_wd_drift_on_frozen_params(self):
        """AdamW with the default optimizer.zero_grad(set_to_none=True)
        skips frozen params entirely — no SGD step, no weight decay.
        Frozen-adapter weights are bit-identical across optimizer steps
        even when weight_decay is non-zero.
        """
        cfg = _tiny_ca_config(freeze=[FreezeSpec("adapter", True)])
        wrapper = _build(cfg, param_dtype=torch.float32)

        # Snapshot adapter weights before the optimizer step.
        before = {
            n: p.detach().clone() for n, p in wrapper.named_parameters() if n.startswith("adapter")
        }
        # Sanity: frozen really means requires_grad=False.
        assert all(
            not p.requires_grad for n, p in wrapper.named_parameters() if n.startswith("adapter")
        )

        optimizer = torch.optim.AdamW(
            [p for p in wrapper.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=0.1,
        )

        # Run 5 forward + backward + optimizer.step cycles.
        for _ in range(5):
            pixels, input_ids, labels = _dummy_batch(wrapper, batch=2, text_len=8)
            logits, _ = wrapper(pixels, input_ids, labels)
            loss = logits.float().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Adapter weights bit-identical (frozen + AdamW skips entire step).
        for n, p in wrapper.named_parameters():
            if n.startswith("adapter"):
                assert torch.equal(p, before[n]), f"Frozen param drifted: {n}"

    def test_ca_save_load_forward_parity(self):
        """A model's state_dict round-trips with bit-equal forward
        output. Catches missing buffer registration, persistent vs
        non-persistent buffer mistakes, etc."""
        cfg = _tiny_ca_config()
        wrapper_a = _build(cfg, param_dtype=torch.float32)
        wrapper_a.eval()
        pixels, input_ids, _ = _dummy_batch(wrapper_a, batch=1, text_len=8)

        with torch.no_grad():
            logits_a, _ = wrapper_a(pixels, input_ids)

        # Save state, build a fresh wrapper, load into it.
        state = wrapper_a.state_dict()
        wrapper_b = _build(cfg, param_dtype=torch.float32)
        wrapper_b.load_state_dict(state, strict=True)
        wrapper_b.eval()

        with torch.no_grad():
            logits_b, _ = wrapper_b(pixels, input_ids)

        torch.testing.assert_close(logits_a, logits_b, atol=0.0, rtol=0.0)

    def test_jd_state_dict_loads_into_jd_clean(self):
        """Sanity: a JD state_dict round-trips into a JD-built wrapper
        without missing/unexpected keys.
        """
        cfg = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=256,
            max_seq_len=128,
            vlm=JointDecoderConfig(
                vision_encoder="random",
                feature_dim=96,
                num_tokens=8,
                max_text_len=32,
            ),
        )
        wrapper_a = _build(cfg)
        state = wrapper_a.state_dict()
        wrapper_b = _build(cfg)
        missing, unexpected = wrapper_b.load_state_dict(state, strict=True)
        assert missing == []
        assert unexpected == []

    def test_jd_state_dict_loads_into_ca_with_strict_false(self):
        """A JD checkpoint state_dict has no cross_attention_layers.*
        keys. Loading it into a CA-built wrapper requires strict=False
        (CA layers stay at their construction defaults, which for Wo
        is zero -> identity at init), and the resulting model is
        functional for forward."""
        jd_cfg = ModelConfig(
            dim=64,
            n_layers=4,
            n_heads=4,
            vocab_size=256,
            max_seq_len=128,
            vlm=JointDecoderConfig(
                vision_encoder="random",
                feature_dim=96,
                num_tokens=8,
                max_text_len=32,
            ),
        )
        ca_cfg = _tiny_ca_config(num_image_tokens=8, feature_dim=96, cadence=2)

        jd_wrapper = _build(jd_cfg)
        ca_wrapper = _build(ca_cfg)

        # Load JD state into CA wrapper (strict=False because CA has
        # extra cross_attention_layers.* keys not in JD).
        jd_state = jd_wrapper.state_dict()
        # Filter out JD-specific items that don't exist in CA (none expected
        # in this setup since the backbone is identical).
        missing, unexpected = ca_wrapper.load_state_dict(jd_state, strict=False)
        # CA-only keys appear in `missing` because JD didn't provide them.
        assert all(k.startswith("transformer.cross_attention_layers") for k in missing)
        assert unexpected == []

        # Forward still works (CA layers at zero-init are identity).
        ca_wrapper.eval()
        pixels, input_ids, _ = _dummy_batch(ca_wrapper)
        with torch.no_grad():
            logits, _ = ca_wrapper(pixels, input_ids)
        assert logits.shape[0] == pixels.shape[0]

    def test_freeze_schedule_with_typo_module_alias_raises(self):
        """A FreezeSpec/FreezeStage referencing an unknown module alias
        (typo in TOML, e.g., "adaptor" instead of "adapter") must raise
        at training-loop start so the user can fix the typo before
        wasting compute. effective_freeze validates against
        mc.vlm.module_patterns keys when the training loop calls it
        with valid_modules set.
        """
        cfg = _tiny_ca_config()
        # FreezeStage with a typo'd module alias: "adaptor" not in
        # module_patterns ({transformer, vision_encoder, adapter,
        # cross_attention} for CrossAttentionConfig).
        cfg.vlm.freeze_schedule = [
            FreezeStage(start_step=5, specs=(FreezeSpec("adaptor", True),)),
        ]
        # The validation fires when effective_freeze is called with
        # valid_modules=set(module_patterns.keys()), which scripts/train.py
        # does at every save and at resume.
        valid_modules = set(cfg.vlm.module_patterns.keys())
        with pytest.raises(ValueError, match="adaptor"):
            effective_freeze(
                step=10,
                base=cfg.vlm.freeze,
                schedule=cfg.vlm.freeze_schedule,
                valid_modules=valid_modules,
            )

    def test_ca_plus_moe_runs_and_learns(self):
        """VLM (Cross-Attention) + MoE in the text TransformerBlocks
        builds, forwards, backwards, and the CE loss decreases over a
        few optimizer steps. set_moe_step routed through
        inner_transformer(model); aux_loss accumulated correctly.

        MoE lives only in TransformerBlocks (frequency=2 means every
        other text block is MoE); CrossAttentionBlocks stay dense.
        """
        from kempnerforge.model.vlm import inner_transformer

        cfg = ModelConfig(
            dim=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=256,
            max_seq_len=128,
            num_experts=4,
            moe_top_k=2,
            moe_frequency=2,
            vlm=CrossAttentionConfig(
                vision_encoder="random",
                feature_dim=96,
                num_tokens=8,
                max_text_len=32,
                cross_attention_every_n_layers=2,
                freeze=[FreezeSpec("vision_encoder", True)],
            ),
        )
        wrapper = _build(cfg, param_dtype=torch.float32)
        # 4 layers, frequency=2 -> 2 MoE blocks (layers 1 and 3).
        from kempnerforge.model.moe import MoEMLP

        moe_blocks = sum(
            1 for layer in wrapper.transformer.layers.values() if isinstance(layer.mlp, MoEMLP)
        )
        assert moe_blocks == 2, f"expected 2 MoE blocks, got {moe_blocks}"
        # 4 layers, CA cadence=2 -> 2 CA blocks. CA blocks remain dense.
        assert len(wrapper.transformer.cross_attention_layers) == 2

        opt = torch.optim.AdamW(wrapper.parameters(), lr=3e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        torch.manual_seed(0)
        pixels = torch.randn(2, 3, 32, 32, device=DEVICE)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)

        inner = inner_transformer(wrapper)
        losses = []
        for step in range(5):
            inner.set_moe_step(step, max_steps=100)  # type: ignore[attr-defined]
            logits, _ = wrapper(pixels, input_ids, input_ids)
            ce = loss_fn(logits.reshape(-1, cfg.vocab_size), input_ids.reshape(-1))
            aux = inner.get_moe_aux_loss()  # type: ignore[attr-defined]
            loss = ce + 0.01 * aux
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append((ce.item(), aux.item()))

        # CE decreases.
        initial_ce, final_ce = losses[0][0], losses[-1][0]
        assert final_ce < initial_ce, (
            f"VLM + MoE: CE did not decrease ({initial_ce:.3f} -> {final_ce:.3f})"
        )
        # aux_loss is non-zero throughout — router is producing real loss.
        assert all(a > 0 for _, a in losses), "aux_loss zeroed out"

    def test_ca_overfits_tiny_dataset(self):
        """Train a tiny CA wrapper on 8 image-caption pairs for 200
        steps with AdamW. Assertions:
        (a) final loss << initial loss (model learns).
        (b) loss with REAL images is lower than loss with zero images
            at the same training step (model is learning *from images*,
            not just from text).
        Catches the silent "image input is structurally connected but
        functionally ignored" failure mode.
        """
        torch.manual_seed(0)
        cfg = _tiny_ca_config(num_image_tokens=8, feature_dim=96, cadence=2)
        wrapper = _build(cfg, param_dtype=torch.float32)
        wrapper.train()

        # 8 distinct (pixels, input_ids) pairs.
        n_pairs = 8
        text_len = 8
        pixels_dataset = torch.randn(n_pairs, 3, 32, 32, device=DEVICE)
        # Force a strong image -> text correlation: each label sequence
        # is determined by the image (via a hash), so fitting the data
        # requires using the image features. Otherwise text-only prior
        # could match if labels were random.
        input_ids_dataset = torch.zeros(n_pairs, text_len, dtype=torch.long, device=DEVICE)
        for i in range(n_pairs):
            seed_val = int(pixels_dataset[i].sum().item() * 1000) % cfg.vocab_size
            input_ids_dataset[i] = (
                torch.arange(text_len, device=DEVICE) + seed_val
            ) % cfg.vocab_size

        optimizer = torch.optim.AdamW(wrapper.parameters(), lr=3e-3)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        def _train_step(pixels, input_ids, labels):
            logits, _ = wrapper(pixels, input_ids, labels)
            loss = loss_fn(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
            return loss

        # Initial loss: forward over the full mini-dataset.
        with torch.no_grad():
            initial_loss = _train_step(pixels_dataset, input_ids_dataset, input_ids_dataset).item()

        # Train.
        for _ in range(200):
            # Shuffle the 8-pair "epoch" each iteration.
            perm = torch.randperm(n_pairs, device=DEVICE)
            pixels_b = pixels_dataset[perm]
            ids_b = input_ids_dataset[perm]
            loss = _train_step(pixels_b, ids_b, ids_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            final_loss = _train_step(pixels_dataset, input_ids_dataset, input_ids_dataset).item()
            zero_pixels = torch.zeros_like(pixels_dataset)
            loss_with_zero_images = _train_step(
                zero_pixels, input_ids_dataset, input_ids_dataset
            ).item()

        # (a) Model learned: final loss is much smaller than initial.
        assert final_loss < 0.5 * initial_loss, (
            f"Model did not learn: initial={initial_loss:.3f}, final={final_loss:.3f}"
        )
        # (b) Model uses images: loss with real images < loss with zero images.
        assert final_loss < loss_with_zero_images, (
            f"Model ignores images: real_img_loss={final_loss:.3f}, "
            f"zero_img_loss={loss_with_zero_images:.3f}"
        )
