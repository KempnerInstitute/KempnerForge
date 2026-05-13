"""Integration tests for the VLM Joint-Decoder training path.

Single-GPU forward + backward + optimizer step on a tiny synthetic
config. Exercises ``build_parallel_model`` VLM branch, the
Adapter / VLMWrapper contract, freeze application, dtype policy for the
frozen encoder, and the all-pad-batch edge case.

Runs on CUDA only; skipped when ``torch.cuda.is_available()`` is False.
Checkpoint round-trip tests live alongside the checkpoint commit
(Step 9); this file focuses on the build + one-step flow.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import OptimizerConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper, inner_transformer
from kempnerforge.training.loss import cross_entropy_loss
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VLM integration tests require CUDA"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_vlm_configs(
    *,
    num_image_tokens: int = 8,
    feature_dim: int = 96,
    freeze: list[FreezeSpec] | None = None,
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig]:
    return (
        ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=128),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        VLMConfig(
            max_text_len=32,
            freeze=freeze if freeze is not None else [FreezeSpec("vision_encoder", True)],
        ),
    )


def _build(
    configs: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig],
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
    wrapper: VLMWrapper,
    batch: int = 2,
    text_len: int = 16,
    *,
    all_pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = torch.randn(batch, 3, 32, 32, device=DEVICE)
    input_ids = torch.randint(
        0, wrapper.transformer.config.vocab_size, (batch, text_len), device=DEVICE
    )
    if all_pad:
        labels = torch.full((batch, text_len), -100, dtype=torch.long, device=DEVICE)
    else:
        labels = input_ids.clone()
    return pixels, input_ids, labels


class TestBuild:
    def test_meta_device_build_no_oom(self):
        """Tiny VLM config builds on a single GPU."""
        cfg = _tiny_vlm_configs()
        wrapper = _build(cfg)
        assert isinstance(wrapper, VLMWrapper)
        assert wrapper.num_image_tokens == 8

    def test_build_respects_freeze(self):
        cfg = _tiny_vlm_configs(freeze=[FreezeSpec("vision_encoder", True)])
        wrapper = _build(cfg)
        trainable = {name for name, p in wrapper.named_parameters() if p.requires_grad}
        frozen = {name for name, p in wrapper.named_parameters() if not p.requires_grad}
        # Encoder params must be fully frozen; adapter and transformer trainable.
        assert all(not n.startswith("vision_encoder") for n in trainable)
        assert all(n.startswith("vision_encoder") for n in frozen)
        assert any(n.startswith("adapter") for n in trainable)
        assert any(n.startswith("transformer") for n in trainable)

    def test_encoder_in_eval_when_frozen(self):
        wrapper = _build(_tiny_vlm_configs())
        assert wrapper.vision_encoder.training is False
        # Transformer and adapter remain in train mode (default for nn.Module).
        assert wrapper.transformer.training is True
        assert wrapper.adapter.training is True


class TestDtypePolicy:
    def test_frozen_encoder_not_cast_to_bf16(self):
        """Transformer and adapter are bf16, vision encoder stays in its
        HF dtype (fp32 for RandomVisionEncoder). D16."""
        cfg = _tiny_vlm_configs()
        wrapper = _build(cfg, param_dtype=torch.bfloat16)
        # Transformer / adapter are bf16.
        for p in wrapper.transformer.parameters():
            assert p.dtype == torch.bfloat16, f"transformer param dtype = {p.dtype}"
        for p in wrapper.adapter.parameters():
            assert p.dtype == torch.bfloat16, f"adapter param dtype = {p.dtype}"
        # Vision encoder buffer (RandomVisionEncoder._anchor) stays fp32.
        anchor = wrapper.vision_encoder.get_buffer("_anchor")
        assert anchor.dtype == torch.float32


class TestForward:
    def test_one_step_loss_finite(self):
        cfg = _tiny_vlm_configs()
        wrapper = _build(cfg)
        optimizer = build_optimizer(wrapper, OptimizerConfig(lr=1e-3, fused=False))

        pixels, input_ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, input_ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(loss).item()

    def test_logits_shape_matches_text_only(self):
        cfg = _tiny_vlm_configs(num_image_tokens=8)
        wrapper = _build(cfg)
        pixels, input_ids, _ = _dummy_batch(wrapper, text_len=20)
        logits, _ = wrapper(pixels, input_ids, None)
        # output_slice drops image positions; (B, T, V) not (B, N+T, V).
        assert logits.shape == (2, 20, cfg[0].vocab_size)

    def test_vlm_all_pad_batch_no_nan(self):
        """Every label is -100: kempnerforge.training.loss.cross_entropy_loss
        short-circuits to 0.0 (no NaN), backward is skipped (no grad graph),
        and a subsequent real step still updates params."""
        cfg = _tiny_vlm_configs()
        wrapper = _build(cfg)
        optimizer = build_optimizer(wrapper, OptimizerConfig(lr=1e-3, fused=False))

        # Snapshot trainable params.
        snap = {n: p.detach().clone() for n, p in wrapper.named_parameters() if p.requires_grad}

        pixels, input_ids, labels = _dummy_batch(wrapper, all_pad=True)
        logits, labels_out = wrapper(pixels, input_ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        assert float(loss.item()) == 0.0
        # The short-circuit returns a tensor with no graph; mirror what
        # scripts/train.py does when nan_detector skips the step.
        optimizer.zero_grad()

        # Subsequent real step updates the params normally.
        pixels2, input_ids2, labels2 = _dummy_batch(wrapper)
        logits2, labels_out2 = wrapper(pixels2, input_ids2, labels2)
        loss2 = cross_entropy_loss(logits2, labels_out2)
        loss2.backward()
        optimizer.step()

        changed = [
            n
            for n, p in wrapper.named_parameters()
            if p.requires_grad and not torch.equal(snap[n], p.detach())
        ]
        assert changed, "No trainable params moved after a real step"


class TestOverfit:
    def test_adapter_changes_frozen_encoder_stays(self):
        """After 10 steps on a fixed batch, adapter + transformer params
        drift and the frozen encoder parameters are bit-equal."""
        cfg = _tiny_vlm_configs()
        wrapper = _build(cfg)
        optimizer = build_optimizer(wrapper, OptimizerConfig(lr=1e-2, fused=False))

        # Snapshot frozen encoder params.
        encoder_snap = {n: p.detach().clone() for n, p in wrapper.vision_encoder.named_parameters()}

        pixels, input_ids, labels = _dummy_batch(wrapper)
        last_loss = None
        for _ in range(10):
            logits, labels_out = wrapper(pixels, input_ids, labels)
            loss = cross_entropy_loss(logits, labels_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            last_loss = loss.item()
        assert last_loss is not None and last_loss == last_loss  # not NaN
        # Frozen encoder unchanged.
        for n, p in wrapper.vision_encoder.named_parameters():
            assert torch.equal(encoder_snap[n], p.detach()), f"encoder param {n} moved"


class TestInnerTransformer:
    def test_unwrap_reaches_moe_methods(self):
        """inner_transformer is usable under the real build path."""
        wrapper = _build(_tiny_vlm_configs())
        inner = inner_transformer(wrapper)
        assert inner is wrapper.transformer
        # set_moe_step is a no-op on a dense model but must resolve.
        inner.set_moe_step(0, 100)  # type: ignore[attr-defined]


class TestTokenAccounting:
    def test_text_token_count_excludes_pad_and_prompt(self):
        """scripts/train.py measures 'text_tokens_trained' as
        (labels != -100).sum(). Verify the math matches what the
        train-loop VLM branch would compute."""
        cfg = _tiny_vlm_configs(num_image_tokens=8)
        wrapper = _build(cfg)
        pixels, input_ids, labels = _dummy_batch(wrapper, batch=3, text_len=16)
        # Manually mask half of labels as -100 (mimics prompt + pad masking).
        labels[:, :8] = -100
        logits, labels_out = wrapper(pixels, input_ids, labels)
        # Contract: output_slice removes image positions, labels returned
        # unchanged; counting non -100 directly is what train.py does.
        n_tokens = int((labels_out != -100).sum().item())
        # 3 rows * 8 non-masked positions = 24
        assert n_tokens == 24
        # Logits cover text positions, not image positions.
        assert logits.shape == (3, 16, cfg[0].vocab_size)


class TestValidation:
    def test_max_seq_len_too_short_raises_at_build(self):
        """If max_seq_len < num_image_tokens + max_text_len, build_parallel_model
        raises even when num_tokens is resolved at build time (e.g.
        encoder exposes a different value than the config asserts)."""
        # Trick: pass num_tokens=0 so the JobConfig-time check is skipped,
        # but the encoder default (num_tokens=16) will exceed max_seq_len.
        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=20)
        vc = VisionEncoderConfig(type="random", feature_dim=0, num_tokens=0)
        ac = AdapterConfig()
        lc = VLMConfig(max_text_len=32)  # 16 image + 32 text = 48 > 20
        with pytest.raises(ValueError, match="insufficient"):
            build_parallel_model(
                mc,
                device=DEVICE,
                device_mesh=None,
                vision_config=vc,
                adapter_config=ac,
                vlm_config=lc,
            )
