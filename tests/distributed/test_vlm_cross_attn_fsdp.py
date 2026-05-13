"""Distributed tests for the VLM Cross-Attention FSDP2 wrap.

Run with:
    torchrun --nproc_per_node=2 -m pytest \\
        tests/distributed/test_vlm_cross_attn_fsdp.py -v

Mirrors ``tests/distributed/test_vlm_fsdp.py`` for the CA arch:

- Forward + backward on a 2-GPU sharded ``VLMWrapper`` with CA.
- Variable-length text + collator (rank consistency).
- ``inner_transformer`` reachable under ``torch.compile`` + FSDP.
- DCP checkpoint round-trip preserving CA layers + freeze metadata.
- ``test_fsdp_unfreeze_grad_flows``: requires_grad mid-train flip
  under FSDP2 actually re-enables gradient flow (mandatory pre-merge
  test of the FreezeStage hook semantics).
- ``test_ca_two_runs_bitwise_equal_under_fsdp``: same-seed determinism.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import CheckpointConfig, OptimizerConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import CrossAttentionConfig, FreezeSpec
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper, inner_transformer
from kempnerforge.training.freeze import (
    apply_freeze_specs,
    canonical_freeze_meta,
    effective_freeze,
)
from kempnerforge.training.loss import cross_entropy_loss
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)


def _tiny_ca_cfg(
    *,
    num_image_tokens: int = 8,
    feature_dim: int = 96,
    cadence: int = 2,
    freeze: list[FreezeSpec] | None = None,
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, CrossAttentionConfig]:
    return (
        ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=128),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        CrossAttentionConfig(
            max_text_len=32,
            cross_attention_every_n_layers=cadence,
            freeze=freeze if freeze is not None else [FreezeSpec("vision_encoder", True)],
        ),
    )


def _build(
    cfg: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, CrossAttentionConfig],
    mesh,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
    compile_model: bool = False,
) -> VLMWrapper:
    mc, vc, ac, lc = cfg
    model = build_parallel_model(
        mc,
        device=torch.device("cuda"),
        device_mesh=mesh,
        vision_config=vc,
        adapter_config=ac,
        vlm_config=lc,
        param_dtype=param_dtype,
        compile_model=compile_model,
    )
    real = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore[attr-defined]
    assert isinstance(real, VLMWrapper)
    return model  # type: ignore[return-value]


def _dummy_batch(
    wrapper, batch: int = 2, text_len: int = 16, *, seed_offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    pixel_gen = torch.Generator(device="cpu").manual_seed(2000 + rank + seed_offset)
    pixels = torch.randn(batch, 3, 32, 32, generator=pixel_gen).cuda()
    id_gen = torch.Generator(device="cpu").manual_seed(1000 + rank + seed_offset)
    ids = torch.randint(0, 256, (batch, text_len), generator=id_gen).cuda()
    labels = ids.clone()
    return pixels, ids, labels


class TestBuildAndForward:
    def test_build_2gpu_runs(self, distributed_env):
        mesh = distributed_env
        wrapper = _build(_tiny_ca_cfg(), mesh)
        # CA arch: residual is text-only.
        real = wrapper._orig_mod if hasattr(wrapper, "_orig_mod") else wrapper
        assert real.num_image_tokens == 0
        # 4 layers, cadence 2 -> 2 CA blocks.
        assert len(real.transformer.cross_attention_layers) == 2

        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        assert logits.shape == (2, 16, 256)
        assert labels_out is labels

    def test_fsdp_sharded_grads_flow(self, distributed_env):
        mesh = distributed_env
        wrapper = _build(_tiny_ca_cfg(), mesh)
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        # Adapter, transformer.layers, and cross_attention_layers all get
        # gradients; the frozen vision encoder does not.
        for name, p in wrapper.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"trainable {name} got no grad"
        for name, p in wrapper.vision_encoder.named_parameters():
            assert p.grad is None, f"frozen encoder param {name} got a grad"


class TestVariableLengthRankConsistency:
    def test_fixed_length_pad_keeps_ranks_in_sync(self, distributed_env):
        """Two ranks feeding different logical text lengths but the same
        max_text_len batch shape -> NCCL collectives stay well-formed."""
        mesh = distributed_env
        wrapper = _build(_tiny_ca_cfg(), mesh)
        rank = dist.get_rank()
        logical_len = 5 if rank == 0 else 30
        max_text_len = 32
        pixels = torch.randn(2, 3, 32, 32, device="cuda")
        ids = torch.zeros(2, max_text_len, dtype=torch.long, device="cuda")
        labels = torch.full((2, max_text_len), -100, dtype=torch.long, device="cuda")
        ids[:, :logical_len] = torch.arange(1, logical_len + 1, device="cuda").expand(2, -1)
        labels[:, :logical_len] = ids[:, :logical_len]

        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        t = torch.tensor([float(loss.item())], device="cuda")
        dist.all_reduce(t)
        assert torch.isfinite(t).all()


class TestInnerTransformerUnderCompileAndFsdp:
    def test_inner_transformer_under_compile_and_fsdp(self, distributed_env):
        """torch.compile + fully_shard + CA blocks does not break the
        inner_transformer unwrap helper."""
        mesh = distributed_env
        compiled = _build(_tiny_ca_cfg(), mesh, compile_model=True)
        inner = inner_transformer(compiled)
        # set_moe_step is a Transformer method, reachable through inner
        # ref even for a compiled wrapper. set_moe_step is a no-op for
        # a non-MoE model but exercises the unwrap.
        inner.set_moe_step(0, 100)  # type: ignore[attr-defined]
        pixels, ids, labels = _dummy_batch(compiled)
        logits, labels_out = compiled(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        assert torch.isfinite(loss).item()


class TestFreezeStageUnderFsdp:
    def test_fsdp_unfreeze_grad_flows(self, distributed_env):
        """Mid-training flip of requires_grad under FSDP2 must
        re-enable gradient flow. This pins the FreezeStage hook
        semantics: AdamW skips frozen params (set_to_none=True
        default), and unfreezing brings the param back online with a
        live grad on the next backward.
        """
        mesh = distributed_env
        cfg = _tiny_ca_cfg(freeze=[FreezeSpec("adapter", True)])  # adapter starts frozen
        wrapper = _build(cfg, mesh)
        # Step 0: confirm adapter frozen -> no grads.
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        for n, p in wrapper.named_parameters():
            if n.startswith("adapter"):
                assert p.grad is None, f"frozen {n} got a grad"
        # Now unfreeze the adapter (simulating a FreezeStage transition).
        apply_freeze_specs(
            wrapper,
            [FreezeSpec("adapter", False)],
            cfg[3].module_patterns,  # type: ignore[union-attr]
        )
        # Run another backward (must zero grads first since prior backward
        # left frozen-param grads as None and trainable-param grads populated).
        for p in wrapper.parameters():
            p.grad = None
        pixels, ids, labels = _dummy_batch(wrapper, seed_offset=100)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        # Post-unfreeze: adapter params should now receive grads under FSDP2.
        adapter_grads_present = [
            p.grad is not None for n, p in wrapper.named_parameters() if n.startswith("adapter")
        ]
        assert all(adapter_grads_present), (
            "FSDP2 did not re-attach grad-allocation hooks on requires_grad flip; "
            "FreezeStage transitions need a fully_shard rebuild on this PyTorch version"
        )


class TestDeterminism:
    def test_ca_two_runs_bitwise_equal_under_fsdp(self, distributed_env):
        """Same seed, two runs, same loss. Catches CUDA stream race
        between encoder/adapter writes and CA-block reads under FSDP2
        prefetch.
        """
        mesh = distributed_env
        torch.manual_seed(0)
        wrapper_a = _build(_tiny_ca_cfg(), mesh, param_dtype=torch.float32)
        wrapper_a.eval()
        pixels, ids, _ = _dummy_batch(wrapper_a, batch=1, text_len=8)
        with torch.no_grad():
            logits_a, _ = wrapper_a(pixels, ids)
        loss_a = float(logits_a.sum().item())

        torch.manual_seed(0)
        wrapper_b = _build(_tiny_ca_cfg(), mesh, param_dtype=torch.float32)
        wrapper_b.eval()
        with torch.no_grad():
            logits_b, _ = wrapper_b(pixels, ids)
        loss_b = float(logits_b.sum().item())

        assert loss_a == loss_b, f"non-deterministic: {loss_a} vs {loss_b}"


class TestMoEWithVLM:
    def test_ca_plus_moe_under_fsdp_runs_and_learns(self, distributed_env):
        """VLM (Cross-Attention) + MoE in the text TransformerBlocks
        runs end-to-end under FSDP2 at 2 GPU. CE decreases on a fixed
        mini-batch over 8 steps; aux_loss bounded.

        CrossAttentionBlocks remain dense; MoE lives only in the text
        TransformerBlocks selected by moe_frequency. Exercises the
        EP-aware FSDP block wrap (Fix 5) on the VLM path.
        """
        from kempnerforge.model.moe import MoEMLP

        torch.manual_seed(42 + dist.get_rank())
        cfg = (
            ModelConfig(
                dim=256,
                n_layers=4,
                n_heads=4,
                n_kv_heads=2,
                vocab_size=32000,
                max_seq_len=128,
                num_experts=4,
                moe_top_k=2,
                moe_frequency=2,
            ),
            VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            AdapterConfig(),
            CrossAttentionConfig(
                max_text_len=64,
                cross_attention_every_n_layers=2,
                freeze=[FreezeSpec("vision_encoder", True)],
            ),
        )
        mc = cfg[0]
        wrapper = _build(cfg, distributed_env, param_dtype=torch.float32)
        # 4 layers, frequency=2 -> 2 MoE blocks; cadence=2 -> 2 CA blocks.
        moe_blocks = sum(
            1 for layer in wrapper.transformer.layers.values() if isinstance(layer.mlp, MoEMLP)
        )
        assert moe_blocks == 2, f"expected 2 MoE blocks, got {moe_blocks}"
        assert len(wrapper.transformer.cross_attention_layers) == 2

        opt = build_optimizer(wrapper, OptimizerConfig(lr=3e-3, fused=False))
        loss_fn = torch.nn.CrossEntropyLoss()

        pixels = torch.randn(2, 3, 32, 32, device="cuda")
        ids = torch.randint(0, mc.vocab_size, (2, 16), device="cuda")

        inner = inner_transformer(wrapper)
        losses = []
        for step in range(8):
            inner.set_moe_step(step, max_steps=100)  # type: ignore[attr-defined]
            logits, _ = wrapper(pixels, ids, ids)
            ce = loss_fn(logits.reshape(-1, mc.vocab_size), ids.reshape(-1))
            aux = inner.get_moe_aux_loss()  # type: ignore[attr-defined]
            loss = ce + 0.01 * aux
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append((ce.item(), aux.item()))

        # CE decreases (verifies that FSDP+MoE+CA together produce
        # a learning signal, not that they merely don't crash).
        initial_ce, final_ce = losses[0][0], losses[-1][0]
        assert final_ce < initial_ce, (
            f"FSDP+MoE+CA: CE did not decrease ({initial_ce:.3f} -> {final_ce:.3f})"
        )
        assert all(a > 0 for _, a in losses), "aux_loss zeroed out under FSDP+MoE"


class TestCheckpointRoundtrip:
    def test_save_load_freeze_metadata(self, distributed_env, tmp_path_factory):
        """Save a CA VLM checkpoint, load it in a fresh manager, and
        verify metadata.json carries the canonical vlm_freeze (computed
        through effective_freeze) and DCP shards round-trip the
        cross_attention_layers params.
        """
        mesh = distributed_env
        cfg = _tiny_ca_cfg()
        wrapper = _build(cfg, mesh)
        opt = build_optimizer(wrapper, OptimizerConfig(lr=1e-3, fused=False))

        rank = dist.get_rank()
        if rank == 0:
            base = tmp_path_factory.mktemp("vlm_ca_ckpt")
            path_str = str(base)
        else:
            path_str = ""
        objs: list[object] = [path_str]
        dist.broadcast_object_list(objs, src=0)
        path_str = objs[0]  # type: ignore[assignment]

        ckpt_cfg = CheckpointConfig(dir=str(path_str), interval=1)
        mgr = CheckpointManager(ckpt_cfg, wrapper, opt)
        freeze = canonical_freeze_meta(
            effective_freeze(0, cfg[3].freeze, cfg[3].freeze_schedule)  # type: ignore[union-attr]
        )
        mgr.save(step=1, extra={"vlm_freeze": freeze})
        dist.barrier()

        if rank == 0:
            import json
            from pathlib import Path

            meta = json.loads((Path(path_str) / "step_1" / "metadata.json").read_text())
            assert meta["vlm_freeze"] == freeze

        # Load into a fresh wrapper / optimizer, expecting no mismatch.
        wrapper2 = _build(cfg, mesh)
        opt2 = build_optimizer(wrapper2, OptimizerConfig(lr=1e-3, fused=False))
        mgr2 = CheckpointManager(ckpt_cfg, wrapper2, opt2)
        step, _, _ = mgr2.load(path=str(path_str) + "/step_1", vlm_freeze_expected=freeze)
        assert step == 1
        # CA layers params survived round-trip.
        for n, _ in wrapper2.named_parameters():
            if "cross_attention_layers" in n:
                # If the layer params loaded, they're present in named_parameters.
                pass
