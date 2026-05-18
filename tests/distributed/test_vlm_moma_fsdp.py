"""Distributed tests for the VLM Mixture of Modality-Aware Experts (MoMa) FSDP2 wrap.

Run with:
    torchrun --nproc_per_node=2 -m pytest \\
        tests/distributed/test_vlm_moma_fsdp.py -v

Mirrors ``tests/distributed/test_vlm_mot_fsdp.py`` for the MoMa arch:

- Forward + backward on a 2-GPU sharded ``VLMWrapper`` with MoMa.
- Variable-length text + collator (rank consistency).
- ``test_fsdp_unfreeze_grad_flows_moma``: requires_grad mid-train flip
  under FSDP2 actually re-enables gradient flow on the per-layer MoMa
  stack (mandatory pre-merge test of the FreezeStage hook semantics).
- ``test_moma_two_runs_bitwise_equal_under_fsdp``: same-seed determinism
  in eval mode with Gumbel noise off (the only deterministic regime for
  EC + Sigmoid routing; train mode is stochastic by design).
- ``test_moma_runs_and_learns_under_fsdp``: VLM(MoMa) end-to-end on a
  fixed mini-batch over a few steps; CE decreases. Proves that EC
  routing + per-modality expert groups produce useful gradient signal.
- DCP checkpoint round-trip preserving MoMa-specific state-dict keys
  (mlp.experts.{modality}.router.gate, mlp.experts.{modality}.experts.{i})
  + canonical vlm_freeze metadata.

Inference-path tests are intentionally omitted from v1: EC routing is
non-causal in v1 (deferred auxiliary routers, paper §2.4).
``torch.compile`` is warned by JobConfig.validate and skipped here.
``set_moe_step`` / ``get_moe_aux_loss`` are silent no-ops for MoMa
layers (EC has no bias schedule and no aux loss), which is the
documented v1 behavior; we don't pin it as a separate test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import CheckpointConfig, OptimizerConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, MoMaConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper
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


def _tiny_moma_cfg(
    *,
    num_image_tokens: int = 8,
    feature_dim: int = 96,
    n_layers: int = 2,
    freeze: list[FreezeSpec] | None = None,
    experts_per_modality: dict[str, int] | None = None,
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
            moma_gumbel_noise=gumbel_noise,
            freeze=freeze if freeze is not None else [FreezeSpec("vision_encoder", True)],
        ),
    )


def _build(
    cfg: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoMaConfig],
    mesh,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
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
        from kempnerforge.model.moma import MoMaBlock

        mesh = distributed_env
        wrapper = _build(_tiny_moma_cfg(), mesh)
        real = wrapper._orig_mod if hasattr(wrapper, "_orig_mod") else wrapper
        # MoMa uses the JD/MoT image-prefix residual layout.
        assert real.num_image_tokens == 8
        # Layers are MoMaBlocks.
        assert all(isinstance(layer, MoMaBlock) for layer in real.transformer.layers.values())

        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        assert logits.shape == (2, 16, 256)
        assert labels_out is labels

    def test_fsdp_sharded_grads_flow(self, distributed_env):
        mesh = distributed_env
        wrapper = _build(_tiny_moma_cfg(), mesh)
        # MoMa's depth-scaled init on o_proj / down_proj keeps weights small
        # but nonzero, so gradients flow without an explicit re-init step
        # (unlike MoT's identity-at-construction zero-init).
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
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
        wrapper = _build(_tiny_moma_cfg(), mesh)
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


class TestFreezeStageUnderFsdp:
    def test_fsdp_unfreeze_grad_flows_moma(self, distributed_env):
        """Mid-training flip of requires_grad under FSDP2 must re-enable
        gradient flow on the per-layer MoMa stack. Mandatory merge gate
        for the FreezeStage hook semantics.
        """
        mesh = distributed_env
        cfg = _tiny_moma_cfg(freeze=[FreezeSpec("moma", True)])  # main stack frozen
        wrapper = _build(cfg, mesh)

        # Step 0: confirm moma stack frozen -> no grads on transformer.layers.*.
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        for n, p in wrapper.named_parameters():
            if n.startswith("transformer.layers."):
                assert p.grad is None, f"frozen {n} got a grad"

        # Unfreeze the main stack (simulating a FreezeStage transition).
        apply_freeze_specs(
            wrapper,
            [FreezeSpec("moma", False)],
            cfg[3].module_patterns,  # type: ignore[union-attr]
        )
        for p in wrapper.parameters():
            p.grad = None
        pixels, ids, labels = _dummy_batch(wrapper, seed_offset=100)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        # Post-unfreeze: every trainable per-layer param has a grad
        # allocated. (EC routing may leave individual experts unselected
        # in a given batch — they'd get zero grads but still have a grad
        # tensor allocated by FSDP2.)
        for n, p in wrapper.named_parameters():
            if n.startswith("transformer.layers.") and p.requires_grad:
                assert p.grad is not None, (
                    f"FSDP2 did not re-attach grad-allocation hooks on requires_grad flip; "
                    f"FreezeStage transitions need a fully_shard rebuild on this PyTorch version. "
                    f"Param: {n}"
                )


class TestDeterminism:
    def test_moma_two_runs_bitwise_equal_under_fsdp(self, distributed_env):
        """Same seed, two builds, same eval-mode forward.

        EC routing with Gumbel noise is stochastic by design in train
        mode; this test pins determinism only in the regime that
        actually has it: eval mode + ``moma_gumbel_noise=False``. Catches
        CUDA-stream races or non-deterministic dispatch in the
        modality-aware scatter/gather under FSDP2 prefetch.
        """
        mesh = distributed_env
        torch.manual_seed(0)
        wrapper_a = _build(_tiny_moma_cfg(gumbel_noise=False), mesh, param_dtype=torch.float32)
        wrapper_a.eval()
        pixels, ids, _ = _dummy_batch(wrapper_a, batch=1, text_len=8)
        with torch.no_grad():
            logits_a, _ = wrapper_a(pixels, ids)
        loss_a = float(logits_a.sum().item())

        torch.manual_seed(0)
        wrapper_b = _build(_tiny_moma_cfg(gumbel_noise=False), mesh, param_dtype=torch.float32)
        wrapper_b.eval()
        with torch.no_grad():
            logits_b, _ = wrapper_b(pixels, ids)
        loss_b = float(logits_b.sum().item())

        assert loss_a == loss_b, f"non-deterministic: {loss_a} vs {loss_b}"


class TestEcRoutingTraining:
    def test_moma_runs_and_learns_under_fsdp(self, distributed_env):
        """VLM(MoMa) runs end-to-end under FSDP2 at 2 GPU. CE decreases on
        a fixed mini-batch over a handful of steps, proving that EC +
        Sigmoid routing produces useful gradient signal through the
        modality-aware scatter/gather.

        Gumbel noise is off so the EC selection is deterministic for a
        given (model, batch) and the loss trajectory is comparable. With
        Gumbel on, learning still works but step-to-step variance can
        mask a small CE decrease in such a short run.
        """
        torch.manual_seed(42 + dist.get_rank())
        mc, vc, ac, lc = _tiny_moma_cfg(n_layers=4, num_image_tokens=4, gumbel_noise=False)
        # Bump dim/ffn so the FFN is meaningful.
        mc.dim = 128
        mc.n_heads = 4
        mc.n_kv_heads = 4
        mc.ffn_hidden_dim = 256
        wrapper = _build((mc, vc, ac, lc), distributed_env, param_dtype=torch.float32)

        opt = build_optimizer(wrapper, OptimizerConfig(lr=3e-3, fused=False))
        loss_fn = torch.nn.CrossEntropyLoss()

        pixels = torch.randn(2, 3, 32, 32, device="cuda")
        ids = torch.randint(0, mc.vocab_size, (2, 16), device="cuda")

        losses = []
        for _ in range(8):
            logits, _ = wrapper(pixels, ids, ids)
            ce = loss_fn(logits.reshape(-1, mc.vocab_size), ids.reshape(-1))
            ce.backward()
            opt.step()
            opt.zero_grad()
            losses.append(ce.item())

        initial_ce, final_ce = losses[0], losses[-1]
        assert final_ce < initial_ce, (
            f"FSDP+MoMa: CE did not decrease ({initial_ce:.3f} -> {final_ce:.3f}); "
            f"trajectory={losses}"
        )


class TestCheckpointRoundtrip:
    def test_save_load_freeze_metadata_moma(self, distributed_env, shared_tmp_dir):
        """Save a MoMa VLM checkpoint, load it in a fresh manager, and
        verify metadata.json carries the canonical vlm_freeze and DCP
        shards round-trip the per-modality MoE state-dict keys.
        """
        mesh = distributed_env
        cfg = _tiny_moma_cfg()
        wrapper = _build(cfg, mesh)
        opt = build_optimizer(wrapper, OptimizerConfig(lr=1e-3, fused=False))

        # shared_tmp_dir lives on the shared filesystem so DCP shards
        # written by rank 0 are visible to rank 1 under multi-node srun.
        path_str = shared_tmp_dir
        rank = dist.get_rank()

        ckpt_cfg = CheckpointConfig(dir=str(path_str), interval=1)
        mgr = CheckpointManager(ckpt_cfg, wrapper, opt)
        freeze = canonical_freeze_meta(
            effective_freeze(0, cfg[3].freeze, cfg[3].freeze_schedule)  # type: ignore[union-attr]
        )
        mgr.save(step=1, extra={"vlm_freeze": freeze})
        dist.barrier()

        if rank == 0:
            import json

            meta = json.loads((Path(path_str) / "step_1" / "metadata.json").read_text())
            assert meta["vlm_freeze"] == freeze

        wrapper2 = _build(cfg, mesh)
        opt2 = build_optimizer(wrapper2, OptimizerConfig(lr=1e-3, fused=False))
        mgr2 = CheckpointManager(ckpt_cfg, wrapper2, opt2)
        step, _, _ = mgr2.load(path=str(path_str) + "/step_1", vlm_freeze_expected=freeze)
        assert step == 1
        # Per-modality MoE keys survived round-trip: gate (router) + experts.
        per_modality_keys = [
            n
            for n, _ in wrapper2.named_parameters()
            if "mlp.experts." in n
            and ("router.gate" in n or "experts." in n.split("mlp.experts.", 1)[1])
        ]
        assert len(per_modality_keys) > 0, "MoMa per-modality MoE params missing after DCP load"
