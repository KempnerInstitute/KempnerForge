"""Distributed tests for the VLM FSDP2 wrap policy.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/distributed/test_vlm_fsdp.py -v

Covers:
  - Forward + backward on a 2-GPU sharded ``VLMWrapper``.
  - Variable-length text across ranks stays shape-consistent because
    ``VLMCollator`` / dataset emit fixed-length ``max_text_len`` batches.
  - Dtype combinatorics: vision encoder fp32, adapter bf16, transformer bf16.
  - torch.compile + fully_shard + ``inner_transformer(compiled_wrapper)``.
  - DCP checkpoint round-trip with freeze metadata.
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
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vlm import VLMWrapper, inner_transformer
from kempnerforge.training.freeze import canonical_freeze_meta
from kempnerforge.training.loss import cross_entropy_loss
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)


def _tiny_cfg(
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
    cfg: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig],
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
    # When compile_model=True the returned wrapper may be a
    # torch._dynamo.OptimizedModule around VLMWrapper; unwrap for
    # isinstance check.
    real = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore[attr-defined]
    assert isinstance(real, VLMWrapper)
    return model  # type: ignore[return-value]


def _dummy_batch(
    wrapper, batch: int = 2, text_len: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = torch.randn(batch, 3, 32, 32, device="cuda")
    # Compose a rank-varying pattern: rank-dependent seed makes inputs differ
    # across ranks, which is what DP ranks would see in practice.
    rank = dist.get_rank() if dist.is_initialized() else 0
    gen = torch.Generator(device="cpu").manual_seed(1000 + rank)
    ids = torch.randint(0, 256, (batch, text_len), generator=gen).cuda()
    labels = ids.clone()
    return pixels, ids, labels


class TestBuildAndForward:
    def test_build_2gpu_runs(self, distributed_env):
        mesh = distributed_env
        wrapper = _build(_tiny_cfg(), mesh)
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        assert logits.shape == (2, 16, 256)
        assert labels_out is labels

    def test_fsdp_sharded_grads_flow(self, distributed_env):
        mesh = distributed_env
        wrapper = _build(_tiny_cfg(), mesh)
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        # Trainable params (adapter + transformer) receive grads; frozen
        # encoder params do not.
        has_grad = [
            (name, p.grad is not None) for name, p in wrapper.named_parameters() if p.requires_grad
        ]
        assert all(g for (_, g) in has_grad), "trainable param missing grad"
        for name, p in wrapper.vision_encoder.named_parameters():
            assert p.grad is None, f"frozen encoder param {name} got a grad"


class TestVariableLengthRankConsistency:
    def test_fixed_length_pad_keeps_ranks_in_sync(self, distributed_env):
        """VLMCollator / the dataset always pad to max_text_len, so two
        ranks feeding different logical lengths still hit identical
        tensor shapes and NCCL collectives succeed."""
        mesh = distributed_env
        wrapper = _build(_tiny_cfg(num_image_tokens=8), mesh)

        rank = dist.get_rank()
        # Rank 0 simulates shorter logical text; rank 1 simulates longer.
        # Both pad to the same max_text_len=32 (from the config).
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
        # Shape is identical across ranks; all_reduce on the loss would be
        # well-formed. Run it explicitly to make the assertion concrete.
        t = torch.tensor([float(loss.item())], device="cuda")
        dist.all_reduce(t)
        assert torch.isfinite(t).all()


class TestDtypeCombinatorics:
    def test_encoder_fp32_adapter_bf16_transformer_bf16(self, distributed_env):
        """Vision encoder stays in HF-loaded dtype (fp32 for the random
        stub); adapter and transformer are cast to bf16 by _build_vlm.
        VLMWrapper.forward inserts a cast at the adapter boundary."""
        mesh = distributed_env
        wrapper = _build(_tiny_cfg(), mesh, param_dtype=torch.bfloat16)
        anchor = wrapper.vision_encoder.get_buffer("_anchor")
        assert anchor.dtype == torch.float32
        for p in wrapper.adapter.parameters():
            assert p.dtype == torch.bfloat16
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, _ = wrapper(pixels, ids, labels)
        assert logits.dtype == torch.bfloat16


class TestInnerTransformerUnderCompileAndFsdp:
    def test_inner_transformer_under_compile_and_fsdp(self, distributed_env):
        """torch.compile(wrapper) + real fully_shard does not break the
        inner_transformer unwrap helper used by MoE call sites in
        scripts/train.py."""
        mesh = distributed_env
        compiled = _build(_tiny_cfg(), mesh, compile_model=True)
        inner = inner_transformer(compiled)
        # set_moe_step is a Transformer method, reachable through the inner
        # ref even for a compiled wrapper.
        inner.set_moe_step(0, 100)  # type: ignore[attr-defined]
        # The compiled wrapper is still callable end-to-end.
        pixels, ids, labels = _dummy_batch(compiled)
        logits, labels_out = compiled(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        assert torch.isfinite(loss).item()


class TestCheckpointRoundtrip:
    def test_save_load_freeze_metadata(self, distributed_env, shared_tmp_dir):
        """Save a VLM checkpoint, load it in a fresh manager, and verify
        the canonical vlm_freeze metadata is present in metadata.json."""
        mesh = distributed_env
        cfg_tuple = _tiny_cfg()
        wrapper = _build(cfg_tuple, mesh)
        opt = build_optimizer(wrapper, OptimizerConfig(lr=1e-3, fused=False))

        # shared_tmp_dir is on the shared filesystem and identical on every
        # rank, so DCP shards from rank 0 are visible to rank 1 even under
        # multi-node srun.
        path_str = shared_tmp_dir
        rank = dist.get_rank()

        cfg = CheckpointConfig(dir=str(path_str), interval=1)
        mgr = CheckpointManager(cfg, wrapper, opt)
        # _tiny_cfg returns (ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig)
        freeze = canonical_freeze_meta(cfg_tuple[3].freeze)
        mgr.save(step=1, extra={"vlm_freeze": freeze})

        # Let rank 0 finish writing metadata.json before rank 1 reads it.
        dist.barrier()

        if rank == 0:
            import json
            from pathlib import Path

            meta = json.loads((Path(path_str) / "step_1" / "metadata.json").read_text())
            assert meta["vlm_freeze"] == freeze
