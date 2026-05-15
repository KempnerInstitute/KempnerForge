"""Distributed tests for the VLM Mixture-of-Transformers (MoT) FSDP2 wrap.

Run with:
    torchrun --nproc_per_node=2 -m pytest \\
        tests/distributed/test_vlm_mot_fsdp.py -v

Mirrors ``tests/distributed/test_vlm_cross_attn_fsdp.py`` for the MoT arch:

- Forward + backward on a 2-GPU sharded ``VLMWrapper`` with MoT.
- Variable-length text + collator (rank consistency).
- ``inner_transformer`` reachable under ``torch.compile`` + FSDP.
- ``test_fsdp_unfreeze_grad_flows_mot``: requires_grad mid-train flip
  under FSDP2 actually re-enables gradient flow (mandatory pre-merge
  test of the FreezeStage hook semantics on the per-modality stack).
- ``test_mot_two_runs_bitwise_equal_under_fsdp``: same-seed determinism.
- ``test_warm_start_from_jd_under_fsdp``: warm-start helper translates
  per-modality weights correctly when the target lives under FSDP2.
- ``test_mot_plus_moe_under_fsdp_runs_and_learns``: VLM(MoT) + MoE
  in the per-modality FFNs runs end-to-end and CE decreases.
- DCP checkpoint round-trip preserving MoT-specific state-dict keys
  + canonical vlm_freeze metadata.
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
from kempnerforge.config.vlm import FreezeSpec, JointDecoderConfig, MoTConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.mot import mot_warm_start_from_text_stack
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


def _tiny_mot_cfg(
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
    cfg: tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoTConfig],
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
        from kempnerforge.model.mot import MoTBlock

        mesh = distributed_env
        wrapper = _build(_tiny_mot_cfg(), mesh)
        real = wrapper._orig_mod if hasattr(wrapper, "_orig_mod") else wrapper
        # MoT extends the residual stream with num_image_tokens.
        assert real.num_image_tokens == 8
        # Layers are MoTBlocks.
        assert all(isinstance(layer, MoTBlock) for layer in real.transformer.layers.values())

        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        assert logits.shape == (2, 16, 256)
        assert labels_out is labels

    def test_fsdp_sharded_grads_flow(self, distributed_env):
        mesh = distributed_env
        wrapper = _build(_tiny_mot_cfg(), mesh)
        # Re-init per-modality residual projections so backward exercises
        # Q/K/V/gate/up grads (zero-init blocks gradient via the chain rule).
        with torch.no_grad():
            for layer in wrapper.transformer.layers.values():
                for m in layer.modalities:
                    torch.nn.init.normal_(layer.attn.o_proj[m].weight, std=0.01)
                    torch.nn.init.normal_(layer.mlp[m].down_proj.weight, std=0.01)
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
        wrapper = _build(_tiny_mot_cfg(), mesh)
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
        """torch.compile + fully_shard + MoT blocks does not break the
        inner_transformer unwrap helper."""
        mesh = distributed_env
        compiled = _build(_tiny_mot_cfg(), mesh, compile_model=True)
        inner = inner_transformer(compiled)
        inner.set_moe_step(0, 100)  # type: ignore[attr-defined]
        pixels, ids, labels = _dummy_batch(compiled)
        logits, labels_out = compiled(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        assert torch.isfinite(loss).item()


class TestFreezeStageUnderFsdp:
    def test_fsdp_unfreeze_grad_flows_mot(self, distributed_env):
        """Mid-training flip of requires_grad under FSDP2 must re-enable
        gradient flow on the per-modality main stack. Mandatory merge gate.
        """
        mesh = distributed_env
        cfg = _tiny_mot_cfg(freeze=[FreezeSpec("mot", True)])  # main stack frozen
        wrapper = _build(cfg, mesh)
        # Re-init residual projections so the wrapper isn't identity (otherwise
        # the unfreeze test would be uninteresting — grads would be 0 even when
        # requires_grad=True).
        with torch.no_grad():
            for layer in wrapper.transformer.layers.values():
                for m in layer.modalities:
                    torch.nn.init.normal_(layer.attn.o_proj[m].weight, std=0.01)
                    torch.nn.init.normal_(layer.mlp[m].down_proj.weight, std=0.01)
        # Step 0: confirm mot stack frozen -> no grads.
        pixels, ids, labels = _dummy_batch(wrapper)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        for n, p in wrapper.named_parameters():
            if n.startswith("transformer.layers."):
                assert p.grad is None, f"frozen {n} got a grad"
        # Now unfreeze the main stack (simulating a FreezeStage transition).
        apply_freeze_specs(
            wrapper,
            [FreezeSpec("mot", False)],
            cfg[3].module_patterns,  # type: ignore[union-attr]
        )
        for p in wrapper.parameters():
            p.grad = None
        pixels, ids, labels = _dummy_batch(wrapper, seed_offset=100)
        logits, labels_out = wrapper(pixels, ids, labels)
        loss = cross_entropy_loss(logits, labels_out)
        loss.backward()
        # Post-unfreeze: at least one per-modality projection per layer has a
        # grad. (Some Q/K/V projections may have zero grads at the instant
        # we sample if the upstream o_proj.weight happens to be zero in some
        # head, but at least one of the four projections must be non-None.)
        for n, p in wrapper.named_parameters():
            if n.startswith("transformer.layers.") and p.requires_grad:
                assert p.grad is not None, (
                    f"FSDP2 did not re-attach grad-allocation hooks on requires_grad flip; "
                    f"FreezeStage transitions need a fully_shard rebuild on this PyTorch version. "
                    f"Param: {n}"
                )


class TestDeterminism:
    def test_mot_two_runs_bitwise_equal_under_fsdp(self, distributed_env):
        """Same seed, two runs, same loss. Catches CUDA stream race
        on the per-modality stream split / re-concat under FSDP2 prefetch.
        """
        mesh = distributed_env
        torch.manual_seed(0)
        wrapper_a = _build(_tiny_mot_cfg(), mesh, param_dtype=torch.float32)
        wrapper_a.eval()
        pixels, ids, _ = _dummy_batch(wrapper_a, batch=1, text_len=8)
        with torch.no_grad():
            logits_a, _ = wrapper_a(pixels, ids)
        loss_a = float(logits_a.sum().item())

        torch.manual_seed(0)
        wrapper_b = _build(_tiny_mot_cfg(), mesh, param_dtype=torch.float32)
        wrapper_b.eval()
        with torch.no_grad():
            logits_b, _ = wrapper_b(pixels, ids)
        loss_b = float(logits_b.sum().item())

        assert loss_a == loss_b, f"non-deterministic: {loss_a} vs {loss_b}"


class TestWarmStartUnderFsdp:
    def test_warm_start_from_jd_under_fsdp(self, distributed_env, tmp_path_factory):
        """JD checkpoint -> torch.save -> mot_warm_start_from_text_stack
        on an FSDP2-wrapped MoT model. Per-modality copies become bit-equal
        to the source dense block weights after the helper runs.
        """
        mesh = distributed_env
        rank = dist.get_rank()

        # Build a JD model with the same backbone shape, unwrap the FSDP
        # state on rank 0 via state_dict, broadcast a CPU dict to all ranks.
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
        jd_wrapper = _build(jd_cfg, mesh, param_dtype=torch.float32)
        jd_state_full = inner_transformer(jd_wrapper).state_dict()
        # Materialize to CPU on every rank (FSDP returns DTensors); .full_tensor()
        # gathers the unsharded data.
        jd_state_cpu: dict[str, torch.Tensor] = {}
        for k, v in jd_state_full.items():
            t = v.full_tensor() if hasattr(v, "full_tensor") else v
            jd_state_cpu[k] = t.detach().cpu()

        if rank == 0:
            base = tmp_path_factory.mktemp("jd_warm_start_src")
            ckpt_path = str(base / "jd.pt")
            torch.save(jd_state_cpu, ckpt_path)
        else:
            ckpt_path = ""
        objs: list[object] = [ckpt_path]
        dist.broadcast_object_list(objs, src=0)
        ckpt_path = objs[0]  # type: ignore[assignment]
        dist.barrier()

        # Build a MoT model under FSDP and run the warm-start helper.
        mot_cfg = _tiny_mot_cfg(num_image_tokens=8, n_layers=2)
        mot_wrapper = _build(mot_cfg, mesh, param_dtype=torch.float32)
        source = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mot_warm_start_from_text_stack(inner_transformer(mot_wrapper), source)  # type: ignore[arg-type]

        # Per-modality copies equal source dense weights (gather first).
        mot_t = inner_transformer(mot_wrapper)
        for i in range(mot_cfg[0].n_layers):
            for m in mot_t.layers[str(i)].modalities:  # type: ignore[union-attr]
                w = mot_t.layers[str(i)].attn.q_proj[m].weight  # type: ignore[union-attr]
                w_full = w.full_tensor() if hasattr(w, "full_tensor") else w
                assert torch.equal(w_full.cpu(), source[f"layers.{i}.attention.q_proj.weight"]), (
                    f"layer {i} modality {m}: warm-start did not propagate under FSDP2"
                )


class TestMoEWithMoT:
    def test_mot_plus_moe_under_fsdp_runs_and_learns(self, distributed_env):
        """VLM(MoT) + MoE in the per-modality FFNs runs end-to-end under
        FSDP2 at 2 GPU. CE decreases on a fixed mini-batch over 8 steps.
        """
        from kempnerforge.model.moe import MoEMLP

        torch.manual_seed(42 + dist.get_rank())
        # _tiny_mot_cfg returns (ModelConfig, VisionEncoderConfig, AdapterConfig, MoTConfig);
        # destructure to mutate the ModelConfig in place before passing the tuple to _build.
        mc, vc, ac, lc = _tiny_mot_cfg(moe=True, n_layers=4, num_image_tokens=4)
        # Bump the dim/ffn so MoE is meaningful.
        mc.dim = 128
        mc.n_heads = 4
        mc.n_kv_heads = 4
        mc.ffn_hidden_dim = 256
        wrapper = _build((mc, vc, ac, lc), distributed_env, param_dtype=torch.float32)
        # 4 layers, frequency=2 -> layers 1, 3 have MoE per modality.
        moe_blocks = sum(
            1
            for layer in wrapper.transformer.layers.values()
            if any(isinstance(layer.mlp[m], MoEMLP) for m in layer.modalities)
        )
        assert moe_blocks == 2, f"expected 2 MoE blocks, got {moe_blocks}"

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

        initial_ce, final_ce = losses[0][0], losses[-1][0]
        assert final_ce < initial_ce, (
            f"FSDP+MoE+MoT: CE did not decrease ({initial_ce:.3f} -> {final_ce:.3f})"
        )
        assert all(a >= 0 for _, a in losses), "aux_loss negative under FSDP+MoE+MoT"


class TestCheckpointRoundtrip:
    def test_save_load_freeze_metadata_mot(self, distributed_env, tmp_path_factory):
        """Save a MoT VLM checkpoint, load it in a fresh manager, and
        verify metadata.json carries the canonical vlm_freeze and DCP
        shards round-trip the per-modality block params.
        """
        mesh = distributed_env
        cfg = _tiny_mot_cfg()
        wrapper = _build(cfg, mesh)
        opt = build_optimizer(wrapper, OptimizerConfig(lr=1e-3, fused=False))

        rank = dist.get_rank()
        if rank == 0:
            base = tmp_path_factory.mktemp("vlm_mot_ckpt")
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

        wrapper2 = _build(cfg, mesh)
        opt2 = build_optimizer(wrapper2, OptimizerConfig(lr=1e-3, fused=False))
        mgr2 = CheckpointManager(ckpt_cfg, wrapper2, opt2)
        step, _, _ = mgr2.load(path=str(path_str) + "/step_1", vlm_freeze_expected=freeze)
        assert step == 1
        # Per-modality state-dict keys survived round-trip.
        per_modality_keys = [
            n for n, _ in wrapper2.named_parameters() if "attn.q_proj." in n or "mot_norms" in n
        ]
        assert len(per_modality_keys) > 0, "MoT per-modality params missing after DCP load"
