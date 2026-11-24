"""A module the build freezes and a freeze stage unfreezes, compiled with inductor on one GPU.

Runs on CUDA only; skipped when ``torch.cuda.is_available()`` is False.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.training.freeze import trainable_for_the_run
from kempnerforge.training.loop import _apply_stage_specs
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")


@pytest.mark.parametrize("compile_model", [False, True])
def test_a_module_a_stage_unfreezes_trains(compile_model: bool) -> None:
    """The adapter is frozen at build and unfrozen by a stage applied as the loop applies
    it. A step before the stage leaves it as it is and a step after trains it, also when
    the model was compiled while the adapter was frozen."""
    stage = FreezeStage(2, (FreezeSpec("adapter", False),))
    vlm = VLMConfig(
        max_text_len=8,
        freeze=[FreezeSpec("vision_encoder"), FreezeSpec("adapter")],
        freeze_schedule=[stage],
    )
    torch.manual_seed(0)
    model = build_parallel_model(
        ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=128, max_seq_len=32),
        torch.device("cuda"),
        None,
        vision_config=VisionEncoderConfig(type="random", feature_dim=32, num_tokens=4),
        adapter_config=AdapterConfig(),
        vlm_config=vlm,
        compile_model=compile_model,
    )
    adapter = getattr(model, "_orig_mod", model).adapter
    with trainable_for_the_run(model, vlm):
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-2))
    before = [p.detach().clone() for p in adapter.parameters()]

    def step(seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        pixels = torch.rand(2, 3, 8, 8, generator=generator).cuda()
        tokens = torch.randint(0, 128, (2, 8), generator=generator).cuda()
        logits, _ = model(pixels, tokens, tokens)
        logits.float().square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

    step(0)
    assert all(torch.equal(p, b) for p, b in zip(adapter.parameters(), before, strict=True))
    _apply_stage_specs(model, stage.specs, vlm.module_patterns)
    step(1)
    assert not any(torch.equal(p, b) for p, b in zip(adapter.parameters(), before, strict=True))
    if compile_model:
        torch.compiler.reset()
