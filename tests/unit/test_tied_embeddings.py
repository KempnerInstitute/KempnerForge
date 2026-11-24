"""A tied token embedding and output head stay tied through every build path (#562)."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

import kempnerforge.distributed.parallel as parallel
from kempnerforge.config.model import ModelConfig
from kempnerforge.distributed.parallel import apply_float8, build_parallel_model
from kempnerforge.model.transformer import Transformer


@pytest.fixture
def info_logged() -> list[str]:
    """Info lines the parallel module logs, captured on its own logger."""
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    log = logging.getLogger(parallel.__name__)
    handler, level = Collect(level=logging.INFO), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    yield messages
    log.removeHandler(handler)
    log.setLevel(level)


def _config(tie: bool = True) -> ModelConfig:
    return ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64, tie_embeddings=tie
    )


def _tied(model: Transformer) -> bool:
    return model.output_head.proj.weight is model.token_embedding.embedding.weight


def _values(weight: torch.Tensor) -> torch.Tensor:
    full = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    return full.detach().float().clone()


def _build_and_step(
    rank: int, world: int, store: str, shape: tuple[int, ...], names, dtype: torch.dtype
) -> None:
    """Build on a gloo mesh, check the tie, and take an optimizer step on the shared weight."""
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"file://{store}",
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
        torch.manual_seed(0)
        model = build_parallel_model(_config(), torch.device("cpu"), mesh, param_dtype=dtype)
        assert _tied(model)
        assert len(list(model.parameters())) == len(list(Transformer(_config()).parameters()))
        weight = model.token_embedding.embedding.weight
        assert weight.dtype == dtype
        before = _values(weight)
        assert torch.isfinite(before).all()
        assert abs(before.std().item() - _config().init_std) < 0.002
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        assert sum(p is weight for group in optimizer.param_groups for p in group["params"]) == 1
        tokens = torch.randint(0, 256, (2, 16))
        F.cross_entropy(model(tokens).flatten(0, 1).float(), tokens.flatten()).backward()
        optimizer.step()
        assert _tied(model)
        assert not torch.equal(_values(model.output_head.proj.weight), before)
    finally:
        dist.destroy_process_group()


def _spawn(store: Path, shape: tuple[int, ...], names: tuple[str, ...], dtype: torch.dtype) -> None:
    """Run ``_build_and_step`` on every rank; stop them if they have not finished in 2 minutes."""
    world = 1
    for size in shape:
        world *= size
    ranks = mp.spawn(
        _build_and_step, args=(world, str(store), shape, names, dtype), nprocs=world, join=False
    )
    deadline = time.monotonic() + 120
    try:
        while not ranks.join(timeout=5):
            assert time.monotonic() < deadline, "the ranks did not finish"
    finally:
        for process in ranks.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        for error_file in ranks.error_files:
            Path(error_file).unlink(missing_ok=True)


class TestMetaBuild:
    def test_materializing_a_meta_build_ties_the_head_again(self) -> None:
        """``to_empty`` gives each module a parameter of its own; init ties them back."""
        with torch.device("meta"):
            model = Transformer(_config())
        model.to_empty(device="cpu")
        assert not _tied(model)
        model.init_weights_and_freqs()
        assert _tied(model)
        assert len(list(model.parameters())) == len(list(Transformer(_config()).parameters()))
        weight = model.token_embedding.embedding.weight
        assert abs(weight.std().item() - _config().init_std) < 0.002

    def test_an_untied_meta_build_stays_untied(self) -> None:
        with torch.device("meta"):
            model = Transformer(_config(tie=False))
        model.to_empty(device="cpu")
        model.init_weights_and_freqs()
        assert not _tied(model)


class TestTensorParallelBuild:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    @pytest.mark.parametrize(
        ("shape", "names"),
        [((2,), ("tp",)), ((2, 2), ("dp_shard", "tp"))],
        ids=["tp", "dp_shard-and-tp"],
    )
    def test_a_tensor_parallel_build_keeps_the_tie_through_a_step(
        self, tmp_path: Path, shape: tuple[int, ...], names: tuple[str, ...], dtype: torch.dtype
    ) -> None:
        """A TP mesh without a data-parallel dimension builds on meta and skips FSDP2."""
        _spawn(tmp_path / "store", shape, names, dtype)


class TestFloat8:
    def test_a_tied_output_head_stays_in_high_precision(self, info_logged: list[str]) -> None:
        model = Transformer(_config())
        apply_float8(model)
        assert _tied(model)
        assert type(model.output_head.proj) is torch.nn.Linear
        assert type(model.layers["0"].attention.q_proj).__name__ == "Float8Linear"
        assert any("output_head.proj stays in high precision" in m for m in info_logged)

    def test_an_untied_output_head_is_converted(self, info_logged: list[str]) -> None:
        model = Transformer(_config(tie=False))
        apply_float8(model)
        assert type(model.output_head.proj).__name__ == "Float8Linear"
        assert not any("stays in high precision" in m for m in info_logged)
