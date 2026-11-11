"""Text warm starts read supported checkpoint formats and fail together across ranks."""

import pickle
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vlm import MoTConfig
from kempnerforge.distributed.utils import agree_across_ranks
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.entry import _mot_text_warm_start

CASES = [(0, False), (1, False), (1, True), (None, False)]


def _alone(error, what):
    if error is not None:
        raise error


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain_state_dict", "model_wrapper"])
def test_saved_text_weights_reach_each_modality(tmp_path, wrapped):
    config = ModelConfig(dim=16, n_layers=1, n_heads=2, vocab_size=32, max_seq_len=16)
    text = Transformer(config)
    source = text.state_dict()
    path = tmp_path / "text.pt"
    torch.save({"model": source} if wrapped else source, path)
    vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(path))
    model = Transformer(config, vlm_config=vlm)

    _mot_text_warm_start(vlm, model, _alone)

    restored = model.state_dict()
    for modality in vlm.mot_modalities:
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            torch.testing.assert_close(
                restored[f"layers.0.attn.{projection}.{modality}.weight"],
                source[f"layers.0.attention.{projection}.weight"],
            )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            torch.testing.assert_close(
                restored[f"layers.0.mlp.{modality}.{projection}.weight"],
                source[f"layers.0.mlp.{projection}.weight"],
            )


def test_missing_source_preserves_local_error(tmp_path):
    path = tmp_path / "missing.pt"
    vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(path))
    with pytest.raises(FileNotFoundError):
        _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), _alone)


@pytest.mark.parametrize(
    "local_failure,remote_failure", [(True, False), (False, True), (False, False)]
)
def test_read_agreement_precedes_copy(tmp_path, local_failure, remote_failure):
    path = tmp_path / "text.pt"
    if not local_failure:
        torch.save({"weight": torch.ones(2, 2)}, path)
    vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(path))
    reported = []

    def agree(error, what):
        reported.append(error)
        if error is not None:
            raise error
        if remote_failure:
            raise RuntimeError(f"{what} failed on ranks [1]")

    with patch("kempnerforge.training.entry.mot_warm_start_from_text_stack") as copy:
        if local_failure:
            with pytest.raises(FileNotFoundError):
                _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), agree)
            assert isinstance(reported[0], FileNotFoundError)
        elif remote_failure:
            with pytest.raises(RuntimeError, match=r"reading .* failed on ranks \[1\]"):
                _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), agree)
            assert reported == [None]
        else:
            _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), agree)
            copy.assert_called_once()
            assert reported == [None]
        if local_failure or remote_failure:
            copy.assert_not_called()


def _read_on_every_rank(rank, root):
    """Run every case of ``CASES`` on a two-rank gloo group.

    A rank-local failure must leave every process able to reach the next
    collective, which the barrier after each case checks.
    """
    root = Path(root)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=10),
    )
    agree = partial(agree_across_ranks, group=None)
    try:
        for bad_rank, corrupt in CASES:
            name = "text.pt"
            if rank == bad_rank:
                name = "corrupt.pt" if corrupt else "missing.pt"
            vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(root / name))
            with patch("kempnerforge.training.entry.mot_warm_start_from_text_stack") as copy:
                if rank == bad_rank:
                    expected = pickle.UnpicklingError if corrupt else FileNotFoundError
                    with pytest.raises(expected):
                        _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), agree)
                elif bad_rank is not None:
                    with pytest.raises(RuntimeError, match=f"on ranks \\[{bad_rank}\\]"):
                        _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), agree)
                else:
                    _mot_text_warm_start(vlm, torch.nn.Linear(2, 2), agree)
                    copy.assert_called_once()
                if bad_rank is not None:
                    copy.assert_not_called()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_source_read_agreement_across_ranks(tmp_path):
    torch.save({"weight": torch.ones(2, 2)}, tmp_path / "text.pt")
    (tmp_path / "corrupt.pt").write_bytes(b"not a torch checkpoint")
    mp.spawn(_read_on_every_rank, args=(str(tmp_path),), nprocs=2)
