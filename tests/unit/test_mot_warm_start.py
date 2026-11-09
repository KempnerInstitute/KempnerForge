"""Text warm starts read supported checkpoint formats and fail together across ranks."""

import pickle
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vlm import MoTConfig
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.entry import _mot_text_warm_start


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain_state_dict", "model_wrapper"])
def test_saved_text_weights_reach_each_modality(tmp_path, wrapped):
    config = ModelConfig(dim=16, n_layers=1, n_heads=2, vocab_size=32, max_seq_len=16)
    text = Transformer(config)
    source = text.state_dict()
    path = tmp_path / "text.pt"
    torch.save({"model": source} if wrapped else source, path)
    vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(path))
    model = Transformer(config, vlm_config=vlm)

    _mot_text_warm_start(vlm, model)

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
        _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))


@pytest.mark.parametrize(
    "local_failure,remote_failure", [(True, False), (False, True), (False, False)]
)
def test_read_agreement_precedes_copy(tmp_path, monkeypatch, local_failure, remote_failure):
    path = tmp_path / "text.pt"
    if not local_failure:
        torch.save({"weight": torch.ones(2, 2)}, path)
    vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(path))
    gathered = []

    def gather(errors, local_error):
        gathered.append(local_error)
        errors[:] = [local_error, "unreadable source" if remote_failure else None]

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    with patch("kempnerforge.training.entry.mot_warm_start_from_text_stack") as copy:
        if local_failure:
            with pytest.raises(FileNotFoundError):
                _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))
            assert gathered[0] is not None
        elif remote_failure:
            with pytest.raises(RuntimeError, match="on ranks \\[1\\]"):
                _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))
            assert gathered == [None]
        else:
            _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))
            copy.assert_called_once()
            assert gathered == [None]
        if local_failure or remote_failure:
            copy.assert_not_called()


def _read_on_every_rank(rank, root, bad_rank, corrupt):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=10),
    )
    try:
        path = root / ("unreadable.pt" if rank == bad_rank else "text.pt")
        vlm = MoTConfig(mot_warm_start_from_text=True, mot_warm_start_path=str(path))
        with patch("kempnerforge.training.entry.mot_warm_start_from_text_stack") as copy:
            if rank == bad_rank:
                expected = pickle.UnpicklingError if corrupt else FileNotFoundError
                with pytest.raises(expected):
                    _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))
            elif bad_rank is not None:
                with pytest.raises(RuntimeError, match=f"on ranks \\[{bad_rank}\\]"):
                    _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))
            else:
                _mot_text_warm_start(vlm, torch.nn.Linear(2, 2))
                copy.assert_called_once()
            if bad_rank is not None:
                copy.assert_not_called()
        # A rank-local failure must leave every process able to reach the next collective.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(45)
@pytest.mark.parametrize("bad_rank,corrupt", [(0, False), (1, False), (1, True), (None, False)])
def test_source_read_agreement_across_ranks(tmp_path, bad_rank, corrupt):
    torch.save({"weight": torch.ones(2, 2)}, tmp_path / "text.pt")
    if corrupt:
        (tmp_path / "unreadable.pt").write_bytes(b"not a torch checkpoint")
    mp.spawn(_read_on_every_rank, args=(str(tmp_path), bad_rank, corrupt), nprocs=2)
