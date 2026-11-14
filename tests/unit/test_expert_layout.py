"""The expert layout's names, param groups and checks, in one process without a mesh.

An unpacked layer is marked as rank 1 of two EP ranks by hand: it keeps experts 2 and 3
of four, as ``apply_expert_parallel`` leaves them, renumbered 0 and 1. Packed tensors and
whole checkpoints are checked over gloo in ``tests/unit/test_ep_checkpoint.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor.placement_types import _StridedShard

from kempnerforge.checkpoint.expert_layout import (
    ExpertLayout,
    _root_mesh,
    _rows_after_ep,
    check_saved_experts,
    expert_layout,
    moe_layers,
)
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import ActivationCheckpointing
from kempnerforge.distributed.parallel import apply_ac
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.transformer import Transformer

LAYER = "layers.0.mlp"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _rank_one_of_two(packed: bool = False) -> MoEMLP:
    """A four-expert layer holding experts 2 and 3 as EP rank 1 of 2 does."""
    moe = build_moe(dim=8, hidden_dim=16, num_experts=4, top_k=2, packed_experts=packed)
    if not packed:
        moe.experts = torch.nn.ModuleList(list(moe.experts)[2:])
    moe.ep_world_size = 2
    moe.local_expert_start = 2
    moe.num_local_experts = 2
    return moe


def _names(experts: list[int]) -> list[str]:
    return [f"{LAYER}.experts.{i}.{p}.weight" for i in experts for p in PROJECTIONS]


def _state(moe: MoEMLP) -> dict:
    """A state dict as rank 1 gives it: its experts under local names 0 and 1."""
    model = {f"{LAYER}.router.gate.weight": torch.zeros(1)}
    model.update({f"{LAYER}.{name}": p for name, p in moe.named_parameters() if "experts" in name})
    names = [f"{LAYER}.router.gate.weight", *_names([0, 1])]
    return {
        "model": model,
        "optimizer": {
            "state": {name: {"exp_avg": torch.zeros(1)} for name in names},
            "param_groups": [{"params": names, "lr": 0.1}],
        },
        "optimizer_never_stepped": {_names([1])[0]: torch.zeros((), dtype=torch.bool)},
    }


class TestMoELayers:
    def test_names_match_the_state_dict_under_ac_and_compile(self):
        """The layer names leave out the AC and compile wrappers, as the state dict does."""
        config = ModelConfig(
            dim=16, n_layers=2, n_heads=2, vocab_size=32, num_experts=4, moe_top_k=2
        )
        model = Transformer(config)
        apply_ac(model, ActivationCheckpointing.full)
        prefixes = {key.rpartition(".mlp.")[0] + ".mlp" for key in get_model_state_dict(model)}
        for wrapped in (model, torch.compile(model)):
            names = [name for name, _ in moe_layers(wrapped)]
            assert names == ["layers.0.mlp", "layers.1.mlp"]
            assert set(names) <= prefixes

    def test_no_layout_without_expert_parallelism(self):
        """With ``ep = 1`` nothing is translated, so the checkpoint keeps its names."""
        config = ModelConfig(dim=16, n_layers=1, n_heads=2, vocab_size=32, num_experts=4)
        assert expert_layout(Transformer(config)) is None


class TestUnpackedNames:
    def test_experts_take_their_global_index(self):
        """Local experts 0 and 1 of rank 1 are saved as experts 2 and 3."""
        moe = _rank_one_of_two()
        state = ExpertLayout([(LAYER, moe)]).to_global(_state(moe))
        expected = {f"{LAYER}.router.gate.weight", *_names([2, 3])}
        assert set(state["model"]) == expected
        assert set(state["optimizer"]["state"]) == expected
        assert set(state["optimizer_never_stepped"]) == {_names([3])[0]}

    def test_tensors_are_the_rank_own(self):
        """Each global name holds the tensor of the local expert it names."""
        moe = _rank_one_of_two()
        local = _state(moe)
        state = ExpertLayout([(LAYER, moe)]).to_global(local)
        for name, tensor in zip(_names([2, 3]), _names([0, 1]), strict=True):
            assert state["model"][name] is local["model"][tensor]

    def test_param_groups_list_every_rank_experts(self):
        """Each rank lists all four experts in global order, as one process does."""
        moe = _rank_one_of_two()
        state = ExpertLayout([(LAYER, moe)]).to_global(_state(moe))
        expected = [f"{LAYER}.router.gate.weight", *_names([0, 1, 2, 3])]
        assert state["optimizer"]["param_groups"][0]["params"] == expected

    def test_to_local_restores_rank_names_and_keeps_loaded_settings(self):
        """After a load, names and param lists are the rank's again; group settings are loaded."""
        moe = _rank_one_of_two()
        local = _state(moe)
        layout = ExpertLayout([(LAYER, moe)])
        loaded = layout.to_global(local)
        loaded["optimizer"]["param_groups"][0]["lr"] = 0.5
        restored = layout.to_local(loaded)
        assert set(restored["model"]) == set(local["model"])
        assert set(restored["optimizer"]["state"]) == set(local["optimizer"]["state"])
        assert restored["optimizer"]["param_groups"] == [
            {"params": local["optimizer"]["param_groups"][0]["params"], "lr": 0.5}
        ]

    def test_input_is_left_as_it_is(self):
        """The translation returns new dicts, so the optimizer's own state is never renamed."""
        moe = _rank_one_of_two()
        local = _state(moe)
        names = list(local["optimizer"]["state"])
        ExpertLayout([(LAYER, moe)]).to_global(local)
        assert list(local["optimizer"]["state"]) == names

    def test_missing_expert_weights_refuse_the_save(self):
        """A model state without the layer's experts would save no expert of this rank."""
        moe = _rank_one_of_two()
        state = _state(moe)
        del state["model"][_names([1])[2]]
        with pytest.raises(ValueError, match="no entry for 1 expert weights"):
            ExpertLayout([(LAYER, moe)]).to_global(state)

    def test_entries_outside_the_split_layers_pass_through(self):
        """A value that maps no names, and another layer's experts, keep their names."""
        moe = _rank_one_of_two()
        local = _state(moe)
        other = "layers.1.mlp.experts.0.up_proj.weight"
        local["model"][other] = torch.zeros(1)
        local["step"] = 7
        layout = ExpertLayout([(LAYER, moe)])
        state = layout.to_global(local)
        assert state["step"] == 7
        assert state["model"][other] is local["model"][other]
        restored = layout.to_local(state)
        assert restored["step"] == 7
        assert restored["model"][other] is local["model"][other]

    def test_a_local_index_past_the_rank_experts_is_refused(self):
        """Rank 1 has no local expert 2, whose global index 4 is past the layer's four experts."""
        moe = _rank_one_of_two()
        state = _state(moe)
        state["model"][_names([2])[0]] = torch.zeros(1)
        with pytest.raises(
            ValueError, match=r"names local expert 2, but layers\.0\.mlp holds 2 experts on this"
        ):
            ExpertLayout([(LAYER, moe)]).to_global(state)


class TestPackedWithoutMesh:
    def test_layer_without_ep_mesh_is_refused(self):
        """A packed layer split by hand records no EP mesh to place its rows by."""
        moe = _rank_one_of_two(packed=True)
        state = {"model": {f"{LAYER}.{name}": p for name, p in moe.named_parameters()}}
        with pytest.raises(ValueError, match="records no ep_mesh"):
            ExpertLayout([(LAYER, moe)]).to_global(state)

    def test_rows_already_strided_are_refused(self):
        """Rows FSDP2 already strides, as it does rows that TP shards, are not strided again."""
        with pytest.raises(
            ValueError, match=r"^layers\.0\.mlp\.up_w is already strided over its rows"
        ):
            _rows_after_ep((_StridedShard(0, split_factor=2),), 2, f"{LAYER}.up_w")

    def test_an_optimizer_state_of_another_shape_is_refused(self):
        """A packed weight's optimizer state is spread like the weight, so it needs its shape."""
        moe = _rank_one_of_two(packed=True)
        name = f"{LAYER}.up_w"
        optimizer = {
            "state": {name: {"step": torch.tensor(1.0), "exp_avg": torch.zeros(3)}},
            "param_groups": [{"params": [name], "lr": 0.1}],
        }
        with pytest.raises(
            ValueError,
            match=r"^the optimizer state 'exp_avg' of layers\.0\.mlp\.up_w has shape \(3,\), "
            r"not \(4, 8, 16\)",
        ):
            ExpertLayout([(LAYER, moe)]).to_global({"optimizer": optimizer})


class TestOlderTorch:
    def test_a_mesh_without_a_root_lookup_names_the_torch_it_needs(self):
        """Before torch 2.10 ``DeviceMesh`` has no ``_get_root_mesh``, and the error says so."""
        with pytest.raises(ValueError, match="need torch 2.10 or newer"):
            _root_mesh(SimpleNamespace())


def _index(experts: int = 4, packed_rows: int | None = None, layers: int = 2) -> dict:
    """A DCP index of an unpacked or packed checkpoint holding ``experts`` or ``packed_rows``."""
    index: dict = {"model.norm.weight": SimpleNamespace(size=torch.Size([16]))}
    for layer in range(layers):
        if packed_rows is not None:
            index[f"model.layers.{layer}.mlp.up_w"] = SimpleNamespace(
                size=torch.Size([packed_rows, 16, 32])
            )
            continue
        for expert in range(experts):
            for p in PROJECTIONS:
                key = f"model.layers.{layer}.mlp.experts.{expert}.{p}.weight"
                index[key] = SimpleNamespace(size=torch.Size([32, 16]))
    return index


def _model(packed: bool = False, num_experts: int = 4) -> Transformer:
    config = ModelConfig(
        dim=16,
        n_layers=2,
        n_heads=2,
        vocab_size=32,
        num_experts=num_experts,
        moe_top_k=1,
        moe_packed_experts=packed,
    )
    return Transformer(config)


class TestCheckSavedExperts:
    @pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
    def test_a_checkpoint_of_every_expert_passes(self, packed):
        index = _index(packed_rows=4) if packed else _index()
        check_saved_experts(_model(packed), index, "ckpt")

    def test_unpacked_checkpoint_of_one_rank_is_refused(self):
        """It names the experts held, how many layers, and the ``ep`` that saves that."""
        with pytest.raises(ValueError) as error:
            check_saved_experts(_model(), _index(experts=2), "ckpt/step_4")
        message = str(error.value)
        assert message.startswith("ckpt/step_4 holds 2 of the 4 experts of 2 MoE layers")
        assert "(layers.0.mlp holds experts 0-1)" in message
        assert "saved with ep=2 holds" in message

    def test_packed_checkpoint_of_one_rank_is_refused(self):
        with pytest.raises(
            ValueError, match=r"\(layers\.0\.mlp\.up_w has 1 row\).*saved with ep=4"
        ):
            check_saved_experts(_model(packed=True), _index(packed_rows=1), "ckpt")

    def test_one_expert_is_named_alone(self):
        with pytest.raises(ValueError, match=r"\(layers\.0\.mlp holds expert 0\)"):
            check_saved_experts(_model(), _index(experts=1), "ckpt")

    def test_layers_the_checkpoint_does_not_hold_are_left_to_the_load(self):
        """A checkpoint without the layer's experts, such as a dense one, is not refused here."""
        check_saved_experts(_model(), _index(experts=0), "ckpt")
        check_saved_experts(_model(packed=True), _index(), "ckpt")


class TestRootLayer:
    """A model that is itself the ``MoEMLP`` names its state-dict entries with no prefix."""

    def test_its_name_is_empty(self):
        moe = build_moe(dim=8, hidden_dim=16, num_experts=4, top_k=2)
        assert list(moe_layers(moe)) == [("", moe)]

    def test_unpacked_experts_take_their_global_index(self):
        moe = _rank_one_of_two()
        names = [
            "router.gate.weight",
            *(f"experts.{i}.{p}.weight" for i in (0, 1) for p in PROJECTIONS),
        ]
        local = {
            "model": {name: p for name, p in moe.named_parameters() if "experts" in name},
            "optimizer": {
                "state": {name: {"exp_avg": torch.zeros(1)} for name in names},
                "param_groups": [{"params": names, "lr": 0.1}],
            },
        }
        layout = ExpertLayout([("", moe)])
        state = layout.to_global(local)
        assert set(state["model"]) == {
            f"experts.{i}.{p}.weight" for i in (2, 3) for p in PROJECTIONS
        }
        assert state["optimizer"]["param_groups"][0]["params"] == [
            "router.gate.weight",
            *(f"experts.{i}.{p}.weight" for i in range(4) for p in PROJECTIONS),
        ]
        assert set(layout.to_local(state)["model"]) == set(local["model"])

    def test_packed_weights_are_named_without_a_leading_dot(self):
        moe = _rank_one_of_two(packed=True)
        state = {"model": dict(moe.named_parameters())}
        with pytest.raises(ValueError, match=r"^(up|down|gate)_w is split over ep=2"):
            ExpertLayout([("", moe)]).to_global(state)

    def test_a_checkpoint_of_one_rank_is_refused(self):
        moe = build_moe(dim=8, hidden_dim=16, num_experts=4, top_k=2)
        size = SimpleNamespace(size=torch.Size([16, 8]))
        held = {f"model.experts.{e}.{p}.weight": size for e in range(2) for p in PROJECTIONS}
        with pytest.raises(ValueError, match=r"\(the model holds experts 0-1\).*saved with ep=2"):
            check_saved_experts(moe, held, "ckpt")
        every = {f"model.experts.{e}.{p}.weight": size for e in range(4) for p in PROJECTIONS}
        check_saved_experts(moe, every, "ckpt")

    def test_a_packed_checkpoint_of_one_rank_is_refused(self):
        moe = build_moe(dim=8, hidden_dim=16, num_experts=4, top_k=2, packed_experts=True)
        held = {"model.up_w": SimpleNamespace(size=torch.Size([1, 16, 8]))}
        with pytest.raises(ValueError, match=r"\(up_w has 1 row\).*saved with ep=4"):
            check_saved_experts(moe, held, "ckpt")
