"""Unit tests for activation extraction hooks."""

from __future__ import annotations

import copy
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.parallel import apply_fsdp2, default_mp_policy
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.model.attention import Attention
from kempnerforge.model.hooks import ActivationStore, extract_representations, save_activations
from kempnerforge.model.position import precompute_rope_frequencies
from kempnerforge.model.transformer import Transformer
from tests.gloo_ranks import gloo_group, spawn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WHOLE_BLOCK_TARGETS = [
    "token_embedding",
    "layers.0",
    "layers.0.attention_norm",
    "layers.0.attention",
    "layers.1.mlp_norm",
    "layers.1.mlp",
    "norm",
    "output_head",
]
PROJECTION_TARGETS = [
    "layers.0.attention.q_proj",
    "layers.0.attention.k_proj",
    "layers.0.attention.v_proj",
    "layers.0.attention.o_proj",
    "layers.1.mlp.gate_proj",
    "layers.1.mlp.up_proj",
    "layers.1.mlp.down_proj",
    "output_head.proj",
]
LOCAL_HEAD_SHARD_TARGETS = ["layers.0.attention.q_norm", "layers.0.attention.k_norm"]
TP_LAYOUTS = {
    "tp2": (("tp",), (2,), False),
    "tp2_tied": (("tp",), (2,), True),
    "dp_shard2_tp2": (("dp_shard", "tp"), (2, 2), False),
}


@pytest.fixture
def small_config():
    return ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=32)


@pytest.fixture
def small_model(small_config):
    return Transformer(small_config).to(DEVICE).eval()


@pytest.fixture
def small_dataset():
    """Simple map-style dataset for testing."""

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=16, seq_len=16, vocab_size=256):
            self._data = torch.randint(0, vocab_size, (n_samples, seq_len))

        def __len__(self):
            return self._data.shape[0]

        def __getitem__(self, idx):
            return {"input_ids": self._data[idx]}

    return _Dataset()


class TestActivationStore:
    def test_capture_single_layer(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        act = store.get("layers.0.attention")
        assert act is not None
        assert act.shape == (2, 16, 64)
        assert act.device == torch.device("cpu")
        store.disable()

    def test_capture_multiple_layers(self, small_model):
        layers = ["layers.0.attention", "layers.1.mlp", "layers.3.attention"]
        store = ActivationStore(small_model, layers=layers)
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        for name in layers:
            assert store.get(name) is not None
        store.disable()

    def test_capture_norm_layer(self, small_model):
        store = ActivationStore(small_model, layers=["norm"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        act = store.get("norm")
        assert act is not None
        assert act.shape == (2, 16, 64)
        store.disable()

    def test_enable_disable(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        assert not store.enabled

        store.enable()
        assert store.enabled
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        assert store.get("layers.0.attention") is not None

        store.disable()
        assert not store.enabled
        store.clear()
        small_model(tokens)
        assert store.get("layers.0.attention") is None

    def test_clear(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        assert store.get("layers.0.attention") is not None
        store.clear()
        assert store.get("layers.0.attention") is None
        assert store.activations == {}
        store.disable()

    def test_activations_property(self, small_model):
        layers = ["layers.0.attention", "layers.1.mlp"]
        store = ActivationStore(small_model, layers=layers)
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        acts = store.activations
        assert set(acts.keys()) == set(layers)
        store.disable()

    def test_layer_names_property(self, small_model):
        layers = ["layers.0.attention", "norm"]
        store = ActivationStore(small_model, layers=layers)
        assert store.layer_names == layers

    def test_invalid_layer_raises(self, small_model):
        store = ActivationStore(small_model, layers=["nonexistent_module"])
        with pytest.raises(ValueError, match="not found"):
            store.enable()

    def test_double_enable_is_idempotent(self, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        assert store.get("layers.0.attention") is not None
        store.disable()

    def test_output_is_detached(self, small_model):
        """Captured activations should not require grad."""
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)
        act = store.get("layers.0.attention")
        assert not act.requires_grad
        store.disable()

    def test_shapes_match_model_dim(self, small_config, small_model):
        """Captured shapes should match model dimension."""
        store = ActivationStore(small_model, layers=["layers.2.mlp"])
        store.enable()
        tokens = torch.randint(0, 256, (4, 8), device=DEVICE)
        small_model(tokens)
        act = store.get("layers.2.mlp")
        assert act.shape == (4, 8, small_config.dim)
        store.disable()

    def test_tuple_output_stores_its_first_element(self):
        """A module returning a tuple has its first element captured; an empty one, nothing."""

        class _Pair(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return (x + 1, x) if x.numel() else ()

        class _Holder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pair = _Pair()
                self.empty = _Pair()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.empty(x[:0])
                return self.pair(x)[0]

        model = _Holder()
        store = ActivationStore(model, layers=["pair", "empty"])
        store.enable()
        x = torch.randn(2, 3)
        model(x)
        assert torch.equal(store.get("pair"), x + 1)
        assert store.get("empty") is None
        store.disable()


class TestAttentionWeightCapture:
    @pytest.fixture
    def attention_setup(self):
        dim, n_heads, n_kv_heads = 64, 4, 4
        attn = Attention(dim, n_heads, n_kv_heads).to(DEVICE).eval()
        rope_cos, rope_sin = precompute_rope_frequencies(head_dim=dim // n_heads, max_seq_len=32)
        rope_cos = rope_cos[:16].to(DEVICE)
        rope_sin = rope_sin[:16].to(DEVICE)
        return attn, rope_cos, rope_sin

    def test_weights_not_captured_by_default(self, attention_setup):
        attn, rope_cos, rope_sin = attention_setup
        x = torch.randn(2, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is None

    def test_capture_attention_weights(self, attention_setup):
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is not None
        assert attn.last_attention_weights.shape == (2, 4, 16, 16)
        assert attn.last_attention_weights.device == torch.device("cpu")
        assert out.shape == (2, 16, 64)

    def test_attention_weights_sum_to_one(self, attention_setup):
        """Attention weights should sum to ~1.0 along key dimension."""
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(2, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        weights = attn.last_attention_weights.float()
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_are_causal(self, attention_setup):
        """Upper triangle (future positions) should be zero."""
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(1, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        weights = attn.last_attention_weights.float()
        seq_len = weights.shape[-1]
        upper_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        upper_values = weights[0, 0][upper_mask]
        assert torch.allclose(upper_values, torch.zeros_like(upper_values), atol=1e-6)

    def test_capture_output_matches_sdpa(self, attention_setup):
        """Manual attention path should produce same output as SDPA."""
        attn, rope_cos, rope_sin = attention_setup
        x = torch.randn(2, 16, 64, device=DEVICE)

        attn.capture_attention_weights = False
        out_sdpa = attn(x, rope_cos, rope_sin)

        attn.capture_attention_weights = True
        out_manual = attn(x, rope_cos, rope_sin)

        assert torch.allclose(out_sdpa, out_manual, atol=1e-4)

    def test_capture_disables_cleanly(self, attention_setup):
        attn, rope_cos, rope_sin = attention_setup
        attn.capture_attention_weights = True
        x = torch.randn(2, 16, 64, device=DEVICE)
        attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is not None

        attn.capture_attention_weights = False
        attn.last_attention_weights = None
        attn(x, rope_cos, rope_sin)
        assert attn.last_attention_weights is None


class TestExtractRepresentations:
    def test_basic_extraction(self, small_model, small_dataset):
        layers = ["layers.0.attention", "layers.1.mlp"]
        reps = extract_representations(small_model, small_dataset, layers, DEVICE, batch_size=8)
        assert set(reps.keys()) == set(layers)
        for name in layers:
            assert reps[name].shape[0] == len(small_dataset)
            assert reps[name].shape[2] == 64

    def test_max_samples(self, small_model, small_dataset):
        reps = extract_representations(
            small_model,
            small_dataset,
            ["layers.0.attention"],
            DEVICE,
            batch_size=4,
            max_samples=8,
        )
        assert reps["layers.0.attention"].shape[0] == 8

    def test_model_restored_to_original_mode(self, small_model, small_dataset):
        small_model.train()
        assert small_model.training
        extract_representations(small_model, small_dataset, ["layers.0.attention"], DEVICE)
        assert small_model.training

    def test_results_on_cpu(self, small_model, small_dataset):
        reps = extract_representations(small_model, small_dataset, ["layers.0.attention"], DEVICE)
        for tensor in reps.values():
            assert tensor.device == torch.device("cpu")

    def test_non_trivial_activations(self, small_model, small_dataset):
        """Activations should not be all zeros."""
        reps = extract_representations(small_model, small_dataset, ["layers.0.attention"], DEVICE)
        act = reps["layers.0.attention"]
        assert act.abs().sum() > 0


class TestSaveActivations:
    def test_save_and_load(self, tmp_path, small_model):
        store = ActivationStore(small_model, layers=["layers.0.attention", "norm"])
        store.enable()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens)

        path = tmp_path / "activations.npz"
        save_activations(store.activations, path)
        store.disable()

        assert path.exists()
        loaded = np.load(str(path))
        assert "layers.0.attention" in loaded
        assert "norm" in loaded
        assert loaded["layers.0.attention"].shape == (2, 16, 64)

    def test_adds_npz_extension(self, tmp_path, small_model):
        store = ActivationStore(small_model, layers=["norm"])
        store.enable()
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        small_model(tokens)

        path = tmp_path / "acts"
        save_activations(store.activations, path)
        store.disable()

        assert (tmp_path / "acts.npz").exists()

    def test_creates_parent_dirs(self, tmp_path, small_model):
        store = ActivationStore(small_model, layers=["norm"])
        store.enable()
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        small_model(tokens)

        path = tmp_path / "subdir" / "deep" / "acts.npz"
        save_activations(store.activations, path)
        store.disable()

        assert path.exists()


class TestActivationStoreIntegration:
    def test_full_workflow(self, small_model):
        """Full workflow: enable → forward → capture → save → load."""
        layers = ["layers.0.attention", "layers.2.mlp", "norm"]
        store = ActivationStore(small_model, layers=layers)
        store.enable()

        tokens = torch.randint(0, 256, (4, 16), device=DEVICE)
        with torch.no_grad():
            small_model(tokens)

        for name in layers:
            act = store.get(name)
            assert act is not None
            assert act.shape[0] == 4
            assert act.shape[1] == 16
            assert act.shape[2] == 64

        store.disable()

    def test_multiple_forward_passes(self, small_model):
        """Each forward overwrites previous activations."""
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()

        tokens1 = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens1)
        act1 = store.get("layers.0.attention").clone()

        tokens2 = torch.randint(0, 256, (2, 16), device=DEVICE)
        small_model(tokens2)
        act2 = store.get("layers.0.attention")

        assert not torch.allclose(act1, act2)
        store.disable()

    def test_no_hooks_leak_after_disable(self, small_model):
        """After disable, no hooks remain on the model."""
        store = ActivationStore(small_model, layers=["layers.0.attention"])
        store.enable()
        store.disable()

        assert len(store._hooks) == 0


def _parallel_model(
    mesh_dims: tuple[str, ...], mesh_sizes: tuple[int, ...], tie_embeddings: bool
) -> tuple[Transformer, Transformer]:
    """A seeded GQA Transformer parallelized over a CPU mesh, and an unparallelized copy."""
    torch.manual_seed(0)
    config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=128,
        max_seq_len=32,
        qk_norm=True,
        tie_embeddings=tie_embeddings,
    )
    model = Transformer(config)
    reference = copy.deepcopy(model)
    mesh = init_device_mesh("cpu", mesh_sizes, mesh_dim_names=mesh_dims)
    if "tp" in mesh_dims:
        apply_tensor_parallel(model, mesh)
    if "dp_shard" in mesh_dims:
        apply_fsdp2(model, mesh, mp_policy=default_mp_policy(torch.float32))
    return model, reference


def _tokens(seed: int) -> torch.Tensor:
    return torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(seed))


def _capture(
    model: torch.nn.Module, layers: list[str], inputs: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """What a store over ``layers`` captures in one forward on ``inputs``, and the output."""
    store = ActivationStore(model, layers=layers)
    store.enable()
    with torch.no_grad():
        output = model(inputs)
    store.disable()
    return store.activations, output


def _assert_full_and_saved(
    captured: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    """Each capture is a plain tensor equal to the TP=1 one and survives an ``.npz`` save."""
    for name, value in expected.items():
        assert not isinstance(captured[name], DTensor), name
        torch.testing.assert_close(captured[name], value, msg=name)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "acts.npz")
        save_activations(captured, path)
        with np.load(path) as loaded:
            for name in expected:
                np.testing.assert_array_equal(loaded[name], captured[name].numpy())


def _check_whole_block_outputs_gathered(rank: int, world: int, port: int, layout: str) -> None:
    mesh_dims, mesh_sizes, tie_embeddings = TP_LAYOUTS[layout]
    with gloo_group(rank, world, port):
        model, reference = _parallel_model(mesh_dims, mesh_sizes, tie_embeddings)
        tokens = _tokens(rank // mesh_sizes[-1])
        captured, _ = _capture(model, WHOLE_BLOCK_TARGETS, tokens)
        expected, _ = _capture(reference, WHOLE_BLOCK_TARGETS, tokens)
        _assert_full_and_saved(captured, expected)


def _check_projection_outputs_gathered(rank: int, world: int, port: int, layout: str) -> None:
    mesh_dims, mesh_sizes, tie_embeddings = TP_LAYOUTS[layout]
    with gloo_group(rank, world, port):
        model, reference = _parallel_model(mesh_dims, mesh_sizes, tie_embeddings)
        tokens = _tokens(rank // mesh_sizes[-1])
        with torch.no_grad():
            logits = model(tokens)
        captured, captured_logits = _capture(model, PROJECTION_TARGETS, tokens)
        expected, _ = _capture(reference, PROJECTION_TARGETS, tokens)
        _assert_full_and_saved(captured, expected)
        assert torch.equal(captured_logits, logits)


def _check_local_head_shards_refused(rank: int, world: int, port: int, layout: str) -> None:
    mesh_dims, mesh_sizes, tie_embeddings = TP_LAYOUTS[layout]
    with gloo_group(rank, world, port):
        model, reference = _parallel_model(mesh_dims, mesh_sizes, tie_embeddings)
        tokens = _tokens(rank // mesh_sizes[-1])
        for name in LOCAL_HEAD_SHARD_TARGETS:
            store = ActivationStore(model, layers=["layers.0", name])
            with pytest.raises(ValueError, match=re.escape(f"Cannot capture '{name}'")):
                store.enable()
            assert not store.enabled
            with torch.no_grad():
                model(tokens)
            assert store.activations == {}, name
        reason = (
            "because it sits next to 'layers.0.attention.q_proj', which holds parameters "
            "sharded over the 'tp' mesh dimension. Capture 'layers.0.attention' or a module "
            "enclosing it instead."
        )
        with pytest.raises(ValueError, match=re.escape(reason)):
            ActivationStore(model, layers=["layers.0.attention.k_norm"]).enable()
        attention = model.get_submodule("layers.0.attention")
        for name in ("q_norm", "k_norm"):
            with pytest.raises(ValueError, match=re.escape(f"Cannot capture '{name}'")):
                ActivationStore(attention, layers=[name]).enable()
        store = ActivationStore(attention, layers=["k_proj"])
        store.enable()
        with torch.no_grad():
            model(tokens)
        store.disable()
        expected, _ = _capture(reference, ["layers.0.attention.k_proj"], tokens)
        torch.testing.assert_close(store.get("k_proj"), expected["layers.0.attention.k_proj"])


def _check_top_level_tp_linears(rank: int, world: int, port: int) -> None:
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 6)
        pair = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.Identity())
        reference_linear, reference_pair = copy.deepcopy(linear), copy.deepcopy(pair)
        parallelize_module(linear, mesh, ColwiseParallel())
        parallelize_module(pair, mesh, {"0": ColwiseParallel(output_layouts=Replicate())})
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            captured, output = _capture(linear, [""], x)
            assert output.shape == (3, 6 // world)
            torch.testing.assert_close(captured[""], reference_linear(x))
            captured, _ = _capture(pair, ["0", ""], x)
            torch.testing.assert_close(captured["0"], reference_pair(x))
            torch.testing.assert_close(captured[""], reference_pair(x))
        reason = "Cannot capture '1': under tensor parallelism it may run on this rank's local"
        with pytest.raises(ValueError, match=re.escape(reason) + ".* next to '0', which holds"):
            ActivationStore(pair, layers=["1"]).enable()


def _check_fsdp_targets_not_refused(rank: int, world: int, port: int) -> None:
    with gloo_group(rank, world, port):
        model, reference = _parallel_model(("dp_shard",), (world,), False)
        layers = [*WHOLE_BLOCK_TARGETS, *PROJECTION_TARGETS, *LOCAL_HEAD_SHARD_TARGETS]
        tokens = _tokens(rank)
        captured, _ = _capture(model, layers, tokens)
        expected, _ = _capture(reference, layers, tokens)
        for name in layers:
            assert torch.equal(captured[name], expected[name]), name


def _check_attention_weights_gathered(rank: int, world: int, port: int, layout: str) -> None:
    mesh_dims, mesh_sizes, tie_embeddings = TP_LAYOUTS[layout]
    with gloo_group(rank, world, port):
        model, reference = _parallel_model(mesh_dims, mesh_sizes, tie_embeddings)
        tokens = _tokens(rank // mesh_sizes[-1])
        for each in (model, reference):
            for layer in each.layers.values():
                layer.attention.capture_attention_weights = True
            with torch.no_grad():
                each(tokens)
        for name, layer in model.layers.items():
            weights = layer.attention.last_attention_weights
            assert weights.shape == (2, 4, 16, 16), name
            expected = reference.layers[name].attention.last_attention_weights
            torch.testing.assert_close(weights, expected, msg=name)


class TestActivationStoreParallel:
    """Captures on a parallelized Transformer over gloo CPU ranks.

    Every capture is stored as the full tensor on every rank, equal to the TP=1 capture. A
    ``DTensor`` output is gathered. A projection, whose weights are sharded over ``tp``, is
    hooked ahead of the tensor-parallel hook that turns its output into this rank's shard,
    and is gathered too, leaving the model's own output untouched. The layouts cover each
    style the tensor-parallel plan uses: ``tp2`` runs sequence parallelism (column-parallel
    projections on sequence-sharded inputs, row-parallel ones with sequence-sharded outputs,
    a column-parallel output head with a replicated output), ``tp2_tied`` runs without it
    (replicated inputs and row-parallel outputs, an output head left unsharded), and
    ``dp_shard2_tp2`` adds FSDP2. A module that may run on a local shard without such
    weights (``q_norm``, ``k_norm``, or any module beside a top-level projection) is refused
    by ``enable()`` before any hook is registered. FSDP2 alone shards parameters, not
    outputs, so it refuses nothing.
    """

    @pytest.mark.parametrize("layout", sorted(TP_LAYOUTS))
    def test_whole_block_outputs_are_gathered_to_the_full_tensor(self, layout: str) -> None:
        spawn(_check_whole_block_outputs_gathered, math.prod(TP_LAYOUTS[layout][1]), layout)

    @pytest.mark.parametrize("layout", sorted(TP_LAYOUTS))
    def test_projection_outputs_are_gathered_to_the_tp1_capture(self, layout: str) -> None:
        spawn(_check_projection_outputs_gathered, math.prod(TP_LAYOUTS[layout][1]), layout)

    @pytest.mark.parametrize("layout", sorted(TP_LAYOUTS))
    def test_local_head_shard_targets_are_refused(self, layout: str) -> None:
        spawn(_check_local_head_shards_refused, math.prod(TP_LAYOUTS[layout][1]), layout)

    def test_top_level_tp_linear_is_gathered_and_its_siblings_refused(self) -> None:
        spawn(_check_top_level_tp_linears, 2)

    def test_fsdp_only_targets_are_not_refused(self) -> None:
        spawn(_check_fsdp_targets_not_refused, 2)


class TestAttentionWeightCaptureParallel:
    """Attention weights captured on a tensor-parallel Transformer over gloo CPU ranks.

    Each rank computes one block of heads; the blocks are gathered over the tensor-parallel
    group, so every rank stores every head in the model's order, equal to the TP=1 capture,
    with and without sequence parallelism and under FSDP2.
    """

    @pytest.mark.parametrize("layout", sorted(TP_LAYOUTS))
    def test_every_head_is_gathered_to_the_tp1_capture(self, layout: str) -> None:
        spawn(_check_attention_weights_gathered, math.prod(TP_LAYOUTS[layout][1]), layout)
