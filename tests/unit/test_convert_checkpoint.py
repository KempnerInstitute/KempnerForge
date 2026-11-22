"""Unit tests for checkpoint format conversion key mapping."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from convert_checkpoint import _build_hf_config, _hf_to_kf_key, _kf_to_hf_key, dcp_to_hf, main

from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer


class TestKFtoHFKeyMapping:
    def test_embedding(self):
        assert _kf_to_hf_key("token_embedding.embedding.weight") == "model.embed_tokens.weight"

    def test_output_head(self):
        assert _kf_to_hf_key("output_head.proj.weight") == "lm_head.weight"

    def test_final_norm(self):
        assert _kf_to_hf_key("norm.weight") == "model.norm.weight"

    def test_attention_proj(self):
        assert (
            _kf_to_hf_key("layers.0.attention.q_proj.weight")
            == "model.layers.0.self_attn.q_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.5.attention.k_proj.weight")
            == "model.layers.5.self_attn.k_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.31.attention.v_proj.weight")
            == "model.layers.31.self_attn.v_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.31.attention.o_proj.weight")
            == "model.layers.31.self_attn.o_proj.weight"
        )

    def test_attention_norm(self):
        assert (
            _kf_to_hf_key("layers.0.attention_norm.weight")
            == "model.layers.0.input_layernorm.weight"
        )

    def test_mlp_norm(self):
        assert (
            _kf_to_hf_key("layers.0.mlp_norm.weight")
            == "model.layers.0.post_attention_layernorm.weight"
        )

    def test_mlp_projections(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.gate_proj.weight") == "model.layers.0.mlp.gate_proj.weight"
        )
        assert _kf_to_hf_key("layers.0.mlp.up_proj.weight") == "model.layers.0.mlp.up_proj.weight"
        assert (
            _kf_to_hf_key("layers.0.mlp.down_proj.weight") == "model.layers.0.mlp.down_proj.weight"
        )


class TestHFtoKFKeyMapping:
    def test_embedding(self):
        assert _hf_to_kf_key("model.embed_tokens.weight") == "token_embedding.embedding.weight"

    def test_output_head(self):
        assert _hf_to_kf_key("lm_head.weight") == "output_head.proj.weight"

    def test_final_norm(self):
        assert _hf_to_kf_key("model.norm.weight") == "norm.weight"

    def test_attention_proj(self):
        assert (
            _hf_to_kf_key("model.layers.0.self_attn.q_proj.weight")
            == "layers.0.attention.q_proj.weight"
        )

    def test_attention_norm(self):
        assert (
            _hf_to_kf_key("model.layers.0.input_layernorm.weight")
            == "layers.0.attention_norm.weight"
        )

    def test_mlp_norm(self):
        assert (
            _hf_to_kf_key("model.layers.0.post_attention_layernorm.weight")
            == "layers.0.mlp_norm.weight"
        )


class TestRoundTrip:
    @pytest.mark.parametrize(
        "kf_key",
        [
            "token_embedding.embedding.weight",
            "output_head.proj.weight",
            "norm.weight",
            "layers.0.attention.q_proj.weight",
            "layers.0.attention.k_proj.weight",
            "layers.0.attention.v_proj.weight",
            "layers.0.attention.o_proj.weight",
            "layers.0.attention_norm.weight",
            "layers.0.mlp_norm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.31.attention.q_proj.weight",
        ],
    )
    def test_kf_to_hf_to_kf(self, kf_key):
        assert _hf_to_kf_key(_kf_to_hf_key(kf_key)) == kf_key


class TestMoEKFtoHFKeyMapping:
    def test_router_gate(self):
        assert _kf_to_hf_key("layers.0.mlp.router.gate.weight") == "model.layers.0.mlp.gate.weight"

    def test_router_expert_bias(self):
        assert _kf_to_hf_key("layers.0.mlp.router.expert_bias") == "model.layers.0.mlp.expert_bias"

    def test_expert_projections(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.0.gate_proj.weight")
            == "model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.3.up_proj.weight")
            == "model.layers.0.mlp.experts.3.up_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.experts.7.down_proj.weight")
            == "model.layers.0.mlp.experts.7.down_proj.weight"
        )

    def test_shared_expert(self):
        assert (
            _kf_to_hf_key("layers.0.mlp.shared_expert.gate_proj.weight")
            == "model.layers.0.mlp.shared_experts.gate_proj.weight"
        )
        assert (
            _kf_to_hf_key("layers.0.mlp.shared_expert.up_proj.weight")
            == "model.layers.0.mlp.shared_experts.up_proj.weight"
        )

    def test_router_gate_different_layers(self):
        assert (
            _kf_to_hf_key("layers.15.mlp.router.gate.weight") == "model.layers.15.mlp.gate.weight"
        )


class TestMoEHFtoKFKeyMapping:
    def test_router_gate(self):
        assert _hf_to_kf_key("model.layers.0.mlp.gate.weight") == "layers.0.mlp.router.gate.weight"

    def test_router_expert_bias(self):
        assert _hf_to_kf_key("model.layers.0.mlp.expert_bias") == "layers.0.mlp.router.expert_bias"

    def test_expert_projections(self):
        assert (
            _hf_to_kf_key("model.layers.0.mlp.experts.0.gate_proj.weight")
            == "layers.0.mlp.experts.0.gate_proj.weight"
        )

    def test_shared_experts(self):
        assert (
            _hf_to_kf_key("model.layers.0.mlp.shared_experts.gate_proj.weight")
            == "layers.0.mlp.shared_expert.gate_proj.weight"
        )

    def test_gate_proj_not_confused_with_router_gate(self):
        """mlp.gate_proj should NOT become mlp.router.gate_proj."""
        assert (
            _hf_to_kf_key("model.layers.0.mlp.gate_proj.weight") == "layers.0.mlp.gate_proj.weight"
        )


class TestMoERoundTrip:
    @pytest.mark.parametrize(
        "kf_key",
        [
            "layers.0.mlp.router.gate.weight",
            "layers.0.mlp.router.expert_bias",
            "layers.0.mlp.experts.0.gate_proj.weight",
            "layers.0.mlp.experts.0.up_proj.weight",
            "layers.0.mlp.experts.0.down_proj.weight",
            "layers.0.mlp.experts.7.gate_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.15.mlp.router.gate.weight",
            "layers.15.mlp.experts.3.down_proj.weight",
        ],
    )
    def test_moe_kf_to_hf_to_kf(self, kf_key):
        assert _hf_to_kf_key(_kf_to_hf_key(kf_key)) == kf_key


def _refuse_to_build(*args, **kwargs):
    raise AssertionError("from_pretrained builds the model and can run the repository's code")


class _Opaque:
    """An object ``torch.load(weights_only=True)`` refuses to unpickle."""


class TestWeightFiles:
    """hf-to-dcp reads the weight files transformers loads by default, locally or from the Hub.

    A Hub model is read without building a Hugging Face model or importing the repository's
    code: each file comes from ``hf_hub_download``.
    """

    @staticmethod
    def _repo(path: Path, files: dict[str, dict], index: dict | None = None) -> Path:
        """A model directory holding ``files``, an index if given, and a config whose auto_map
        names the repository's own module, which writes ``ran.txt`` beside it if imported."""
        from safetensors.torch import save_file

        path.mkdir(parents=True)
        auto_map = {"AutoModelForCausalLM": "probe.ProbeForCausalLM"}
        (path / "config.json").write_text(json.dumps({"model_type": "probe", "auto_map": auto_map}))
        marker = str(path / "ran.txt")
        (path / "probe.py").write_text(
            f"import pathlib\npathlib.Path({marker!r}).write_text('ran')\n"
        )
        for name, weights in files.items():
            if name.endswith(".safetensors"):
                save_file(weights, path / name)
            else:
                torch.save(weights, path / name)
        if index is not None:
            (path / index["name"]).write_text(json.dumps({"weight_map": index["weight_map"]}))
        return path

    @staticmethod
    def _hub(monkeypatch, snapshots: list[Path], moves_to: str | None = None) -> list[tuple]:
        """Serve snapshot directories, each named for its commit, through ``hf_hub_download``.

        ``main`` is at the first snapshot; with ``moves_to`` it moves to that commit after the
        first file is downloaded. Returns each request's ``(filename, revision)``.
        """
        import huggingface_hub
        import transformers
        from huggingface_hub.errors import EntryNotFoundError

        commits = {snapshot.name: snapshot for snapshot in snapshots}
        refs = {"main": snapshots[0].name}
        asked: list[tuple] = []

        def hf_hub_download(repo_id, filename, revision=None, **kwargs):
            assert repo_id == "org/model"
            asked.append((filename, revision))
            commit = revision or refs["main"]
            if not (commits[commit] / filename).exists():
                raise EntryNotFoundError(f"{filename} is not in {repo_id} at {commit}")
            if moves_to is not None:
                refs["main"] = moves_to
            return str(commits[commit] / filename)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _refuse_to_build)
        return asked

    INDEX = {
        "name": "model.safetensors.index.json",
        "weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"},
    }

    def test_a_hub_id_reads_model_safetensors_and_not_an_adapter(self, tmp_path, monkeypatch):
        from convert_checkpoint import _load_hf_hub

        model = {"model.norm.weight": torch.arange(4, dtype=torch.float32)}
        adapter = {"model.norm.weight": torch.zeros(4)}
        files = {"model.safetensors": model, "adapter_model.safetensors": adapter}
        repo = self._repo(tmp_path / "c0", files)
        asked = self._hub(monkeypatch, [repo])

        loaded = _load_hf_hub("org/model")

        assert asked == [("model.safetensors", None)]
        assert torch.equal(loaded["model.norm.weight"], model["model.norm.weight"])
        assert loaded["model.norm.weight"].dtype == torch.float32
        assert not (repo / "ran.txt").exists(), "the repository's module was imported"

    def test_a_hub_index_gives_its_shards_and_no_other_file(self, tmp_path, monkeypatch):
        from convert_checkpoint import _load_hf_hub

        files = {
            "model-1.safetensors": {"a": torch.ones(2)},
            "model-2.safetensors": {"b": torch.full((2,), 2.0)},
            "stray.safetensors": {"a": torch.zeros(2)},
        }
        repo = self._repo(tmp_path / "c0", files, self.INDEX)
        asked = self._hub(monkeypatch, [repo])

        loaded = _load_hf_hub("org/model")

        assert asked == [
            ("model.safetensors", None),
            (self.INDEX["name"], None),
            ("model-1.safetensors", "c0"),
            ("model-2.safetensors", "c0"),
        ]
        assert torch.equal(loaded["a"], torch.ones(2))
        assert torch.equal(loaded["b"], files["model-2.safetensors"]["b"])

    def test_a_hub_shard_comes_from_its_index_commit_while_main_moves(self, tmp_path, monkeypatch):
        from convert_checkpoint import _load_hf_hub

        old = {
            "model-1.safetensors": {"a": torch.ones(2)},
            "model-2.safetensors": {"b": torch.ones(2)},
        }
        new = {
            "model-1.safetensors": {"a": torch.zeros(2)},
            "model-2.safetensors": {"b": torch.zeros(2)},
        }
        snapshots = [
            self._repo(tmp_path / "old", old, self.INDEX),
            self._repo(tmp_path / "new", new, self.INDEX),
        ]
        asked = self._hub(monkeypatch, snapshots, moves_to="new")

        loaded = _load_hf_hub("org/model")

        assert torch.equal(loaded["a"], torch.ones(2)) and torch.equal(loaded["b"], torch.ones(2))
        assert asked[1:] == [
            (self.INDEX["name"], None),
            ("model-1.safetensors", "old"),
            ("model-2.safetensors", "old"),
        ]

    def test_a_single_file_is_read_before_an_index_beside_it(self, tmp_path, monkeypatch):
        """Transformers takes ``model.safetensors`` before its index, so a stale index is unused."""
        from convert_checkpoint import _load_hf_hub, _load_hf_local

        files = {
            "model.safetensors": {"a": torch.ones(2)},
            "model-1.safetensors": {"a": torch.zeros(2)},
        }
        index = {"name": "model.safetensors.index.json", "weight_map": {"a": "model-1.safetensors"}}
        repo = self._repo(tmp_path / "c0", files, index)
        self._hub(monkeypatch, [repo])

        assert torch.equal(_load_hf_hub("org/model")["a"], torch.ones(2))
        assert torch.equal(_load_hf_local(repo)["a"], torch.ones(2))

    def test_a_hub_shard_that_cannot_be_had_fails_the_load(self, tmp_path, monkeypatch):
        """A shard missing from the repository or from an offline cache is not left out."""
        from convert_checkpoint import _load_hf_hub
        from huggingface_hub.errors import EntryNotFoundError

        files = {"model-1.safetensors": {"a": torch.ones(2)}}
        repo = self._repo(tmp_path / "c0", files, self.INDEX)
        self._hub(monkeypatch, [repo])

        with pytest.raises(EntryNotFoundError, match="model-2.safetensors"):
            _load_hf_hub("org/model")

    def test_a_hub_repository_without_safetensors_gives_its_bin(self, tmp_path, monkeypatch):
        from convert_checkpoint import _load_hf_hub

        files = {
            "pytorch_model.bin": {"model.norm.weight": torch.ones(4)},
            "adapter_model.safetensors": {"model.norm.weight": torch.zeros(4)},
        }
        repo = self._repo(tmp_path / "c0", files)
        asked = self._hub(monkeypatch, [repo])

        loaded = _load_hf_hub("org/model")

        assert asked[-1] == ("pytorch_model.bin", None)
        assert torch.equal(loaded["model.norm.weight"], torch.ones(4))

    def test_a_hub_repository_without_weight_files_is_refused(self, tmp_path, monkeypatch):
        from convert_checkpoint import _load_hf_hub

        repo = self._repo(tmp_path / "c0", {"adapter_model.safetensors": {"x": torch.ones(1)}})
        self._hub(monkeypatch, [repo])

        with pytest.raises(FileNotFoundError, match="Found none of model.safetensors"):
            _load_hf_hub("org/model")

    def test_a_local_directory_reads_its_bin_and_not_an_adapter(self, tmp_path):
        """Beside a ``pytorch_model.bin``, an adapter's safetensors file no longer replaces it."""
        from convert_checkpoint import _load_hf_local

        files = {
            "pytorch_model.bin": {"model.norm.weight": torch.ones(4)},
            "adapter_model.safetensors": {"model.norm.weight": torch.zeros(4)},
        }
        loaded = _load_hf_local(self._repo(tmp_path / "local", files))
        assert torch.equal(loaded["model.norm.weight"], torch.ones(4))

    def test_a_local_index_gives_its_shards_and_no_other_file(self, tmp_path):
        from convert_checkpoint import _load_hf_local

        files = {
            "model-1.safetensors": {"a": torch.ones(2)},
            "model-2.safetensors": {"b": torch.ones(2)},
            "stray.safetensors": {"a": torch.zeros(2)},
        }
        loaded = _load_hf_local(self._repo(tmp_path / "local", files, self.INDEX))
        assert torch.equal(loaded["a"], torch.ones(2)) and set(loaded) == {"a", "b"}

    def test_a_local_directory_without_standard_names_reads_every_safetensors(self, tmp_path):
        from convert_checkpoint import _load_hf_local

        files = {
            "part-a.safetensors": {"a": torch.ones(2)},
            "part-b.safetensors": {"b": torch.ones(2)},
        }
        loaded = _load_hf_local(self._repo(tmp_path / "local", files))
        assert set(loaded) == {"a", "b"}

    def test_a_bin_file_holding_other_objects_is_refused(self, tmp_path):
        """``.bin`` files load with ``weights_only=True``: tensors and plain data only."""
        import pickle

        from convert_checkpoint import _load_hf_local

        local = self._repo(tmp_path / "local", {"pytorch_model.bin": {"x": _Opaque()}})
        with pytest.raises(pickle.UnpicklingError):
            _load_hf_local(local)

    def test_hf_to_dcp_imports_the_weights_of_a_hub_model(self, tmp_path, monkeypatch):
        from convert_checkpoint import hf_to_dcp

        config = ModelConfig(dim=16, n_layers=1, n_heads=2, vocab_size=32, max_seq_len=16)
        torch.manual_seed(0)
        source = Transformer(config).state_dict()
        weights = {_kf_to_hf_key(key): value.clone() for key, value in source.items()}
        repo = self._repo(tmp_path / "c0", {"model.safetensors": weights})
        self._hub(monkeypatch, [repo])

        hf_to_dcp("org/model", str(tmp_path / "dcp"), config)

        torch.manual_seed(1)
        state = {"model": Transformer(config).state_dict()}
        dcp.load(state, checkpoint_id=str(tmp_path / "dcp"))
        for key, value in source.items():
            assert torch.equal(state["model"][key], value), key
        assert not (repo / "ran.txt").exists(), "the repository's module was imported"


class TestBuildHFConfig:
    def test_config_fields(self):
        mc = ModelConfig(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=32000,
            max_seq_len=4096,
        )
        hf_cfg = _build_hf_config(mc)

        assert hf_cfg["model_type"] == "llama"
        assert hf_cfg["architectures"] == ["LlamaForCausalLM"]
        assert hf_cfg["hidden_size"] == 4096
        assert hf_cfg["num_hidden_layers"] == 32
        assert hf_cfg["num_attention_heads"] == 32
        assert hf_cfg["num_key_value_heads"] == 8
        assert hf_cfg["vocab_size"] == 32000
        assert hf_cfg["max_position_embeddings"] == 4096
        assert hf_cfg["torch_dtype"] == "bfloat16"

    def test_tie_embeddings_propagated(self):
        mc = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, tie_embeddings=True)
        hf_cfg = _build_hf_config(mc)
        assert hf_cfg["tie_word_embeddings"] is True

        mc2 = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, tie_embeddings=False)
        hf_cfg2 = _build_hf_config(mc2)
        assert hf_cfg2["tie_word_embeddings"] is False

    def test_moe_config(self):
        mc = ModelConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=32000,
            num_experts=8,
            moe_top_k=2,
            moe_aux_loss_weight=0.01,
        )
        hf_cfg = _build_hf_config(mc)

        assert hf_cfg["model_type"] == "mixtral"
        assert hf_cfg["architectures"] == ["MixtralForCausalLM"]
        assert hf_cfg["num_local_experts"] == 8
        assert hf_cfg["num_experts_per_tok"] == 2
        assert hf_cfg["router_aux_loss_coef"] == 0.01

    def test_head_dim_is_exported(self):
        """A decoupled width is not recoverable from dim // n_heads, so the
        export must carry it explicitly."""
        coupled = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        assert _build_hf_config(coupled)["head_dim"] == 16

        decoupled = ModelConfig(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim_override=32, vocab_size=256
        )
        hf_cfg = _build_hf_config(decoupled)
        assert hf_cfg["head_dim"] == 32 != decoupled.dim // decoupled.n_heads

    def test_moe_head_dim_is_exported(self):
        mc = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim_override=32,
            vocab_size=256,
            num_experts=4,
            moe_top_k=2,
        )
        assert _build_hf_config(mc)["head_dim"] == 32

    def test_dim_indivisible_by_n_heads_is_rejected(self):
        """``head_dim_override`` lifts that requirement for training, but the
        target architectures reject it even with an explicit head_dim."""
        mc = ModelConfig(
            dim=100, n_layers=2, n_heads=8, n_kv_heads=8, head_dim_override=16, vocab_size=256
        )
        with pytest.raises(ValueError, match="not divisible by n_heads"):
            _build_hf_config(mc)

    @pytest.mark.parametrize("override", [0, 32])
    def test_exported_config_rebuilds_the_attention_shapes(self, override):
        """The end the export exists for: a model built from config.json must
        take the checkpoint's attention weights."""
        import torch

        transformers = pytest.importorskip("transformers")

        from kempnerforge.model.transformer import Transformer

        mc = ModelConfig(
            dim=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim_override=override,
            vocab_size=256,
            max_seq_len=64,
        )
        hf_cfg = _build_hf_config(mc)
        cfg = transformers.LlamaConfig(
            hidden_size=hf_cfg["hidden_size"],
            num_attention_heads=hf_cfg["num_attention_heads"],
            num_key_value_heads=hf_cfg["num_key_value_heads"],
            head_dim=hf_cfg["head_dim"],
            intermediate_size=hf_cfg["intermediate_size"],
            num_hidden_layers=hf_cfg["num_hidden_layers"],
            vocab_size=hf_cfg["vocab_size"],
            max_position_embeddings=hf_cfg["max_position_embeddings"],
        )
        with torch.device("meta"):
            kf = Transformer(mc).state_dict()
            hf = transformers.LlamaForCausalLM(cfg).state_dict()
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert (
                hf[f"model.layers.0.self_attn.{proj}.weight"].shape
                == kf[f"layers.0.attention.{proj}.weight"].shape
            ), proj

    def test_dense_has_no_moe_fields(self):
        mc = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256)
        hf_cfg = _build_hf_config(mc)
        assert "num_local_experts" not in hf_cfg
        assert hf_cfg["model_type"] == "llama"


def _tiny_dcp(path: Path, model_config: ModelConfig) -> None:
    """Save a freshly initialized model as a single-process DCP checkpoint."""
    dcp.save({"model": Transformer(model_config).state_dict()}, checkpoint_id=str(path))


def _written(hf_dir: Path) -> tuple[set[torch.dtype], str]:
    """The dtypes of an export's tensors and the ``torch_dtype`` its config.json names."""
    weights = load_file(hf_dir / "model.safetensors")
    hf_cfg = json.loads((hf_dir / "config.json").read_text())
    return {t.dtype for t in weights.values()}, hf_cfg["torch_dtype"]


class TestExportDtypeMetadata:
    @pytest.mark.parametrize(
        ("dtype", "name"),
        [(torch.float32, "float32"), (torch.bfloat16, "bfloat16"), (torch.float16, "float16")],
    )
    def test_config_json_names_the_written_dtype(self, tmp_path, tiny_model_config, dtype, name):
        """torch_dtype names the dtype the tensors were written in, not a fixed bfloat16."""
        _tiny_dcp(tmp_path / "dcp", tiny_model_config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), tiny_model_config, dtype=dtype)
        assert _written(tmp_path / "hf") == ({dtype}, name)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_transformers_loads_the_written_dtype(self, tmp_path, tiny_model_config, dtype):
        """``from_pretrained(dtype="auto")`` follows torch_dtype, so a fixed
        bfloat16 there would load these weights as bf16."""
        transformers = pytest.importorskip("transformers")
        _tiny_dcp(tmp_path / "dcp", tiny_model_config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), tiny_model_config, dtype=dtype)
        model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf", dtype="auto")
        assert {p.dtype for p in model.parameters()} == {dtype}


_TINY_TOML = """\
[model]
dim = 64
n_layers = 2
n_heads = 2
n_kv_heads = 2
vocab_size = 256
max_seq_len = 64
"""


def _run_dcp_to_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_dtype: str | None, flag: str | None
) -> tuple[set[torch.dtype], str]:
    """Export a tiny checkpoint through the CLI and return what it wrote.

    ``export_dtype`` goes in the TOML's ``[checkpoint]`` table and ``flag`` is
    passed as ``--dtype``; ``None`` leaves either out.
    """
    toml = tmp_path / "job.toml"
    table = "" if export_dtype is None else f'[checkpoint]\nexport_dtype = "{export_dtype}"\n'
    toml.write_text(_TINY_TOML + table)
    _tiny_dcp(tmp_path / "dcp", load_config(toml, cli_args=[]).model)
    argv = ["convert_checkpoint.py", "dcp-to-hf", "--config", str(toml)]
    argv += ["--dcp-dir", str(tmp_path / "dcp"), "--hf-dir", str(tmp_path / "hf")]
    if flag is not None:
        argv += ["--dtype", flag]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return _written(tmp_path / "hf")


class TestExportDtypePrecedence:
    """dcp-to-hf takes ``--dtype``, then ``checkpoint.export_dtype``, then bfloat16."""

    def test_flag_alone_sets_the_dtype(self, tmp_path, monkeypatch):
        written = _run_dcp_to_hf(tmp_path, monkeypatch, None, "float16")
        assert written == ({torch.float16}, "float16")

    def test_config_alone_sets_the_dtype(self, tmp_path, monkeypatch):
        written = _run_dcp_to_hf(tmp_path, monkeypatch, "float32", None)
        assert written == ({torch.float32}, "float32")

    @pytest.mark.parametrize(
        ("flag", "dtype"), [("bfloat16", torch.bfloat16), ("float16", torch.float16)]
    )
    def test_flag_overrides_the_config(self, tmp_path, monkeypatch, flag, dtype):
        """Even a flag that names the default wins over the TOML."""
        written = _run_dcp_to_hf(tmp_path, monkeypatch, "float32", flag)
        assert written == ({dtype}, flag)

    def test_bfloat16_without_flag_or_config(self, tmp_path, monkeypatch):
        written = _run_dcp_to_hf(tmp_path, monkeypatch, None, None)
        assert written == ({torch.bfloat16}, "bfloat16")

    @pytest.mark.parametrize(
        ("export_dtype", "flag", "source"),
        [(None, "float32", "--dtype"), ("float32", None, "checkpoint.export_dtype")],
    )
    def test_log_names_where_the_dtype_came_from(
        self, tmp_path, monkeypatch, caplog, export_dtype, flag, source
    ):
        """The log line says whether the flag or the config chose the export dtype."""
        with caplog.at_level(logging.INFO, logger="convert_checkpoint"):
            _run_dcp_to_hf(tmp_path, monkeypatch, export_dtype, flag)
        assert f"Export dtype: float32 (from {source})" in caplog.text


def _random_dcp(path: Path, model_config: ModelConfig) -> Transformer:
    """Save a model whose every weight, norms included, is drawn at random; return it."""
    model = Transformer(model_config)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(generator=generator)
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(path))
    return model


class TestTiedExport:
    """A model with tied embeddings exports at every dtype, float32 included."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_head_is_written_as_a_copy_of_the_embedding(self, tmp_path, tiny_model_config, dtype):
        """At float32 the cast leaves the tied head on the embedding's storage, which
        safetensors refuses, so the export writes the head as its own copy."""
        config = replace(tiny_model_config, tie_embeddings=True)
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=dtype)
        weights = load_file(tmp_path / "hf" / "model.safetensors")
        embedding = source.state_dict()["token_embedding.embedding.weight"].to(dtype)
        assert torch.equal(weights["model.embed_tokens.weight"], embedding)
        assert torch.equal(weights["lm_head.weight"], embedding)
        assert json.loads((tmp_path / "hf" / "config.json").read_text())["tie_word_embeddings"]

    def test_transformers_loads_a_float32_export_tied(self, tmp_path, tiny_model_config):
        """transformers ties the copied head back to the embedding, at float32."""
        transformers = pytest.importorskip("transformers")
        config = replace(tiny_model_config, tie_embeddings=True)
        source = _random_dcp(tmp_path / "dcp", config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), config, dtype=torch.float32)
        model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf", dtype="auto")
        head, embedding = model.get_output_embeddings().weight, model.get_input_embeddings().weight
        assert head is embedding
        assert embedding.dtype == torch.float32
        assert torch.equal(embedding, source.state_dict()["token_embedding.embedding.weight"])


def _llama_weights(model: Transformer) -> dict[str, torch.Tensor]:
    """A dense model's weights under the names ``LlamaForCausalLM`` loads them by."""
    assert model.token_embedding is not None and model.output_head is not None
    weights = {
        "model.embed_tokens.weight": model.token_embedding.embedding.weight,
        "model.norm.weight": model.norm.weight,
        "lm_head.weight": model.output_head.proj.weight,
    }
    for i, block in model.layers.items():
        prefix = f"model.layers.{i}"
        weights[f"{prefix}.input_layernorm.weight"] = block.get_parameter("attention_norm.weight")
        weights[f"{prefix}.post_attention_layernorm.weight"] = block.get_parameter(
            "mlp_norm.weight"
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.self_attn.{proj}.weight"] = getattr(block.attention, proj).weight
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{prefix}.mlp.{proj}.weight"] = getattr(block.mlp, proj).weight
    return weights


class TestExportValues:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_exported_tensors_are_the_source_weights(self, tmp_path, tiny_model_config, dtype):
        """Each exported tensor is the source weight its name stands for, cast to the
        export dtype, so a key mapping that swaps or drops weights fails here even
        though every dtype comes out right."""
        source = _random_dcp(tmp_path / "dcp", tiny_model_config)
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), tiny_model_config, dtype=dtype)
        weights = load_file(tmp_path / "hf" / "model.safetensors")
        expected = _llama_weights(source)
        assert weights.keys() == expected.keys()
        for name, tensor in expected.items():
            assert torch.equal(weights[name], tensor.detach().to(dtype)), name
