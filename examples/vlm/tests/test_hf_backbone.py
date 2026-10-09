"""Tests for ``scripts/convert_hf_backbone.py``.

Every test builds a tiny random source offline with ``transformers``: a round
trip through ``CheckpointManager.load``, each refusal, and determinism.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.loader import load_config
from kempnerforge.model.vlm import build_vlm_wrapper

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_hf_backbone.py"
_spec = importlib.util.spec_from_file_location("_vlm_convert_hf_backbone", SCRIPT)
assert _spec is not None and _spec.loader is not None
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

SOURCE_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 96,
    # The epsilon every norm of the built transformer uses, so the logits below
    # match bit for bit.
    "rms_norm_eps": 1e-5,
    "rope_theta": 1e6,
    "max_position_embeddings": 64,
}
TARGET_MODEL = {
    "dim": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "vocab_size": 128,
    "ffn_hidden_dim": 96,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "rope_theta": 1e6,
    "tie_embeddings": True,
    "max_seq_len": 64,
}

Sources = dict[bool, tuple[Path, Any]]


def _save_source(path: Path, *, tie: bool) -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(tie_word_embeddings=tie, **SOURCE_CONFIG)).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:  # all-ones norms would hide a swapped norm mapping
                param.copy_(1 + 0.5 * torch.randn_like(param))
    model.save_pretrained(path)
    return model


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value) if isinstance(value, (str, list)) else repr(value)


def _target(
    tmp_path: Path,
    *,
    model: dict[str, Any] | None = None,
    vlm: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> str:
    """Write a tiny VLM config over ``TARGET_MODEL``; ``vlm={}`` drops the VLM sections."""
    sections: dict[str, dict[str, Any]] = {"model": {**TARGET_MODEL, **(model or {})}}
    if vlm != {}:
        sections["vision_encoder"] = {"type": "random", "num_tokens": 4, "feature_dim": 32}
        sections["vlm"] = {"arch": "joint_decoder", "max_text_len": 32, **(vlm or {})}
    sections["train"] = {"seq_len": 64, "seed": 7}
    sections["checkpoint"] = {"dir": str(tmp_path / "run"), **(checkpoint or {})}
    path = tmp_path / "target.toml"
    path.write_text(
        "\n".join(
            f"[{name}]\n" + "".join(f"{k} = {_toml(v)}\n" for k, v in body.items())
            for name, body in sections.items()
        )
    )
    return str(path)


def _read_dcp(path: Path) -> dict[str, torch.Tensor]:
    metadata = FileSystemReader(str(path)).read_metadata().state_dict_metadata
    state = {
        key: torch.empty(meta.size, dtype=meta.properties.dtype)
        for key, meta in metadata.items()
        if isinstance(meta, TensorStorageMetadata)
    }
    dcp.load(state, checkpoint_id=str(path))
    return state


def _refuses(source: Path, config: str, out: Path, exc: type[Exception], match: str) -> str:
    with pytest.raises(exc, match=match) as info:
        conv.convert(str(source), config, str(out))
    assert not out.exists(), "a refused conversion must write nothing"
    return str(info.value)


def _edit_weights(source: Path, edit: Callable[[dict[str, torch.Tensor]], Any]) -> None:
    weights = load_file(source / "model.safetensors")
    edit(weights)
    save_file(weights, source / "model.safetensors")


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Sources:
    """Tied and untied tiny sources, keyed by ``tie_word_embeddings``."""
    root = tmp_path_factory.mktemp("sources")
    return {tie: (root / str(tie), _save_source(root / str(tie), tie=tie)) for tie in (True, False)}


@pytest.fixture
def tied(sources: Sources) -> Path:
    return sources[True][0]


@pytest.fixture
def tied_copy(tied: Path, tmp_path: Path) -> Path:
    """A private copy of the tied source, safe to edit."""
    return Path(shutil.copytree(tied, tmp_path / "source"))


class TestMapKey:
    @pytest.mark.parametrize(
        ("hf_key", "kf_key"),
        [
            ("model.embed_tokens.weight", "token_embedding.embedding.weight"),
            ("model.norm.weight", "norm.weight"),
            ("lm_head.weight", "output_head.proj.weight"),
            ("model.layers.11.self_attn.q_proj.weight", "layers.11.attention.q_proj.weight"),
            ("model.layers.0.self_attn.k_proj.weight", "layers.0.attention.k_proj.weight"),
            ("model.layers.0.self_attn.v_proj.weight", "layers.0.attention.v_proj.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "layers.0.attention.o_proj.weight"),
            ("model.layers.3.self_attn.q_norm.weight", "layers.3.attention.q_norm.weight"),
            ("model.layers.3.self_attn.k_norm.weight", "layers.3.attention.k_norm.weight"),
            ("model.layers.1.input_layernorm.weight", "layers.1.attention_norm.weight"),
            ("model.layers.1.post_attention_layernorm.weight", "layers.1.mlp_norm.weight"),
            ("model.layers.2.mlp.gate_proj.weight", "layers.2.mlp.gate_proj.weight"),
            ("model.layers.2.mlp.up_proj.weight", "layers.2.mlp.up_proj.weight"),
            ("model.layers.2.mlp.down_proj.weight", "layers.2.mlp.down_proj.weight"),
        ],
    )
    def test_maps(self, hf_key: str, kf_key: str) -> None:
        assert conv.map_key(hf_key) == kf_key

    @pytest.mark.parametrize(
        "hf_key",
        [
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
            "model.layers.x.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight.extra",
            "model.rotary_emb.inv_freq",
            "lm_head.bias",
        ],
    )
    def test_unknown_keys_have_no_counterpart(self, hf_key: str) -> None:
        assert conv.map_key(hf_key) is None


class TestRoundTrip:
    @pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
    def test_warm_start_reproduces_source_logits(
        self, sources: Sources, tmp_path: Path, tie: bool
    ) -> None:
        source, hf_model = sources[tie]
        config_path = _target(tmp_path, model={"tie_embeddings": tie})
        out = tmp_path / "init"
        conv.main(["--hf-dir", str(source), "--config", config_path, "--out", str(out)])

        config = load_config(config_path, cli_args=[])
        assert config.vlm is not None and config.vision_encoder is not None
        assert config.adapter is not None
        torch.manual_seed(1234)  # a different init, so only the load can make them agree
        model = build_vlm_wrapper(config.model, config.vision_encoder, config.adapter, config.vlm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        manager = CheckpointManager(config.checkpoint, model, optimizer)
        assert manager.load(path=str(out), exclude_keys=["optimizer"]) == (0, 0, {})

        hf_state = hf_model.state_dict()
        transformer = model.transformer.state_dict()
        assert {conv.map_key(k) for k in hf_state} == set(transformer)
        for hf_key, tensor in hf_state.items():
            assert torch.equal(transformer[conv.map_key(hf_key)], tensor), hf_key
        head, embed = model.transformer.output_head, model.transformer.token_embedding
        assert head is not None and embed is not None
        assert (head.proj.weight is embed.embedding.weight) is tie

        saved = _read_dcp(out)
        for key, tensor in model.adapter.state_dict().items():
            assert torch.equal(tensor, saved[f"model.adapter.{key}"]), key

        tokens = torch.randint(0, 128, (2, 48), generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            assert torch.equal(model.transformer(tokens), hf_model(input_ids=tokens).logits)

    def test_older_config_layout_converts(self, tied_copy: Path, tmp_path: Path) -> None:
        """Top-level ``rope_theta`` and a null ``rope_scaling``, as older exports write it."""
        config_file = tied_copy / "config.json"
        hf_config = json.loads(config_file.read_text())
        hf_config["rope_theta"] = hf_config.pop("rope_parameters")["rope_theta"]
        hf_config["rope_scaling"] = None
        config_file.write_text(json.dumps(hf_config))
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert (out / ".metadata").is_file()

    def test_hub_model_id_resolves_through_snapshot_download(
        self, tied: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        calls: list[dict[str, Any]] = []

        def fake_snapshot_download(**kwargs: Any) -> str:
            calls.append(kwargs)
            return str(tied)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        out = conv.convert("some-org/some-model", _target(tmp_path), str(tmp_path / "out"))
        assert calls == [
            {
                "repo_id": "some-org/some-model",
                "allow_patterns": p,
                "ignore_patterns": ["*/*"],
                "revision": None,  # the snapshot is not named for a commit here
            }
            for p in (
                ["config.json"],
                ["*.safetensors", "*.safetensors.index*.json"],
            )
        ]
        assert (out / ".metadata").is_file()

    def test_hub_weights_are_fetched_at_the_commit_the_config_came_from(
        self, tied: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A moving default branch must not check one revision and convert another."""
        import huggingface_hub

        commit = "0" * 40
        snapshot = tmp_path / "cache" / commit
        shutil.copytree(tied, snapshot)
        calls: list[Any] = []

        def fake_snapshot_download(**kwargs: Any) -> str:
            calls.append(kwargs.get("revision"))
            return str(snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        out = conv.convert("some-org/some-model", _target(tmp_path), str(tmp_path / "out"))
        assert calls == [None, commit], calls
        assert (out / ".metadata").is_file()

    def test_a_source_that_moves_between_the_two_fetches_is_refused(
        self, tied: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        snapshots = [tmp_path / "cache" / ("0" * 40), tmp_path / "cache" / ("1" * 40)]
        for snapshot in snapshots:
            shutil.copytree(tied, snapshot)
        served = iter(snapshots)

        def fake_snapshot_download(**kwargs: Any) -> str:
            return str(next(served))

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="moved between the two requests"):
            conv.convert("some-org/some-model", _target(tmp_path), str(out))
        assert not out.exists(), "a refused conversion must write nothing"

    def test_hub_config_is_checked_before_any_weights_are_fetched(
        self, tied: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        patterns: list[list[str]] = []

        def fake_snapshot_download(**kwargs: Any) -> str:
            patterns.append(kwargs["allow_patterns"])
            return str(tied)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        config = _target(tmp_path, model={"rope_theta": 1e4})
        with pytest.raises(ValueError, match="rope_theta"):
            conv.convert("some-org/some-model", config, str(tmp_path / "out"))
        assert patterns == [["config.json"]]

    def test_weights_in_subfolders_are_ignored(self, tied_copy: Path, tmp_path: Path) -> None:
        (tied_copy / "sub").mkdir()
        save_file({"model.norm.weight": torch.zeros(64)}, tied_copy / "sub" / "x.safetensors")
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert not torch.equal(_read_dcp(out)["model.transformer.norm.weight"], torch.zeros(64))

    def test_warm_start_from_the_config_restores_the_converted_checkpoint(
        self, tied: Path, tmp_path: Path
    ) -> None:
        """The README path: ``load_path`` plus ``exclude_from_loading = ["optimizer"]``."""
        from kempnerforge.training.entry import restore_checkpoint

        out = tmp_path / "init"
        warm = {"load_path": str(out), "exclude_from_loading": ["optimizer"]}
        config_path = _target(tmp_path, checkpoint=warm)
        conv.convert(str(tied), config_path, str(out))

        config = load_config(config_path, cli_args=[])
        assert config.vlm is not None and config.vision_encoder is not None
        assert config.adapter is not None
        torch.manual_seed(1234)
        model = build_vlm_wrapper(config.model, config.vision_encoder, config.adapter, config.vlm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        manager = CheckpointManager(config.checkpoint, model, optimizer)
        assert restore_checkpoint(config, model, None, manager) == (0, 0)

        saved = _read_dcp(out)
        state = model.state_dict()
        assert {f"model.{key}" for key in state} == set(saved)
        for key, tensor in state.items():
            assert torch.equal(tensor, saved[f"model.{key}"]), key
        assert not optimizer.state


class TestConfigRefusals:
    @pytest.mark.parametrize(
        ("override", "problems"),
        [
            ({"rope_theta": 1e4}, ["rope_theta=1000000.0 but model.rope_theta=10000.0"]),
            ({"norm_eps": 1e-6}, ["rms_norm_eps=1e-05 but model.norm_eps=1e-06"]),
            ({"n_layers": 3}, ["num_hidden_layers=2 but model.n_layers=3"]),
            ({"n_kv_heads": 4}, ["num_key_value_heads=2 but model.n_kv_heads=4"]),
            (
                {"n_heads": 8},
                ["num_attention_heads=4 but model.n_heads=8", "head_dim=16 but model.head_dim=8"],
            ),
            (
                {"ffn_hidden_dim": 128},
                ["intermediate_size=96 but model.computed_ffn_hidden_dim=128"],
            ),
            ({"vocab_size": 256}, ["vocab_size=128 but model.vocab_size=256"]),
            (
                {"dim": 128},
                ["hidden_size=64 but model.dim=128", "head_dim=16 but model.head_dim=32"],
            ),
            ({"activation": "gelu"}, ["hidden_act='silu' but model.activation='gelu'"]),
            (
                {"tie_embeddings": False},
                ["tie_word_embeddings=True but model.tie_embeddings=False"],
            ),
            ({"qk_norm": False}, ["model.qk_norm must be true"]),
            ({"norm_type": "layernorm"}, ["model.norm_type='layernorm'"]),
            ({"num_experts": 2}, ["model.num_experts > 0"]),
        ],
        ids=[
            "rope_theta",
            "norm_eps",
            "n_layers",
            "n_kv_heads",
            "n_heads",
            "ffn_hidden_dim",
            "vocab_size",
            "dim",
            "activation",
            "tie_embeddings",
            "qk_norm",
            "norm_type",
            "num_experts",
        ],
    )
    def test_target_disagrees_with_the_source(
        self, tied: Path, tmp_path: Path, override: dict[str, Any], problems: list[str]
    ) -> None:
        config = _target(tmp_path, model=override)
        message = _refuses(tied, config, tmp_path / "out", ValueError, "does not match")
        assert len(message.splitlines()) == 1 + len(problems), message
        for problem in problems:
            assert problem in message, message

    def test_untied_source_into_tied_target(self, sources: Sources, tmp_path: Path) -> None:
        match = "tie_word_embeddings=False but model.tie_embeddings=True"
        _refuses(sources[False][0], _target(tmp_path), tmp_path / "out", ValueError, match)

    @pytest.mark.parametrize(
        ("edit", "match"),
        [
            (
                lambda c: c.update(model_type="unsupported"),
                "model_type 'unsupported' is not supported",
            ),
            (lambda c: c.pop("head_dim"), "config.json has no head_dim"),
            (
                lambda c: c["rope_parameters"].update(rope_type="yarn", factor=2.0),
                "the source uses rope_type='yarn'",
            ),
            (
                lambda c: c.update(rope_scaling={"type": "linear", "factor": 2.0}),
                "the source uses rope_type='linear'",
            ),
            (
                # The library's own validation is the source's problem, and refuses it.
                lambda c: c["rope_parameters"].update(rope_type="yarn"),
                "the installed transformers does not accept the source's config",
            ),
            (lambda c: c.update(use_sliding_window=True), "use_sliding_window is set"),
            (lambda c: c.update(attention_bias=True), "attention_bias is set"),
            (
                # Each spelling of the theta decides the positions, and which one a
                # release reads differs, so a source that disagrees with itself is
                # refused rather than resolved.
                lambda c: c.update(rope_theta=1e6, rope_parameters={"rope_theta": 1e4}),
                "the source gives more than one RoPE theta",
            ),
            (
                lambda c: c.update(rope_theta=1e6, rope_scaling={"rope_theta": 1e4}),
                "rope_scaling.rope_theta=10000.0",
            ),
            (
                lambda c: c.update(rope_scaling={"rope_type": "default", "type": "linear"}),
                "rope_scaling gives both rope_type='default' and type='linear'",
            ),
        ],
        ids=[
            "model_type",
            "missing_key",
            "rope_parameters",
            "rope_scaling",
            "rope_parameters_the_library_rejects",
            "sliding_window",
            "attention_bias",
            "rope_theta_disagrees_with_itself",
            "rope_scaling_carries_another_theta",
            "rope_scaling_names_two_types",
        ],
    )
    def test_source_the_target_cannot_express(
        self, tied_copy: Path, tmp_path: Path, edit: Callable[[dict[str, Any]], Any], match: str
    ) -> None:
        config_file = tied_copy / "config.json"
        hf_config = json.loads(config_file.read_text())
        edit(hf_config)
        config_file.write_text(json.dumps(hf_config))
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, match)

    def test_the_theta_compared_is_the_one_the_library_resolves(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        """Whichever field the installed transformers reads is what the target must match."""
        from transformers import AutoConfig

        config_file = tied_copy / "config.json"
        hf_config = json.loads(config_file.read_text())
        for field in ("rope_theta", "rope_parameters", "rope_scaling"):
            hf_config.pop(field, None)
        hf_config["rope_scaling"] = {"rope_type": "default", "rope_theta": 1e4}
        config_file.write_text(json.dumps(hf_config))
        resolved = AutoConfig.for_model(
            "qwen3", **{k: v for k, v in hf_config.items() if k != "model_type"}
        )
        assert (resolved.rope_parameters or {})["rope_theta"] == 1e4

        _refuses(
            tied_copy,
            _target(tmp_path, model={"rope_theta": 1e6}),
            tmp_path / "out",
            ValueError,
            "rope_theta=10000.0 but model.rope_theta=1000000.0",
        )
        out = conv.convert(
            str(tied_copy), _target(tmp_path, model={"rope_theta": 1e4}), str(tmp_path / "out")
        )
        assert (out / ".metadata").is_file()

    def test_the_nested_rope_theta_is_the_one_compared(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        """``rope_parameters`` overrides the top-level field, so that is what must match."""
        config_file = tied_copy / "config.json"
        hf_config = json.loads(config_file.read_text())
        hf_config["rope_parameters"] = {"rope_type": "default", "rope_theta": 1e4}
        hf_config.pop("rope_theta", None)
        config_file.write_text(json.dumps(hf_config))
        _refuses(
            tied_copy,
            _target(tmp_path, model={"rope_theta": 1e6}),
            tmp_path / "out",
            ValueError,
            r"rope_theta=10000.0 but model.rope_theta=1000000.0",
        )
        out = conv.convert(
            str(tied_copy), _target(tmp_path, model={"rope_theta": 1e4}), str(tmp_path / "out")
        )
        assert (out / ".metadata").is_file()

    def test_text_only_config(self, tied: Path, tmp_path: Path) -> None:
        config = _target(tmp_path, vlm={})
        _refuses(tied, config, tmp_path / "out", ValueError, r"has no \[vlm\] section")


class TestNormEpsilon:
    """The source normalises with one epsilon; every norm of the target must use it."""

    @staticmethod
    def _transformer(eps: float) -> torch.nn.Module:
        from kempnerforge.config.model import ModelConfig
        from kempnerforge.model.transformer import Transformer

        return Transformer(ModelConfig(**{**TARGET_MODEL, "norm_eps": eps}))

    def test_a_norm_with_another_epsilon_is_named(self) -> None:
        transformer = self._transformer(1e-5)
        conv.check_norm_eps(transformer, 1e-5)  # the built model agrees with the source

        transformer.layers["0"].attention_norm.eps = 1e-6
        with pytest.raises(ValueError, match=r"layers.0.attention_norm"):
            conv.check_norm_eps(transformer, 1e-5)

    def test_the_epsilon_is_read_off_the_model_not_the_config(
        self, tied_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A norm that takes its epsilon elsewhere is caught before anything is written."""
        build = conv.build_vlm_wrapper

        def tampered(*args: Any, **kwargs: Any) -> Any:
            wrapper = build(*args, **kwargs)
            wrapper.transformer.norm.eps = 1e-3
            return wrapper

        monkeypatch.setattr(conv, "build_vlm_wrapper", tampered)
        _refuses(
            tied_copy,
            _target(tmp_path),
            tmp_path / "out",
            ValueError,
            r"rms_norm_eps=1e-05, but 1 norms of the built transformer use another epsilon",
        )

    def test_a_source_at_another_epsilon_converts_only_if_it_reproduces(
        self, sources: Sources, tmp_path: Path
    ) -> None:
        """At an epsilon of its own, the conversion either declines or reproduces the source.

        What it must never do is write a checkpoint whose outputs differ, so the
        test asserts the outputs rather than re-deriving the condition to refuse on.
        """
        source = Path(shutil.copytree(sources[True][0], tmp_path / "source"))
        hf_config = json.loads((source / "config.json").read_text())
        hf_config["rms_norm_eps"] = 1e-6
        (source / "config.json").write_text(json.dumps(hf_config))
        from transformers import Qwen3ForCausalLM

        hf_model = Qwen3ForCausalLM.from_pretrained(source).eval()
        config_path = _target(tmp_path, model={"norm_eps": 1e-6})
        out = tmp_path / "out"

        try:
            conv.convert(str(source), config_path, str(out))
        except ValueError as error:
            assert "use another epsilon" in str(error), error
            assert not out.exists(), "a refused conversion must write nothing"
            return

        config = load_config(config_path, cli_args=[])
        assert config.vlm is not None and config.vision_encoder is not None
        assert config.adapter is not None
        torch.manual_seed(1234)
        model = build_vlm_wrapper(config.model, config.vision_encoder, config.adapter, config.vlm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        CheckpointManager(config.checkpoint, model, optimizer).load(
            path=str(out), exclude_keys=["optimizer"]
        )
        tokens = torch.randint(0, 128, (2, 48), generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            assert torch.equal(model.transformer(tokens), hf_model(input_ids=tokens).logits)


class TestWeightRefusals:
    def test_unmapped_source_key(self, tied_copy: Path, tmp_path: Path) -> None:
        bias = {"model.layers.0.self_attn.q_proj.bias": torch.zeros(64)}
        save_file(bias, tied_copy / "extra.safetensors")
        match = r"1 source keys have no transformer key: \['model.layers.0.self_attn.q_proj.bias'\]"
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, match)

    def test_key_repeated_across_files(self, tied_copy: Path, tmp_path: Path) -> None:
        save_file({"model.norm.weight": torch.ones(64)}, tied_copy / "z.safetensors")
        match = r"z.safetensors repeats keys from another file: \['model.norm.weight'\]"
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, match)

    def test_missing_source_weight(self, tied_copy: Path, tmp_path: Path) -> None:
        _edit_weights(tied_copy, lambda w: w.pop("model.layers.1.mlp.down_proj.weight"))
        match = r"1 transformer keys missing from the source \['layers.1.mlp.down_proj.weight'\]"
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, match)

    def test_target_needs_weights_the_source_lacks(self, tied: Path, tmp_path: Path) -> None:
        cross = {"arch": "cross_attention", "cross_attention_every_n_layers": 1}
        message = _refuses(
            tied,
            _target(tmp_path, vlm=cross),
            tmp_path / "out",
            ValueError,
            "transformer keys missing from the source",
        )
        assert "cross_attention_layers." in message

    def test_shape_mismatch(self, tied_copy: Path, tmp_path: Path) -> None:
        q_norm = "model.layers.0.self_attn.q_norm.weight"
        _edit_weights(tied_copy, lambda w: w.update({q_norm: torch.ones(8)}))
        match = r"\[\('layers.0.attention.q_norm.weight', \(8,\), \(16,\)\)\]"
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, match)

    def test_tied_source_with_a_distinct_head(self, tied_copy: Path, tmp_path: Path) -> None:
        _edit_weights(tied_copy, lambda w: w.update({"lm_head.weight": torch.randn(128, 64)}))
        match = "the source ties its output head, yet lm_head.weight differs"
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, match)

    def test_tied_source_whose_head_equals_the_embedding(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        embed = "model.embed_tokens.weight"
        _edit_weights(tied_copy, lambda w: w.update({"lm_head.weight": w[embed].clone()}))
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert (out / ".metadata").is_file()

    def test_untied_source_without_a_head(self, sources: Sources, tmp_path: Path) -> None:
        source = Path(shutil.copytree(sources[False][0], tmp_path / "source"))
        _edit_weights(source, lambda w: w.pop("lm_head.weight"))
        config = _target(tmp_path, model={"tie_embeddings": False})
        match = r"1 transformer keys missing from the source \['output_head.proj.weight'\]"
        _refuses(source, config, tmp_path / "out", ValueError, match)

    @pytest.mark.parametrize(
        ("name", "match"),
        [("config.json", "no config.json"), ("model.safetensors", r"no \*.safetensors")],
        ids=["config", "weights"],
    )
    def test_missing_source_file(
        self, tied_copy: Path, tmp_path: Path, name: str, match: str
    ) -> None:
        (tied_copy / name).unlink()
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", FileNotFoundError, match)


def _shard(
    source: Path, split: Callable[[str], str], index: str = "model.safetensors.index.json"
) -> dict[str, str]:
    """Re-save the source's weights as the shards ``split`` assigns each tensor to."""
    weights = load_file(source / "model.safetensors")
    (source / "model.safetensors").unlink()
    weight_map = {key: split(key) for key in weights}
    for name in sorted(set(weight_map.values())):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v for k, v in weights.items() if weight_map[k] == name},
            path,
            metadata={"format": "pt"},
        )
    (source / index).write_text(json.dumps({"weight_map": weight_map}))
    return weight_map


class TestShardIndex:
    """A sharded source names its files and tensors; the files on disk must match."""

    @staticmethod
    def _two_shards(key: str) -> str:
        tail = "model-00002-of-00002.safetensors"
        return tail if key.startswith("model.layers.1.") else "model-00001-of-00002.safetensors"

    def test_an_indexed_source_converts(self, tied_copy: Path, tmp_path: Path) -> None:
        _shard(tied_copy, self._two_shards)
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert (out / ".metadata").is_file()

    def test_a_shard_replaced_by_another_file_is_refused(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        """A stale file whose tensors happen to fit does not stand in for a named shard."""
        weight_map = _shard(tied_copy, self._two_shards)
        second = tied_copy / "model-00002-of-00002.safetensors"
        stale = load_file(second)
        second.unlink()
        save_file(stale, tied_copy / "model.safetensors", metadata={"format": "pt"})
        message = _refuses(
            tied_copy, _target(tmp_path), tmp_path / "out", ValueError, "lists 2 shards"
        )
        assert "model-00002-of-00002.safetensors" in message, message
        assert set(weight_map.values()) == {
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        }

    def test_a_variant_index_is_followed_too(self, tied_copy: Path, tmp_path: Path) -> None:
        """An index named for a weights variant is still the authority on the shards."""
        _shard(tied_copy, self._two_shards, index="model.safetensors.index.fp16.json")
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert (out / ".metadata").is_file()

        (tied_copy / "model-00002-of-00002.safetensors").unlink()
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out2", ValueError, "lists 2 shards")

    def test_two_indexes_are_refused(self, tied_copy: Path, tmp_path: Path) -> None:
        _shard(tied_copy, self._two_shards)
        (tied_copy / "model.safetensors.index.fp16.json").write_text(
            (tied_copy / "model.safetensors.index.json").read_text()
        )
        _refuses(
            tied_copy, _target(tmp_path), tmp_path / "out", ValueError, "holds 2 shard indexes"
        )

    @pytest.mark.parametrize("prefix", ["./", "shards/"], ids=["relative", "subdirectory"])
    def test_an_index_name_is_read_relative_to_the_source(
        self, tied_copy: Path, tmp_path: Path, prefix: str
    ) -> None:
        """A loader joins the name to the source directory, so these shards are present."""
        _shard(tied_copy, lambda key: f"{prefix}{self._two_shards(key)}")
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert (out / ".metadata").is_file()

    def test_an_index_name_outside_the_source_is_refused(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        _shard(tied_copy, lambda key: f"../{self._two_shards(key)}")
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, "which is outside")

    def test_a_file_the_index_does_not_name_is_left_alone(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        """A second weights variant beside the indexed one is nothing the load reads."""
        _shard(tied_copy, self._two_shards)
        save_file(
            {"model.norm.weight": torch.zeros(64)},
            tied_copy / "model.fp16.safetensors",
            metadata={"format": "pt"},
        )
        out = conv.convert(str(tied_copy), _target(tmp_path), str(tmp_path / "out"))
        assert not torch.equal(_read_dcp(out)["model.transformer.norm.weight"], torch.zeros(64))

    def test_a_shard_holding_other_tensors_is_refused(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        _shard(tied_copy, self._two_shards)
        second = tied_copy / "model-00002-of-00002.safetensors"
        weights = load_file(second)
        weights.pop(next(iter(weights)))
        save_file(weights, second, metadata={"format": "pt"})
        _refuses(
            tied_copy,
            _target(tmp_path),
            tmp_path / "out",
            ValueError,
            r"model-00002-of-00002.safetensors holds \d+ tensors but the index assigns it",
        )

    def test_an_index_without_a_weight_map_is_refused(
        self, tied_copy: Path, tmp_path: Path
    ) -> None:
        (tied_copy / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
        _refuses(tied_copy, _target(tmp_path), tmp_path / "out", ValueError, "has no weight_map")


class TestOutputRefusals:
    def test_non_empty_directory(self, tied: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "keep.txt").write_text("x")
        with pytest.raises(FileExistsError, match="not an empty directory"):
            conv.convert(str(tied), _target(tmp_path), str(out))
        assert [p.name for p in out.iterdir()] == ["keep.txt"]

    def test_existing_file(self, tied: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.write_text("x")
        with pytest.raises(FileExistsError, match="not an empty directory"):
            conv.convert(str(tied), _target(tmp_path), str(out))
        assert out.read_text() == "x"

    def test_empty_directory_is_accepted(self, tied: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        conv.convert(str(tied), _target(tmp_path), str(out))
        assert (out / ".metadata").is_file()

    @pytest.mark.parametrize("inside", ["init", "nested/init"])
    def test_inside_the_source(self, tied_copy: Path, tmp_path: Path, inside: str) -> None:
        before = sorted(p.name for p in tied_copy.iterdir())
        with pytest.raises(ValueError, match="inside the source model directory"):
            conv.convert(str(tied_copy), _target(tmp_path), str(tied_copy / inside))
        assert sorted(p.name for p in tied_copy.iterdir()) == before


class TestDeterminism:
    def _convert(self, source: Path, tmp_path: Path, name: str, **kwargs: Any) -> dict:
        return _read_dcp(
            conv.convert(str(source), _target(tmp_path), str(tmp_path / name), **kwargs)
        )

    def test_same_seed_same_checkpoint(self, tied: Path, tmp_path: Path) -> None:
        first = self._convert(tied, tmp_path, "a", seed=3)
        second = self._convert(tied, tmp_path, "b", seed=3)
        assert first.keys() == second.keys()
        assert any(key.startswith("model.adapter.") for key in first)
        for key in first:
            assert torch.equal(first[key], second[key]), key

    def test_seed_drives_only_the_fresh_weights(self, tied: Path, tmp_path: Path) -> None:
        first = self._convert(tied, tmp_path, "a", seed=3)
        second = self._convert(tied, tmp_path, "b", seed=4)
        for key in first:
            same = torch.equal(first[key], second[key])
            assert same is key.startswith("model.transformer."), key

    def test_default_seed_is_the_train_seed(self, tied: Path, tmp_path: Path) -> None:
        default = self._convert(tied, tmp_path, "a")
        explicit = self._convert(tied, tmp_path, "b", seed=7)  # the target's [train].seed
        for key in default:
            assert torch.equal(default[key], explicit[key]), key
