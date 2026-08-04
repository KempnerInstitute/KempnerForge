"""Component registry for extensible model/optimizer/scheduler construction.

Provides a simple name → callable mapping so researchers can register custom
components without modifying core code.

Usage:
    from kempnerforge.config.registry import registry

    # Register a custom model
    @registry.register_model("my_transformer")
    def build_my_transformer(config):
        return MyTransformer(config)

    # Retrieve it
    builder = registry.get_model("my_transformer")
    model = builder(config)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """Central registry for named components."""

    def __init__(self) -> None:
        self._stores: dict[str, dict[str, Any]] = {}

    def _get_store(self, category: str) -> dict[str, Any]:
        if category not in self._stores:
            self._stores[category] = {}
        return self._stores[category]

    def register(self, category: str, name: str, obj: Any) -> Any:
        """Register an object under category/name."""
        store = self._get_store(category)
        if name in store:
            raise ValueError(f"{category}/{name} is already registered")
        store[name] = obj
        return obj

    def get(self, category: str, name: str) -> Any:
        """Retrieve a registered object."""
        store = self._get_store(category)
        if name not in store:
            available = list(store.keys())
            raise KeyError(f"Unknown {category}: '{name}'. Available: {available}")
        return store[name]

    def list(self, category: str) -> list[str]:
        """List all registered names in a category."""
        return list(self._get_store(category).keys())

    # Convenience methods for common categories

    def register_model(self, name: str) -> Callable:
        """Decorator to register a model builder."""

        def decorator(fn: Callable) -> Callable:
            self.register("model", name, fn)
            return fn

        return decorator

    def get_model(self, name: str) -> Callable:
        return self.get("model", name)

    def register_optimizer(self, name: str) -> Callable:
        """Decorator to register an optimizer builder."""

        def decorator(fn: Callable) -> Callable:
            self.register("optimizer", name, fn)
            return fn

        return decorator

    def get_optimizer(self, name: str) -> Callable:
        return self.get("optimizer", name)

    def register_scheduler(self, name: str) -> Callable:
        """Decorator to register a scheduler builder."""

        def decorator(fn: Callable) -> Callable:
            self.register("scheduler", name, fn)
            return fn

        return decorator

    def get_scheduler(self, name: str) -> Callable:
        return self.get("scheduler", name)

    def register_loss(self, name: str) -> Callable:
        """Decorator to register a loss function."""

        def decorator(fn: Callable) -> Callable:
            self.register("loss", name, fn)
            return fn

        return decorator

    def get_loss(self, name: str) -> Callable:
        return self.get("loss", name)

    def register_vision_encoder(self, name: str) -> Callable:
        """Decorator to register a vision encoder builder.

        Builders take ``(path: str, **kwargs)`` and return a ``VisionEncoder``
        that produces ``(B, num_tokens, feature_dim)`` patch tokens from
        ``(B, 3, H, W)`` pixel values.
        """

        def decorator(fn: Callable) -> Callable:
            self.register("vision_encoder", name, fn)
            return fn

        return decorator

    def get_vision_encoder(self, name: str) -> Callable:
        return self.get("vision_encoder", name)

    def list_vision_encoders(self) -> list[str]:
        return self.list("vision_encoder")

    def register_vlm_config(self, name: str) -> Callable:
        """Decorator to register a ``VLMConfig`` subclass.

        Registers an arch discriminator (e.g. ``"joint_decoder"``,
        ``"cross_attention"``) so the loader can dispatch a TOML
        ``[vlm]`` table with ``arch = "..."`` to the right subclass
        and ``VLMConfig.for_arch`` can resolve programmatically.
        """

        def decorator(cls: Any) -> Any:
            self.register("vlm_config", name, cls)
            return cls

        return decorator

    def get_vlm_config(self, name: str) -> Any:
        return self.get("vlm_config", name)

    def list_vlm_configs(self) -> list[str]:
        return self.list("vlm_config")

    def register_modality_strategy(self, name: str) -> Callable:
        """Decorator to register a ``ModalityStrategy`` for a VLM arch.

        The strategy prepares a ``ModalityContext`` from raw
        ``(pixel_values, input_ids)`` for one arch (Joint-Decoder,
        Cross-Attention, MoT, ...). ``VLMWrapper`` looks up its
        strategy by arch name.
        """

        def decorator(cls: Any) -> Any:
            self.register("modality_strategy", name, cls)
            return cls

        return decorator

    def get_modality_strategy(self, name: str) -> Any:
        return self.get("modality_strategy", name)

    def list_modality_strategies(self) -> list[str]:
        return self.list("modality_strategy")

    def register_adapter(self, name: str) -> Callable:
        """Decorator to register an adapter builder.

        Adapters project image features (``(B, N, feature_dim)``) into the
        LLM embedding space (``(B, N, model.dim)``). Builders take
        ``(in_dim, out_dim, **kwargs)`` and return an ``nn.Module`` with the
        expected forward shape.
        """

        def decorator(fn: Callable) -> Callable:
            self.register("adapter", name, fn)
            return fn

        return decorator

    def get_adapter(self, name: str) -> Callable:
        return self.get("adapter", name)

    def list_adapters(self) -> list[str]:
        return self.list("adapter")

    def register_video_dataset(self, name: str) -> Callable:
        """Decorator to register a video-dataset builder.

        Builders take ``(video_config, tokenizer_path, max_text_len)`` and return
        a map-style ``Dataset`` whose samples ``VideoCollator`` batches (see
        ``kempnerforge.data.video_dataset.VideoDataset``). Selected by
        ``[video].dataset_type``.
        """

        def decorator(fn: Callable) -> Callable:
            self.register("video_dataset", name, fn)
            return fn

        return decorator

    def get_video_dataset(self, name: str) -> Callable:
        return self.get("video_dataset", name)

    def list_video_datasets(self) -> list[str]:
        return self.list("video_dataset")

    def register_sampling_policy(self, name: str) -> Callable:
        """Decorator to register a frame-sampling policy.

        Policies take ``(duration_s, fps, min_frames, max_frames)`` and return a
        sorted list of timestamps (seconds) to sample from a clip. Selected by
        ``[video].sampling_policy``.
        """

        def decorator(fn: Callable) -> Callable:
            self.register("sampling_policy", name, fn)
            return fn

        return decorator

    def get_sampling_policy(self, name: str) -> Callable:
        return self.get("sampling_policy", name)

    def list_sampling_policies(self) -> list[str]:
        return self.list("sampling_policy")

    def register_time_embedding(self, name: str) -> Callable:
        """Decorator to register a time-embedding builder.

        Builders take ``(dim, **kwargs)`` and return an ``nn.Module`` mapping
        per-frame timestamps ``(B, F)`` in seconds to an additive embedding
        ``(B, F, dim)`` (and exposing ``reset_parameters()`` for meta-device
        builds). Selected by ``[time_embedding].type`` on the VLM video path.
        """

        def decorator(fn: Callable) -> Callable:
            self.register("time_embedding", name, fn)
            return fn

        return decorator

    def get_time_embedding(self, name: str) -> Callable:
        return self.get("time_embedding", name)

    def list_time_embeddings(self) -> list[str]:
        return self.list("time_embedding")

    def register_dyn_ckpt_strategy(self, name: str) -> Callable:
        """Decorator to register a dynamic-checkpointing-window strategy.

        Strategies take ``(window: DynamicCheckpointWindow, step: int)`` and
        return ``True`` iff ``step`` should be saved by the dynamic window.
        ``DynamicCheckpointWindow.is_milestone`` looks the strategy up here by
        name; ``CheckpointManager._cleanup`` exempts every step the strategy
        fires on from ``keep_last_n`` retention.

        Ships with ``"power2"`` registered by default in
        ``kempnerforge/config/checkpoint.py``.
        """

        def decorator(fn: Callable) -> Callable:
            self.register("dyn_ckpt_strategy", name, fn)
            return fn

        return decorator

    def get_dyn_ckpt_strategy(self, name: str) -> Callable:
        return self.get("dyn_ckpt_strategy", name)

    def list_dyn_ckpt_strategies(self) -> list[str]:
        return self.list("dyn_ckpt_strategy")


# Global registry instance
registry = Registry()
