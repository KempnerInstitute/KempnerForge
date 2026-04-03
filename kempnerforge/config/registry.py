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
            raise KeyError(
                f"Unknown {category}: '{name}'. Available: {available}"
            )
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


# Global registry instance
registry = Registry()
