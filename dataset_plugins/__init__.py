"""Dataset plugin registry."""

from __future__ import annotations

from typing import Callable, Dict, Any, List, Tuple

Example = Dict[str, List[bool]]
DatasetResult = Tuple[List[Example], int, int]
PluginBuilder = Callable[[Dict[str, Any]], DatasetResult]

_REGISTRY: Dict[str, PluginBuilder] = {}


def register_plugin(name: str, builder: PluginBuilder) -> None:
    """Register a dataset builder under a given name."""
    if name in _REGISTRY:
        raise ValueError(f"Dataset plugin already registered: {name}")
    _REGISTRY[name] = builder


def get_plugin(name: str) -> PluginBuilder:
    """Return the plugin builder for ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset plugin '{name}'") from exc


def available_plugins() -> Dict[str, PluginBuilder]:
    """Return a copy of the registered plugins."""
    return dict(_REGISTRY)
