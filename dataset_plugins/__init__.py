"""Dataset plugin registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import pkgutil
from typing import Callable, Dict, Any, List, Optional, Tuple

IOList = List[Optional[bool]]
Example = Dict[str, IOList]
DatasetResult = Tuple[List[Example], int, int]
PluginBuilder = Callable[[Dict[str, Any]], DatasetResult]


@dataclass(frozen=True)
class DatasetPlugin:
    name: str
    builder: PluginBuilder
    default_config: Dict[str, Any]


_REGISTRY: Dict[str, DatasetPlugin] = {}


def register_plugin(name: str, builder: PluginBuilder, default_config: Dict[str, Any]) -> None:
    """Register a dataset builder under a given name."""
    if name in _REGISTRY:
        raise ValueError(f"Dataset plugin already registered: {name}")

    config = {"type": name}
    config.update(deepcopy(default_config))
    if config["type"] != name:
        raise ValueError(f"Default config for {name!r} must use the same type")

    _REGISTRY[name] = DatasetPlugin(name=name, builder=builder, default_config=config)


def get_plugin(name: str) -> PluginBuilder:
    """Return the plugin builder for ``name``."""
    try:
        return _REGISTRY[name].builder
    except KeyError as exc:
        raise KeyError(f"Unknown dataset plugin '{name}'") from exc


def get_plugin_config(name: str) -> Dict[str, Any]:
    """Return a copy of the built-in config for ``name``."""
    try:
        return deepcopy(_REGISTRY[name].default_config)
    except KeyError as exc:
        raise KeyError(f"Unknown dataset plugin '{name}'") from exc


def available_plugins() -> Dict[str, PluginBuilder]:
    """Return a copy of the registered plugins."""
    return {name: plugin.builder for name, plugin in _REGISTRY.items()}


def _load_builtin_plugins() -> None:
    """Import package modules so they can register their built-in datasets."""
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")


_load_builtin_plugins()
