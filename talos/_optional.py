from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require_optional(module_name: str, package_name: str, extra_name: str) -> ModuleType:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_root = module_name.partition('.')[0]
        if exc.name == missing_root:
            raise ImportError(
                f"{package_name} is required for this Talos surface. Install it with `pip install talos[{extra_name}]`."
            ) from exc
        raise
