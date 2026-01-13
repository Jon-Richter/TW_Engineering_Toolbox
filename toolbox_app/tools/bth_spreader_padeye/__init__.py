"""Engineering Toolbox plugin."""
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "TOOL":
        from .tool import TOOL

        return TOOL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TOOL"]
