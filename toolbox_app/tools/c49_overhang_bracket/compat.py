from __future__ import annotations

"""Compatibility shims for running inside the Engineering Toolbox host app.

This tool is distributed as a plugin under `toolbox_app.tools.<tool_id>`.
In the host application, core helpers live under `toolbox_app.core`.

For local/CI execution (e.g., running tests_smoke.py standalone), these
imports may not be available. In that case we provide minimal fallbacks.

Do NOT import Qt here.
"""

from dataclasses import dataclass
from pathlib import Path
import os


# -----------------------------
# Tool metadata (preferred: host)
# -----------------------------
try:
    # Host app (preferred)
    from toolbox_app.core.tool_base import ToolMeta as ToolMeta  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(frozen=True)
    class ToolMeta:
        id: str
        name: str
        category: str
        version: str
        description: str


# -----------------------------
# Paths (preferred: host)
# -----------------------------
def user_data_dir() -> Path:
    """Return the Engineering Toolbox local user data root.

    Host app provides `toolbox_app.core.paths.user_data_dir()`.
    Fallback uses LOCALAPPDATA/APPDATA/home.
    """
    try:
        from toolbox_app.core.paths import user_data_dir as _udd  # type: ignore

        return Path(_udd())
    except Exception:  # pragma: no cover
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "EngineeringToolbox"
