from __future__ import annotations

from pathlib import Path


def appdata_root() -> Path:
    try:
        from toolbox_app.core.paths import user_data_dir  # type: ignore

        return user_data_dir()
    except Exception:
        from pathlib import Path as _Path
        import os

        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(_Path.home())
        root = _Path(base) / "EngineeringToolbox"
        root.mkdir(parents=True, exist_ok=True)
        return root
