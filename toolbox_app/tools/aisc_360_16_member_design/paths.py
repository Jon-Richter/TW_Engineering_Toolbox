from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def appdata_root() -> Path:
    """
    Return the EngineeringToolbox user-writable root folder.
    Prefers the toolbox's core path helper when available.
    """
    try:
        from toolbox_app.core.paths import user_data_dir  # type: ignore

        return user_data_dir()
    except Exception:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        root = Path(base) / "EngineeringToolbox"
        root.mkdir(parents=True, exist_ok=True)
        return root


def tool_output_root(tool_id: str) -> Path:
    p = appdata_root() / tool_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_dir(tool_id: str) -> Path:
    """
    Create a per-run folder under:
      %LOCALAPPDATA%\\EngineeringToolbox\\<tool_id>\\runs\\YYYYMMDD_HHMMSS
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = tool_output_root(tool_id) / "runs" / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text_log(run_dir: Path, filename: str, text: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(text, encoding="utf-8")
    return path
