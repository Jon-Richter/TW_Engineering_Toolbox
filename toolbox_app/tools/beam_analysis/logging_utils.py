from __future__ import annotations

import logging
import os
from pathlib import Path


def get_local_appdata_dir(tool_id: str) -> Path:
    """Return a user-writable folder for tool outputs/logs.

    Windows: %LOCALAPPDATA%\\EngineeringToolbox\\<tool_id>
    Fallback: ~/.local/share/EngineeringToolbox/<tool_id>
    """
    local = os.getenv("LOCALAPPDATA")
    if local:
        base = Path(local) / "EngineeringToolbox"
    else:
        base = Path.home() / ".local" / "share" / "EngineeringToolbox"
    out = base / tool_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def configure_file_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if user launches multiple windows
    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
        for h in logger.handlers
    )
    if not already:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
