from __future__ import annotations
import os
from pathlib import Path

APP_NAME = "EngineeringToolbox"

def user_data_dir() -> Path:
    """
    Writable location for logs/cache/settings. Avoid writing to the SharePoint code folder.
    Windows default: %LOCALAPPDATA%\\EngineeringToolbox\\
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    p = Path(base) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p

def logs_dir() -> Path:
    p = user_data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def cache_dir() -> Path:
    p = user_data_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

def settings_path() -> Path:
    return user_data_dir() / "settings.json"
