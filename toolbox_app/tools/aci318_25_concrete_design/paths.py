\
from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _get_local_appdata_dir() -> Path:
    """
    Return base directory for persistent, user-local data.

    Windows:  %LOCALAPPDATA%\EngineeringToolbox
    Fallback: ~/.local/share/EngineeringToolbox
    """
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "EngineeringToolbox"
    # Cross-platform fallback (tests / non-Windows)
    return Path.home() / ".local" / "share" / "EngineeringToolbox"


def tool_data_dir(tool_id: str) -> Path:
    d = _get_local_appdata_dir() / tool_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_root(tool_id: str) -> Path:
    d = tool_data_dir(tool_id) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def create_run_dir(tool_id: str, seed: Optional[str] = None) -> Path:
    """
    Create a run directory:
      %LOCALAPPDATA%\EngineeringToolbox\<tool_id>\runs\YYYYMMDD_HHMMSS_<short_hash>\
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    if seed is None:
        seed = str(uuid.uuid4())
    h = short_hash(seed, 8)
    d = runs_root(tool_id) / f"{ts}_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def server_state_dir(tool_id: str) -> Path:
    d = tool_data_dir(tool_id) / "server"
    d.mkdir(parents=True, exist_ok=True)
    return d


def server_pid_file(tool_id: str) -> Path:
    return server_state_dir(tool_id) / "server.pid"


def server_port_file(tool_id: str) -> Path:
    return server_state_dir(tool_id) / "server_port.txt"


def normalize_inputs_for_hash(inputs: dict) -> str:
    """
    Deterministic normalization for hashing: sort keys, JSON encode.
    """
    def _normalize(o):
        if isinstance(o, dict):
            return {k: _normalize(o[k]) for k in sorted(o.keys())}
        if isinstance(o, list):
            return [_normalize(x) for x in o]
        if isinstance(o, float):
            # avoid floating noise: round to 12 significant digits
            return float(f"{o:.12g}")
        return o

    normalized = _normalize(inputs)
    return __import__("json").dumps(normalized, sort_keys=True, separators=(",", ":"))


def input_hash(inputs: dict) -> str:
    s = normalize_inputs_for_hash(inputs)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
