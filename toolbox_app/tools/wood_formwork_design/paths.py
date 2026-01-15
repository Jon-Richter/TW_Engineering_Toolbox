from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _local_appdata() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if root:
        return Path(root) / "EngineeringToolbox"
    return Path.home() / ".local" / "share" / "EngineeringToolbox"


def tool_root_local(tool_id: str) -> Path:
    d = _local_appdata() / tool_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_root(tool_id: str) -> Path:
    d = tool_root_local(tool_id) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    def norm(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: norm(value[k]) for k in sorted(value.keys())}
        if isinstance(value, list):
            return [norm(v) for v in value]
        if isinstance(value, float):
            return float(f"{value:.12g}")
        return value

    return norm(inputs)


def input_hash(inputs: Dict[str, Any]) -> str:
    blob = json.dumps(normalize_inputs(inputs), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def create_run_dir(tool_id: str, inputs_for_hash: Dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = input_hash(inputs_for_hash)
    d = runs_root(tool_id) / f"{ts}_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def server_state_dir(tool_id: str) -> Path:
    d = tool_root_local(tool_id) / "server"
    d.mkdir(parents=True, exist_ok=True)
    return d
