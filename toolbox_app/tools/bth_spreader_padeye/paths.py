from __future__ import annotations
import os, json, hashlib
from datetime import datetime
from pathlib import Path

def _local_appdata() -> Path:
    p = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(p)

def tool_root_local(tool_id: str) -> Path:
    return _local_appdata() / "EngineeringToolbox" / tool_id

def runs_root(tool_id: str) -> Path:
    return tool_root_local(tool_id) / "runs"

def normalize_inputs(inputs: dict) -> dict:
    def norm(v):
        if isinstance(v, dict):
            return {k: norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, float):
            return float(f"{v:.12g}")
        return v
    return norm(inputs)

def input_hash(inputs: dict) -> str:
    blob = json.dumps(normalize_inputs(inputs), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

def create_run_dir(tool_id: str, inputs_for_hash: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = input_hash(inputs_for_hash)
    d = runs_root(tool_id) / f"{ts}_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d
