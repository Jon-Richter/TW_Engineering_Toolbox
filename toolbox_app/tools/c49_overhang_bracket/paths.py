from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .compat import user_data_dir

TOOL_ID = "c49_overhang_bracket"


def create_run_dir(tool_id: str = TOOL_ID, input_hash: str | None = None) -> Path:
    """Authoritative run directory creator.

    Location:
      %LOCALAPPDATA%\EngineeringToolbox\<tool_id>\runs\YYYYMMDD_HHMMSS_<short_hash>\

    Notes:
      - Always writes to the user data directory (never to the code directory).
      - Includes a timestamp and short hash to avoid collisions.
    """
    root = user_data_dir() / tool_id / "runs"
    root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collision-resistant suffix
    seed = f"{ts}:{os.getpid()}:{time.time_ns()}"
    rand = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]

    if input_hash:
        short = f"{str(input_hash)[:6]}{rand[:2]}"
    else:
        short = rand

    run_dir = root / f"{ts}_{short}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def compute_input_hash(inputs: Dict[str, Any]) -> str:
    """Deterministic input hash computed from normalized, sorted keys."""
    norm: Dict[str, Any] = {}
    for k in sorted(inputs.keys()):
        v = inputs[k]
        if isinstance(v, float):
            # Stable float repr for hashing (keeps determinism across platforms)
            norm[k] = float(f"{v:.12g}")
        else:
            norm[k] = v

    payload = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
