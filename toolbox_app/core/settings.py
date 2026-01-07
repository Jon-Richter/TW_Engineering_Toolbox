from __future__ import annotations

import json
from typing import Any, Dict

from toolbox_app.core.paths import settings_path


def load_settings() -> Dict[str, Any]:
    p = settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: Dict[str, Any]) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
