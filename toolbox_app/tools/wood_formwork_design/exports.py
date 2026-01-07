from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import csv
import json
import os
import re
from datetime import datetime

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover
    Workbook = None  # type: ignore


def _local_appdata_dir() -> Path:
    """
    Returns a user-writable base directory under:
      %LOCALAPPDATA%\EngineeringToolbox
    """
    root = os.environ.get("LOCALAPPDATA")
    if root:
        return Path(root) / "EngineeringToolbox"
    # Fallbacks (non-Windows / misconfigured env)
    return Path.home() / "AppData" / "Local" / "EngineeringToolbox"


def tool_export_dir(tool_id: str) -> Path:
    d = _local_appdata_dir() / "tools" / tool_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def timestamp_slug(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def _safe_key(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "value"


def flatten(obj: Any, prefix: str = "", max_depth: int = 6) -> Dict[str, Any]:
    """
    Flattens nested dict/list structures into a key->value dict suitable for CSV/Excel.
    Keys use dot + [idx] notation.
    """
    out: Dict[str, Any] = {}

    def _walk(o: Any, p: str, depth: int) -> None:
        if depth > max_depth:
            out[p or "value"] = json.dumps(o, default=str)
            return

        if isinstance(o, dict):
            if not o:
                out[p or "value"] = ""
                return
            for k, v in o.items():
                kk = _safe_key(str(k))
                _walk(v, f"{p}.{kk}" if p else kk, depth + 1)
            return

        if isinstance(o, (list, tuple)):
            if not o:
                out[p or "value"] = ""
                return
            for i, v in enumerate(o):
                _walk(v, f"{p}[{i}]" if p else f"[{i}]", depth + 1)
            return

        out[p or "value"] = o

    _walk(obj, prefix, 0)
    return out


def write_json(payload: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def write_flat_csv(flat: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k in sorted(flat.keys()):
            v = flat[k]
            if isinstance(v, (dict, list, tuple)):
                v = json.dumps(v, default=str)
            w.writerow([k, v])
    return out_path


def export_excel(payload: Any, out_path: Path) -> Path:
    """
    Exports payload into an .xlsx file with:
      - 'meta' sheet: capture metadata (if present)
      - 'flat' sheet: flattened key/value pairs
    """
    if Workbook is None:
        raise RuntimeError("openpyxl is required for Excel export but is not installed.")

    wb = Workbook()
    ws_meta = wb.active
    ws_meta.title = "meta"
    ws_flat = wb.create_sheet("flat")

    # Meta sheet
    if isinstance(payload, dict):
        meta = payload.get("meta", {})
    else:
        meta = {}

    ws_meta.append(["key", "value"])
    for k, v in sorted(flatten(meta).items()):
        ws_meta.append([k, v])

    # Flat sheet
    ws_flat.append(["key", "value"])
    for k, v in sorted(flatten(payload).items()):
        ws_flat.append([k, v])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


def mathcad_handoff(payload: Any, out_dir: Path) -> Dict[str, Path]:
    """
    Writes a Mathcad-friendly handoff pack:
      - handoff.json: full payload
      - handoff.csv: flattened key/value
      - assignments.txt: simple assignment lines (key:=value)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "handoff.json"
    csv_path = out_dir / "handoff.csv"
    txt_path = out_dir / "assignments.txt"

    write_json(payload, json_path)

    flat = flatten(payload)
    write_flat_csv(flat, csv_path)

    lines = []
    for k in sorted(flat.keys()):
        v = flat[k]
        key = _safe_key(k)
        if v is None:
            rhs = "0"
        elif isinstance(v, bool):
            rhs = "1" if v else "0"
        elif isinstance(v, (int, float)):
            rhs = str(v)
        else:
            s = str(v)
            s = s.replace('"', '""')
            rhs = f"\"{s}\""
        lines.append(f"{key}:={rhs}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "assignments": txt_path}
