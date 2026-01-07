from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import csv

@dataclass
class ShapeProps:
    name: str
    shape_type: str
    A: float
    W: float
    d: float
    bf: float
    tw: float
    tf: float
    Ix: float
    Iy: float
    Sx: float
    Sy: float
    rx: float
    ry: float
    J: float
    H: float

def load_shapes(tool_dir: Path) -> Dict[str, ShapeProps]:
    csv_path = tool_dir / "assets" / "aisc_shapes_database_v16.csv"
    if not csv_path.exists():
        raise FileNotFoundError("aisc_shapes_database_v16.csv not found in tool assets folder.")
    with csv_path.open("r", encoding="cp1252", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        rows = list(reader)

    index_map: Dict[str, List[int]] = {}
    for i, h in enumerate(headers):
        index_map.setdefault(h, []).append(i)

    def _raw(row: list[str], key: str, default: str = "") -> str:
        for idx in index_map.get(key, []):
            if idx < len(row):
                val = row[idx].strip()
                if val not in ("", None):
                    return val
        return default

    def g(row: list[str], *keys, default=0.0) -> float:
        for k in keys:
            val = _raw(row, k, default="")
            if val not in ("", None):
                try:
                    return float(val)
                except Exception:
                    pass
        return float(default)

    def gs(row: list[str], *keys, default="") -> str:
        for k in keys:
            val = _raw(row, k, default="")
            if val:
                return str(val)
        return default

    out: Dict[str, ShapeProps] = {}
    for row in rows:
        name = gs(row, "AISC_Manual_Label", "EDI_Std_Nomenclature", "DESIGNATION", "Shape")
        if not name:
            continue
        st = gs(row, "Type", "SHAPE_TYPE", default="Unknown")
        props = ShapeProps(
            name=name,
            shape_type=st,
            A=g(row,"A","AREA"),
            W=g(row,"W","WGT","WEIGHT"),
            d=g(row,"d","D"),
            bf=g(row,"bf","B"),
            tw=g(row,"tw","t_w","TW"),
            tf=g(row,"tf","t_f","TF"),
            Ix=g(row,"Ix","I_x","IX"),
            Iy=g(row,"Iy","I_y","IY"),
            Sx=g(row,"Sx","S_x","SX"),
            Sy=g(row,"Sy","S_y","SY"),
            rx=g(row,"rx","r_x","RX"),
            ry=g(row,"ry","r_y","RY"),
            J=g(row,"J"),
            H=g(row,"H"),
        )
        out[name] = props
        name_upper = name.upper()
        if name_upper not in out:
            out[name_upper] = props
    return out

def search_shapes(shapes: Dict[str, ShapeProps], q: str, limit: int = 50) -> List[ShapeProps]:
    q2 = (q or "").strip().upper()
    if not q2:
        return list(shapes.values())[:limit]
    hits = [s for k,s in shapes.items() if q2 in k.upper()]
    return hits[:limit]
