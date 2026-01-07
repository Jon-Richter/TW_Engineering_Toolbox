from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .paths import appdata_root


class ShapeNotFoundError(RuntimeError):
    pass


def _tool_data_db_path() -> Path:
    # Tool-relative DB path (packaged with tool)
    return Path(__file__).resolve().parent / "data" / "aisc_shapes_database_v16.csv"


def _user_override_db_path() -> Optional[Path]:
    # Optional user override: %LOCALAPPDATA%\EngineeringToolbox\databases\aisc\aisc_shapes_database_v16.csv
    p = appdata_root() / "databases" / "aisc" / "aisc_shapes_database_v16.csv"
    return p if p.exists() else None


def _canonical(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _label_sort_key(label: str, type_code: str) -> tuple:
    s = (label or "").upper().replace(" ", "")
    tc = (type_code or "").upper()
    if tc and s.startswith(tc):
        s = s[len(tc):]
    tokens = re.findall(r"\d+(?:\.\d+)?(?:/\d+)?", s)
    nums = tuple(_to_float(t) or 0.0 for t in tokens)
    return (nums, s)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    s = value.strip()
    if s in {"", "-", "\u2013", "\u2014"}:
        return None
    s = s.replace(",", "")
    s = " ".join(s.split())
    sign = 1.0
    if s.startswith(("+", "-")):
        if s[0] == "-":
            sign = -1.0
        s = s[1:].strip()
    if s == "":
        return None
    if " " in s and "/" in s:
        parts = s.split()
        if len(parts) >= 2 and "/" in parts[1]:
            whole = float(parts[0])
            num, den = parts[1].split("/", 1)
            return sign * (whole + (float(num) / float(den)))
    if "/" in s:
        num, den = s.split("/", 1)
        return sign * (float(num) / float(den))
    return sign * float(s)


@dataclass(frozen=True)
class Shape:
    label: str
    type_code: str

    # Basic section properties (AISC database units assumed)
    A_in2: float
    Ix_in4: float
    Iy_in4: float
    Sx_in3: float
    Sy_in3: float
    Zx_in3: float
    Zy_in3: float
    rx_in: float
    ry_in: float

    # Optional properties used for LTB (I-shapes primarily)
    J_in4: Optional[float] = None
    Cw_in6: Optional[float] = None
    rts_in: Optional[float] = None
    h0_in: Optional[float] = None
    Lp_in: Optional[float] = None
    Lr_in: Optional[float] = None

    # Principal-axis properties (angles)
    Iw_in4: Optional[float] = None
    Iz_in4: Optional[float] = None
    SwA_in3: Optional[float] = None
    SwB_in3: Optional[float] = None
    SwC_in3: Optional[float] = None
    SzA_in3: Optional[float] = None
    SzB_in3: Optional[float] = None
    SzC_in3: Optional[float] = None
    wA_in: Optional[float] = None
    wB_in: Optional[float] = None
    wC_in: Optional[float] = None
    zA_in: Optional[float] = None
    zB_in: Optional[float] = None
    zC_in: Optional[float] = None
    tan_theta: Optional[float] = None

    # Geometry helpful for shear area fallback
    d_in: Optional[float] = None
    tw_in: Optional[float] = None
    bf_in: Optional[float] = None
    tf_in: Optional[float] = None
    t_in: Optional[float] = None

    # Slenderness ratios from DB (B4.1 tables)
    bf_2tf: Optional[float] = None
    h_tw: Optional[float] = None
    b_t: Optional[float] = None
    b_t_des: Optional[float] = None
    h_t_des: Optional[float] = None
    D_t: Optional[float] = None

    # HSS geometry (outside + clear) and wall thicknesses
    B_in: Optional[float] = None
    H_in: Optional[float] = None
    b_in: Optional[float] = None
    h_in: Optional[float] = None
    t_nom_in: Optional[float] = None
    t_des_in: Optional[float] = None
    OD_in: Optional[float] = None
    ro_in: Optional[float] = None
    H_const: Optional[float] = None

    # Centroid/location data for singly-symmetric sections
    x_in: Optional[float] = None
    y_in: Optional[float] = None
    eo_in: Optional[float] = None
    xp_in: Optional[float] = None
    yp_in: Optional[float] = None

    def to_public_dict(self) -> Dict[str, float | str | None]:
        return {
            "label": self.label,
            "type_code": self.type_code,
            "A_in2": self.A_in2,
            "Ix_in4": self.Ix_in4,
            "Iy_in4": self.Iy_in4,
            "Sx_in3": self.Sx_in3,
            "Sy_in3": self.Sy_in3,
            "Zx_in3": self.Zx_in3,
            "Zy_in3": self.Zy_in3,
            "rx_in": self.rx_in,
            "ry_in": self.ry_in,
            "J_in4": self.J_in4,
            "Cw_in6": self.Cw_in6,
            "rts_in": self.rts_in,
            "h0_in": self.h0_in,
            "Lp_in": self.Lp_in,
            "Lr_in": self.Lr_in,
            "Iw_in4": self.Iw_in4,
            "Iz_in4": self.Iz_in4,
            "SwA_in3": self.SwA_in3,
            "SwB_in3": self.SwB_in3,
            "SwC_in3": self.SwC_in3,
            "SzA_in3": self.SzA_in3,
            "SzB_in3": self.SzB_in3,
            "SzC_in3": self.SzC_in3,
            "wA_in": self.wA_in,
            "wB_in": self.wB_in,
            "wC_in": self.wC_in,
            "zA_in": self.zA_in,
            "zB_in": self.zB_in,
            "zC_in": self.zC_in,
            "tan_theta": self.tan_theta,
            "d_in": self.d_in,
            "tw_in": self.tw_in,
            "bf_in": self.bf_in,
            "tf_in": self.tf_in,
            "t_in": self.t_in,
            "bf_2tf": self.bf_2tf,
            "h_tw": self.h_tw,
            "b_t": self.b_t,
            "b_t_des": self.b_t_des,
            "h_t_des": self.h_t_des,
            "D_t": self.D_t,
            "B_in": self.B_in,
            "H_in": self.H_in,
            "b_in": self.b_in,
            "h_in": self.h_in,
            "t_nom_in": self.t_nom_in,
            "t_des_in": self.t_des_in,
            "OD_in": self.OD_in,
            "ro_in": self.ro_in,
            "H_const": self.H_const,
            "x_in": self.x_in,
            "y_in": self.y_in,
            "eo_in": self.eo_in,
            "xp_in": self.xp_in,
            "yp_in": self.yp_in,
        }


class ShapeDatabase:
    """
    Lightweight CSV reader for AISC Shapes Database v16.

    Expectations:
    - CSV contains a label column (commonly 'AISC_Manual_Label' or similar)
    - CSV contains 'Type' or 'Type' code
    - Units are AISC standard database units: in, in^2, in^3, in^4, in^6.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or _user_override_db_path() or _tool_data_db_path()
        if not self.db_path.exists():
            raise FileNotFoundError(
                "AISC Shapes DB not found. Expected one of:\n"
                f"1) {str(_tool_data_db_path())}\n"
                f"2) {str(appdata_root() / 'databases/aisc/aisc_shapes_database_v16.csv')}\n"
                "Place your CSV at one of these locations."
            )
        self._rows = self._read_all_rows()

        # Pre-index by canonical label for fast lookup
        self._index: Dict[str, Dict[str, str]] = {}
        for r in self._rows:
            label = r.get("AISC_Manual_Label") or r.get("EDI_Std_Nomenclature") or r.get("Label") or r.get("DESIGNATION")
            if not label:
                continue
            self._index[_canonical(label)] = r

    def _read_all_rows(self) -> List[Dict[str, str]]:
        encodings = ("utf-8-sig", "cp1252", "latin-1")
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                with self.db_path.open("r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        return []
                    # Prefer first occurrence of duplicate columns (CSV contains repeated US/metric blocks)
                    seen: set[str] = set()
                    keys: List[str] = []
                    idxs: List[int] = []
                    for i, key in enumerate(header):
                        k = (key or "").strip()
                        if not k or k in seen:
                            continue
                        seen.add(k)
                        keys.append(k)
                        idxs.append(i)
                    rows: List[Dict[str, str]] = []
                    for row in reader:
                        if not row:
                            continue
                        d: Dict[str, str] = {}
                        for i, k in zip(idxs, keys):
                            d[k] = row[i] if i < len(row) else ""
                        rows.append(d)
                    return rows
            except UnicodeDecodeError as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        raise RuntimeError(f"Failed reading AISC shapes DB: {self.db_path}")

    def _col(self, row: Dict[str, str], *names: str, required: bool = True) -> Optional[str]:
        # Prefer exact (case-insensitive) name match first to avoid collisions (e.g., "T_F" vs "tf").
        missing = {"-", "\u2013", "\u2014"}
        for n in names:
            for key in row.keys():
                if key.strip().lower() != n.strip().lower():
                    continue
                v = row.get(key, "")
                if v is None:
                    v = ""
                v = v.strip()
                if v == "" or v in missing:
                    continue
                return v
        # Fallback: canonical match (first non-empty)
        for n in names:
            ck = _canonical(n)
            for key in row.keys():
                if _canonical(key) != ck:
                    continue
                v = row.get(key, "")
                if v is None:
                    v = ""
                v = v.strip()
                if v == "" or v in missing:
                    continue
                return v
        if required:
            raise KeyError(f"Missing required column(s): {names}. Available keys: {list(row.keys())[:30]} ...")
        return None

    def get_shape(self, label: str) -> Shape:
        key = _canonical(label)
        row = self._index.get(key)
        if not row:
            # Try relaxed match (remove spaces, common typos)
            candidates = self.suggest_labels(label, limit=12)
            raise ShapeNotFoundError(
                f"Section '{label}' not found in AISC DB. "
                f"Closest matches: {', '.join(candidates) if candidates else '(none)'}"
            )

        type_code = (row.get("Type") or row.get("TYPE") or "").strip()

        def f(*colnames: str, required: bool = True) -> Optional[float]:
            s = self._col(row, *colnames, required=required)
            v = _to_float(s)
            if required and v is None:
                raise ValueError(f"Missing numeric value for {colnames}")
            return v

        # Required
        A = f("A", "A_in2", "Area", required=True)
        Ix = f("Ix", "I_x", "Ix_in4", required=True)
        Iy = f("Iy", "I_y", "Iy_in4", required=True)
        Sx = f("Sx", "S_x", "Sx_in3", required=True)
        Sy = f("Sy", "S_y", "Sy_in3", required=True)
        Zx = f("Zx", "Z_x", "Zx_in3", required=True)
        Zy = f("Zy", "Z_y", "Zy_in3", required=True)
        rx = f("rx", "r_x", "rx_in", required=True)
        ry = f("ry", "r_y", "ry_in", required=True)

        shp_label = (
            row.get("AISC_Manual_Label")
            or row.get("EDI_Std_Nomenclature")
            or row.get("Label")
            or label
        ).strip()

        Ht = f("Ht", required=False)
        H_col = f("H", required=False)
        x_in = f("x", required=False)
        y_in = f("y", required=False)
        eo_in = f("eo", required=False)
        ro_in = f("ro", required=False)
        if ro_in is None and A and Ix and Iy:
            tcode = type_code.strip().upper()
            symmetric = tcode in {"W", "S", "M", "HP", "HSS", "PIPE"}
            x0 = x_in
            y0 = y_in
            if x0 is None and tcode in {"C", "MC"} and eo_in is not None:
                x0 = eo_in
            if x0 is None and symmetric:
                x0 = 0.0
            if y0 is None and (symmetric or tcode in {"C", "MC", "WT", "MT", "ST"}):
                y0 = 0.0
            if x0 is not None and y0 is not None:
                # E4-9: r0^2 = x0^2 + y0^2 + (Ix + Iy)/A
                ro_in = math.sqrt(float(x0) ** 2 + float(y0) ** 2 + (float(Ix) + float(Iy)) / float(A))

        return Shape(
            label=shp_label,
            type_code=type_code,
            A_in2=A,
            Ix_in4=Ix,
            Iy_in4=Iy,
            Sx_in3=Sx,
            Sy_in3=Sy,
            Zx_in3=Zx,
            Zy_in3=Zy,
            rx_in=rx,
            ry_in=ry,
            J_in4=f("J", "J_in4", required=False),
            Cw_in6=f("Cw", "C_w", "Cw_in6", required=False),
            rts_in=f("rts", "r_ts", "rts_in", required=False),
            h0_in=f("h0", "ho", "h_o", "h0_in", required=False),
            Lp_in=f("Lp", "L_p", "Lp_in", required=False),
            Lr_in=f("Lr", "L_r", "Lr_in", required=False),
            Iw_in4=f("Iw", "I_w", "Iw_in4", required=False),
            Iz_in4=f("Iz", "I_z", "Iz_in4", required=False),
            SwA_in3=f("SwA", "Sw_A", "SwA_in3", required=False),
            SwB_in3=f("SwB", "Sw_B", "SwB_in3", required=False),
            SwC_in3=f("SwC", "Sw_C", "SwC_in3", required=False),
            SzA_in3=f("SzA", "Sz_A", "SzA_in3", required=False),
            SzB_in3=f("SzB", "Sz_B", "SzB_in3", required=False),
            SzC_in3=f("SzC", "Sz_C", "SzC_in3", required=False),
            wA_in=f("wA", "w_A", required=False),
            wB_in=f("wB", "w_B", required=False),
            wC_in=f("wC", "w_C", required=False),
            zA_in=f("zA", "z_A", required=False),
            zB_in=f("zB", "z_B", required=False),
            zC_in=f("zC", "z_C", required=False),
            tan_theta=f("tan(?)", "tan_theta", "tan", required=False),
            d_in=f("d", "Depth", required=False),
            tw_in=f("tw", "t_w", required=False),
            bf_in=f("bf", "b_f", required=False),
            tf_in=f("tf", "t_f", required=False),
            t_in=f("t", required=False),
            bf_2tf=f("bf/2tf", required=False),
            h_tw=f("h/tw", required=False),
            b_t=f("b/t", required=False),
            b_t_des=f("b/tdes", required=False),
            h_t_des=f("h/tdes", required=False),
            D_t=f("D/t", required=False),
            B_in=f("B", required=False),
            H_in=Ht or H_col,
            b_in=f("b", required=False),
            h_in=f("h", required=False),
            t_nom_in=f("tnom", required=False),
            t_des_in=f("tdes", required=False),
            OD_in=f("OD", "O.D.", "Outside_Diameter", required=False),
            ro_in=ro_in,
            H_const=H_col,
            x_in=x_in,
            y_in=y_in,
            eo_in=eo_in,
            xp_in=f("xp", required=False),
            yp_in=f("yp", required=False),
        )

    def list_labels_by_typecode(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for r in self._rows:
            t = (r.get("Type") or r.get("TYPE") or "").strip()
            label = r.get("AISC_Manual_Label") or r.get("Label") or r.get("EDI_Std_Nomenclature")
            if not t or not label:
                continue
            out.setdefault(t, []).append(label.strip())
        # Sort for stable UI dropdowns
        for k in out:
            uniq = sorted(set(out[k]))
            out[k] = sorted(uniq, key=lambda s: _label_sort_key(s, k))
        return out

    def partition_hss_labels(self, labels: List[str]) -> Tuple[List[str], List[str]]:
        rect: List[str] = []
        rnd: List[str] = []
        for lb in labels:
            s = lb.upper().replace(" ", "")
            if not s.startswith("HSS"):
                rect.append(lb)
                continue
            core = s[3:]
            # HSS round often has pattern like HSS6.625X0.280 (one 'X')
            if core.count("X") == 1:
                rnd.append(lb)
            else:
                rect.append(lb)
        def dedupe(seq: List[str]) -> List[str]:
            seen: set[str] = set()
            out: List[str] = []
            for item in seq:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out

        rect = dedupe(rect)
        rnd = dedupe(rnd)
        rect = sorted(rect, key=lambda s: _label_sort_key(s, "HSS"))
        rnd = sorted(rnd, key=lambda s: _label_sort_key(s, "HSS"))
        return rect, rnd

    def suggest_labels(self, label: str, limit: int = 10) -> List[str]:
        # Simple suggestion: share alphanumeric prefix
        key = _canonical(label)
        if not key:
            return []
        candidates = []
        for k, r in self._index.items():
            if k.startswith(key[: max(2, len(key) // 2)]):
                candidates.append(r.get("AISC_Manual_Label") or r.get("EDI_Std_Nomenclature") or "")
        candidates = [c for c in candidates if c]
        return sorted(candidates)[:limit]
