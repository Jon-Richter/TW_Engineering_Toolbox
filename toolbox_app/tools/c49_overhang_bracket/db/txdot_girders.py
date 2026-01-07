from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GirderProfile:
    """TxDOT Tx-girder profile parameters from Standard Sheet A-21.

    This object is used for:
      - Constructability screening / bracket placement geometry.
      - Drawing an accurate section outline (piecewise linear, no radii).

    Coordinate convention used by the geometry builder:
      - y = 0 at top of girder, +y downward
      - x = 0 at exterior face of web, +x outboard (toward overhang)

    Notes:
      - Dimensions are taken directly from A-21 "Girder Dimensions".
      - The outline generator uses only the A-21 dimension set (D, B, C, E, F)
        plus fixed dimensions shown on the same sheet (top flange thickness 3.5 in,
        web thickness 7 in, bottom flange width 32 in, etc.).
      - Fillets/chamfers are not explicitly modeled; curved transitions are
        represented by straight segments using the governing A-21 dimensions.
    """

    name: str

    # A-21 table parameters
    depth_in: float  # "D"
    B_in: float      # "B" (vertical)
    C_in: float      # "C" (horizontal)
    E_in: float      # "E" (vertical)
    F_in: float      # "F" (vertical)

    # Fixed / derived geometry parameters (A-21 sheet callouts)
    top_flange_width_in: float  # 36 in for TX28-TX54; 42 in for TX62-TX70
    bottom_flange_width_in: float  # 32 in (fixed)
    web_thickness_in: float  # 7 in (fixed)

    # Optional section metadata
    area_sq_in: Optional[float] = None
    weight_plf: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return {
            "depth_in": float(self.depth_in),
            "B_in": float(self.B_in),
            "C_in": float(self.C_in),
            "E_in": float(self.E_in),
            "F_in": float(self.F_in),
            "top_flange_width_in": float(self.top_flange_width_in),
            "bottom_flange_width_in": float(self.bottom_flange_width_in),
            "web_thickness_in": float(self.web_thickness_in),
            "area_sq_in": float(self.area_sq_in) if self.area_sq_in is not None else None,
            "weight_plf": float(self.weight_plf) if self.weight_plf is not None else None,
        }


# TxDOT Standard Sheet A-21 (Girder Dimensions):
# Table provides D, B, C, E, F, Area, Weight.
# Sheet callouts provide:
#   - Top flange width = 36 in (TX28-TX54) or 42 in (TX62-TX70)
#   - Web thickness = 7 in
#   - Bottom flange width = 32 in
#   - Top flange edge thickness = 3.5 in
#   - Additional fixed offsets/dimensions used by outline builder are in analysis.girder_outline
_TX_A21: Dict[str, Dict[str, float]] = {
    "TX28": {"D": 28.0, "B": 6.0,  "C": 12.5, "E": 2.0, "F": 6.75, "A": 585.0, "W": 630.0, "TFW": 36.0},
    "TX34": {"D": 34.0, "B": 12.0, "C": 12.5, "E": 2.0, "F": 6.75, "A": 627.0, "W": 675.0, "TFW": 36.0},
    "TX40": {"D": 40.0, "B": 18.0, "C": 12.5, "E": 2.0, "F": 6.75, "A": 669.0, "W": 720.0, "TFW": 36.0},
    "TX46": {"D": 46.0, "B": 22.0, "C": 12.5, "E": 2.0, "F": 8.75, "A": 761.0, "W": 819.0, "TFW": 36.0},
    "TX54": {"D": 54.0, "B": 30.0, "C": 12.5, "E": 2.0, "F": 8.75, "A": 817.0, "W": 880.0, "TFW": 36.0},
    "TX62": {"D": 62.0, "B": 37.5, "C": 15.5, "E": 2.5, "F": 8.75, "A": 910.0, "W": 980.0, "TFW": 42.0},
    "TX70": {"D": 70.0, "B": 45.5, "C": 15.5, "E": 2.5, "F": 8.75, "A": 966.0, "W": 1040.0, "TFW": 42.0},
}


# Fixed dimensions shown on A-21.
_A21_FIXED = {
    "WEB_THK_IN": 7.0,
    "BOTTOM_FLANGE_WIDTH_IN": 32.0,
}


def get_txdot_profile(name: str, overrides: Optional[Dict[str, float]] = None) -> GirderProfile:
    """Return a TxDOT Tx-girder profile by name (e.g., 'TX54').

    Supported override keys (in):
      - depth_in, top_flange_width_in, bottom_flange_width_in, web_thickness_in

    Note:
      - A-21 parameters B_in/C_in/E_in/F_in are not exposed via the current UI override
        fields, because the tool is intended to use the standard shapes directly.
        If you need custom shapes, extend models.C49Inputs and this function.
    """

    key = name.upper().replace(" ", "")
    if key not in _TX_A21:
        raise ValueError(
            f"Unsupported TxDOT girder type '{name}'. Supported: {', '.join(sorted(_TX_A21))}"
        )

    rec = _TX_A21[key]
    prof = GirderProfile(
        name=key,
        depth_in=float(rec["D"]),
        B_in=float(rec["B"]),
        C_in=float(rec["C"]),
        E_in=float(rec["E"]),
        F_in=float(rec["F"]),
        top_flange_width_in=float(rec["TFW"]),
        bottom_flange_width_in=float(_A21_FIXED["BOTTOM_FLANGE_WIDTH_IN"]),
        web_thickness_in=float(_A21_FIXED["WEB_THK_IN"]),
        area_sq_in=float(rec["A"]),
        weight_plf=float(rec["W"]),
    )

    if overrides:
        d = prof.as_dict()
        # Only allow overriding the externally-visible geometry values.
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in (
                "depth_in",
                "top_flange_width_in",
                "bottom_flange_width_in",
                "web_thickness_in",
            ):
                continue
            d[k] = float(v)

        prof = GirderProfile(
            name=key,
            depth_in=float(d["depth_in"]),
            B_in=prof.B_in,
            C_in=prof.C_in,
            E_in=prof.E_in,
            F_in=prof.F_in,
            top_flange_width_in=float(d["top_flange_width_in"]),
            bottom_flange_width_in=float(d["bottom_flange_width_in"]),
            web_thickness_in=float(d["web_thickness_in"]),
            area_sq_in=prof.area_sq_in,
            weight_plf=prof.weight_plf,
        )

    return prof


def list_supported_txdot_girders() -> Dict[str, GirderProfile]:
    """Return dict of supported girder names to default profiles."""
    return {k: get_txdot_profile(k) for k in sorted(_TX_A21)}
