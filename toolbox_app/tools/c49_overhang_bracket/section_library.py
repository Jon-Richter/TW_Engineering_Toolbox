from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .db.txdot_girders import GirderProfile, get_txdot_profile
from .analysis.girder_outline import (
    GirderGeometry,
    build_a21_exterior_half_outline,
    build_rectilinear_exterior_half_outline,
)


@dataclass(frozen=True)
class TxDotGirder:
    profile: GirderProfile

    def geometry(self) -> GirderGeometry:
        """Return the constructability/diagram geometry for the overhang-side half section."""
        # For TxDOT Tx-girders, we prefer the A-21 piecewise-linear outline.
        # If a caller provides unusual overrides that omit A-21 parameters, we fall back.
        d = self.profile.as_dict()
        if all(k in d for k in ("B_in", "C_in", "E_in", "F_in")):
            return build_a21_exterior_half_outline(d)
        return build_rectilinear_exterior_half_outline(d)

    def metadata(self) -> Dict[str, Any]:
        """Metadata table exported in CalcTrace."""
        return {
            "name": self.profile.name,
            "depth_in": self.profile.depth_in,
            "B_in": self.profile.B_in,
            "C_in": self.profile.C_in,
            "E_in": self.profile.E_in,
            "F_in": self.profile.F_in,
            "top_flange_width_in": self.profile.top_flange_width_in,
            "bottom_flange_width_in": self.profile.bottom_flange_width_in,
            "web_thickness_in": self.profile.web_thickness_in,
            "area_sq_in": self.profile.area_sq_in,
            "weight_plf": self.profile.weight_plf,
            "source": "TxDOT Std. Sheet A-21",
        }


def get_txdot_girder(girder_type: str, overrides: Optional[Dict[str, float]] = None) -> TxDotGirder:
    """Return a TxDOT girder object with profile and geometry.

    `overrides` uses GirderProfile override keys from db.txdot_girders.get_txdot_profile.
    """

    prof = get_txdot_profile(girder_type, overrides=overrides)
    return TxDotGirder(profile=prof)
