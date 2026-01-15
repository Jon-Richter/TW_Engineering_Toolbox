from __future__ import annotations

import datetime as dt
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from toolbox_app.blocks import NDS_SUPP_db as nds_db


TOOL_ID = "wood_formwork_design"
TOOL_VERSION = "1.1.0"

TOOL_DIR = Path(__file__).resolve().parents[1]


# -----------------------------
# Helpers / unit conversions
# -----------------------------
def in_to_ft(x_in: float) -> float:
    return float(x_in) / 12.0


def ft_to_in(x_ft: float) -> float:
    return float(x_ft) * 12.0


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Section properties (nominal dimensions)
# -----------------------------
DIMLUMBER_DB: Dict[str, Tuple[float, float]] = {
    "2x4": (1.5, 3.5),
    "2x6": (1.5, 5.5),
    "2x8": (1.5, 7.25),
    "2x10": (1.5, 9.25),
    "2x12": (1.5, 11.25),
    "4x4": (3.5, 3.5),
    "4x6": (3.5, 5.5),
    "4x8": (3.5, 7.25),
}

DIMLUMBER_OPTIONS = ["2x4", "2x6", "2x8", "2x10", "2x12", "4x4", "4x6"]
PLYWOOD_THICKNESS_OPTIONS = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125]


def rect_I(b_in: float, d_in: float) -> float:
    return b_in * d_in**3 / 12.0


def rect_S(b_in: float, d_in: float) -> float:
    return b_in * d_in**2 / 6.0


def rect_A(b_in: float, d_in: float) -> float:
    return b_in * d_in


def member_props_dimlumber(size: str, strong_axis: bool = True) -> Tuple[float, float, float]:
    """
    Returns (S, I, A) in in^3, in^4, in^2.
    strong_axis=True: bending about strong axis (depth = larger dim)
    """
    if size not in DIMLUMBER_DB:
        raise ValueError(f"Unknown size: {size}")
    b, d = DIMLUMBER_DB[size]
    if not strong_axis and d > b:
        b, d = d, b
    S = rect_S(b, d)
    I = rect_I(b, d)
    A = rect_A(b, d)
    return S, I, A


def member_props_plywood_strip(t_in: float, strip_width_in: float = 12.0) -> Tuple[float, float, float]:
    """
    1-ft wide strip by default.
    Returns (S, I, A) in in^3, in^4, in^2.
    """
    b = float(strip_width_in)
    t = float(t_in)
    S = b * t**2 / 6.0
    I = b * t**3 / 12.0
    A = b * t
    return S, I, A


# -----------------------------
# Simple beam analysis (numeric) for arbitrary distributed load
# -----------------------------
@dataclass
class BeamResult:
    Mmax_lbft: float
    Vmax_lb: float
    defl_max_in: float


def beam_simply_supported_distributed(
    L_ft: float,
    w_func_lbft: np.ndarray,
    x_ft: np.ndarray,
    E_psi: float,
    I_in4: float,
) -> BeamResult:
    """
    Numeric solution:
    - reactions from statics
    - shear/moment from integration
    - deflection from curvature integration: y'' = M / (E I)
    """
    L_ft = float(L_ft)
    if L_ft <= 0 or len(x_ft) < 3:
        return BeamResult(float("nan"), float("nan"), float("nan"))

    W = np.trapezoid(w_func_lbft, x_ft)
    M_about_left = np.trapezoid(w_func_lbft * x_ft, x_ft)

    R2 = M_about_left / L_ft if L_ft > 0 else float("nan")
    R1 = W - R2

    w_int = np.cumsum((w_func_lbft[:-1] + w_func_lbft[1:]) * 0.5 * np.diff(x_ft))
    w_int = np.concatenate([[0.0], w_int])
    V = R1 - w_int

    V_mid = (V[:-1] + V[1:]) * 0.5
    M_int = np.cumsum(V_mid * np.diff(x_ft))
    M = np.concatenate([[0.0], M_int])

    Mmax = float(np.max(np.abs(M)))
    Vmax = float(np.max(np.abs(V)))

    if E_psi <= 0 or I_in4 <= 0:
        dmax = float("nan")
    else:
        x_in = x_ft * 12.0
        M_lbin = M * 12.0
        k = M_lbin / (E_psi * I_in4)

        k_int = np.cumsum((k[:-1] + k[1:]) * 0.5 * np.diff(x_in))
        k_int = np.concatenate([[0.0], k_int])

        A = np.cumsum((k_int[:-1] + k_int[1:]) * 0.5 * np.diff(x_in))
        A = np.concatenate([[0.0], A])

        L_in = float(L_ft * 12.0)
        C1 = -A[-1] / L_in if L_in > 0 else 0.0
        y = C1 * x_in + A
        dmax = float(np.max(np.abs(y)))

    return BeamResult(Mmax_lbft=Mmax, Vmax_lb=Vmax, defl_max_in=dmax)


def beam_uniform(
    L_ft: float,
    w_lbft: float,
    E_psi: float,
    I_in4: float,
) -> BeamResult:
    L_ft = float(L_ft)
    w = float(w_lbft)
    if L_ft <= 0:
        return BeamResult(float("nan"), float("nan"), float("nan"))
    Mmax = abs(w) * L_ft**2 / 8.0
    Vmax = abs(w) * L_ft / 2.0

    if E_psi <= 0 or I_in4 <= 0:
        dmax = float("nan")
    else:
        L_in = L_ft * 12.0
        w_lbin = w / 12.0
        dmax = 5.0 * w_lbin * L_in**4 / (384.0 * E_psi * I_in4)

    return BeamResult(Mmax_lbft=float(Mmax), Vmax_lb=float(Vmax), defl_max_in=float(dmax))


# -----------------------------
# ACI 347R-14 pressure (inch-pound)
# -----------------------------
def cw_unit_weight_coeff(w_pcf: float) -> float:
    if w_pcf <= 0:
        return float("nan")
    return w_pcf / 145.0


def cc_chem_coeff(mix_category: str, retarder: bool) -> float:
    base = {
        "normal": 1.00,
        "high_cementitious": 1.10,
        "retarded": 1.20,
        "accelerated": 0.90,
    }.get(mix_category, 1.00)
    if retarder and mix_category != "retarded":
        base = max(base, 1.20)
    return float(base)


def aci347r14_lateral_pressure_psf(
    *,
    element_type: str,
    height_ft: float,
    w_pcf: float,
    T_F: float,
    R_ftph: float,
    slump_in: float,
    internal_vib_depth_ft: float,
    mix_category: str,
    retarder_included: bool,
    is_scc: bool,
    pumped_from_base: bool,
) -> Dict[str, float | str]:
    height_ft = max(float(height_ft), 0.0)
    w_pcf = max(float(w_pcf), 0.0)
    T_F = max(float(T_F), 1.0)
    R_ftph = max(float(R_ftph), 0.0)

    Cc = cc_chem_coeff(mix_category, retarder_included)
    Cw = cw_unit_weight_coeff(w_pcf)

    p_hydro = w_pcf * height_ft

    if is_scc:
        p_cap = p_hydro
        eq = "Hydrostatic (SCC selected)"
        if pumped_from_base:
            p_cap = 1.25 * p_cap
            eq += " + 25% pumping surge"
        return {
            "p_cap_psf": p_cap,
            "p_hydro_psf": p_hydro,
            "p_empirical_psf": float("nan"),
            "p_min_psf": float("nan"),
            "Cc": Cc,
            "Cw": Cw,
            "controlling_eq": eq,
        }

    if slump_in > 7.0 or internal_vib_depth_ft > 4.0:
        p_cap = p_hydro
        eq = "Hydrostatic (slump > 7 in or vib depth > 4 ft)"
        if pumped_from_base:
            p_cap = 1.25 * p_cap
            eq += " + 25% pumping surge"
        return {
            "p_cap_psf": p_cap,
            "p_hydro_psf": p_hydro,
            "p_empirical_psf": float("nan"),
            "p_min_psf": float("nan"),
            "Cc": Cc,
            "Cw": Cw,
            "controlling_eq": eq,
        }

    p_min = 600.0 * Cw

    if element_type == "wall":
        p_emp = Cc * Cw * (150.0 + 9000.0 * R_ftph / T_F)
        eq = "ACI 347R-14 (Wall): Cc*Cw*(150 + 9000R/T)"
    else:
        p_emp = Cc * Cw * (150.0 + 43400.0 / T_F + 2800.0 * R_ftph / T_F)
        eq = "ACI 347R-14 (Column): Cc*Cw*(150 + 43400/T + 2800R/T)"

    p_cap = min(max(p_min, p_emp), p_hydro)

    if pumped_from_base:
        p_cap = 1.25 * p_cap
        eq += " + 25% pumping surge"
        p_cap = min(p_cap, 1.25 * p_hydro)

    return {
        "p_cap_psf": float(p_cap),
        "p_hydro_psf": float(p_hydro),
        "p_empirical_psf": float(p_emp),
        "p_min_psf": float(p_min),
        "Cc": float(Cc),
        "Cw": float(Cw),
        "controlling_eq": eq,
    }


def pressure_at_depth_psf(depth_ft: float, w_pcf: float, p_cap_psf: float) -> float:
    return float(min(w_pcf * max(depth_ft, 0.0), p_cap_psf))


def build_pressure_profile(
    height_ft: float,
    w_pcf: float,
    p_cap_psf: float,
    seg_ft: float,
) -> pd.DataFrame:
    H = max(float(height_ft), 0.0)
    seg = max(float(seg_ft), 0.25)
    n = int(math.ceil(H / seg)) if H > 0 else 1
    bounds = [min(i * seg, H) for i in range(n + 1)]

    rows = []
    for i in range(n):
        z0 = bounds[i]
        z1 = bounds[i + 1]
        zm = 0.5 * (z0 + z1)
        p0 = pressure_at_depth_psf(z0, w_pcf, p_cap_psf)
        p1 = pressure_at_depth_psf(z1, w_pcf, p_cap_psf)
        pm = pressure_at_depth_psf(zm, w_pcf, p_cap_psf)

        zs = np.linspace(z0, z1, 31)
        ps = np.minimum(w_pcf * zs, p_cap_psf)
        pavg = float(np.trapezoid(ps, zs) / max((z1 - z0), 1e-9))

        rows.append(
            {
                "Segment": i + 1,
                "DepthTop_ft": z0,
                "DepthBot_ft": z1,
                "ElevTop_ft": H - z0,
                "ElevBot_ft": H - z1,
                "p_top_psf": p0,
                "p_bot_psf": p1,
                "p_mid_psf": pm,
                "p_avg_psf": pavg,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# NDS presets (dimension lumber)
# -----------------------------
NDS_TABLES = ("4A", "4B")


def _table_for_nominal_size(nominal_size: str) -> str:
    size = (nominal_size or "").strip().lower()
    return "4B" if size.startswith("4x") else "4A"


def _get_record_for_size(species: str, grade: str, nominal_size: str) -> nds_db.WoodRecord:
    table = _table_for_nominal_size(nominal_size)
    try:
        return nds_db.get_record_auto_size_class(table, species, grade, nominal_size)
    except Exception:
        other = "4A" if table == "4B" else "4B"
        return nds_db.get_record_auto_size_class(other, species, grade, nominal_size)


def nds_species_options() -> List[str]:
    species = {k[1] for k in nds_db.ALLOWABLES.keys() if k[0] in NDS_TABLES}
    return sorted(species)


def nds_grade_options_for_species(species: str) -> List[str]:
    grades = {k[2] for k in nds_db.ALLOWABLES.keys() if k[1] == species and k[0] in NDS_TABLES}
    return sorted(grades)


def nds_cm_factor(is_wet: bool, cm_table: float) -> float:
    if not is_wet:
        return 1.0
    return cm_table if math.isfinite(cm_table) and cm_table > 0 else 1.0


def nds_ct_factor(is_hot: bool) -> float:
    return 0.90 if is_hot else 1.0


def apply_nds_factors(base: Dict[str, float], CD: float, CM: float, Ct: float, Cf: float) -> Dict[str, float]:
    """
    Simplified factor application for formwork (ASD-style).
    Fb' = Fb * CD * CM * Ct * Cf
    Fv' = Fv * CD * CM * Ct
    E'  = E  * CM * Ct
    """
    Fb = base.get("Fb", float("nan"))
    Fv = base.get("Fv", float("nan"))
    E = base.get("E", float("nan"))
    return {"Fb": Fb * CD * CM * Ct * Cf, "Fv": Fv * CD * CM * Ct, "E": E * CM * Ct}


def resolve_cd_value(cd_preset: float, cd_custom: float) -> Tuple[float, str]:
    cd_preset_val = safe_float(cd_preset, 1.25)
    cd_custom_val = safe_float(cd_custom, float("nan"))
    if math.isfinite(cd_custom_val) and cd_custom_val > 0:
        return cd_custom_val, "custom"
    return cd_preset_val, "preset"


def compute_nds_adjustments(
    *,
    use_nds: bool,
    species: str,
    grade: str,
    stud_size: str,
    waler_size: str,
    cd_preset: float,
    cd_custom: float,
    moisture: str,
    temp: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not use_nds:
        return None, "NDS lookup disabled: using manual stud/waler allowables."

    species = species or ""
    grade = grade or ""
    stud_size = stud_size or ""
    waler_size = waler_size or ""

    try:
        stud_record = _get_record_for_size(species, grade, stud_size)
    except Exception as exc:
        return None, f"NDS lookup failed for stud size {stud_size!r}: {exc}"
    try:
        waler_record = _get_record_for_size(species, grade, waler_size)
    except Exception as exc:
        return None, f"NDS lookup failed for waler size {waler_size!r}: {exc}"

    base_s = {
        "Fb": safe_float(stud_record.reference.get("Fb_psi")),
        "Fv": safe_float(stud_record.reference.get("Fv_psi")),
        "E": safe_float(stud_record.reference.get("E_psi")),
    }
    base_w = {
        "Fb": safe_float(waler_record.reference.get("Fb_psi")),
        "Fv": safe_float(waler_record.reference.get("Fv_psi")),
        "E": safe_float(waler_record.reference.get("E_psi")),
    }

    CD, cd_source = resolve_cd_value(cd_preset, cd_custom)
    is_wet = (moisture or "").lower() == "yes"
    is_hot = (temp or "").lower() == "yes"
    Cm_s = nds_cm_factor(is_wet, safe_float(stud_record.factors.get("Cm", 1.0), 1.0))
    Cm_w = nds_cm_factor(is_wet, safe_float(waler_record.factors.get("Cm", 1.0), 1.0))
    Ct = nds_ct_factor(is_hot)
    Cf_s = safe_float(stud_record.factors.get("Cf", 1.0), 1.0)
    Cf_w = safe_float(waler_record.factors.get("Cf", 1.0), 1.0)

    adj_s = apply_nds_factors(base_s, CD, Cm_s, Ct, Cf_s)
    adj_w = apply_nds_factors(base_w, CD, Cm_w, Ct, Cf_w)

    summary = {
        "enabled": True,
        "cd_source": cd_source,
        "factors": {
            "CD": CD,
            "CM": Cm_s,
            "CM_waler": Cm_w,
            "Ct": Ct,
            "Cf_stud": Cf_s,
            "Cf_waler": Cf_w,
            "Cr_stud": safe_float(stud_record.factors.get("Cr", 1.0), 1.0),
            "Cr_waler": safe_float(waler_record.factors.get("Cr", 1.0), 1.0),
            "Cfu_stud": safe_float(stud_record.factors.get("Cfu", 1.0), 1.0),
            "Cfu_waler": safe_float(waler_record.factors.get("Cfu", 1.0), 1.0),
        },
        "stud": {
            "size": stud_size,
            "table": stud_record.table,
            "size_classification": stud_record.size_classification,
            "base": base_s,
            "adjusted": adj_s,
        },
        "waler": {
            "size": waler_size,
            "table": waler_record.table,
            "size_classification": waler_record.size_classification,
            "base": base_w,
            "adjusted": adj_w,
        },
    }
    return summary, None


# -----------------------------
# Tie capacities (SWL) presets (lb)
# -----------------------------
SNAP_TIE_SWL_LB = {"snap_tie": 2250, "heavy_snap_tie": 3750}
TAPER_TIE_SWL_LB = {
    '1/2" taper tie': 7000,
    '5/8" taper tie': 12000,
    '3/4" taper tie': 17000,
}
COIL_ROD_SWL_LB = {
    '3/8"': 4000,
    '1/2"': 6500,
    '5/8"': 9500,
    '3/4"': 14000,
    '7/8"': 18500,
    '1"': 25000,
}


def tie_swl_lb(tie_type: str, tie_size: Optional[str]) -> float:
    if tie_type in ("snap_tie", "heavy_snap_tie"):
        return float(SNAP_TIE_SWL_LB.get(tie_type, float("nan")))
    if tie_type == "taper_tie":
        return float(TAPER_TIE_SWL_LB.get(tie_size or "", float("nan")))
    if tie_type == "coil_rod":
        return float(COIL_ROD_SWL_LB.get(tie_size or "", float("nan")))
    return float("nan")


# -----------------------------
# Utilization calculations
# -----------------------------
def util_ratio(demand: float, capacity: float) -> float:
    if capacity <= 0 or math.isnan(capacity) or math.isnan(demand):
        return float("nan")
    return demand / capacity


def defl_allowable_in(L_in: float, limit_ratio: float) -> float:
    if limit_ratio <= 0:
        return float("nan")
    return L_in / limit_ratio


@dataclass
class MemberUtil:
    member: str
    check: str
    demand: float
    capacity: float
    util: float
    units: str


def check_beam_section(
    *,
    member_name: str,
    beam: BeamResult,
    S_in3: float,
    A_in2: float,
    Fb_allow: float,
    Fv_allow: float,
    L_in: float,
    defl_limit_ratio: float,
    V_shear_override_lb: Optional[float] = None,
) -> List[MemberUtil]:
    out: List[MemberUtil] = []
    M_lbin = beam.Mmax_lbft * 12.0
    fb = M_lbin / S_in3 if S_in3 > 0 else float("nan")
    out.append(
        MemberUtil(
            member=member_name,
            check="Bending (fb <= Fb)",
            demand=fb,
            capacity=Fb_allow,
            util=util_ratio(fb, Fb_allow),
            units="psi",
        )
    )
    V_use = beam.Vmax_lb if V_shear_override_lb is None else float(V_shear_override_lb)
    tau = 1.5 * V_use / A_in2 if A_in2 > 0 else float("nan")
    out.append(
        MemberUtil(
            member=member_name,
            check="Shear (tau <= Fv)",
            demand=tau,
            capacity=Fv_allow,
            util=util_ratio(tau, Fv_allow),
            units="psi",
        )
    )
    da = defl_allowable_in(L_in, defl_limit_ratio)
    out.append(
        MemberUtil(
            member=member_name,
            check=f"Deflection (delta <= L/{int(defl_limit_ratio)})",
            demand=beam.defl_max_in,
            capacity=da,
            util=util_ratio(beam.defl_max_in, da),
            units="in",
        )
    )
    return out


def compute_member_utils_uniform(
    *,
    p_psf: float,
    plywood_t_in: float,
    stud_size: str,
    waler_size: str,
    stud_spacing_in: float,
    waler_spacing_in: float,
    tie_spacing_in: float,
    ply_Fb: float,
    ply_Fv: float,
    ply_E: float,
    stud_Fb: float,
    stud_Fv: float,
    stud_E: float,
    waler_Fb: float,
    waler_Fv: float,
    waler_E: float,
    stud_orientation: str,
    defl_ratio_ply: float,
    defl_ratio_stud: float,
    defl_ratio_waler: float,
) -> List[MemberUtil]:
    p = max(float(p_psf), 0.0)

    s_stud_ft = in_to_ft(max(stud_spacing_in, 0.0))
    s_waler_ft = in_to_ft(max(waler_spacing_in, 0.0))
    s_tie_ft = in_to_ft(max(tie_spacing_in, 0.0))

    S_ply, I_ply, A_ply = member_props_plywood_strip(plywood_t_in, 12.0)
    if stud_orientation == "vertical":
        L_ply_ft = s_stud_ft
        w_ply_lbft = p * s_stud_ft
    else:
        L_ply_ft = s_stud_ft
        w_ply_lbft = p * 1.0
    beam_ply = beam_uniform(L_ply_ft, w_ply_lbft, ply_E, I_ply)
    utils = check_beam_section(
        member_name="Plywood",
        beam=beam_ply,
        S_in3=S_ply,
        A_in2=A_ply,
        Fb_allow=ply_Fb,
        Fv_allow=ply_Fv,
        L_in=ft_to_in(L_ply_ft),
        defl_limit_ratio=defl_ratio_ply,
    )

    S_stud, I_stud, A_stud = member_props_dimlumber(stud_size, strong_axis=True)
    if stud_orientation == "vertical":
        L_stud_ft = s_waler_ft
        w_stud_lbft = p * s_stud_ft
    else:
        L_stud_ft = s_waler_ft
        w_stud_lbft = p * s_stud_ft
    beam_stud = beam_uniform(L_stud_ft, w_stud_lbft, stud_E, I_stud)
    utils += check_beam_section(
        member_name="Stud",
        beam=beam_stud,
        S_in3=S_stud,
        A_in2=A_stud,
        Fb_allow=stud_Fb,
        Fv_allow=stud_Fv,
        L_in=ft_to_in(L_stud_ft),
        defl_limit_ratio=defl_ratio_stud,
    )

    S_w, I_w, A_w = member_props_dimlumber(waler_size, strong_axis=True)
    S_w *= 2.0
    I_w *= 2.0
    A_w *= 2.0

    L_w_ft = s_tie_ft
    w_w_lbft = p * s_waler_ft
    beam_w = beam_uniform(L_w_ft, w_w_lbft, waler_E, I_w)
    d_in = DIMLUMBER_DB.get(waler_size, (0.0, 0.0))[1]
    d_ft = float(d_in) / 12.0
    Vd_lb = abs(w_w_lbft) * max(0.0, (L_w_ft / 2.0 - d_ft))
    utils += check_beam_section(
        member_name="Double waler",
        beam=beam_w,
        S_in3=S_w,
        A_in2=A_w,
        Fb_allow=waler_Fb,
        Fv_allow=waler_Fv,
        L_in=ft_to_in(L_w_ft),
        defl_limit_ratio=defl_ratio_waler,
        V_shear_override_lb=Vd_lb,
    )

    return utils


def compute_segment_checks(
    *,
    profile: pd.DataFrame,
    height_ft: float,
    w_pcf: float,
    p_cap_psf: float,
    plywood_t_in: float,
    stud_size: str,
    waler_size: str,
    stud_spacing_in: float,
    waler_spacing_in: float,
    tie_spacing_in: float,
    ply_Fb: float,
    ply_Fv: float,
    ply_E: float,
    stud_Fb: float,
    stud_Fv: float,
    stud_E: float,
    waler_Fb: float,
    waler_Fv: float,
    waler_E: float,
    stud_orientation: str,
    defl_ratio_ply: float,
    defl_ratio_stud: float,
    defl_ratio_waler: float,
    tie_type: str,
    tie_size: Optional[str],
) -> pd.DataFrame:
    H = max(float(height_ft), 0.0)
    s_stud_ft = in_to_ft(max(stud_spacing_in, 0.0))
    s_waler_ft = in_to_ft(max(waler_spacing_in, 0.0))
    s_tie_ft = in_to_ft(max(tie_spacing_in, 0.0))

    S_ply, I_ply, A_ply = member_props_plywood_strip(plywood_t_in, 12.0)
    S_stud, I_stud, A_stud = member_props_dimlumber(stud_size, strong_axis=True)
    S_w, I_w, A_w = member_props_dimlumber(waler_size, strong_axis=True)
    S_w *= 2.0
    I_w *= 2.0
    A_w *= 2.0

    tie_cap = tie_swl_lb(tie_type, tie_size)

    def max_util_from_utils(utils: List[MemberUtil], member: str) -> float:
        vals = [u.util for u in utils if u.member == member and not math.isnan(u.util)]
        return float(max(vals)) if vals else float("nan")

    def util_vertical_span(
        *,
        member: str,
        z_top: float,
        z_bot: float,
        trib_width_ft: float,
        S_in3: float,
        I_in4: float,
        A_in2: float,
        Fb_allow: float,
        Fv_allow: float,
        E_psi: float,
        defl_ratio: float,
    ) -> float:
        L = max(z_bot - z_top, 0.0)
        if L <= 0:
            return float("nan")
        x = np.linspace(0.0, L, 201)
        z = z_top + x
        p = np.minimum(w_pcf * z, p_cap_psf)
        w_lbft = p * trib_width_ft
        beam = beam_simply_supported_distributed(L, w_lbft, x, E_psi, I_in4)
        utils = check_beam_section(
            member_name=member,
            beam=beam,
            S_in3=S_in3,
            A_in2=A_in2,
            Fb_allow=Fb_allow,
            Fv_allow=Fv_allow,
            L_in=ft_to_in(L),
            defl_limit_ratio=defl_ratio,
        )
        return max_util_from_utils(utils, member)

    def ply_util_at(depth_mid: float) -> float:
        p = pressure_at_depth_psf(depth_mid, w_pcf, p_cap_psf)
        if stud_orientation == "vertical":
            L = s_stud_ft
            w_lbft = p * s_stud_ft
            beam = beam_uniform(L, w_lbft, ply_E, I_ply)
            utils = check_beam_section(
                member_name="Plywood",
                beam=beam,
                S_in3=S_ply,
                A_in2=A_ply,
                Fb_allow=ply_Fb,
                Fv_allow=ply_Fv,
                L_in=ft_to_in(L),
                defl_limit_ratio=defl_ratio_ply,
            )
            return max_util_from_utils(utils, "Plywood")
        if s_stud_ft <= 0:
            return float("nan")
        i = int(depth_mid // s_stud_ft)
        z0 = min(i * s_stud_ft, H)
        z1 = min((i + 1) * s_stud_ft, H)
        return util_vertical_span(
            member="Plywood",
            z_top=z0,
            z_bot=z1,
            trib_width_ft=1.0,
            S_in3=S_ply,
            I_in4=I_ply,
            A_in2=A_ply,
            Fb_allow=ply_Fb,
            Fv_allow=ply_Fv,
            E_psi=ply_E,
            defl_ratio=defl_ratio_ply,
        )

    def stud_util_at(depth_mid: float) -> float:
        p = pressure_at_depth_psf(depth_mid, w_pcf, p_cap_psf)
        if stud_orientation == "vertical":
            if s_waler_ft <= 0:
                return float("nan")
            i = int(depth_mid // s_waler_ft)
            z0 = min(i * s_waler_ft, H)
            z1 = min((i + 1) * s_waler_ft, H)
            return util_vertical_span(
                member="Stud",
                z_top=z0,
                z_bot=z1,
                trib_width_ft=s_stud_ft,
                S_in3=S_stud,
                I_in4=I_stud,
                A_in2=A_stud,
                Fb_allow=stud_Fb,
                Fv_allow=stud_Fv,
                E_psi=stud_E,
                defl_ratio=defl_ratio_stud,
            )
        L = s_waler_ft
        w_lbft = p * s_stud_ft
        beam = beam_uniform(L, w_lbft, stud_E, I_stud)
        utils = check_beam_section(
            member_name="Stud",
            beam=beam,
            S_in3=S_stud,
            A_in2=A_stud,
            Fb_allow=stud_Fb,
            Fv_allow=stud_Fv,
            L_in=ft_to_in(L),
            defl_limit_ratio=defl_ratio_stud,
        )
        return max_util_from_utils(utils, "Stud")

    def waler_util_at(depth_mid: float) -> float:
        p = pressure_at_depth_psf(depth_mid, w_pcf, p_cap_psf)
        if stud_orientation == "vertical":
            L = s_tie_ft
            w_lbft = p * s_waler_ft
            beam = beam_uniform(L, w_lbft, waler_E, I_w)
            utils = check_beam_section(
                member_name="Double waler",
                beam=beam,
                S_in3=S_w,
                A_in2=A_w,
                Fb_allow=waler_Fb,
                Fv_allow=waler_Fv,
                L_in=ft_to_in(L),
                defl_limit_ratio=defl_ratio_waler,
            )
            return max_util_from_utils(utils, "Double waler")
        if s_tie_ft <= 0:
            return float("nan")
        i = int(depth_mid // s_tie_ft)
        z0 = min(i * s_tie_ft, H)
        z1 = min((i + 1) * s_tie_ft, H)
        return util_vertical_span(
            member="Double waler",
            z_top=z0,
            z_bot=z1,
            trib_width_ft=s_waler_ft,
            S_in3=S_w,
            I_in4=I_w,
            A_in2=A_w,
            Fb_allow=waler_Fb,
            Fv_allow=waler_Fv,
            E_psi=waler_E,
            defl_ratio=defl_ratio_waler,
        )

    def tie_utils_at(depth_mid: float) -> Dict[str, float]:
        p = pressure_at_depth_psf(depth_mid, w_pcf, p_cap_psf)
        A_int = s_tie_ft * s_waler_ft
        A_edge = 0.5 * s_tie_ft * s_waler_ft
        A_corner = 0.25 * s_tie_ft * s_waler_ft
        A_tb_edge = s_tie_ft * (0.5 * s_waler_ft)

        dem_int = p * A_int
        dem_edge = p * A_edge
        dem_corner = p * A_corner
        dem_tb = p * A_tb_edge
        return {
            "Tie (interior)": util_ratio(dem_int, tie_cap),
            "Tie (edge)": util_ratio(dem_edge, tie_cap),
            "Tie (corner)": util_ratio(dem_corner, tie_cap),
            "Tie (top/bottom edge)": util_ratio(dem_tb, tie_cap),
        }

    seg_rows = []
    for _, r in profile.iterrows():
        depth_mid = float(r["DepthTop_ft"] + r["DepthBot_ft"]) * 0.5
        pu = ply_util_at(depth_mid)
        su = stud_util_at(depth_mid)
        wu = waler_util_at(depth_mid)
        tu = tie_utils_at(depth_mid)

        controls = {"Plywood": pu, "Stud": su, "Double waler": wu, **tu}
        best = None
        maxu = -1.0
        for k, v in controls.items():
            if v is None or math.isnan(v):
                continue
            if v > maxu:
                maxu = v
                best = k

        seg_rows.append(
            {
                **{k: r[k] for k in r.index},
                "ply_util": pu,
                "stud_util": su,
                "waler_util": wu,
                "tie_util_interior": tu["Tie (interior)"],
                "tie_util_edge": tu["Tie (edge)"],
                "tie_util_corner": tu["Tie (corner)"],
                "tie_util_tb_edge": tu["Tie (top/bottom edge)"],
                "controlling_member": best or "",
                "controlling_util": maxu if maxu >= 0 else float("nan"),
            }
        )

    return pd.DataFrame(seg_rows)


def build_report_payload(
    *,
    inputs: Dict[str, Any],
    pressure_summary: Dict[str, Any],
    util_summary: Dict[str, Any],
    segment_table: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "meta": {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "inputs": inputs,
        "pressure_summary": pressure_summary,
        "util_summary": util_summary,
        "segment_table": segment_table,
    }


def build_pdf_bytes(report: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Wood Formwork Design Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    meta = report.get("meta", {})
    story.append(Paragraph(f"Generated: {meta.get('generated_at','')}", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Inputs", styles["Heading2"]))
    inputs = report.get("inputs", {})
    input_rows = [["Field", "Value"]]
    for k, v in inputs.items():
        input_rows.append([str(k), str(v)])
    t = Table(input_rows, colWidths=[2.7 * inch, 3.6 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Pressure Summary", styles["Heading2"]))
    ps = report.get("pressure_summary", {})
    ps_rows = [["Item", "Value"]]
    for k, v in ps.items():
        ps_rows.append([str(k), str(v)])
    t2 = Table(ps_rows, colWidths=[2.7 * inch, 3.6 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Governing Utilizations", styles["Heading2"]))
    us = report.get("util_summary", {})
    us_rows = [["Member / Case", "Utilization (max)"]]
    for k, v in us.items():
        val = f"{v:.3f}" if isinstance(v, (int, float)) and not math.isnan(v) else str(v)
        us_rows.append([str(k), val])
    t3 = Table(us_rows, colWidths=[3.2 * inch, 3.1 * inch])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 0.2 * inch))

    seg = report.get("segment_table", [])
    story.append(Paragraph("Segment Checks (top and bottom)", styles["Heading2"]))
    if seg:
        df = pd.DataFrame(seg)
        cols = ["Segment", "DepthTop_ft", "DepthBot_ft", "p_mid_psf", "controlling_member", "controlling_util"]
        df2 = df[cols].copy()
        if len(df2) > 60:
            df2 = pd.concat([df2.head(30), df2.tail(30)], ignore_index=True)
        tbl = [cols] + df2.round(3).astype(str).values.tolist()
        t4 = Table(tbl, colWidths=[0.7 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 2.2 * inch, 0.9 * inch])
        t4.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(t4)
        if len(seg) > 60:
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph("Note: Segment table truncated in PDF. Download CSV for full detail.", styles["Italic"]))
    else:
        story.append(Paragraph("No segment data available.", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


def default_inputs() -> Dict[str, Any]:
    species_list = nds_species_options()
    default_species = species_list[0] if species_list else ""
    grade_list = nds_grade_options_for_species(default_species)
    default_grade = grade_list[0] if grade_list else ""

    return {
        "element_type": "wall",
        "form_height_ft": 12,
        "segment_ft": 1.0,
        "unit_weight_pcf": 150,
        "temp_F": 70,
        "rate_ftph": 7,
        "slump_in": 5,
        "vib_depth_ft": 4,
        "mix_category": "normal",
        "special_conditions": [],
        "ply_thk": 0.75,
        "stud_size": "2x4",
        "waler_size": "2x6",
        "stud_spacing_in": 16,
        "waler_spacing_in": 24,
        "tie_spacing_in": 24,
        "stud_orientation": "vertical",
        "ply_Fb": 1500,
        "ply_Fv": 120,
        "ply_E": 1_200_000,
        "stud_Fb": 900,
        "stud_Fv": 180,
        "stud_E": 1_600_000,
        "waler_Fb": 900,
        "waler_Fv": 180,
        "waler_E": 1_600_000,
        "defl_ply": 240,
        "defl_stud": 240,
        "defl_waler": 240,
        "tie_type": "snap_tie",
        "tie_size": None,
        "tie_swl_override": None,
        "use_nds": True,
        "nds_species": default_species,
        "nds_grade": default_grade,
        "nds_cd_preset": 1.25,
        "nds_cd_custom": None,
        "nds_moisture": "no",
        "nds_temp": "no",
    }


def get_config() -> Dict[str, Any]:
    species = nds_species_options()
    grades_by_species = {s: nds_grade_options_for_species(s) for s in species}
    return {
        "tool_version": TOOL_VERSION,
        "defaults": default_inputs(),
        "options": {
            "element_types": [
                {"label": "Wall", "value": "wall"},
                {"label": "Column", "value": "column"},
            ],
            "mix_categories": [
                {"label": "Normal", "value": "normal"},
                {"label": "High cementitious (slower set)", "value": "high_cementitious"},
                {"label": "Retarded (explicit)", "value": "retarded"},
                {"label": "Accelerated", "value": "accelerated"},
            ],
            "special_conditions": [
                {"label": "Retarder included", "value": "retarder"},
                {"label": "SCC (hydrostatic)", "value": "scc"},
                {"label": "Pumped from base (+25%)", "value": "pumped"},
            ],
            "plywood_thickness": PLYWOOD_THICKNESS_OPTIONS,
            "dimlumber_sizes": DIMLUMBER_OPTIONS,
            "stud_orientations": [
                {"label": "Vertical studs (common)", "value": "vertical"},
                {"label": "Horizontal studs", "value": "horizontal"},
            ],
            "tie_types": [
                {"label": "Snap tie", "value": "snap_tie"},
                {"label": "Heavy duty snap tie", "value": "heavy_snap_tie"},
                {"label": "Taper tie (select size)", "value": "taper_tie"},
                {"label": "Coil rod (select diameter)", "value": "coil_rod"},
            ],
            "tie_sizes": {
                "taper_tie": [
                    {"label": f'{k} (SWL {v:,} lb)', "value": k} for k, v in TAPER_TIE_SWL_LB.items()
                ],
                "coil_rod": [
                    {"label": f'{k} (SWL {v:,} lb)', "value": k} for k, v in COIL_ROD_SWL_LB.items()
                ],
            },
            "nds": {
                "species": species,
                "grades_by_species": grades_by_species,
                "cd_presets": [0.9, 1.0, 1.15, 1.25, 1.6],
                "yes_no": ["no", "yes"],
            },
        },
    }


def solve(inputs: Dict[str, Any]) -> Dict[str, Any]:
    data = default_inputs()
    data.update({k: v for k, v in inputs.items() if v is not None})

    sc = set(data.get("special_conditions") or [])
    retarder = "retarder" in sc
    is_scc = "scc" in sc
    pumped = "pumped" in sc

    use_nds = bool(data.get("use_nds"))
    nds_summary, nds_note = compute_nds_adjustments(
        use_nds=use_nds,
        species=str(data.get("nds_species") or ""),
        grade=str(data.get("nds_grade") or ""),
        stud_size=str(data.get("stud_size") or ""),
        waler_size=str(data.get("waler_size") or ""),
        cd_preset=safe_float(data.get("nds_cd_preset"), 1.25),
        cd_custom=safe_float(data.get("nds_cd_custom"), float("nan")),
        moisture=str(data.get("nds_moisture") or "no"),
        temp=str(data.get("nds_temp") or "no"),
    )

    if nds_summary:
        data["stud_Fb"] = nds_summary["stud"]["adjusted"]["Fb"]
        data["stud_Fv"] = nds_summary["stud"]["adjusted"]["Fv"]
        data["stud_E"] = nds_summary["stud"]["adjusted"]["E"]
        data["waler_Fb"] = nds_summary["waler"]["adjusted"]["Fb"]
        data["waler_Fv"] = nds_summary["waler"]["adjusted"]["Fv"]
        data["waler_E"] = nds_summary["waler"]["adjusted"]["E"]

    pres = aci347r14_lateral_pressure_psf(
        element_type=data.get("element_type") or "wall",
        height_ft=safe_float(data.get("form_height_ft"), 0.0),
        w_pcf=safe_float(data.get("unit_weight_pcf"), 150.0),
        T_F=safe_float(data.get("temp_F"), 70.0),
        R_ftph=safe_float(data.get("rate_ftph"), 0.0),
        slump_in=safe_float(data.get("slump_in"), 5.0),
        internal_vib_depth_ft=safe_float(data.get("vib_depth_ft"), 4.0),
        mix_category=data.get("mix_category") or "normal",
        retarder_included=retarder,
        is_scc=is_scc,
        pumped_from_base=pumped,
    )
    p_cap = float(pres["p_cap_psf"])
    p_hydro = float(pres["p_hydro_psf"])
    p_emp = float(pres.get("p_empirical_psf", float("nan")))
    p_min = float(pres.get("p_min_psf", float("nan")))
    Cc = float(pres["Cc"])
    Cw = float(pres["Cw"])
    eq = str(pres["controlling_eq"])

    H = safe_float(data.get("form_height_ft"), 0.0)
    seg = safe_float(data.get("segment_ft"), 1.0)
    w = safe_float(data.get("unit_weight_pcf"), 150.0)
    profile = build_pressure_profile(H, w, p_cap, seg)

    swl = tie_swl_lb(data.get("tie_type") or "snap_tie", data.get("tie_size"))
    tie_override = safe_float(data.get("tie_swl_override"), float("nan"))
    if math.isfinite(tie_override) and tie_override > 0:
        swl = tie_override
        tie_note = "override"
    else:
        tie_note = "preset"

    utils_uniform = compute_member_utils_uniform(
        p_psf=p_cap,
        plywood_t_in=safe_float(data.get("ply_thk"), 0.75),
        stud_size=data.get("stud_size") or "2x4",
        waler_size=data.get("waler_size") or "2x6",
        stud_spacing_in=safe_float(data.get("stud_spacing_in"), 16),
        waler_spacing_in=safe_float(data.get("waler_spacing_in"), 24),
        tie_spacing_in=safe_float(data.get("tie_spacing_in"), 24),
        ply_Fb=safe_float(data.get("ply_Fb"), 1500),
        ply_Fv=safe_float(data.get("ply_Fv"), 120),
        ply_E=safe_float(data.get("ply_E"), 1_200_000),
        stud_Fb=safe_float(data.get("stud_Fb"), 900),
        stud_Fv=safe_float(data.get("stud_Fv"), 180),
        stud_E=safe_float(data.get("stud_E"), 1_600_000),
        waler_Fb=safe_float(data.get("waler_Fb"), 900),
        waler_Fv=safe_float(data.get("waler_Fv"), 180),
        waler_E=safe_float(data.get("waler_E"), 1_600_000),
        stud_orientation=data.get("stud_orientation") or "vertical",
        defl_ratio_ply=safe_float(data.get("defl_ply"), 240),
        defl_ratio_stud=safe_float(data.get("defl_stud"), 240),
        defl_ratio_waler=safe_float(data.get("defl_waler"), 240),
    )

    s_waler_ft = in_to_ft(max(safe_float(data.get("waler_spacing_in"), 24), 0.0))
    s_tie_ft = in_to_ft(max(safe_float(data.get("tie_spacing_in"), 24), 0.0))
    tie_area_int = s_waler_ft * s_tie_ft
    tie_cases = {
        "Tie (interior)": 1.00,
        "Tie (edge)": 0.50,
        "Tie (corner)": 0.25,
        "Tie (top/bottom edge)": 0.50,
    }
    for name, factor in tie_cases.items():
        demand = p_cap * tie_area_int * factor
        utils_uniform.append(
            MemberUtil(
                member="Tie",
                check=f"{name} (T <= SWL)",
                demand=demand,
                capacity=swl,
                util=util_ratio(demand, swl),
                units="lb",
            )
        )

    seg_df = compute_segment_checks(
        profile=profile,
        height_ft=H,
        w_pcf=w,
        p_cap_psf=p_cap,
        plywood_t_in=safe_float(data.get("ply_thk"), 0.75),
        stud_size=data.get("stud_size") or "2x4",
        waler_size=data.get("waler_size") or "2x6",
        stud_spacing_in=safe_float(data.get("stud_spacing_in"), 16),
        waler_spacing_in=safe_float(data.get("waler_spacing_in"), 24),
        tie_spacing_in=safe_float(data.get("tie_spacing_in"), 24),
        ply_Fb=safe_float(data.get("ply_Fb"), 1500),
        ply_Fv=safe_float(data.get("ply_Fv"), 120),
        ply_E=safe_float(data.get("ply_E"), 1_200_000),
        stud_Fb=safe_float(data.get("stud_Fb"), 900),
        stud_Fv=safe_float(data.get("stud_Fv"), 180),
        stud_E=safe_float(data.get("stud_E"), 1_600_000),
        waler_Fb=safe_float(data.get("waler_Fb"), 900),
        waler_Fv=safe_float(data.get("waler_Fv"), 180),
        waler_E=safe_float(data.get("waler_E"), 1_600_000),
        stud_orientation=data.get("stud_orientation") or "vertical",
        defl_ratio_ply=safe_float(data.get("defl_ply"), 240),
        defl_ratio_stud=safe_float(data.get("defl_stud"), 240),
        defl_ratio_waler=safe_float(data.get("defl_waler"), 240),
        tie_type=data.get("tie_type") or "snap_tie",
        tie_size=data.get("tie_size"),
    )

    depths = np.concatenate([[0.0], seg_df["DepthBot_ft"].values])
    p_vals = np.array([pressure_at_depth_psf(z, w, p_cap) for z in depths])

    util_summary = {
        "Plywood (max seg)": float(np.nanmax(seg_df["ply_util"].values)),
        "Stud (max seg)": float(np.nanmax(seg_df["stud_util"].values)),
        "Double waler (max seg)": float(np.nanmax(seg_df["waler_util"].values)),
        "Tie (interior max seg)": float(np.nanmax(seg_df["tie_util_interior"].values)),
        "Tie (edge max seg)": float(np.nanmax(seg_df["tie_util_edge"].values)),
        "Tie (corner max seg)": float(np.nanmax(seg_df["tie_util_corner"].values)),
        "Tie (top/bottom edge max seg)": float(np.nanmax(seg_df["tie_util_tb_edge"].values)),
    }

    pressure_summary = {
        "p_cap_psf": p_cap,
        "p_hydro_psf": p_hydro,
        "p_empirical_psf": None if math.isnan(p_emp) else p_emp,
        "p_min_psf": None if math.isnan(p_min) else p_min,
        "Cc": Cc,
        "Cw": Cw,
        "case": eq,
    }

    util_table = [
        {
            "member": u.member,
            "check": u.check,
            "demand": u.demand,
            "capacity": u.capacity,
            "utilization": u.util,
            "units": u.units,
        }
        for u in utils_uniform
    ]

    return {
        "ok": True,
        "tool_version": TOOL_VERSION,
        "inputs": data,
        "pressure_summary": pressure_summary,
        "tie_capacity": {"swl": swl, "note": tie_note},
        "util_table": util_table,
        "segment_table": seg_df.to_dict("records"),
        "pressure_profile": {
            "depths_ft": depths.tolist(),
            "pressures_psf": p_vals.tolist(),
        },
        "util_summary": util_summary,
        "nds_summary": nds_summary,
        "nds_note": nds_note,
    }
