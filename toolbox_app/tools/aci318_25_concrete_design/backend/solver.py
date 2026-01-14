\
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure tool root is importable when backend modules run as scripts
_TOOL_ROOT = _Path(__file__).resolve().parents[1]
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))


import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from calc_trace import CalcTrace, TraceAssumption, TraceInput, TraceMeta, TraceSummary, compute_step
from exports import export_all
from paths import create_run_dir, input_hash

from models import (
    BeamFlexureInputs,
    ColumnAxialInputs,
    DevLengthTensionInputs,
    DevelopmentLengthSpliceInputs,
    AnchorsCh17Inputs,
    PunchingShearInputs,
    SlabOnewayFlexureInputs,
    WallSlenderInputs,
)


TOOL_VERSION = "0.4.0"
REPORT_VERSION = "0.4.0"
CODE_BASIS = "ACI 318-25"
UNITS_SYSTEM = "US"


# Simple US rebar database (nominal)
REBAR_DB = {
    "#3": {"db_in": 0.375, "area_in2": 0.11},
    "#4": {"db_in": 0.500, "area_in2": 0.20},
    "#5": {"db_in": 0.625, "area_in2": 0.31},
    "#6": {"db_in": 0.750, "area_in2": 0.44},
    "#7": {"db_in": 0.875, "area_in2": 0.60},
    "#8": {"db_in": 1.000, "area_in2": 0.79},
    "#9": {"db_in": 1.128, "area_in2": 1.00},
    "#10": {"db_in": 1.270, "area_in2": 1.27},
    "#11": {"db_in": 1.410, "area_in2": 1.56},
    "#14": {"db_in": 1.693, "area_in2": 2.25},
    "#18": {"db_in": 2.257, "area_in2": 4.00},
}


def _reinf_layer_area_in2(layer: dict) -> float:
    """Return reinforcement area for a layer dict from pydantic model_dump."""
    if layer.get("As_override_in2") is not None:
        return float(layer["As_override_in2"])
    n = float(layer.get("n_bars") or 0.0)
    db = float(layer.get("bar_dia_in") or 0.0)
    return n * (math.pi * db * db / 4.0)


def _reinf_layer_y_from_top_in(layer: dict, h_in: float) -> float:
    face = layer.get("face", "bottom")
    off = float(layer.get("offset_in") or 0.0)
    if face == "top":
        return off
    return h_in - off


def _phi_flexure_from_eps_t(eps_t: float, eps_ty: float, transverse_type: str) -> float:
    """ACI 318-25 Table 21.2.2 for flexure-controlled members."""
    phi_cc = 0.75 if transverse_type == "spiral" else 0.65
    if eps_t <= eps_ty:
        return phi_cc
    if eps_t >= eps_ty + 0.003:
        return 0.90
    if transverse_type == "spiral":
        return 0.75 + 0.15 * (eps_t - eps_ty) / 0.003
    return 0.65 + 0.25 * (eps_t - eps_ty) / 0.003


def _solve_c_equilibrium_multilayer(
    *,
    fc_psi: float,
    b_in: float,
    h_in: float,
    beta1: float,
    fy_psi: float,
    Es_psi: float,
    layers_As_y: list[tuple[float, float]],
    eps_cu: float = 0.003,
) -> float:
    """Solve neutral axis depth c (from top) for pure bending (Pn=0) using bisection.

    Equilibrium: 0 = Cc + sum(Fsi)
    where Cc = 0.85 fc b a, a=beta1*c
    and Fsi = As_i * fsi, fsi=clamp(Es*eps_si, -fy, fy), eps_si = eps_cu*(c - y_i)/c.
    """

    def Pn(c: float) -> float:
        a = beta1 * c
        Cc = 0.85 * fc_psi * b_in * a  # lb
        P = Cc
        for As, y in layers_As_y:
            eps_s = eps_cu * (c - y) / c
            fs = max(-fy_psi, min(fy_psi, Es_psi * eps_s))
            P += As * fs
        return P

    c_lo = 1e-6
    c_hi = max(h_in * 2.0, 1.0)
    f_lo = Pn(c_lo)
    f_hi = Pn(c_hi)

    # Expand bracket if needed
    it = 0
    while f_lo * f_hi > 0 and it < 30:
        c_hi *= 1.5
        f_hi = Pn(c_hi)
        it += 1

    if f_lo * f_hi > 0:
        # No sign change; return a conservative large c within section depth
        return float('nan')

    for _ in range(80):
        c_mid = 0.5 * (c_lo + c_hi)
        f_mid = Pn(c_mid)
        if abs(f_mid) < 1e-3:
            return c_mid
        if f_lo * f_mid <= 0:
            c_hi = c_mid
            f_hi = f_mid
        else:
            c_lo = c_mid
            f_lo = f_mid
    return 0.5 * (c_lo + c_hi)


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"aci318_tool_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _make_trace(tool_id: str, inputs: Dict[str, Any]) -> CalcTrace:
    ih = input_hash(inputs)
    meta = TraceMeta(
        tool_id=tool_id,
        tool_version=TOOL_VERSION,
        report_version=REPORT_VERSION,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        units_system=UNITS_SYSTEM,
        code_basis=CODE_BASIS,
        input_hash=ih,
        app_build=None,
    )
    return CalcTrace(meta=meta)


def _add_inputs(trace: CalcTrace, inputs: Dict[str, Any]) -> None:
    for k, v in inputs.items():
        if v is None:
            continue
        # units: inferred from name suffix if present
        units = ""
        if k.endswith("_psi"):
            units = "psi"
        elif k.endswith("_kipft"):
            units = "kip-ft"
        elif k.endswith("_kip"):
            units = "kip"
        elif k.endswith("_lbin"):
            units = "lb-in"
        elif k.endswith("_lb"):
            units = "lb"
        elif k.endswith("_in2"):
            units = "in^2"
        elif k.endswith("_in"):
            units = "in"
        elif k.endswith("_per_ft"):
            units = "/ft"
        val = v
        if isinstance(v, (list, dict)):
            val = json.dumps(v, ensure_ascii=False)
            units = units or "json"
        trace.inputs.append(TraceInput(id=k, label=k, value=val, units=units, source="user"))


def _phi_from_epsilon_t(trace: CalcTrace, *, step_id: str, eps_t: float, eps_ty: float, transverse_type: str) -> float:
    """
    Table 21.2.2 — strength reduction factor phi based on net tensile strain eps_t.
    This helper is currently not used by the main modules (they compute φ explicitly per-iteration),
    but is retained as a utility.
    """
    phi_cc = 0.75 if transverse_type == "spiral" else 0.65
    transition_expr = "0.75 + 0.15(\\varepsilon_t-\\varepsilon_{ty})/0.003" if transverse_type == "spiral" else "0.65 + 0.25(\\varepsilon_t-\\varepsilon_{ty})/0.003"

    def _compute():
        if eps_t <= eps_ty:
            return phi_cc
        if eps_t >= eps_ty + 0.003:
            return 0.90
        if transverse_type == "spiral":
            return 0.75 + 0.15 * (eps_t - eps_ty) / 0.003
        return 0.65 + 0.25 * (eps_t - eps_ty) / 0.003

    return compute_step(
        trace,
        id=step_id,
        section="Strength reduction factor",
        title="Strength reduction factor φ from net tensile strain εt",
        output_symbol="\\phi",
        output_description="Strength reduction factor",
        equation_latex=(
            "\\phi = \\begin{cases}"
            + f"{phi_cc} & \\text{{if }} \\varepsilon_t \\le \\varepsilon_{{ty}} \\\\"
            + "0.90 & \\text{if } \\varepsilon_t \\ge \\varepsilon_{ty}+0.003 \\\\"
            + f"{transition_expr} & \\text{{otherwise}}"
            + "\\end{cases}"
        ),
        variables=[
            {"symbol": "\\varepsilon_t", "description": "Net tensile strain at nominal strength", "value": eps_t, "units": "", "source": f"step:{step_id}_eps_t"},
            {"symbol": "\\varepsilon_{ty}", "description": "Yield strain of reinforcement", "value": eps_ty, "units": "", "source": f"step:{step_id}_eps_ty"},
        ],
        compute_fn=_compute,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2"}],
    )

def _beta1(trace: CalcTrace, fc_psi: float) -> float:
    """
    Table 22.2.2.4.3 — beta1 as a function of f'c.
    """
    def _compute():
        if fc_psi <= 4000.0:
            return 0.85
        b1 = 0.85 - 0.05 * ((fc_psi - 4000.0) / 1000.0)
        return max(0.65, b1)

    return compute_step(
        trace,
        id="beta1",
        section="Concrete stress block",
        title="Equivalent rectangular stress block factor β1",
        output_symbol="\\beta_1",
        output_description="Stress block factor",
        equation_latex="\\beta_1 = \\max\\left(0.65, 0.85 - 0.05\\frac{f'_c-4000}{1000}\\right) \\; (f'_c>4000\\,\\mathrm{psi});\\; \\beta_1=0.85\\; (f'_c\\le 4000\\,\\mathrm{psi})",
        variables=[
            {"symbol": "f'_c", "description": "Specified compressive strength of concrete", "value": fc_psi, "units": "psi", "source": "input:fc_psi"},
        ],
        compute_fn=_compute,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 Table 22.2.2.4.3"}],
    )


def _Ec_normalweight(trace: CalcTrace, fc_psi: float) -> float:
    """Concrete modulus of elasticity for normalweight concrete.

    ACI 318-25 19.2.2 provides Ec expressions. This tool uses the common normalweight
    approximation Ec = 57,000 * sqrt(f'c) (psi).
    """

    return compute_step(
        trace,
        id="Ec",
        section="Materials",
        title="Concrete modulus of elasticity (normalweight)",
        output_symbol="E_c",
        output_description="Concrete modulus of elasticity",
        equation_latex="E_c = 57000\,\sqrt{f'_c}",
        variables=[
            {"symbol": "f'_c", "description": "Specified concrete compressive strength", "value": fc_psi, "units": "psi", "source": "input:fc_psi"},
        ],
        compute_fn=lambda: 57000.0 * math.sqrt(fc_psi),
        units="psi",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 19.2.2 (Ec for normalweight concrete)"}],
    )


def _wall_k_factor(trace: CalcTrace, end_bottom: str, end_top: str) -> float:
    """Effective length factor k based on simplified wall boundary conditions.

    Uses ACI 318-25 Table 11.5.3.2 (simplified wall design).

    Mapping used (per user-requested 2 end-condition dropdowns):
    - If either end is "unbraced" -> k = 2.0
    - Else if either end is "fixed" -> k = 0.8
    - Else (both pinned) -> k = 1.0
    """

    def _compute():
        if end_bottom == "unbraced" or end_top == "unbraced":
            return 2.0
        if end_bottom == "fixed" or end_top == "fixed":
            return 0.8
        return 1.0

    return compute_step(
        trace,
        id="k",
        section="Slenderness",
        title="Effective length factor k from end conditions",
        output_symbol="k",
        output_description="Effective length factor",
        equation_latex=(
            "k = \\begin{cases}"
            "2.0 & \\text{if wall not braced against translation}\\\\"
            "0.8 & \\text{if braced and restrained against rotation at one or both ends}\\\\"
            "1.0 & \\text{if braced and unrestrained against rotation at both ends}"
            "\\end{cases}"
        ),
        variables=[
            {"symbol": "e_{bot}", "description": f"Bottom end condition code (0=pinned, 1=fixed, 2=unbraced); selected={end_bottom}", "value": {"pinned":0.0,"fixed":1.0,"unbraced":2.0}[end_bottom], "units": "", "source": "input:end_bottom"},
            {"symbol": "e_{top}", "description": f"Top end condition code (0=pinned, 1=fixed, 2=unbraced); selected={end_top}", "value": {"pinned":0.0,"fixed":1.0,"unbraced":2.0}[end_top], "units": "", "source": "input:end_top"},
        ],
        compute_fn=_compute,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 11.5.3.2"}],
    )


def _phi_table_21_2_2(*, eps_t: float, eps_ty: float, transverse_type: str) -> float:
    """Table 21.2.2 phi for combined moment and axial force."""
    phi_cc = 0.75 if transverse_type == "spiral" else 0.65
    if eps_t <= eps_ty:
        return phi_cc
    if eps_t >= eps_ty + 0.003:
        return 0.90
    if transverse_type == "spiral":
        return 0.75 + 0.15 * (eps_t - eps_ty) / 0.003
    return 0.65 + 0.25 * (eps_t - eps_ty) / 0.003


def _section_capacity_points_rect(
    *,
    b_in: float,
    h_in: float,
    As_total_in2: float,
    cover_in: float,
    bar_dia_in: float,
    fc_psi: float,
    fy_psi: float,
    Es_psi: float,
    transverse_type: str,
    beta1: float,
    n_points: int = 240,
) -> Tuple[List[float], List[float], List[float]]:
    """Generate phi*Pn and phi*Mn interaction points for symmetric 2-face reinforcement.

    Returns lists: phiPn_kip, phiMn_kipft, eps_t.
    """
    eps_cu = 0.003
    As_face = 0.5 * As_total_in2
    y_comp = cover_in + 0.5 * bar_dia_in
    y_tens = h_in - y_comp
    eps_ty = fy_psi / Es_psi

    def steel_stress(eps: float) -> float:
        fs = Es_psi * eps
        if fs > fy_psi:
            return fy_psi
        if fs < -fy_psi:
            return -fy_psi
        return fs

    phiPn: List[float] = []
    phiMn: List[float] = []
    eps_t_list: List[float] = []

    # c from very small to large (include pure compression region)
    c_min = 0.05 * h_in
    c_max = 6.0 * h_in
    for i in range(n_points):
        c = c_min + (c_max - c_min) * (i / (n_points - 1))
        a = beta1 * c
        if a > h_in:
            a = h_in

        Cc_lb = 0.85 * fc_psi * b_in * a
        # strains at steel layers
        eps_s_comp = eps_cu * (1.0 - y_comp / c)
        eps_s_tens = eps_cu * (1.0 - y_tens / c)
        fs_comp = steel_stress(eps_s_comp)
        fs_tens = steel_stress(eps_s_tens)
        Fs_comp_lb = fs_comp * As_face
        Fs_tens_lb = fs_tens * As_face

        Pn_lb = Cc_lb + Fs_comp_lb + Fs_tens_lb

        # moment about centroid
        y_cent = 0.5 * h_in
        M_lb_in = (
            Cc_lb * ((a / 2.0) - y_cent)
            + Fs_comp_lb * (y_comp - y_cent)
            + Fs_tens_lb * (y_tens - y_cent)
        )

        eps_t = max(0.0, eps_cu * (y_tens - c) / c) if y_tens > c else 0.0
        phi = _phi_table_21_2_2(eps_t=eps_t, eps_ty=eps_ty, transverse_type=transverse_type)

        phiPn.append(phi * Pn_lb / 1000.0)  # kip
        phiMn.append(abs(phi * M_lb_in / (1000.0 * 12.0)))  # kip-ft
        eps_t_list.append(eps_t)

    return phiPn, phiMn, eps_t_list


def _interp_capacity_moment(phiPn_kip: List[float], phiMn_kipft: List[float], Pu_kip: float) -> float:
    """Interpolate capacity moment at a given Pu based on phiPn/phiMn arrays."""
    # sort by phiPn
    pts = sorted(zip(phiPn_kip, phiMn_kipft), key=lambda x: x[0])
    if Pu_kip <= pts[0][0]:
        return pts[0][1]
    if Pu_kip >= pts[-1][0]:
        return pts[-1][1]
    for (p1, m1), (p2, m2) in zip(pts[:-1], pts[1:]):
        if p1 <= Pu_kip <= p2:
            if abs(p2 - p1) < 1e-9:
                return min(m1, m2)
            t = (Pu_kip - p1) / (p2 - p1)
            return m1 + t * (m2 - m1)
    return pts[-1][1]


def _moment_magnification_nonsway(
    trace: CalcTrace,
    *,
    prefix: str,
    Pu_kip: float,
    M_top_kipft: float,
    M_bot_kipft: float,
    h_in: float,
    Ec_psi: float,
    I_in4: float,
    beta_dns: float,
    k: float,
    lu_in: float,
    transverse_loads_between_ends: bool,
) -> Tuple[float, float, float, float]:
    """ACI 318-25 moment magnification for nonsway members (6.6.4.5).

    Returns (Cm, Pc_kip, delta_ns, M2_design_kipft).
    """
    # (EI)eff = Ec*I/(1+beta_dns) per 6.6.4.4.4(c)
    EI_eff = compute_step(
        trace,
        id=f"{prefix}_EIeff",
        section="Slenderness",
        title=f"Effective stiffness (EI)eff for moment magnification ({prefix})",
        output_symbol="(EI)_{eff}",
        output_description="Effective flexural rigidity",
        equation_latex="(EI)_{eff} = \\frac{E_c I}{1+\\beta_{dns}}",
        variables=[
            {"symbol": "E_c", "description": "Concrete modulus", "value": Ec_psi, "units": "psi", "source": "step:Ec"},
            {"symbol": "I", "description": "Effective moment of inertia", "value": I_in4, "units": "in^4", "source": f"step:{prefix}_I"},
            {"symbol": "\\beta_{dns}", "description": "Sustained/total axial load ratio", "value": beta_dns, "units": "", "source": "input:beta_dns"},
        ],
        compute_fn=lambda: (Ec_psi * I_in4) / (1.0 + beta_dns),
        units="lb-in^2",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 Eq. (6.6.4.4.4c)"}],
    )

    # Pc per Eq. (6.6.4.4.2)
    Pc_kip = compute_step(
        trace,
        id=f"{prefix}_Pc",
        section="Slenderness",
        title=f"Critical buckling load Pc ({prefix})",
        output_symbol="P_c",
        output_description="Euler critical buckling load",
        equation_latex="P_c = \\frac{\\pi^2 (EI)_{eff}}{(k\\ell_u)^2}",
        variables=[
            {"symbol": "(EI)_{eff}", "description": "Effective stiffness", "value": EI_eff, "units": "lb-in^2", "source": f"step:{prefix}_EIeff"},
            {"symbol": "k", "description": "Effective length factor", "value": k, "units": "", "source": "step:k"},
            {"symbol": "\\ell_u", "description": "Unsupported length", "value": lu_in, "units": "in", "source": "input:lu_in"},
        ],
        compute_fn=lambda: (math.pi**2 * EI_eff) / ((k * lu_in) ** 2) / 1000.0,
        units="kip",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 Eq. (6.6.4.4.2)"}],
    )

    # Determine M1 and M2 per 6.6.4.5.3; for transverse loads between ends, Cm=1.0
    def _compute_Cm():
        if transverse_loads_between_ends:
            return 1.0
        # M2 is larger absolute, M1 is smaller with sign
        if abs(M_top_kipft) >= abs(M_bot_kipft):
            M2 = M_top_kipft
            M1 = M_bot_kipft
        else:
            M2 = M_bot_kipft
            M1 = M_top_kipft
        if abs(M2) < 1e-9:
            return 1.0
        r = M1 / M2
        return max(0.4, 0.6 - 0.4 * r)

    Cm = compute_step(
        trace,
        id=f"{prefix}_Cm",
        section="Slenderness",
        title=f"Moment gradient factor Cm ({prefix})",
        output_symbol="C_m",
        output_description="Moment gradient factor",
        equation_latex="C_m = \\max\\left(0.4,\\, 0.6 - 0.4\\frac{M_1}{M_2}\\right)\\;\\text{(if end moments only)};\\; C_m=1.0\\;\\text{(if transverse loads between ends)}",
        variables=[
            {"symbol": "M_{top}", "description": "End moment at top", "value": M_top_kipft, "units": "kip-ft", "source": f"input:{'M_top_oop_kipft' if prefix=='oop' else 'M_top_ip_kipft'}"},
            {"symbol": "M_{bot}", "description": "End moment at bottom", "value": M_bot_kipft, "units": "kip-ft", "source": f"input:{'M_bot_oop_kipft' if prefix=='oop' else 'M_bot_ip_kipft'}"},
        ],
        compute_fn=_compute_Cm,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 6.6.4.5.3"}],
    )

    # M2 (max end moment) and M2,min per Eq. (6.6.4.5.4)
    def _compute_M2():
        return max(abs(M_top_kipft), abs(M_bot_kipft))

    M2 = compute_step(
        trace,
        id=f"{prefix}_M2",
        section="Slenderness",
        title=f"Second-order end moment basis M2 ({prefix})",
        output_symbol="M_2",
        output_description="Larger end moment magnitude",
        equation_latex="M_2 = \\max\\left(|M_{top}|,|M_{bot}|\\right)",
        variables=[
            {"symbol": "M_{top}", "description": "End moment at top", "value": M_top_kipft, "units": "kip-ft", "source": f"input:{'M_top_oop_kipft' if prefix=='oop' else 'M_top_ip_kipft'}"},
            {"symbol": "M_{bot}", "description": "End moment at bottom", "value": M_bot_kipft, "units": "kip-ft", "source": f"input:{'M_bot_oop_kipft' if prefix=='oop' else 'M_bot_ip_kipft'}"},
        ],
        compute_fn=_compute_M2,
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "End moment selection for moment magnification"}],
    )

    M2min = compute_step(
        trace,
        id=f"{prefix}_M2min",
        section="Slenderness",
        title=f"Minimum design moment M2,min ({prefix})",
        output_symbol="M_{2,min}",
        output_description="Minimum design moment",
        equation_latex="M_{2,min} = P_u\\left(0.6 + 0.03h\\right)",
        variables=[
            {"symbol": "P_u", "description": "Factored axial load", "value": Pu_kip, "units": "kip", "source": "input:Pu_kip"},
            {"symbol": "h", "description": "Overall thickness/depth about bending axis", "value": h_in, "units": "in", "source": f"step:{prefix}_h"},
        ],
        compute_fn=lambda: (Pu_kip * 1000.0) * (0.6 + 0.03 * h_in) / (1000.0 * 12.0),
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 Eq. (6.6.4.5.4)"}],
    )

    M2_design = compute_step(
        trace,
        id=f"{prefix}_M2design",
        section="Slenderness",
        title=f"Design moment for magnification max(M2,M2,min) ({prefix})",
        output_symbol="M_{2,design}",
        output_description="Moment basis for magnification",
        equation_latex="M_{2,design} = \\max\\left(M_2,\, M_{2,min}\\right)",
        variables=[
            {"symbol": "M_2", "description": "End moment basis", "value": M2, "units": "kip-ft", "source": f"step:{prefix}_M2"},
            {"symbol": "M_{2,min}", "description": "Minimum moment", "value": M2min, "units": "kip-ft", "source": f"step:{prefix}_M2min"},
        ],
        compute_fn=lambda: max(M2, M2min),
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 6.6.4.5.4"}],
    )

    # delta_ns per Eq. (6.6.4.5.2)
    delta_ns = compute_step(
        trace,
        id=f"{prefix}_delta",
        section="Slenderness",
        title=f"Moment magnification factor δns ({prefix})",
        output_symbol="\\delta_{ns}",
        output_description="Nonsway moment magnification factor",
        equation_latex="\\delta_{ns} = \\frac{C_m}{1 - \\frac{P_u}{0.75P_c}} \\ge 1.0",
        variables=[
            {"symbol": "C_m", "description": "Moment gradient factor", "value": Cm, "units": "", "source": f"step:{prefix}_Cm"},
            {"symbol": "P_u", "description": "Factored axial load", "value": Pu_kip, "units": "kip", "source": "input:Pu_kip"},
            {"symbol": "P_c", "description": "Critical buckling load", "value": Pc_kip, "units": "kip", "source": f"step:{prefix}_Pc"},
        ],
        compute_fn=lambda: max(1.0, Cm / (1.0 - (Pu_kip / (0.75 * Pc_kip))) ),
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 Eq. (6.6.4.5.2)"}],
    )

    return Cm, Pc_kip, delta_ns, M2_design


def solve_beam_flexure(tool_id: str, inputs: BeamFlexureInputs) -> Tuple[Dict[str, Any], CalcTrace, Path]:
    inp = inputs.model_dump()
    trace = _make_trace(tool_id, {"module": "beam_flexure", **inp})
    _add_inputs(trace, {"module": "beam_flexure", **inp})

    trace.assumptions.extend([
        TraceAssumption(id="A1", text="Strength design (factored loads) for nonprestressed, singly reinforced rectangular section. Compression reinforcement is ignored."),
        TraceAssumption(id="A2", text="Plane sections remain plane; linear strain distribution; maximum concrete compressive strain at crushing εcu = 0.003."),
        TraceAssumption(id="A3", text="Equivalent rectangular stress block per ACI 318-25 Chapter 22; steel stress taken as min(fy, Es·εs)."),
        TraceAssumption(id="A4", text="Shear, torsion, deflection, crack control, bar spacing/constructability, and seismic detailing are outside this module’s scope."),
    ])

    warnings: List[str] = []
    if inputs.compression_reinf:
        warnings.append("Compression reinforcement option is set true, but this version models singly reinforced sections only. Results may be unconservative for doubly reinforced behavior.")

    # Step: Mu conversion to lb-in
    Mu_lbin = compute_step(
        trace,
        id="Mu_conv",
        section="Loads",
        title="Convert factored moment to consistent units",
        output_symbol="M_u",
        output_description="Factored design moment",
        equation_latex="M_{u,lb\\cdot in} = M_{u,kip\\cdot ft}\\,(1000\\,\\mathrm{lb/kip})\\,(12\\,\\mathrm{in/ft})",
        variables=[
            {"symbol": "M_{u,kip\\cdot ft}", "description": "Factored moment input", "value": inputs.Mu_kipft, "units": "kip-ft", "source": "input:Mu_kipft"},
        ],
        compute_fn=lambda: inputs.Mu_kipft * 1000.0 * 12.0,
        units="lb-in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "Unit conversion (kip-ft to lb-in)"}],
    )

    # Step: effective depth d
    d_in = compute_step(
        trace,
        id="d",
        section="Section geometry",
        title="Effective depth to tension steel centroid",
        output_symbol="d",
        output_description="Effective depth",
        equation_latex="d = h - c_{cover} - d_{st} - \\frac{d_b}{2}",
        variables=[
            {"symbol": "h", "description": "Overall depth", "value": inputs.h_in, "units": "in", "source": "input:h_in"},
            {"symbol": "c_{cover}", "description": "Cover to stirrup outside", "value": inputs.cover_in, "units": "in", "source": "input:cover_in"},
            {"symbol": "d_{st}", "description": "Stirrup diameter", "value": inputs.stirrup_dia_in, "units": "in", "source": "input:stirrup_dia_in"},
            {"symbol": "d_b", "description": "Main bar diameter", "value": inputs.bar_dia_in, "units": "in", "source": "input:bar_dia_in"},
        ],
        compute_fn=lambda: inputs.h_in - inputs.cover_in - inputs.stirrup_dia_in - inputs.bar_dia_in / 2.0,
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Geometry (effective depth to bar centroid)"}],
    )

    # Step: epsilon_ty
    eps_ty = compute_step(
        trace,
        id="eps_ty",
        section="Materials",
        title="Steel yield strain",
        output_symbol="\\varepsilon_{ty}",
        output_description="Yield strain of reinforcement",
        equation_latex="\\varepsilon_{ty} = \\frac{f_y}{E_s}",
        variables=[
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
        ],
        compute_fn=lambda: inputs.fy_psi / inputs.Es_psi,
        units="",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "Yield strain definition εy = fy/Es (Es per ACI 318-25 20.2.2.2)"}],
    )

    beta1 = _beta1(trace, inputs.fc_psi)
    # Multi-layer reinforcement capacity check (tension + compression layers)
    if inputs.reinf_layers:
        # Update assumption to reflect multi-layer capability
        trace.assumptions = [a for a in trace.assumptions if a.id != "A1" and a.id != "A4"] + [
            TraceAssumption(id="A1", text="Strength design check for nonprestressed rectangular section with user-defined reinforcement layers (tension and compression permitted)."),
            TraceAssumption(id="A4", text="This module checks flexure strength only. Shear, torsion, deflection, crack control, bar spacing/constructability, and seismic detailing are outside this module’s scope."),
        ]

        layers = [ly.model_dump() for ly in inputs.reinf_layers]

        # Layer geometry: compute y_i from top and layer area
        layer_As = []
        layer_y = []
        for j, ly in enumerate(layers, start=1):
            As_j = compute_step(
                trace,
                id=f"L{j}_As",
                section="Reinforcement",
                title=f"Layer {j} reinforcement area",
                output_symbol=f"A_{{s,{j}}}",
                output_description=f"Reinforcement area of layer {j}",
                equation_latex=("A_{s,i} = A_{s,override}\;\\text{(if provided)}\;\\text{else}\; n\,(\\pi d_b^2/4)"),
                variables=[
                    {"symbol": "A_{s,override}", "description": "Override layer area (Option 2)", "value": (ly.get("As_override_in2") if ly.get("As_override_in2") is not None else 0.0), "units": "in^2", "source": "input:reinf_layers"},
                    {"symbol": "n", "description": "Number of bars (Option 1)", "value": float(ly.get("n_bars") or 0.0), "units": "", "source": "input:reinf_layers"},
                    {"symbol": "d_b", "description": "Bar diameter (Option 1)", "value": float(ly.get("bar_dia_in") or 0.0), "units": "in", "source": "input:reinf_layers"},
                ],
                compute_fn=lambda ly=ly: _reinf_layer_area_in2(ly),
                units="in^2",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
                references=[{"type": "derived", "ref": "Layer area definition (Option 1 or override Option 2)"}],
            )
            y_j = compute_step(
                trace,
                id=f"L{j}_y",
                section="Reinforcement",
                title=f"Layer {j} centroid location from top",
                output_symbol=f"y_{{{j}}}",
                output_description=f"Distance from top fiber to centroid of layer {j}",
                equation_latex="y_i = \\begin{cases}\\mathrm{offset} & (\\text{top})\\ h-\\mathrm{offset} & (\\text{bottom})\\end{cases}",
                variables=[
                    {"symbol": "h", "description": "Overall depth", "value": inputs.h_in, "units": "in", "source": "input:h_in"},
                    {"symbol": "offset", "description": "Offset from selected face to bar centroid", "value": float(ly.get("offset_in") or 0.0), "units": "in", "source": "input:reinf_layers"},
                ],
                compute_fn=lambda ly=ly: _reinf_layer_y_from_top_in(ly, inputs.h_in),
                units="in",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
                references=[{"type": "derived", "ref": "Layer coordinate transform"}],
            )
            layer_As.append(As_j)
            layer_y.append(y_j)

        As_total = compute_step(
            trace,
            id="As_total",
            section="Reinforcement",
            title="Total reinforcement area",
            output_symbol="A_s",
            output_description="Total reinforcement area (sum of layers)",
            equation_latex="A_s = \\sum_i A_{s,i}",
            variables=[{ "symbol": f"A_{{s,{j}}}", "description": f"Layer {j} area", "value": layer_As[j-1], "units": "in^2", "source": f"step:L{j}_As" } for j in range(1, len(layer_As)+1)],
            compute_fn=lambda: float(sum(layer_As)),
            units="in^2",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Sum of layer areas"}],
        )

        d_eff = compute_step(
            trace,
            id="d_eff",
            section="Section geometry",
            title="Effective depth to extreme tension reinforcement",
            output_symbol="d",
            output_description="Effective depth to extreme tension reinforcement (max y from top)",
            equation_latex="d = \\max(y_i)",
            variables=[{ "symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y" } for j in range(1, len(layer_y)+1)],
            compute_fn=lambda: float(max(layer_y)),
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Extreme tension layer governs d"}],
        )

        eps_ty_local = eps_ty  # computed above

        # Solve for neutral axis depth c from equilibrium (Pn=0)
        c_eq = compute_step(
            trace,
            id="c_eq",
            section="Flexure check",
            title="Neutral axis depth from force equilibrium (multi-layer)",
            output_symbol="c",
            output_description="Neutral axis depth from top (solved)",
            equation_latex="0 = C_c + \\sum_i A_{s,i} f_{s,i};\; C_c=0.85 f'_c b a,\; a=\\beta_1 c,\; f_{s,i}=\\mathrm{clamp}(E_s\\varepsilon_{s,i},-f_y,f_y),\;\\varepsilon_{s,i}=0.003(c-y_i)/c",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
                {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
            ] + [{"symbol": f"A_{{s,{j}}}", "description": f"Layer {j} area", "value": layer_As[j-1], "units": "in^2", "source": f"step:L{j}_As"} for j in range(1,len(layer_As)+1)]
              + [{"symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y"} for j in range(1,len(layer_y)+1)],
            compute_fn=lambda: _solve_c_equilibrium_multilayer(
                fc_psi=inputs.fc_psi,
                b_in=inputs.b_in,
                h_in=inputs.h_in,
                beta1=beta1,
                fy_psi=inputs.fy_psi,
                Es_psi=inputs.Es_psi,
                layers_As_y=[(float(layer_As[j-1]), float(layer_y[j-1])) for j in range(1, len(layer_As)+1)],
            ),
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Force equilibrium solved by bisection (_solve_c_equilibrium_multilayer)"}],
            warnings=(["Equilibrium root not bracketed; check reinforcement layout or load sign conventions."] if math.isnan(_solve_c_equilibrium_multilayer(fc_psi=inputs.fc_psi,b_in=inputs.b_in,h_in=inputs.h_in,beta1=beta1,fy_psi=inputs.fy_psi,Es_psi=inputs.Es_psi,layers_As_y=[(float(layer_As[j-1]), float(layer_y[j-1])) for j in range(1, len(layer_As)+1)])) else None),
        )

        a_eq = compute_step(
            trace,
            id="a_eq",
            section="Flexure check",
            title="Compression block depth a",
            output_symbol="a",
            output_description="Equivalent rectangular stress block depth",
            equation_latex="a = \\beta_1 c",
            variables=[
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_eq, "units": "in", "source": "step:c_eq"},
            ],
            compute_fn=lambda: beta1 * c_eq,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1 and 22.2.2.4.3"}],
        )

        # Layer strains, stresses, and forces at equilibrium
        eps_layers = []
        fs_layers = []
        F_layers = []
        for j in range(1, len(layer_As)+1):
            eps_sj = compute_step(
                trace,
                id=f"L{j}_eps",
                section="Flexure check",
                title=f"Layer {j} steel strain",
                output_symbol=f"\\varepsilon_{{s,{j}}}",
                output_description=f"Steel strain in layer {j} at nominal strength",
                equation_latex="\\varepsilon_{s,i} = 0.003\\left(\\frac{c-y_i}{c}\\right)",
                variables=[
                    {"symbol": "c", "description": "Neutral axis depth", "value": c_eq, "units": "in", "source": "step:c_eq"},
                    {"symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y"},
                ],
                compute_fn=lambda y=float(layer_y[j-1]): 0.003 * ((c_eq - y)/c_eq),
                units="",
                rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 5},
                references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1 (linear strain; εcu = 0.003)"}],
            )
            fs_sj = compute_step(
                trace,
                id=f"L{j}_fs",
                section="Flexure check",
                title=f"Layer {j} steel stress",
                output_symbol=f"f_{{s,{j}}}",
                output_description=f"Steel stress in layer {j} (limited to ±fy)",
                equation_latex="f_{s,i} = \\max(-f_y,\\min(f_y, E_s\\varepsilon_{s,i}))",
                variables=[
                    {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
                    {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
                    {"symbol": f"\\varepsilon_{{s,{j}}}", "description": f"Layer {j} strain", "value": eps_sj, "units": "", "source": f"step:L{j}_eps"},
                ],
                compute_fn=lambda eps=float(eps_sj): max(-inputs.fy_psi, min(inputs.fy_psi, inputs.Es_psi * eps)),
                units="psi",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
                references=[{"type": "code", "ref": "ACI 318-25 22.2.2.3 (steel stress-strain relationship)"}],
            )
            F_sj = compute_step(
                trace,
                id=f"L{j}_F",
                section="Flexure check",
                title=f"Layer {j} steel force",
                output_symbol=f"F_{{s,{j}}}",
                output_description=f"Steel force in layer {j} (compression +)",
                equation_latex="F_{s,i} = A_{s,i} f_{s,i}",
                variables=[
                    {"symbol": f"A_{{s,{j}}}", "description": f"Layer {j} area", "value": layer_As[j-1], "units": "in^2", "source": f"step:L{j}_As"},
                    {"symbol": f"f_{{s,{j}}}", "description": f"Layer {j} stress", "value": fs_sj, "units": "psi", "source": f"step:L{j}_fs"},
                ],
                compute_fn=lambda As=float(layer_As[j-1]), fs=float(fs_sj): As * fs,
                units="lb",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
                references=[{"type": "derived", "ref": "Force = stress × area"}],
            )
            eps_layers.append(eps_sj)
            fs_layers.append(fs_sj)
            F_layers.append(F_sj)

        Cc_eq = compute_step(
            trace,
            id="Cc_eq",
            section="Flexure check",
            title="Concrete compression resultant",
            output_symbol="C_c",
            output_description="Concrete compression force (resultant)",
            equation_latex="C_c = 0.85 f'_c b a",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
                {"symbol": "a", "description": "Stress block depth", "value": a_eq, "units": "in", "source": "step:a_eq"},
            ],
            compute_fn=lambda: 0.85 * inputs.fc_psi * inputs.b_in * a_eq,
            units="lb",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1"}],
        )

        Pn_check = compute_step(
            trace,
            id="Pn_check",
            section="Flexure check",
            title="Check force equilibrium (Pn≈0)",
            output_symbol="P_n",
            output_description="Resultant axial force (should be ~0 for pure bending)",
            equation_latex="P_n = C_c + \\sum_i F_{s,i}",
            variables=[
                {"symbol": "C_c", "description": "Concrete compression resultant", "value": Cc_eq, "units": "lb", "source": "step:Cc_eq"},
            ] + [{"symbol": f"F_{{s,{j}}}", "description": f"Layer {j} force", "value": F_layers[j-1], "units": "lb", "source": f"step:L{j}_F"} for j in range(1, len(F_layers)+1)],
            compute_fn=lambda: float(Cc_eq + sum(F_layers)),
            units="lb",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "derived", "ref": "Equilibrium check"}],
        )

        Mn_lbin = compute_step(
            trace,
            id="Mn_eq",
            section="Flexure check",
            title="Nominal moment strength from internal forces",
            output_symbol="M_n",
            output_description="Nominal moment strength about top fiber",
            equation_latex="M_n = |C_c(a/2) + \\sum_i F_{s,i} y_i|",
            variables=[
                {"symbol": "C_c", "description": "Concrete compression resultant", "value": Cc_eq, "units": "lb", "source": "step:Cc_eq"},
                {"symbol": "a", "description": "Stress block depth", "value": a_eq, "units": "in", "source": "step:a_eq"},
            ] + [
                {"symbol": f"F_{{s,{j}}}", "description": f"Layer {j} force", "value": F_layers[j-1], "units": "lb", "source": f"step:L{j}_F"}
                for j in range(1,len(F_layers)+1)
            ] + [
                {"symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y"}
                for j in range(1,len(layer_y)+1)
            ],
            compute_fn=lambda: abs(float(Cc_eq*(a_eq/2.0) + sum(float(F_layers[j-1])*float(layer_y[j-1]) for j in range(1,len(F_layers)+1)))),
            units="lb-in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "derived", "ref": "Internal moment equilibrium about top fiber"}],
        )

        Mn_kipft = compute_step(
            trace,
            id="Mn_kipft",
            section="Flexure check",
            title="Convert Mn to kip-ft",
            output_symbol="M_n",
            output_description="Nominal moment strength",
            equation_latex="M_{n,kip\cdot ft} = M_{n,lb\cdot in}/(12\,1000)",
            variables=[{"symbol": "M_{n,lb\cdot in}", "description": "Nominal moment", "value": Mn_lbin, "units": "lb-in", "source": "step:Mn_eq"}],
            compute_fn=lambda: float(Mn_lbin)/(12.0*1000.0),
            units="kip-ft",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "derived", "ref": "Unit conversion"}],
        )

        eps_t_eff = compute_step(
            trace,
            id="eps_t_eff",
            section="Flexure check",
            title="Net tensile strain at extreme tension layer",
            output_symbol="\\varepsilon_t",
            output_description="Net tensile strain at extreme tension layer",
            equation_latex="\\varepsilon_t = 0.003\\left(\\frac{d-c}{c}\\right)",
            variables=[
                {"symbol": "d", "description": "Effective depth", "value": d_eff, "units": "in", "source": "step:d_eff"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_eq, "units": "in", "source": "step:c_eq"},
            ],
            compute_fn=lambda: 0.003 * ((d_eff - c_eq)/c_eq),
            units="",
            rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 5},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1"}],
        )

        phi_cap = compute_step(
            trace,
            id="phi_cap",
            section="Strength reduction factor",
            title="Compute φ from εt (multi-layer)",
            output_symbol="\phi",
            output_description="Strength reduction factor per Table 21.2.2",
            equation_latex=(
                "\phi = \\begin{cases}"
                + ("0.75" if inputs.transverse_type=="spiral" else "0.65") + " & \\text{if } \\varepsilon_t \le \\varepsilon_{ty} \\n"
                + "0.90 & \\text{if } \\varepsilon_t \ge \\varepsilon_{ty}+0.003 \\n"
                + ("0.75 + 0.15(\\varepsilon_t-\\varepsilon_{ty})/0.003" if inputs.transverse_type=="spiral" else "0.65 + 0.25(\\varepsilon_t-\\varepsilon_{ty})/0.003")
                + " & \\text{otherwise}\\end{cases}"
            ),
            variables=[
                {"symbol": "\\varepsilon_t", "description": "Net tensile strain", "value": eps_t_eff, "units": "", "source": "step:eps_t_eff"},
                {"symbol": "\\varepsilon_{ty}", "description": "Yield strain", "value": eps_ty_local, "units": "", "source": "step:eps_ty"},
            ],
            compute_fn=lambda: _phi_flexure_from_eps_t(float(eps_t_eff), float(eps_ty_local), inputs.transverse_type),
            units="",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2"}],
        )

        phiMn_lbin = compute_step(
            trace,
            id="phiMn_eq",
            section="Flexure check",
            title="Design flexural strength",
            output_symbol="\phi M_n",
            output_description="Design moment strength",
            equation_latex="\phi M_n = \phi M_n",
            variables=[
                {"symbol": "\phi", "description": "Strength reduction factor", "value": phi_cap, "units": "", "source": "step:phi_cap"},
                {"symbol": "M_n", "description": "Nominal moment strength", "value": Mn_lbin, "units": "lb-in", "source": "step:Mn_eq"},
            ],
            compute_fn=lambda: float(phi_cap)*float(Mn_lbin),
            units="lb-in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "code", "ref": "ACI 318-25 21.2.1"}],
            checks_builder=lambda val: [{
                "label": "Flexure strength (multi-layer)",
                "demand": float(Mu_lbin),
                "capacity": float(val),
                "ratio": float(Mu_lbin/val) if val else float('inf'),
                "pass_fail": "PASS" if val >= Mu_lbin else "FAIL",
            }],
        )

        phiMn_kipft = compute_step(
            trace,
            id="phiMn_kipft",
            section="Flexure check",
            title="Convert φMn to kip-ft",
            output_symbol="\phi M_n",
            output_description="Design moment strength",
            equation_latex="(\phi M_n)_{kip\cdot ft} = (\phi M_n)_{lb\cdot in}/(12\,1000)",
            variables=[{"symbol": "\phi M_n", "description": "Design strength", "value": phiMn_lbin, "units": "lb-in", "source": "step:phiMn_eq"}],
            compute_fn=lambda: float(phiMn_lbin)/(12.0*1000.0),
            units="kip-ft",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "derived", "ref": "Unit conversion"}],
        )

        ok = float(phiMn_lbin) >= float(Mu_lbin)

        results = {
            "ok": bool(ok),
            "module": "beam_flexure",
            "inputs": inp,
            "outputs": {
                "beta1": beta1,
                "phi": float(phi_cap),
                "eps_t": float(eps_t_eff),
                "c_in": float(c_eq),
                "a_in": float(a_eq),
                "As_total_in2": float(As_total),
                "phiMn_kipft": float(phiMn_kipft),
                "Mu_kipft": float(inputs.Mu_kipft),
                "utilization": float(inputs.Mu_kipft)/float(phiMn_kipft) if float(phiMn_kipft) else float('inf'),
                "layers": layers,
            },
            "summary_text": (
                f"Beam flexure (rectangular, multi-layer reinforcement)\n"
                f"φ = {float(phi_cap):.3f}\n"
                f"As,total = {float(As_total):.4f} in^2\n"
                f"Design strength φMn = {float(phiMn_kipft):.2f} kip-ft; Demand Mu = {inputs.Mu_kipft:.2f} kip-ft\n"
                f"Status: {'PASS' if ok else 'FAIL'}\n"
            ),
            "warnings": warnings,
        }

        trace.summary.key_outputs = {
            "phi": {"value": float(phi_cap), "units": ""},
            "As_total": {"value": float(As_total), "units": "in^2"},
            "phiMn": {"value": float(phiMn_kipft), "units": "kip-ft"},
            "Mu": {"value": float(inputs.Mu_kipft), "units": "kip-ft"},
        }
        trace.summary.warnings = warnings
        trace.summary.governing_checks = [{"label": "Flexure strength (multi-layer)", "ratio": float(Mu_lbin/phiMn_lbin) if float(phiMn_lbin) else float('inf'), "status": "PASS" if ok else "FAIL"}]
        trace.summary.controlling_step_ids = ["phiMn_eq"]

        run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
        logger = _setup_logger(run_dir)
        logger.info("Solve beam_flexure (multi-layer) inputs=%s", json.dumps(inp))
        if warnings:
            for w in warnings:
                logger.warning(w)
        export_all(trace, results, run_dir)
        return results, trace, run_dir


    # Iterative phi update (explicit each iteration)
    phi = 0.90
    c_in = None
    a_in = None
    eps_t = None

    for i in range(1, 4):
        phi_i = compute_step(
            trace,
            id=f"phi_assumed_{i}",
            section="Strength reduction factor",
            title=f"Assumed φ for iteration {i}",
            output_symbol="\\phi",
            output_description="Assumed strength reduction factor",
            equation_latex="\\phi = \\phi_{assumed}",
            variables=[{"symbol": "\\phi_{assumed}", "description": "Assumed φ for iteration", "value": phi, "units": "", "source": "assumption:A_phi_start" if i == 1 else f"step:phi_calc_{i-1}"}],
            compute_fn=lambda phi=phi: phi,
            units="",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2 (φ depends on εt; iterated)"}],
        )

        Mn_req = compute_step(
            trace,
            id=f"Mn_req_{i}",
            section="Flexure design",
            title=f"Required nominal moment for iteration {i}",
            output_symbol="M_n",
            output_description="Required nominal moment strength",
            equation_latex="M_n = \\frac{M_u}{\\phi}",
            variables=[
                {"symbol": "M_u", "description": "Factored design moment (lb-in)", "value": Mu_lbin, "units": "lb-in", "source": "step:Mu_conv"},
                {"symbol": "\\phi", "description": "Strength reduction factor", "value": phi_i, "units": "", "source": f"step:phi_assumed_{i}"},
            ],
            compute_fn=lambda Mu=Mu_lbin, phi_i=phi_i: Mu / phi_i,
            units="lb-in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "code", "ref": "ACI 318-25 21.2.1 (φMn ≥ Mu)"}],
        )

        # Solve for c from Mn = 0.85 f'c b a (d - a/2), with a=beta1 c
        def _compute_c():
            fc = inputs.fc_psi
            b = inputs.b_in
            d = d_in
            b1 = beta1
            K1 = 0.85 * fc * b * b1  # lb/in
            A = 0.85 * fc * b * (b1 ** 2) / 2.0  # lb/in
            B = -0.85 * fc * b * b1 * d  # lb
            C = Mn_req  # lb-in
            disc = B * B - 4.0 * A * C
            if disc < 0:
                # No real solution -> demand too high for this section under assumptions
                return float("nan")
            c_small = (-B - math.sqrt(disc)) / (2.0 * A)
            return c_small

        c_in = compute_step(
            trace,
            id=f"c_{i}",
            section="Flexure design",
            title=f"Neutral axis depth from required Mn (iteration {i})",
            output_symbol="c",
            output_description="Neutral axis depth",
            equation_latex="c = \\frac{-B - \\sqrt{B^2 - 4AC}}{2A},\\quad A=\\frac{0.85 f'_c b \\beta_1^2}{2},\\; B=-0.85 f'_c b \\beta_1 d,\\; C=M_n",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
                {"symbol": "M_n", "description": "Required nominal moment", "value": Mn_req, "units": "lb-in", "source": f"step:Mn_req_{i}"},
            ],
            compute_fn=_compute_c,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Derived from ACI 318-25 22.2.2 assumptions (equilibrium + stress block)"}],
            warnings=(["No real solution for c (B^2 - 4AC < 0). Demand exceeds this section’s singly reinforced capacity under assumptions."] if math.isnan(_compute_c()) else None),
        )

        a_in = compute_step(
            trace,
            id=f"a_{i}",
            section="Flexure design",
            title=f"Compression block depth a (iteration {i})",
            output_symbol="a",
            output_description="Equivalent stress block depth",
            equation_latex="a = \\beta_1 c",
            variables=[
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_in, "units": "in", "source": f"step:c_{i}"},
            ],
            compute_fn=lambda: beta1 * c_in,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1 and 22.2.2.4.3"}],
        )

        eps_t = compute_step(
            trace,
            id=f"eps_t_{i}",
            section="Flexure design",
            title=f"Net tensile strain εt at nominal strength (iteration {i})",
            output_symbol="\\varepsilon_t",
            output_description="Net tensile strain at tension reinforcement",
            equation_latex="\\varepsilon_t = 0.003\\left(\\frac{d-c}{c}\\right)",
            variables=[
                {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_in, "units": "in", "source": f"step:c_{i}"},
            ],
            compute_fn=lambda: 0.003 * ((d_in - c_in) / c_in),
            units="",
            rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1 (linear strain; εcu = 0.003)"}],
        )

        # phi from Table 21.2.2 — record eps_t and eps_ty through dummy step ids for source tracking
        # (compute_step requires the variables' sources; we already computed eps_t and eps_ty in earlier steps)
        # We'll store these values in variables in phi step with artificial source step ids.
        phi_cc_val = 0.75 if inputs.transverse_type == "spiral" else 0.65
        transition_expr = "0.75 + 0.15(\\varepsilon_t-\\varepsilon_{ty})/0.003" if inputs.transverse_type == "spiral" else "0.65 + 0.25(\\varepsilon_t-\\varepsilon_{ty})/0.003"

        phi_new = compute_step(
            trace,
            id=f"phi_calc_{i}",
            section="Strength reduction factor",
            title=f"Compute φ from εt (iteration {i})",
            output_symbol="\\phi",
            output_description="Strength reduction factor per Table 21.2.2",
            equation_latex=(
                "\\phi = \\begin{cases}"
                + f"{phi_cc_val} & \\text{{if }} \\varepsilon_t \\le \\varepsilon_{{ty}} \\\\"
                + "0.90 & \\text{if } \\varepsilon_t \\ge \\varepsilon_{ty}+0.003 \\\\"
                + f"{transition_expr} & \\text{{otherwise}}"
                + "\\end{cases}"
            ),
            variables=[
                {"symbol": "\\varepsilon_t", "description": "Net tensile strain at nominal strength", "value": eps_t, "units": "", "source": f"step:eps_t_{i}"},
                {"symbol": "\\varepsilon_{ty}", "description": "Yield strain of reinforcement", "value": eps_ty, "units": "", "source": "step:eps_ty"},
            ],
            compute_fn=lambda eps_t=eps_t, eps_ty=eps_ty: (
                (0.75 if inputs.transverse_type == "spiral" else 0.65) if eps_t <= eps_ty
                else 0.90 if eps_t >= eps_ty + 0.003
                else (0.75 + 0.15*(eps_t - eps_ty)/0.003) if inputs.transverse_type == "spiral"
                else (0.65 + 0.25*(eps_t - eps_ty)/0.003)
            ),
            units="",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2"}],
        )

        if abs(phi_new - phi) < 0.001:
            phi = phi_new
            break
        phi = phi_new

    # Compression force Cc and steel stress fs
    Cc_lb = compute_step(
        trace,
        id="Cc",
        section="Flexure design",
        title="Concrete compression resultant",
        output_symbol="C_c",
        output_description="Concrete compression force (resultant)",
        equation_latex="C_c = 0.85 f'_c\\, b\\, a",
        variables=[
            {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
            {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
            {"symbol": "a", "description": "Stress block depth", "value": a_in, "units": "in", "source": f"step:a_{i}"},
        ],
        compute_fn=lambda: 0.85 * inputs.fc_psi * inputs.b_in * a_in,
        units="lb",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1"}],
    )

    fs_psi = compute_step(
        trace,
        id="fs",
        section="Flexure design",
        title="Tension steel stress at nominal strength",
        output_symbol="f_s",
        output_description="Steel stress (limited to fy)",
        equation_latex="f_s = \\min(f_y, E_s\\varepsilon_t)",
        variables=[
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
            {"symbol": "\\varepsilon_t", "description": "Net tensile strain", "value": eps_t, "units": "", "source": f"step:eps_t_{i}"},
        ],
        compute_fn=lambda: min(inputs.fy_psi, inputs.Es_psi * eps_t),
        units="psi",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.3 (steel stress-strain relationship)"}],
    )

    As_req_in2 = compute_step(
        trace,
        id="As_req",
        section="Flexure design",
        title="Required tension reinforcement area",
        output_symbol="A_s",
        output_description="Required tension reinforcement area (analysis)",
        equation_latex="A_s = \\frac{C_c}{f_s}",
        variables=[
            {"symbol": "C_c", "description": "Concrete compression force", "value": Cc_lb, "units": "lb", "source": "step:Cc"},
            {"symbol": "f_s", "description": "Steel stress", "value": fs_psi, "units": "psi", "source": "step:fs"},
        ],
        compute_fn=lambda: Cc_lb / fs_psi,
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "Equilibrium: T = As·fs = Cc (ACI 318-25 22.2.2)"}],
    )

    # Minimum flexural reinforcement As,min per 9.6.1.2
    sqrt_fc = compute_step(
        trace,
        id="sqrt_fc",
        section="Flexure detailing",
        title="Concrete strength square-root for minimum reinforcement",
        output_symbol="\\sqrt{f'_c}",
        output_description="Square root of f'c",
        equation_latex="\\sqrt{f'_c} = \\sqrt{f'_c}",
        variables=[{"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"}],
        compute_fn=lambda: math.sqrt(inputs.fc_psi),
        units="psi^0.5",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Math"}],
    )

    As_min_a = compute_step(
        trace,
        id="As_min_a",
        section="Flexure detailing",
        title="Minimum reinforcement expression (a)",
        output_symbol="A_{s,min(a)}",
        output_description="Minimum As per 9.6.1.2(a)",
        equation_latex="A_{s,min(a)} = \\frac{3\\sqrt{f'_c}}{f_y} b_w d",
        variables=[
            {"symbol": "\\sqrt{f'_c}", "description": "Square root of concrete strength", "value": sqrt_fc, "units": "psi^0.5", "source": "step:sqrt_fc"},
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "b_w", "description": "Web width (taken as b for rectangular beam)", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
            {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
        ],
        compute_fn=lambda: (3.0 * sqrt_fc / inputs.fy_psi) * inputs.b_in * d_in,
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 9.6.1.2(a)"}],
    )

    As_min_b = compute_step(
        trace,
        id="As_min_b",
        section="Flexure detailing",
        title="Minimum reinforcement expression (b)",
        output_symbol="A_{s,min(b)}",
        output_description="Minimum As per 9.6.1.2(b)",
        equation_latex="A_{s,min(b)} = \\frac{200}{f_y} b_w d",
        variables=[
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "b_w", "description": "Web width (taken as b for rectangular beam)", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
            {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
        ],
        compute_fn=lambda: (200.0 / inputs.fy_psi) * inputs.b_in * d_in,
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 9.6.1.2(b)"}],
    )

    As_min = compute_step(
        trace,
        id="As_min",
        section="Flexure detailing",
        title="Minimum tension reinforcement",
        output_symbol="A_{s,min}",
        output_description="Minimum tension reinforcement area",
        equation_latex="A_{s,min} = \\max\\left(A_{s,min(a)}, A_{s,min(b)}\\right)",
        variables=[
            {"symbol": "A_{s,min(a)}", "description": "Minimum As expression (a)", "value": As_min_a, "units": "in^2", "source": "step:As_min_a"},
            {"symbol": "A_{s,min(b)}", "description": "Minimum As expression (b)", "value": As_min_b, "units": "in^2", "source": "step:As_min_b"},
        ],
        compute_fn=lambda: max(As_min_a, As_min_b),
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 9.6.1.2"}],
    )

    As_prov = compute_step(
        trace,
        id="As_prov",
        section="Flexure design",
        title="Required provided As (governing by analysis vs minimum)",
        output_symbol="A_{s,prov}",
        output_description="Required provided tension reinforcement area",
        equation_latex="A_{s,prov} = \\max\\left(A_{s,req}, A_{s,min}\\right)",
        variables=[
            {"symbol": "A_{s,req}", "description": "Required As by analysis", "value": As_req_in2, "units": "in^2", "source": "step:As_req"},
            {"symbol": "A_{s,min}", "description": "Minimum As", "value": As_min, "units": "in^2", "source": "step:As_min"},
        ],
        compute_fn=lambda: max(As_req_in2, As_min),
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 9.6.1.2 and strength design"}],
    )

    # Suggest number of bars of selected diameter (area estimated using db^2*pi/4 if not standard)
    Ab_est = math.pi * (inputs.bar_dia_in ** 2) / 4.0
    n_bars = max(1, math.ceil(As_prov / Ab_est))

    # Capacity check using derived Mn with As_prov and assuming yielding (common for flexure)
    # c = As*fs / (0.85 f'c b beta1). Use fs = min(fy, Es*εt(c)) -> solve with yielding check.
    # For practical use and determinism, compute capacity using yielding assumption and verify fs >= fy.
    c_y = compute_step(
        trace,
        id="c_y",
        section="Flexure check",
        title="Neutral axis depth assuming tension steel yields",
        output_symbol="c_y",
        output_description="Neutral axis depth (yielding assumption)",
        equation_latex="c_y = \\frac{A_{s,prov} f_y}{0.85 f'_c b \\beta_1}",
        variables=[
            {"symbol": "A_{s,prov}", "description": "Provided tension steel area", "value": As_prov, "units": "in^2", "source": "step:As_prov"},
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
            {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
            {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
        ],
        compute_fn=lambda: (As_prov * inputs.fy_psi) / (0.85 * inputs.fc_psi * inputs.b_in * beta1),
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "Equilibrium with fs=fy (ACI 318-25 22.2.2)"}],
    )

    eps_t_y = compute_step(
        trace,
        id="eps_t_y",
        section="Flexure check",
        title="Net tensile strain using c_y",
        output_symbol="\\varepsilon_{t,y}",
        output_description="Net tensile strain (yielding assumption)",
        equation_latex="\\varepsilon_{t,y} = 0.003\\left(\\frac{d-c_y}{c_y}\\right)",
        variables=[
            {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
            {"symbol": "c_y", "description": "Neutral axis depth (yield assumption)", "value": c_y, "units": "in", "source": "step:c_y"},
        ],
        compute_fn=lambda: 0.003 * ((d_in - c_y) / c_y),
        units="",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1 (linear strain; εcu = 0.003)"}],
    )

    fs_elastic = compute_step(
        trace,
        id="fs_elastic",
        section="Flexure check",
        title="Elastic steel stress based on εt",
        output_symbol="f_{s,elastic}",
        output_description="Elastic steel stress (Es·εt)",
        equation_latex="f_{s,elastic} = E_s\\varepsilon_{t,y}",
        variables=[
            {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
            {"symbol": "\\varepsilon_{t,y}", "description": "Net tensile strain", "value": eps_t_y, "units": "", "source": "step:eps_t_y"},
        ],
        compute_fn=lambda: inputs.Es_psi * eps_t_y,
        units="psi",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "Hooke's law (elastic range)"}],
    )

    yield_ratio = compute_step(
        trace,
        id="yield_ratio",
        section="Flexure check",
        title="Check if tension steel yields",
        output_symbol="r_y",
        output_description="Yield ratio (fs_elastic / fy)",
        equation_latex="r_y = \\frac{f_{s,elastic}}{f_y}",
        variables=[
            {"symbol": "f_{s,elastic}", "description": "Elastic steel stress", "value": fs_elastic, "units": "psi", "source": "step:fs_elastic"},
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
        ],
        compute_fn=lambda: fs_elastic / inputs.fy_psi,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Yield check"}],
    )

    if yield_ratio < 1.0:
        warnings.append("Tension steel may not reach yield with the provided As and geometry (fs_elastic < fy). Capacity check uses elastic steel stress solution (explicit quadratic) for equilibrium.")
        # Solve quadratic equilibrium for c: 0.85 f'c b beta1 c^2 + As Es 0.003 c - As Es 0.003 d = 0
        def _compute_c_elastic():
            A = 0.85 * inputs.fc_psi * inputs.b_in * beta1
            B = As_prov * inputs.Es_psi * 0.003
            C = -As_prov * inputs.Es_psi * 0.003 * d_in
            disc = B*B - 4*A*C
            cpos = (-B + math.sqrt(disc)) / (2*A)
            return cpos

        c_cap = compute_step(
            trace,
            id="c_cap",
            section="Flexure check",
            title="Neutral axis depth from equilibrium (elastic steel case)",
            output_symbol="c",
            output_description="Neutral axis depth satisfying equilibrium (elastic steel)",
            equation_latex="c = \\frac{-B + \\sqrt{B^2 - 4AC}}{2A},\\; A=0.85 f'_c b \\beta_1,\\; B=A_{s,prov}E_s(0.003),\\; C=-A_{s,prov}E_s(0.003)d",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "A_{s,prov}", "description": "Provided steel area", "value": As_prov, "units": "in^2", "source": "step:As_prov"},
                {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
                {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
            ],
            compute_fn=_compute_c_elastic,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Equilibrium with fs=Es·εs (ACI 318-25 22.2.2)"}],
        )
    else:
        c_cap = c_y

    a_cap = compute_step(
        trace,
        id="a_cap",
        section="Flexure check",
        title="Compression block depth for provided As",
        output_symbol="a",
        output_description="Equivalent stress block depth",
        equation_latex="a = \\beta_1 c",
        variables=[
            {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
            {"symbol": "c", "description": "Neutral axis depth", "value": c_cap, "units": "in", "source": "step:c_cap" if yield_ratio < 1.0 else "step:c_y"},
        ],
        compute_fn=lambda: beta1 * c_cap,
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1"}],
    )

    eps_t_cap = compute_step(
        trace,
        id="eps_t_cap",
        section="Flexure check",
        title="Net tensile strain for provided As",
        output_symbol="\\varepsilon_t",
        output_description="Net tensile strain",
        equation_latex="\\varepsilon_t = 0.003\\left(\\frac{d-c}{c}\\right)",
        variables=[
            {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
            {"symbol": "c", "description": "Neutral axis depth", "value": c_cap, "units": "in", "source": "step:c_cap" if yield_ratio < 1.0 else "step:c_y"},
        ],
        compute_fn=lambda: 0.003 * ((d_in - c_cap) / c_cap),
        units="",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1"}],
    )

    phi_cc_val = 0.75 if inputs.transverse_type == "spiral" else 0.65
    transition_expr = "0.75 + 0.15(\\varepsilon_t-\\varepsilon_{ty})/0.003" if inputs.transverse_type == "spiral" else "0.65 + 0.25(\\varepsilon_t-\\varepsilon_{ty})/0.003"

    phi_cap = compute_step(
        trace,
        id="phi_cap",
        section="Strength reduction factor",
        title="Strength reduction factor φ for provided As",
        output_symbol="\\phi",
        output_description="Strength reduction factor",
        equation_latex=(
            "\\phi = \\begin{cases}"
            + f"{phi_cc_val} & \\text{{if }} \\varepsilon_t \\le \\varepsilon_{{ty}} \\\\"
            + "0.90 & \\text{if } \\varepsilon_t \\ge \\varepsilon_{ty}+0.003 \\\\"
            + f"{transition_expr} & \\text{{otherwise}}"
            + "\\end{cases}"
        ),
        variables=[
            {"symbol": "\\varepsilon_t", "description": "Net tensile strain", "value": eps_t_cap, "units": "", "source": "step:eps_t_cap"},
            {"symbol": "\\varepsilon_{ty}", "description": "Yield strain", "value": eps_ty, "units": "", "source": "step:eps_ty"},
        ],
        compute_fn=lambda: (
            (0.75 if inputs.transverse_type == "spiral" else 0.65) if eps_t_cap <= eps_ty
            else 0.90 if eps_t_cap >= eps_ty + 0.003
            else (0.75 + 0.15*(eps_t_cap - eps_ty)/0.003) if inputs.transverse_type == "spiral"
            else (0.65 + 0.25*(eps_t_cap - eps_ty)/0.003)
        ),
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2"}],
    )

    Mn_cap = compute_step(
        trace,
        id="Mn_cap",
        section="Flexure check",
        title="Nominal moment strength for provided As",
        output_symbol="M_n",
        output_description="Nominal moment strength",
        equation_latex="M_n = 0.85 f'_c b a\\left(d - \\frac{a}{2}\\right)",
        variables=[
            {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
            {"symbol": "b", "description": "Section width", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
            {"symbol": "a", "description": "Stress block depth", "value": a_cap, "units": "in", "source": "step:a_cap"},
            {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
        ],
        compute_fn=lambda: 0.85 * inputs.fc_psi * inputs.b_in * a_cap * (d_in - a_cap / 2.0),
        units="lb-in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "Derived from ACI 318-25 22.2.2 stress block and internal equilibrium"}],
    )

    phiMn = compute_step(
        trace,
        id="phiMn",
        section="Flexure check",
        title="Design flexural strength",
        output_symbol="\\phi M_n",
        output_description="Design moment strength",
        equation_latex="\\phi M_n = \\phi\\, M_n",
        variables=[
            {"symbol": "\\phi", "description": "Strength reduction factor", "value": phi_cap, "units": "", "source": "step:phi_cap"},
            {"symbol": "M_n", "description": "Nominal moment strength", "value": Mn_cap, "units": "lb-in", "source": "step:Mn_cap"},
        ],
        compute_fn=lambda: phi_cap * Mn_cap,
        units="lb-in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 21.2.1 (φMn ≥ Mu)"}],
        checks_builder=lambda val: [{
            "label": "Flexure strength",
            "demand": float(Mu_lbin),
            "capacity": float(val),
            "ratio": float(Mu_lbin / val) if val else float("inf"),
            "pass_fail": "PASS" if val >= Mu_lbin else "FAIL",
        }],
    )

    ok = phiMn >= Mu_lbin

    # Summary text
    phiMn_kipft = phiMn / (12.0 * 1000.0)
    results = {
        "ok": bool(ok),
        "module": "beam_flexure",
        "inputs": inp,
        "outputs": {
            "d_in": d_in,
            "beta1": beta1,
            "phi": phi_cap,
            "eps_t": eps_t_cap,
            "As_required_in2": As_req_in2,
            "As_min_in2": As_min,
            "As_provided_required_in2": As_prov,
            "suggested_n_bars": int(n_bars),
            "bar_dia_in": inputs.bar_dia_in,
            "bar_area_est_in2": Ab_est,
            "phiMn_kipft": phiMn_kipft,
        },
        "summary_text": (
            f"Beam flexure (rectangular singly reinforced)\n"
            f"φ = {phi_cap:.3f}\n"
            f"As,req = {As_req_in2:.4f} in^2; As,min = {As_min:.4f} in^2; As,prov(required) = {As_prov:.4f} in^2\n"
            f"Design strength φMn = {phiMn_kipft:.2f} kip-ft; Demand Mu = {inputs.Mu_kipft:.2f} kip-ft\n"
            f"Status: {'PASS' if ok else 'FAIL'}\n"
        ),
        "warnings": warnings,
    }

    trace.summary.key_outputs = {
        "phi": {"value": phi_cap, "units": ""},
        "As_provided_required": {"value": As_prov, "units": "in^2"},
        "phiMn": {"value": phiMn_kipft, "units": "kip-ft"},
        "Mu": {"value": inputs.Mu_kipft, "units": "kip-ft"},
    }
    trace.summary.warnings = warnings
    trace.summary.governing_checks = [{"label": "Flexure strength", "ratio": float(Mu_lbin / phiMn) if phiMn else float("inf"), "status": "PASS" if ok else "FAIL"}]
    trace.summary.controlling_step_ids = ["phiMn"]

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
    logger = _setup_logger(run_dir)
    logger.info("Solve beam_flexure inputs=%s", json.dumps(inp))
    if warnings:
        for w in warnings:
            logger.warning(w)

    export_all(trace, results, run_dir)
    return results, trace, run_dir


def solve_slab_oneway_flexure(tool_id: str, inputs: SlabOnewayFlexureInputs) -> Tuple[Dict[str, Any], CalcTrace, Path]:
    # Treat as 12-in strip beam with b=12 in, no stirrup term; effective depth uses cover and bar_dia/2.
    inp = inputs.model_dump()
    trace = _make_trace(tool_id, {"module": "slab_oneway_flexure", **inp})
    _add_inputs(trace, {"module": "slab_oneway_flexure", **inp})

    trace.assumptions.extend([
        TraceAssumption(id="A1", text="One-way slab designed as a 12-in wide strip; singly reinforced; compression steel ignored."),
        TraceAssumption(id="A2", text="Plane sections remain plane; εcu = 0.003; stress block per Chapter 22."),
        TraceAssumption(id="A3", text="This module does not check minimum thickness, shear, deflection, temperature/shrinkage reinforcement, or bar spacing limits."),
    ])

    Mu_lbin_per_ft = compute_step(
        trace,
        id="Mu_strip",
        section="Loads",
        title="Convert Mu per ft to lb-in for a 12-in strip (per ft width)",
        output_symbol="M_u",
        output_description="Factored moment per foot width",
        equation_latex="M_{u,lb\\cdot in/ft} = M_{u,kip\\cdot ft/ft}\\,(1000)\\,(12)",
        variables=[
            {"symbol": "M_{u,kip\\cdot ft/ft}", "description": "Factored moment per foot width", "value": inputs.Mu_kipft_per_ft, "units": "kip-ft/ft", "source": "input:Mu_kipft_per_ft"},
        ],
        compute_fn=lambda: inputs.Mu_kipft_per_ft * 1000.0 * 12.0,
        units="lb-in/ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "Unit conversion"}],
    )

    b_in = 12.0
    d_in = compute_step(
        trace,
        id="d",
        section="Section geometry",
        title="Effective depth to slab tension steel centroid (12-in strip)",
        output_symbol="d",
        output_description="Effective depth",
        equation_latex="d = h - c_{cover} - \\frac{d_b}{2}",
        variables=[
            {"symbol": "h", "description": "Slab thickness", "value": inputs.thickness_in, "units": "in", "source": "input:thickness_in"},
            {"symbol": "c_{cover}", "description": "Cover to bar", "value": inputs.cover_in, "units": "in", "source": "input:cover_in"},
            {"symbol": "d_b", "description": "Bar diameter", "value": inputs.bar_dia_in, "units": "in", "source": "input:bar_dia_in"},
        ],
        compute_fn=lambda: inputs.thickness_in - inputs.cover_in - inputs.bar_dia_in / 2.0,
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Geometry"}],
    )

    eps_ty = compute_step(
        trace,
        id="eps_ty",
        section="Materials",
        title="Steel yield strain",
        output_symbol="\\varepsilon_{ty}",
        output_description="Yield strain of reinforcement",
        equation_latex="\\varepsilon_{ty} = \\frac{f_y}{E_s}",
        variables=[
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
        ],
        compute_fn=lambda: inputs.fy_psi / inputs.Es_psi,
        units="",
        rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "Yield strain definition εy = fy/Es (Es per ACI 318-25 20.2.2.2)"}],
    )

    beta1 = _beta1(trace, inputs.fc_psi)
    # Multi-layer reinforcement capacity check for 12-in slab strip
    if inputs.reinf_layers:
        layers = [ly.model_dump() for ly in inputs.reinf_layers]
        b_in = 12.0
        h_in = inputs.thickness_in

        layer_As = []
        layer_y = []
        for j, ly in enumerate(layers, start=1):
            As_j = compute_step(
                trace,
                id=f"L{j}_As",
                section="Reinforcement",
                title=f"Layer {j} reinforcement area",
                output_symbol=f"A_{{s,{j}}}",
                output_description=f"Reinforcement area of layer {j}",
                equation_latex=("A_{s,i} = A_{s,override}\;\\text{(if provided)}\;\\text{else}\; n\,(\\pi d_b^2/4)"),
                variables=[
                    {"symbol": "A_{s,override}", "description": "Override layer area (Option 2)", "value": (ly.get("As_override_in2") if ly.get("As_override_in2") is not None else 0.0), "units": "in^2", "source": "input:reinf_layers"},
                    {"symbol": "n", "description": "Number of bars (Option 1)", "value": float(ly.get("n_bars") or 0.0), "units": "", "source": "input:reinf_layers"},
                    {"symbol": "d_b", "description": "Bar diameter (Option 1)", "value": float(ly.get("bar_dia_in") or 0.0), "units": "in", "source": "input:reinf_layers"},
                ],
                compute_fn=lambda ly=ly: _reinf_layer_area_in2(ly),
                units="in^2",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
                references=[{"type": "derived", "ref": "Layer area definition (Option 1 or override Option 2)"}],
            )
            y_j = compute_step(
                trace,
                id=f"L{j}_y",
                section="Reinforcement",
                title=f"Layer {j} centroid location from top",
                output_symbol=f"y_{{{j}}}",
                output_description=f"Distance from top fiber to centroid of layer {j}",
                equation_latex="y_i = \\begin{cases}\\mathrm{offset} & (\\text{top})\\ h-\\mathrm{offset} & (\\text{bottom})\\end{cases}",
                variables=[
                    {"symbol": "h", "description": "Overall depth", "value": h_in, "units": "in", "source": "input:thickness_in"},
                    {"symbol": "offset", "description": "Offset from selected face to bar centroid", "value": float(ly.get("offset_in") or 0.0), "units": "in", "source": "input:reinf_layers"},
                ],
                compute_fn=lambda ly=ly: _reinf_layer_y_from_top_in(ly, h_in),
                units="in",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
                references=[{"type": "derived", "ref": "Layer coordinate transform"}],
            )
            layer_As.append(As_j)
            layer_y.append(y_j)

        As_total = compute_step(
            trace,
            id="As_total",
            section="Reinforcement",
            title="Total reinforcement area",
            output_symbol="A_s",
            output_description="Total reinforcement area (sum of layers)",
            equation_latex="A_s = \\sum_i A_{s,i}",
            variables=[{ "symbol": f"A_{{s,{j}}}", "description": f"Layer {j} area", "value": layer_As[j-1], "units": "in^2", "source": f"step:L{j}_As" } for j in range(1, len(layer_As)+1)],
            compute_fn=lambda: float(sum(layer_As)),
            units="in^2",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Sum of layer areas"}],
        )

        d_eff = compute_step(
            trace,
            id="d_eff",
            section="Section geometry",
            title="Effective depth to extreme tension reinforcement",
            output_symbol="d",
            output_description="Effective depth to extreme tension reinforcement (max y from top)",
            equation_latex="d = \\max(y_i)",
            variables=[{ "symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y" } for j in range(1, len(layer_y)+1)],
            compute_fn=lambda: float(max(layer_y)),
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Extreme tension layer governs d"}],
        )

        eps_ty = compute_step(
            trace,
            id="eps_ty",
            section="Materials",
            title="Steel yield strain",
            output_symbol="\\varepsilon_{ty}",
            output_description="Yield strain of reinforcement",
            equation_latex="\\varepsilon_{ty} = f_y/E_s",
            variables=[
                {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
                {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
            ],
            compute_fn=lambda: inputs.fy_psi/inputs.Es_psi,
            units="",
            rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Yield strain definition εy = fy/Es"}],
        )

        c_eq = compute_step(
            trace,
            id="c_eq",
            section="Flexure check",
            title="Neutral axis depth from force equilibrium (multi-layer)",
            output_symbol="c",
            output_description="Neutral axis depth from top (solved)",
            equation_latex="0 = C_c + \\sum_i A_{s,i} f_{s,i};\; C_c=0.85 f'_c b a,\; a=\\beta_1 c,\; f_{s,i}=\\mathrm{clamp}(E_s\\varepsilon_{s,i},-f_y,f_y),\;\\varepsilon_{s,i}=0.003(c-y_i)/c",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Strip width", "value": b_in, "units": "in", "source": "assumption:strip_width"},
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
                {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
            ] + [{"symbol": f"A_{{s,{j}}}", "description": f"Layer {j} area", "value": layer_As[j-1], "units": "in^2", "source": f"step:L{j}_As"} for j in range(1,len(layer_As)+1)]
              + [{"symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y"} for j in range(1,len(layer_y)+1)],
            compute_fn=lambda: _solve_c_equilibrium_multilayer(
                fc_psi=inputs.fc_psi,
                b_in=b_in,
                h_in=h_in,
                beta1=beta1,
                fy_psi=inputs.fy_psi,
                Es_psi=inputs.Es_psi,
                layers_As_y=[(float(layer_As[j-1]), float(layer_y[j-1])) for j in range(1, len(layer_As)+1)],
            ),
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Force equilibrium solved by bisection (_solve_c_equilibrium_multilayer)"}],
        )

        a_eq = compute_step(
            trace,
            id="a_eq",
            section="Flexure check",
            title="Compression block depth a",
            output_symbol="a",
            output_description="Equivalent rectangular stress block depth",
            equation_latex="a = \\beta_1 c",
            variables=[
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_eq, "units": "in", "source": "step:c_eq"},
            ],
            compute_fn=lambda: beta1 * c_eq,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1 and 22.2.2.4.3"}],
        )

        # Concrete compression resultant
        Cc_eq = compute_step(
            trace,
            id="Cc_eq",
            section="Flexure check",
            title="Concrete compression resultant",
            output_symbol="C_c",
            output_description="Concrete compression force (resultant)",
            equation_latex="C_c = 0.85 f'_c b a",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Strip width", "value": b_in, "units": "in", "source": "assumption:strip_width"},
                {"symbol": "a", "description": "Stress block depth", "value": a_eq, "units": "in", "source": "step:a_eq"},
            ],
            compute_fn=lambda: 0.85 * inputs.fc_psi * b_in * a_eq,
            units="lb",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1"}],
        )

        # Layer forces
        F_layers = []
        for j in range(1, len(layer_As)+1):
            eps_sj = compute_step(
                trace,
                id=f"L{j}_eps",
                section="Flexure check",
                title=f"Layer {j} steel strain",
                output_symbol=f"\\varepsilon_{{s,{j}}}",
                output_description=f"Steel strain in layer {j}",
                equation_latex="\\varepsilon_{s,i} = 0.003\\left(\\frac{c-y_i}{c}\\right)",
                variables=[
                    {"symbol": "c", "description": "Neutral axis depth", "value": c_eq, "units": "in", "source": "step:c_eq"},
                    {"symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y"},
                ],
                compute_fn=lambda y=float(layer_y[j-1]): 0.003 * ((c_eq - y)/c_eq),
                units="",
                rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 5},
                references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1"}],
            )
            fs_sj = compute_step(
                trace,
                id=f"L{j}_fs",
                section="Flexure check",
                title=f"Layer {j} steel stress",
                output_symbol=f"f_{{s,{j}}}",
                output_description=f"Steel stress in layer {j} (±fy)",
                equation_latex="f_{s,i} = \\max(-f_y,\\min(f_y, E_s\\varepsilon_{s,i}))",
                variables=[
                    {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
                    {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
                    {"symbol": f"\\varepsilon_{{s,{j}}}", "description": f"Layer {j} strain", "value": eps_sj, "units": "", "source": f"step:L{j}_eps"},
                ],
                compute_fn=lambda eps=float(eps_sj): max(-inputs.fy_psi, min(inputs.fy_psi, inputs.Es_psi * eps)),
                units="psi",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
                references=[{"type": "code", "ref": "ACI 318-25 22.2.2.3"}],
            )
            F_sj = compute_step(
                trace,
                id=f"L{j}_F",
                section="Flexure check",
                title=f"Layer {j} steel force",
                output_symbol=f"F_{{s,{j}}}",
                output_description=f"Steel force in layer {j}",
                equation_latex="F_{s,i} = A_{s,i} f_{s,i}",
                variables=[
                    {"symbol": f"A_{{s,{j}}}", "description": f"Layer {j} area", "value": layer_As[j-1], "units": "in^2", "source": f"step:L{j}_As"},
                    {"symbol": f"f_{{s,{j}}}", "description": f"Layer {j} stress", "value": fs_sj, "units": "psi", "source": f"step:L{j}_fs"},
                ],
                compute_fn=lambda As=float(layer_As[j-1]), fs=float(fs_sj): As*fs,
                units="lb",
                rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
                references=[{"type": "derived", "ref": "Force = stress × area"}],
            )
            F_layers.append(F_sj)

        Mn_lbin = compute_step(
            trace,
            id="Mn_eq",
            section="Flexure check",
            title="Nominal moment strength",
            output_symbol="M_n",
            output_description="Nominal moment strength about top fiber",
            equation_latex="M_n = |C_c(a/2) + \\sum_i F_{s,i} y_i|",
            variables=[
                {"symbol": "C_c", "description": "Concrete compression resultant", "value": Cc_eq, "units": "lb", "source": "step:Cc_eq"},
                {"symbol": "a", "description": "Stress block depth", "value": a_eq, "units": "in", "source": "step:a_eq"},
            ] + [{"symbol": f"F_{{s,{j}}}", "description": f"Layer {j} force", "value": F_layers[j-1], "units": "lb", "source": f"step:L{j}_F"} for j in range(1,len(F_layers)+1)]
              + [{"symbol": f"y_{{{j}}}", "description": f"Layer {j} location", "value": layer_y[j-1], "units": "in", "source": f"step:L{j}_y"} for j in range(1,len(layer_y)+1)],
            compute_fn=lambda: abs(float(Cc_eq*(a_eq/2.0) + sum(float(F_layers[j-1])*float(layer_y[j-1]) for j in range(1,len(F_layers)+1)))),
            units="lb-in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "derived", "ref": "Internal moment equilibrium"}],
        )

        Mn_kipft = compute_step(
            trace,
            id="Mn_kipft",
            section="Flexure check",
            title="Convert Mn to kip-ft",
            output_symbol="M_n",
            output_description="Nominal moment strength",
            equation_latex="M_{n,kip\cdot ft} = M_{n,lb\cdot in}/(12\,1000)",
            variables=[{"symbol": "M_{n,lb\cdot in}", "description": "Nominal moment", "value": Mn_lbin, "units": "lb-in", "source": "step:Mn_eq"}],
            compute_fn=lambda: float(Mn_lbin)/(12.0*1000.0),
            units="kip-ft",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "derived", "ref": "Unit conversion"}],
        )

        eps_t_eff = compute_step(
            trace,
            id="eps_t_eff",
            section="Flexure check",
            title="Net tensile strain at extreme tension layer",
            output_symbol="\\varepsilon_t",
            output_description="Net tensile strain at extreme tension layer",
            equation_latex="\\varepsilon_t = 0.003\\left(\\frac{d-c}{c}\\right)",
            variables=[
                {"symbol": "d", "description": "Effective depth", "value": d_eff, "units": "in", "source": "step:d_eff"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_eq, "units": "in", "source": "step:c_eq"},
            ],
            compute_fn=lambda: 0.003 * ((d_eff - c_eq)/c_eq),
            units="",
            rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 5},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1"}],
        )

        phi_cap = compute_step(
            trace,
            id="phi_cap",
            section="Strength reduction factor",
            title="Compute φ from εt",
            output_symbol="\phi",
            output_description="Strength reduction factor per Table 21.2.2",
            equation_latex=(
                "\phi = \\begin{cases}0.65 & \\varepsilon_t \le \\varepsilon_{ty}\\0.90 & \\varepsilon_t \ge \\varepsilon_{ty}+0.003\\0.65 + 0.25(\\varepsilon_t-\\varepsilon_{ty})/0.003 & \\text{otherwise}\\end{cases}"
            ),
            variables=[
                {"symbol": "\\varepsilon_t", "description": "Net tensile strain", "value": eps_t_eff, "units": "", "source": "step:eps_t_eff"},
                {"symbol": "\\varepsilon_{ty}", "description": "Yield strain", "value": eps_ty, "units": "", "source": "step:eps_ty"},
            ],
            compute_fn=lambda: _phi_flexure_from_eps_t(float(eps_t_eff), float(eps_ty), "other"),
            units="",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2"}],
        )

        phiMn_lbin = compute_step(
            trace,
            id="phiMn_eq",
            section="Flexure check",
            title="Design flexural strength",
            output_symbol="\phi M_n",
            output_description="Design moment strength",
            equation_latex="\phi M_n = \phi M_n",
            variables=[
                {"symbol": "\phi", "description": "Strength reduction factor", "value": phi_cap, "units": "", "source": "step:phi_cap"},
                {"symbol": "M_n", "description": "Nominal moment strength", "value": Mn_lbin, "units": "lb-in", "source": "step:Mn_eq"},
            ],
            compute_fn=lambda: float(phi_cap)*float(Mn_lbin),
            units="lb-in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "code", "ref": "ACI 318-25 21.2.1"}],
        )

        phiMn_kipft = compute_step(
            trace,
            id="phiMn_kipft",
            section="Flexure check",
            title="Convert φMn to kip-ft",
            output_symbol="\phi M_n",
            output_description="Design moment strength",
            equation_latex="(\phi M_n)_{kip\cdot ft} = (\phi M_n)_{lb\cdot in}/(12\,1000)",
            variables=[{"symbol": "\phi M_n", "description": "Design strength", "value": phiMn_lbin, "units": "lb-in", "source": "step:phiMn_eq"}],
            compute_fn=lambda: float(phiMn_lbin)/(12.0*1000.0),
            units="kip-ft",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "derived", "ref": "Unit conversion"}],
        )

        Mu_strip_kipft = float(inputs.Mu_kipft_per_ft)
        ok = float(phiMn_kipft) >= Mu_strip_kipft

        results = {
            "ok": bool(ok),
            "module": "slab_oneway_flexure",
            "inputs": inp,
            "outputs": {
                "b_in": b_in,
                "beta1": beta1,
                "phi": float(phi_cap),
                "As_total_in2": float(As_total),
                "phiMn_kipft_per_ft": float(phiMn_kipft),
                "Mu_kipft_per_ft": Mu_strip_kipft,
                "utilization": Mu_strip_kipft/float(phiMn_kipft) if float(phiMn_kipft) else float('inf'),
                "layers": layers,
            },
            "summary_text": (
                f"One-way slab strip flexure (12-in strip, multi-layer reinforcement)\n"
                f"φ = {float(phi_cap):.3f}\n"
                f"As,total = {float(As_total):.4f} in^2 per ft\n"
                f"Design strength φMn = {float(phiMn_kipft):.2f} kip-ft/ft; Demand Mu = {Mu_strip_kipft:.2f} kip-ft/ft\n"
                f"Status: {'PASS' if ok else 'FAIL'}\n"
            ),
            "warnings": [],
        }

        trace.summary.key_outputs = {
            "phi": {"value": float(phi_cap), "units": ""},
            "As_total": {"value": float(As_total), "units": "in^2"},
            "phiMn": {"value": float(phiMn_kipft), "units": "kip-ft/ft"},
            "Mu": {"value": Mu_strip_kipft, "units": "kip-ft/ft"},
        }
        trace.summary.governing_checks = [{"label": "Flexure strength (multi-layer)", "ratio": Mu_strip_kipft/float(phiMn_kipft) if float(phiMn_kipft) else float('inf'), "status": "PASS" if ok else "FAIL"}]
        trace.summary.controlling_step_ids = ["phiMn_eq"]

        run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
        logger = _setup_logger(run_dir)
        logger.info("Solve slab_oneway_flexure (multi-layer) inputs=%s", json.dumps(inp))
        export_all(trace, results, run_dir)
        return results, trace, run_dir


    # phi iteration as in beam (assume transverse_type 'other')
    phi = 0.90
    c_in = None
    a_in = None
    eps_t = None

    for i in range(1, 4):
        phi_i = compute_step(
            trace,
            id=f"phi_assumed_{i}",
            section="Strength reduction factor",
            title=f"Assumed φ for iteration {i}",
            output_symbol="\\phi",
            output_description="Assumed strength reduction factor",
            equation_latex="\\phi = \\phi_{assumed}",
            variables=[{"symbol": "\\phi_{assumed}", "description": "Assumed φ for iteration", "value": phi, "units": "", "source": "assumption:A_phi_start" if i == 1 else f"step:phi_calc_{i-1}"}],
            compute_fn=lambda phi=phi: phi,
            units="",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2 (φ depends on εt; iterated)"}],
        )

        Mn_req = compute_step(
            trace,
            id=f"Mn_req_{i}",
            section="Flexure design",
            title=f"Required nominal moment for iteration {i}",
            output_symbol="M_n",
            output_description="Required nominal moment strength",
            equation_latex="M_n = \\frac{M_u}{\\phi}",
            variables=[
                {"symbol": "M_u", "description": "Factored design moment (lb-in/ft)", "value": Mu_lbin_per_ft, "units": "lb-in/ft", "source": "step:Mu_strip"},
                {"symbol": "\\phi", "description": "Strength reduction factor", "value": phi_i, "units": "", "source": f"step:phi_assumed_{i}"},
            ],
            compute_fn=lambda Mu=Mu_lbin_per_ft, phi_i=phi_i: Mu / phi_i,
            units="lb-in/ft",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
            references=[{"type": "code", "ref": "ACI 318-25 21.2.1 (φMn ≥ Mu)"}],
        )

        def _compute_c():
            fc = inputs.fc_psi
            b = b_in
            d = d_in
            b1 = beta1
            A = 0.85 * fc * b * (b1 ** 2) / 2.0
            B = -0.85 * fc * b * b1 * d
            C = Mn_req
            disc = B * B - 4.0 * A * C
            if disc < 0:
                return float("nan")
            return (-B - math.sqrt(disc)) / (2.0 * A)

        c_in = compute_step(
            trace,
            id=f"c_{i}",
            section="Flexure design",
            title=f"Neutral axis depth from required Mn (iteration {i})",
            output_symbol="c",
            output_description="Neutral axis depth",
            equation_latex="c = \\frac{-B - \\sqrt{B^2 - 4AC}}{2A},\\quad A=\\frac{0.85 f'_c b \\beta_1^2}{2},\\; B=-0.85 f'_c b \\beta_1 d,\\; C=M_n",
            variables=[
                {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
                {"symbol": "b", "description": "Strip width", "value": b_in, "units": "in", "source": "derived:strip_12in"},
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
                {"symbol": "M_n", "description": "Required nominal moment", "value": Mn_req, "units": "lb-in/ft", "source": f"step:Mn_req_{i}"},
            ],
            compute_fn=_compute_c,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "derived", "ref": "Derived from ACI 318-25 22.2.2 assumptions (equilibrium + stress block)"}],
            warnings=(["No real solution for c (B^2 - 4AC < 0). Demand exceeds this section’s singly reinforced capacity under assumptions."] if math.isnan(_compute_c()) else None),
        )

        a_in = compute_step(
            trace,
            id=f"a_{i}",
            section="Flexure design",
            title=f"Compression block depth a (iteration {i})",
            output_symbol="a",
            output_description="Equivalent stress block depth",
            equation_latex="a = \\beta_1 c",
            variables=[
                {"symbol": "\\beta_1", "description": "Stress block factor", "value": beta1, "units": "", "source": "step:beta1"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_in, "units": "in", "source": f"step:c_{i}"},
            ],
            compute_fn=lambda: beta1 * c_in,
            units="in",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1 and 22.2.2.4.3"}],
        )

        eps_t = compute_step(
            trace,
            id=f"eps_t_{i}",
            section="Flexure design",
            title=f"Net tensile strain εt at nominal strength (iteration {i})",
            output_symbol="\\varepsilon_t",
            output_description="Net tensile strain at tension reinforcement",
            equation_latex="\\varepsilon_t = 0.003\\left(\\frac{d-c}{c}\\right)",
            variables=[
                {"symbol": "d", "description": "Effective depth", "value": d_in, "units": "in", "source": "step:d"},
                {"symbol": "c", "description": "Neutral axis depth", "value": c_in, "units": "in", "source": f"step:c_{i}"},
            ],
            compute_fn=lambda: 0.003 * ((d_in - c_in) / c_in),
            units="",
            rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 22.2.2.1"}],
        )

        phi_new = compute_step(
            trace,
            id=f"phi_calc_{i}",
            section="Strength reduction factor",
            title=f"Compute φ from εt (iteration {i})",
            output_symbol="\\phi",
            output_description="Strength reduction factor per Table 21.2.2",
            equation_latex=(
                "\\phi = \\begin{cases}"
                "0.65 & \\text{if } \\varepsilon_t \\le \\varepsilon_{ty} \\\\"
                "0.90 & \\text{if } \\varepsilon_t \\ge \\varepsilon_{ty}+0.003 \\\\"
                "0.65 + 0.25(\\varepsilon_t-\\varepsilon_{ty})/0.003 & \\text{otherwise}"
                "\\end{cases}"
            ),
            variables=[
                {"symbol": "\\varepsilon_t", "description": "Net tensile strain at nominal strength", "value": eps_t, "units": "", "source": f"step:eps_t_{i}"},
                {"symbol": "\\varepsilon_{ty}", "description": "Yield strain of reinforcement", "value": eps_ty, "units": "", "source": "step:eps_ty"},
            ],
            compute_fn=lambda eps_t=eps_t, eps_ty=eps_ty: (
                0.65 if eps_t <= eps_ty else 0.90 if eps_t >= eps_ty + 0.003 else 0.65 + 0.25*(eps_t - eps_ty)/0.003
            ),
            units="",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
            references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2"}],
        )

        if abs(phi_new - phi) < 0.001:
            phi = phi_new
            break
        phi = phi_new

    Cc_lb_per_ft = compute_step(
        trace,
        id="Cc",
        section="Flexure design",
        title="Concrete compression resultant (per ft strip)",
        output_symbol="C_c",
        output_description="Concrete compression force per foot",
        equation_latex="C_c = 0.85 f'_c\\, b\\, a",
        variables=[
            {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
            {"symbol": "b", "description": "Strip width", "value": b_in, "units": "in", "source": "derived:strip_12in"},
            {"symbol": "a", "description": "Stress block depth", "value": a_in, "units": "in", "source": f"step:a_{i}"},
        ],
        compute_fn=lambda: 0.85 * inputs.fc_psi * b_in * a_in,
        units="lb/ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.4.1"}],
    )

    fs_psi = compute_step(
        trace,
        id="fs",
        section="Flexure design",
        title="Tension steel stress at nominal strength",
        output_symbol="f_s",
        output_description="Steel stress (limited to fy)",
        equation_latex="f_s = \\min(f_y, E_s\\varepsilon_t)",
        variables=[
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "E_s", "description": "Steel modulus", "value": inputs.Es_psi, "units": "psi", "source": "input:Es_psi"},
            {"symbol": "\\varepsilon_t", "description": "Net tensile strain", "value": eps_t, "units": "", "source": f"step:eps_t_{i}"},
        ],
        compute_fn=lambda: min(inputs.fy_psi, inputs.Es_psi * eps_t),
        units="psi",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 22.2.2.3"}],
    )

    As_req_in2_per_ft = compute_step(
        trace,
        id="As_req",
        section="Flexure design",
        title="Required tension reinforcement area per foot width",
        output_symbol="A_s",
        output_description="Required As per foot width",
        equation_latex="A_s = \\frac{C_c}{f_s}",
        variables=[
            {"symbol": "C_c", "description": "Concrete compression force", "value": Cc_lb_per_ft, "units": "lb/ft", "source": "step:Cc"},
            {"symbol": "f_s", "description": "Steel stress", "value": fs_psi, "units": "psi", "source": "step:fs"},
        ],
        compute_fn=lambda: Cc_lb_per_ft / fs_psi,
        units="in^2/ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "Equilibrium: T = As·fs = Cc (ACI 318-25 22.2.2)"}],
    )

    # Suggest spacing for selected bar size
    Ab_est = math.pi * (inputs.bar_dia_in ** 2) / 4.0
    s_in = 12.0 * Ab_est / As_req_in2_per_ft
    s_suggest = max(1.0, math.floor(s_in * 2.0) / 2.0)  # round down to 0.5 in

    results = {
        "ok": True,
        "module": "slab_oneway_flexure",
        "inputs": inp,
        "outputs": {
            "d_in": d_in,
            "beta1": beta1,
            "phi": phi,
            "eps_t": eps_t,
            "As_required_in2_per_ft": As_req_in2_per_ft,
            "bar_dia_in": inputs.bar_dia_in,
            "bar_area_est_in2": Ab_est,
            "spacing_suggest_in": s_suggest,
        },
        "summary_text": (
            f"One-way slab strip (12-in) flexure\n"
            f"As,req = {As_req_in2_per_ft:.4f} in^2/ft\n"
            f"Suggested spacing for db={inputs.bar_dia_in:.3f} in (A≈{Ab_est:.3f} in^2): s ≈ {s_suggest:.1f} in\n"
        ),
        "warnings": [],
    }
    trace.summary.key_outputs = {
        "As_required": {"value": As_req_in2_per_ft, "units": "in^2/ft"},
        "spacing_suggest": {"value": s_suggest, "units": "in"},
    }

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
    logger = _setup_logger(run_dir)
    logger.info("Solve slab_oneway_flexure inputs=%s", json.dumps(inp))
    export_all(trace, results, run_dir)
    return results, trace, run_dir


def solve_column_axial(tool_id: str, inputs: ColumnAxialInputs) -> Tuple[Dict[str, Any], CalcTrace, Path]:
    inp = inputs.model_dump()
    trace = _make_trace(tool_id, {"module": "column_axial", **inp})
    _add_inputs(trace, {"module": "column_axial", **inp})

    trace.assumptions.extend([
        TraceAssumption(id="A1", text="Short column; concentric axial compression only; slenderness effects ignored."),
        TraceAssumption(id="A2", text="Nonprestressed reinforcement; nominal axial strength per Eq. (22.4.2.2) and maximum per Table 22.4.2.1."),
        TraceAssumption(id="A3", text="P-Δ, minimum eccentricity, moment interaction, second-order analysis, and seismic design are outside this module’s scope."),
    ])

    Pu_lb = compute_step(
        trace,
        id="Pu_conv",
        section="Loads",
        title="Convert factored axial load to consistent units",
        output_symbol="P_u",
        output_description="Factored axial compressive load",
        equation_latex="P_{u,lb} = P_{u,kip}\\,(1000\\,\\mathrm{lb/kip})",
        variables=[
            {"symbol": "P_{u,kip}", "description": "Factored axial load", "value": inputs.Pu_kip, "units": "kip", "source": "input:Pu_kip"},
        ],
        compute_fn=lambda: inputs.Pu_kip * 1000.0,
        units="lb",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "Unit conversion (kip to lb)"}],
    )

    def _compute_Ag():
        if inputs.shape == "rectangular":
            return inputs.b_in * inputs.h_in
        return math.pi * (inputs.D_in ** 2) / 4.0

    Ag_in2 = compute_step(
        trace,
        id="Ag",
        section="Section geometry",
        title="Gross concrete area",
        output_symbol="A_g",
        output_description="Gross area",
        equation_latex="A_g = \\begin{cases} b h & \\text{rectangular} \\\\ \\pi D^2/4 & \\text{circular}\\end{cases}",
        variables=[
            {"symbol": "b", "description": "Width (rectangular)", "value": inputs.b_in, "units": "in", "source": "input:b_in"},
            {"symbol": "h", "description": "Depth (rectangular)", "value": inputs.h_in, "units": "in", "source": "input:h_in"},
            {"symbol": "D", "description": "Diameter (circular)", "value": inputs.D_in, "units": "in", "source": "input:D_in"},
        ],
        compute_fn=_compute_Ag,
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Geometry"}],
    )

    Po_lb = compute_step(
        trace,
        id="Po",
        section="Axial strength",
        title="Nominal axial strength Po",
        output_symbol="P_o",
        output_description="Nominal axial compressive strength",
        equation_latex="P_o = 0.85 f'_c (A_g - A_{st}) + f_y A_{st}",
        variables=[
            {"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"},
            {"symbol": "A_g", "description": "Gross area", "value": Ag_in2, "units": "in^2", "source": "step:Ag"},
            {"symbol": "A_{st}", "description": "Longitudinal steel area", "value": inputs.Ast_in2, "units": "in^2", "source": "input:Ast_in2"},
            {"symbol": "f_y", "description": "Steel yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
        ],
        compute_fn=lambda: 0.85 * inputs.fc_psi * (Ag_in2 - inputs.Ast_in2) + inputs.fy_psi * inputs.Ast_in2,
        units="lb",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 Eq. (22.4.2.2)"}],
    )

    kmax = 0.85 if inputs.transverse_type == "spiral" else 0.80
    Pn_max_lb = compute_step(
        trace,
        id="Pn_max",
        section="Axial strength",
        title="Maximum nominal axial strength",
        output_symbol="P_{n,max}",
        output_description="Maximum nominal axial compressive strength",
        equation_latex="P_{n,max} = k\\,P_o",
        variables=[
            {"symbol": "k", "description": "Maximum factor (spiral 0.85, ties 0.80)", "value": kmax, "units": "", "source": "code:Table22.4.2.1"},
            {"symbol": "P_o", "description": "Nominal axial strength", "value": Po_lb, "units": "lb", "source": "step:Po"},
        ],
        compute_fn=lambda: kmax * Po_lb,
        units="lb",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 Table 22.4.2.1"}],
    )

    phi = 0.75 if inputs.transverse_type == "spiral" else 0.65
    phi_col = compute_step(
        trace,
        id="phi",
        section="Strength reduction factor",
        title="Strength reduction factor for compression-controlled axial strength",
        output_symbol="\\phi",
        output_description="Strength reduction factor",
        equation_latex="\\phi = \\phi_{cc}",
        variables=[{"symbol": "\\phi_{cc}", "description": "Compression-controlled φ", "value": phi, "units": "", "source": "code:Table21.2.2"}],
        compute_fn=lambda: phi,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 Table 21.2.2 (compression-controlled)"}],
    )

    phiPn_lb = compute_step(
        trace,
        id="phiPn",
        section="Axial strength",
        title="Design axial strength",
        output_symbol="\\phi P_n",
        output_description="Design axial compressive strength",
        equation_latex="\\phi P_n = \\phi\\, P_{n,max}",
        variables=[
            {"symbol": "\\phi", "description": "Strength reduction factor", "value": phi_col, "units": "", "source": "step:phi"},
            {"symbol": "P_{n,max}", "description": "Maximum nominal axial strength", "value": Pn_max_lb, "units": "lb", "source": "step:Pn_max"},
        ],
        compute_fn=lambda: phi_col * Pn_max_lb,
        units="lb",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "code", "ref": "ACI 318-25 21.2.1 (φPn ≥ Pu)"}],
        checks_builder=lambda val: [{
            "label": "Axial compression",
            "demand": float(Pu_lb),
            "capacity": float(val),
            "ratio": float(Pu_lb / val) if val else float("inf"),
            "pass_fail": "PASS" if val >= Pu_lb else "FAIL",
        }],
    )

    ok = phiPn_lb >= Pu_lb

    results = {
        "ok": bool(ok),
        "module": "column_axial",
        "inputs": inp,
        "outputs": {
            "Ag_in2": Ag_in2,
            "Po_kip": Po_lb / 1000.0,
            "Pn_max_kip": Pn_max_lb / 1000.0,
            "phi": phi_col,
            "phiPn_kip": phiPn_lb / 1000.0,
            "utilization": Pu_lb / phiPn_lb if phiPn_lb else float("inf"),
        },
        "summary_text": (
            f"Column axial compression (concentric)\n"
            f"Ag = {Ag_in2:.2f} in^2; Ast = {inputs.Ast_in2:.2f} in^2\n"
            f"Po = {Po_lb/1000.0:.1f} kip; Pn,max = {Pn_max_lb/1000.0:.1f} kip\n"
            f"φ = {phi_col:.3f}; φPn = {phiPn_lb/1000.0:.1f} kip; Pu = {inputs.Pu_kip:.1f} kip\n"
            f"Status: {'PASS' if ok else 'FAIL'}\n"
        ),
        "warnings": [],
    }

    trace.summary.key_outputs = {
        "phiPn": {"value": phiPn_lb / 1000.0, "units": "kip"},
        "Pu": {"value": inputs.Pu_kip, "units": "kip"},
    }

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
    logger = _setup_logger(run_dir)
    logger.info("Solve column_axial inputs=%s", json.dumps(inp))
    export_all(trace, results, run_dir)
    return results, trace, run_dir


def solve_dev_length_tension(tool_id: str, inputs: DevLengthTensionInputs) -> Tuple[Dict[str, Any], CalcTrace, Path]:
    inp = inputs.model_dump()
    trace = _make_trace(tool_id, {"module": "development_length_tension", **inp})
    _add_inputs(trace, {"module": "development_length_tension", **inp})

    trace.assumptions.extend([
        TraceAssumption(id="A1", text="Straight deformed bar development length in tension calculated using Table 25.4.2.3 and modification factors in Table 25.4.2.5."),
        TraceAssumption(id="A2", text="This module does not calculate (cb + Ktr)/db explicitly; it uses the two-tier simplified table approach."),
        TraceAssumption(id="A3", text="Headed/hooked bars, bundled bars, bar groups, post-installed bars, and seismic-specific anchorage checks are outside this module’s scope."),
    ])

    # sqrt(fc') cap
    sqrt_fc = compute_step(
        trace,
        id="sqrt_fc",
        section="Development length",
        title="Square root of concrete strength",
        output_symbol="\\sqrt{f'_c}",
        output_description="Square root of f'c",
        equation_latex="\\sqrt{f'_c} = \\sqrt{f'_c}",
        variables=[{"symbol": "f'_c", "description": "Concrete strength", "value": inputs.fc_psi, "units": "psi", "source": "input:fc_psi"}],
        compute_fn=lambda: math.sqrt(inputs.fc_psi),
        units="psi^0.5",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Math"}],
    )

    sqrt_fc_used = compute_step(
        trace,
        id="sqrt_fc_used",
        section="Development length",
        title="Limit √f'c for development length calculations",
        output_symbol="\\sqrt{f'_c}_{used}",
        output_description="Capped √f'c used for development length",
        equation_latex="\\sqrt{f'_c}_{used} = \\min(\\sqrt{f'_c}, 100\\,\\mathrm{psi})",
        variables=[
            {"symbol": "\\sqrt{f'_c}", "description": "Square root of concrete strength", "value": sqrt_fc, "units": "psi^0.5", "source": "step:sqrt_fc"},
        ],
        compute_fn=lambda: min(sqrt_fc, 100.0),
        units="psi^0.5",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 25.4.1.4"}],
    )

    # Modification factors per Table 25.4.2.5
    psi_t = compute_step(
        trace,
        id="psi_t",
        section="Development length",
        title="Reinforcement location factor ψt",
        output_symbol="\\psi_t",
        output_description="Top reinforcement factor",
        equation_latex="\\psi_t = \\begin{cases}1.3 & \\text{top reinforcement} \\\\ 1.0 & \\text{other}\u00a0\\end{cases}",
        variables=[{"symbol": "top", "description": "Top reinforcement?", "value": 1.0 if inputs.is_top_bar else 0.0, "units": "", "source": "input:is_top_bar"}],
        compute_fn=lambda: 1.3 if inputs.is_top_bar else 1.0,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.4.2.5 (ψt)"}],
    )

    psi_e = compute_step(
        trace,
        id="psi_e",
        section="Development length",
        title="Epoxy coating factor ψe",
        output_symbol="\\psi_e",
        output_description="Epoxy coating factor",
        equation_latex="\\psi_e = \\begin{cases}1.0 & \\text{uncoated} \\\\ 1.5 & \\text{epoxy and (cover<3db or spacing<6db)} \\\\ 1.2 & \\text{epoxy and otherwise}\\end{cases}",
        variables=[
            {"symbol": "epoxy", "description": "Epoxy coated?", "value": 1.0 if inputs.is_epoxy else 0.0, "units": "", "source": "input:is_epoxy"},
            {"symbol": "cond", "description": "Epoxy condition (cover<3db or spacing<6db)?", "value": 1.0 if inputs.epoxy_cover_lt_3db_or_spacing_lt_6db else 0.0, "units": "", "source": "input:epoxy_cover_lt_3db_or_spacing_lt_6db"},
        ],
        compute_fn=lambda: 1.0 if not inputs.is_epoxy else (1.5 if inputs.epoxy_cover_lt_3db_or_spacing_lt_6db else 1.2),
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.4.2.5 (ψe)"}],
    )

    no6_or_smaller = inputs.bar_size in ["#3", "#4", "#5", "#6"]
    psi_s = compute_step(
        trace,
        id="psi_s",
        section="Development length",
        title="Reinforcement size factor ψs",
        output_symbol="\\psi_s",
        output_description="Size factor",
        equation_latex="\\psi_s = \\begin{cases}0.8 & \\text{No. 6 and smaller bars and deformed wire} \\\\ 1.0 & \\text{No. 7 and larger bars}\\end{cases}",
        variables=[{"symbol": "No\\le6", "description": "No.6 and smaller?", "value": 1.0 if no6_or_smaller else 0.0, "units": "", "source": "input:bar_size"}],
        compute_fn=lambda: 0.8 if no6_or_smaller else 1.0,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.4.2.5 (ψs)"}],
    )

    psi_g = compute_step(
        trace,
        id="psi_g",
        section="Development length",
        title="Reinforcement grade factor ψg",
        output_symbol="\\psi_g",
        output_description="Grade factor",
        equation_latex="\\psi_g = \\begin{cases}1.0 & f_y \\le 60,000\\,\\mathrm{psi} \\\\ 1.15 & f_y > 60,000\\,\\mathrm{psi}\\end{cases}",
        variables=[{"symbol": "f_y", "description": "Yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"}],
        compute_fn=lambda: 1.15 if inputs.fy_psi > 60000.0 else 1.0,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.4.2.5 (ψg)"}],
    )

    lam = compute_step(
        trace,
        id="lambda",
        section="Development length",
        title="Lightweight concrete factor λ",
        output_symbol="\\lambda",
        output_description="Lightweight concrete factor",
        equation_latex="\\lambda = \\lambda",
        variables=[{"symbol": "\\lambda", "description": "Lightweight concrete factor", "value": inputs.lambda_factor, "units": "", "source": "input:lambda_factor"}],
        compute_fn=lambda: inputs.lambda_factor,
        units="",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.4.2.5 (λ)"}],
    )

    # Table 25.4.2.3 coefficients
    cover_ok = inputs.cover_ge_db_and_spacing_ge_2db
    denom = None
    mult = 1.0
    if cover_ok:
        denom = 25.0 if no6_or_smaller else 20.0
        mult = 1.0
    else:
        denom = 50.0 if no6_or_smaller else 40.0
        mult = 3.0

    ld_in = compute_step(
        trace,
        id="ld_table",
        section="Development length",
        title="Development length ℓd from Table 25.4.2.3",
        output_symbol="\\ell_d",
        output_description="Development length in tension",
        equation_latex="\\ell_d = \\frac{m\\, f_y\\,\\psi_t\\,\\psi_e\\,\\psi_g}{k\\,\\lambda\\,\\sqrt{f'_c}_{used}} d_b",
        variables=[
            {"symbol": "m", "description": "Table multiplier (1.0 or 3.0)", "value": mult, "units": "", "source": "code:Table25.4.2.3"},
            {"symbol": "f_y", "description": "Yield strength", "value": inputs.fy_psi, "units": "psi", "source": "input:fy_psi"},
            {"symbol": "\\psi_t", "description": "Top factor", "value": psi_t, "units": "", "source": "step:psi_t"},
            {"symbol": "\\psi_e", "description": "Epoxy factor", "value": psi_e, "units": "", "source": "step:psi_e"},
            {"symbol": "\\psi_g", "description": "Grade factor", "value": psi_g, "units": "", "source": "step:psi_g"},
            {"symbol": "k", "description": "Table denominator (20/25/40/50)", "value": denom, "units": "", "source": "code:Table25.4.2.3"},
            {"symbol": "\\lambda", "description": "Lightweight factor", "value": lam, "units": "", "source": "step:lambda"},
            {"symbol": "\\sqrt{f'_c}_{used}", "description": "Capped sqrt(fc')", "value": sqrt_fc_used, "units": "psi^0.5", "source": "step:sqrt_fc_used"},
            {"symbol": "d_b", "description": "Bar diameter", "value": inputs.db_in, "units": "in", "source": "input:db_in"},
        ],
        compute_fn=lambda: (mult * inputs.fy_psi * psi_t * psi_e * psi_g) / (denom * lam * sqrt_fc_used) * inputs.db_in,
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.4.2.3"}],
    )

    ld_min12 = compute_step(
        trace,
        id="ld_min",
        section="Development length",
        title="Apply minimum development length",
        output_symbol="\\ell_d",
        output_description="Development length in tension (minimum applied)",
        equation_latex="\\ell_d = \\max(\\ell_{d,table}, 12\\,\\mathrm{in})",
        variables=[
            {"symbol": "\\ell_{d,table}", "description": "Development length from Table 25.4.2.3", "value": ld_in, "units": "in", "source": "step:ld_table"},
        ],
        compute_fn=lambda: max(ld_in, 12.0),
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 25.4.2.1(b)"}],
    )

    # Lap splice length in tension per Table 25.5.2.1 (using ld from 25.4.2.1(a))
    ratio_As = inputs.As_provided_over_As_required
    classA = ratio_As >= 2.0
    # Table 25.5.2.1: If As_provided/As_required >=2.0 -> Class A (>=1.0ld and 12 in) or Class B (>=1.3ld and 12 in) depending on percent spliced.
    # For simplicity, tool reports both Class A and Class B and selects based on ratio only.
    lst_classA = compute_step(
        trace,
        id="lst_A",
        section="Lap splices",
        title="Tension lap splice length — Class A",
        output_symbol="\\ell_{st,A}",
        output_description="Class A lap splice length (tension)",
        equation_latex="\\ell_{st,A} = \\max(1.0\\,\\ell_d, 12\\,\\mathrm{in})",
        variables=[{"symbol": "\\ell_d", "description": "Development length", "value": ld_min12, "units": "in", "source": "step:ld_min"}],
        compute_fn=lambda: max(1.0 * ld_min12, 12.0),
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.5.2.1 (Class A)"}],
    )

    lst_classB = compute_step(
        trace,
        id="lst_B",
        section="Lap splices",
        title="Tension lap splice length — Class B",
        output_symbol="\\ell_{st,B}",
        output_description="Class B lap splice length (tension)",
        equation_latex="\\ell_{st,B} = \\max(1.3\\,\\ell_d, 12\\,\\mathrm{in})",
        variables=[{"symbol": "\\ell_d", "description": "Development length", "value": ld_min12, "units": "in", "source": "step:ld_min"}],
        compute_fn=lambda: max(1.3 * ld_min12, 12.0),
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "code", "ref": "ACI 318-25 Table 25.5.2.1 (Class B)"}],
    )

    lst_selected = lst_classA if classA else lst_classB

    results = {
        "ok": True,
        "module": "development_length_tension",
        "inputs": inp,
        "outputs": {
            "sqrt_fc_used": sqrt_fc_used,
            "psi_t": psi_t,
            "psi_e": psi_e,
            "psi_s": psi_s,
            "psi_g": psi_g,
            "lambda": lam,
            "ld_in": ld_min12,
            "lap_splice_classA_in": lst_classA,
            "lap_splice_classB_in": lst_classB,
            "lap_splice_selected_in": lst_selected,
            "lap_splice_selected_class": "Class A" if classA else "Class B",
        },
        "summary_text": (
            f"Development length (tension) — Table 25.4.2.3 + 25.4.2.1(b)\n"
            f"ℓd = {ld_min12:.2f} in\n"
            f"Lap splice (tension) per Table 25.5.2.1: Class A = {lst_classA:.2f} in; Class B = {lst_classB:.2f} in; Selected = {lst_selected:.2f} in ({'Class A' if classA else 'Class B'})\n"
        ),
        "warnings": [],
    }

    trace.summary.key_outputs = {
        "ld": {"value": ld_min12, "units": "in"},
        "lst_selected": {"value": lst_selected, "units": "in"},
    }

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
    logger = _setup_logger(run_dir)
    logger.info("Solve development_length_tension inputs=%s", json.dumps(inp))
    export_all(trace, results, run_dir)
    return results, trace, run_dir


def solve_wall_slender(tool_id: str, inputs: WallSlenderInputs) -> Tuple[Dict[str, Any], CalcTrace, Path]:
    """Slender wall check for combined axial + in-plane + out-of-plane bending.

    Implementation notes (conservative):
    - Second-order effects are handled using ACI 318-25 moment magnification for nonsway members (6.6.4.5).
    - EIeff uses Eq. (6.6.4.4.4c) with I from Table 6.6.3.1.1(a) (walls). If cracked_section=True, I=0.35Ig.
    - Strength is evaluated using strain compatibility and ACI 318-25 Chapter 22 assumptions with ϕ per Table 21.2.2.
    - Interaction is approximated by interpolating the ϕPn-ϕMn curve for the demanded Pu.
    """
    inp = inputs.model_dump()
    trace = _make_trace(tool_id, {"module": "wall_slender", **inp})
    _add_inputs(trace, {"module": "wall_slender", **inp})

    trace.assumptions.extend([
        TraceAssumption(id="W1", text="Wall is treated as a prismatic member with constant properties along height for each principal axis check."),
        TraceAssumption(id="W2", text="Second-order effects evaluated by moment magnification method for nonsway members (ACI 318-25 6.6.4.5)."),
        TraceAssumption(id="W3", text="Effective stiffness uses (EI)eff = Ec I /(1+βdns) (ACI 318-25 Eq. 6.6.4.4.4c) with I from Table 6.6.3.1.1(a)."),
        TraceAssumption(id="W4", text="Section strength uses Chapter 22 equivalent rectangular stress block with εcu=0.003 and steel stress limited to ±fy. Reinforcement is assumed symmetrically distributed in two face layers."),
        TraceAssumption(id="W5", text="Out-of-plane and in-plane flexure are checked independently (biaxial interaction not yet implemented)."),
    ])

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
    logger = _setup_logger(run_dir)

    warnings: List[str] = []
    if not inputs.member_nonsway:
        warnings.append("Sway (moment magnification for sway members / second-order frame analysis) is not implemented; results assume nonsway per 6.6.4.5.")
    if inputs.end_bottom == "unbraced" or inputs.end_top == "unbraced":
        warnings.append("End condition set to 'unbraced' -> k=2.0 per Table 11.5.3.2. Ensure this matches actual lateral translation restraint.")

    # Geometry basics
    Ag_in2 = compute_step(
        trace,
        id="Ag",
        section="Section geometry",
        title="Gross wall area",
        output_symbol="A_g",
        output_description="Gross cross-sectional area",
        equation_latex="A_g = \ell_w t",
        variables=[
            {"symbol": "\ell_w", "description": "Wall length", "value": inputs.lw_in, "units": "in", "source": "input:lw_in"},
            {"symbol": "t", "description": "Wall thickness", "value": inputs.t_in, "units": "in", "source": "input:t_in"},
        ],
        compute_fn=lambda: inputs.lw_in * inputs.t_in,
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "Geometry"}],
    )

    As_v_in2 = compute_step(
        trace,
        id="Asv",
        section="Reinforcement",
        title="Total vertical reinforcement area",
        output_symbol="A_{s,v}",
        output_description="Total vertical reinforcement area",
        equation_latex="A_{s,v} = \rho_v A_g",
        variables=[
            {"symbol": "\rho_v", "description": "Vertical reinforcement ratio", "value": inputs.rho_v, "units": "", "source": "input:rho_v"},
            {"symbol": "A_g", "description": "Gross area", "value": Ag_in2, "units": "in^2", "source": "step:Ag"},
        ],
        compute_fn=lambda: inputs.rho_v * Ag_in2,
        units="in^2",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "Reinforcement ratio definition"}],
    )

    beta1 = _beta1(trace, inputs.fc_psi)
    Ec = _Ec_normalweight(trace, inputs.fc_psi)
    k = _wall_k_factor(trace, inputs.end_bottom, inputs.end_top)

    def _I_effective(axis: str, b_in: float, h_in: float) -> float:
        Ig = b_in * (h_in**3) / 12.0
        factor = 0.35 if inputs.cracked_section else 0.70
        return compute_step(
            trace,
            id=f"{axis}_I",
            section="Slenderness",
            title=f"Effective moment of inertia I for {axis} axis",
            output_symbol="I",
            output_description="Effective moment of inertia",
            equation_latex="I = k_I I_g;\\; I_g=\\frac{b h^3}{12}",
            variables=[
                {"symbol": "b", "description": "Section width", "value": b_in, "units": "in", "source": "derived"},
                {"symbol": "h", "description": "Section depth", "value": h_in, "units": "in", "source": "derived"},
                {"symbol": "k_I", "description": "Inertia factor (0.70 uncracked or 0.35 cracked)", "value": factor, "units": "", "source": "assumption:W3"},
            ],
            compute_fn=lambda: factor * Ig,
            units="in^4",
            rounding_rule={"rule": "sigfigs", "decimals_or_sigfigs": 4},
            references=[{"type": "code", "ref": "ACI 318-25 Table 6.6.3.1.1(a) (walls)"}],
        )

    # Axis definitions
    # Out-of-plane bending about thickness: b=lw, h=t
    I_oop = _I_effective("oop", inputs.lw_in, inputs.t_in)
    Cm_oop, Pc_oop, delta_oop, M2_oop = _moment_magnification_nonsway(
        trace,
        prefix="oop",
        Pu_kip=inputs.Pu_kip,
        M_top_kipft=inputs.M_top_oop_kipft,
        M_bot_kipft=inputs.M_bot_oop_kipft,
        h_in=inputs.t_in,
        Ec_psi=Ec,
        I_in4=I_oop,
        beta_dns=inputs.beta_dns,
        k=k,
        lu_in=inputs.lu_in,
        transverse_loads_between_ends=inputs.transverse_loads_between_ends,
    )
    Mu_oop = compute_step(
        trace,
        id="oop_Mu",
        section="Slenderness",
        title="Magnified out-of-plane design moment",
        output_symbol="M_{u,o}",
        output_description="Magnified out-of-plane design moment",
        equation_latex="M_{u,o} = \\delta_{ns} M_2",
        variables=[
            {"symbol": "\\delta_{ns}", "description": "Moment magnification factor", "value": delta_oop, "units": "", "source": "step:oop_delta"},
            {"symbol": "M_2", "description": "Governing first-order moment", "value": M2_oop, "units": "kip-ft", "source": "step:oop_M2_design"},
        ],
        compute_fn=lambda: delta_oop * M2_oop,
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 6.6.4.5.2"}],
    )

    # In-plane bending about wall length: b=t, h=lw
    I_ip = _I_effective("ip", inputs.t_in, inputs.lw_in)
    Cm_ip, Pc_ip, delta_ip, M2_ip = _moment_magnification_nonsway(
        trace,
        prefix="ip",
        Pu_kip=inputs.Pu_kip,
        M_top_kipft=inputs.M_top_ip_kipft,
        M_bot_kipft=inputs.M_bot_ip_kipft,
        h_in=inputs.lw_in,
        Ec_psi=Ec,
        I_in4=I_ip,
        beta_dns=inputs.beta_dns,
        k=k,
        lu_in=inputs.lu_in,
        transverse_loads_between_ends=inputs.transverse_loads_between_ends,
    )
    Mu_ip = compute_step(
        trace,
        id="ip_Mu",
        section="Slenderness",
        title="Magnified in-plane design moment",
        output_symbol="M_{u,i}",
        output_description="Magnified in-plane design moment",
        equation_latex="M_{u,i} = \\delta_{ns} M_2",
        variables=[
            {"symbol": "\\delta_{ns}", "description": "Moment magnification factor", "value": delta_ip, "units": "", "source": "step:ip_delta"},
            {"symbol": "M_2", "description": "Governing first-order moment", "value": M2_ip, "units": "kip-ft", "source": "step:ip_M2_design"},
        ],
        compute_fn=lambda: delta_ip * M2_ip,
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "code", "ref": "ACI 318-25 6.6.4.5.2"}],
    )

    # Strength check per axis using interaction curve
    def _check_axis(axis: str, b_in: float, h_in: float, Mu_kipft: float) -> Dict[str, Any]:
        phiPn, phiMn, _ = _section_capacity_points_rect(
            b_in=b_in,
            h_in=h_in,
            As_total_in2=As_v_in2,
            cover_in=inputs.cover_in,
            bar_dia_in=inputs.bar_dia_in,
            fc_psi=inputs.fc_psi,
            fy_psi=inputs.fy_psi,
            Es_psi=inputs.Es_psi,
            transverse_type=inputs.transverse_type,
            beta1=beta1,
        )
        Mn_cap = _interp_capacity_moment(phiPn, phiMn, inputs.Pu_kip)
        ratio = Mu_kipft / Mn_cap if Mn_cap > 1e-9 else 1e9
        return {
            "phiMn_kipft": Mn_cap,
            "Mu_kipft": Mu_kipft,
            "ratio": ratio,
            "pass": ratio <= 1.0,
        }

    oop_chk = _check_axis("oop", inputs.lw_in, inputs.t_in, Mu_oop)
    ip_chk = _check_axis("ip", inputs.t_in, inputs.lw_in, Mu_ip)

    controlling = "oop" if oop_chk["ratio"] >= ip_chk["ratio"] else "ip"
    overall_pass = bool(oop_chk["pass"] and ip_chk["pass"])

    results = {
        "ok": True,
        "module": "wall_slender",
        "inputs": inp,
        "outputs": {
            "Mu_oop_kipft": Mu_oop,
            "Mu_ip_kipft": Mu_ip,
            "oop": oop_chk,
            "ip": ip_chk,
            "controlling": controlling,
            "overall_pass": overall_pass,
        },
        "summary_text": (
            f"Slender wall check: overall {'PASS' if overall_pass else 'FAIL'}; controlling axis: {controlling}. "
            f"Out-of-plane Mu/ϕMn={oop_chk['ratio']:.3f}, In-plane Mu/ϕMn={ip_chk['ratio']:.3f}."
        ),
        "warnings": warnings,
    }

    # Summary + tables
    trace.summary = TraceSummary(
        governing_checks=[
            {"label": "Out-of-plane flexure", "ratio": oop_chk["ratio"], "pass": oop_chk["pass"], "step_ids": ["oop_Mu"]},
            {"label": "In-plane flexure", "ratio": ip_chk["ratio"], "pass": ip_chk["pass"], "step_ids": ["ip_Mu"]},
        ],
        controlling_step_ids=["oop_Mu"] if controlling == "oop" else ["ip_Mu"],
        key_outputs={
            "Mu_oop_kipft": Mu_oop,
            "Mu_ip_kipft": Mu_ip,
            "phiMn_oop_kipft": oop_chk["phiMn_kipft"],
            "phiMn_ip_kipft": ip_chk["phiMn_kipft"],
        },
        warnings=warnings,
    )

    export_all(trace, results, run_dir)
    logger.info("wall_slender completed: %s", results["summary_text"])

def _anchor_phi_factors(*, redundant: bool, steel_ductile_tension: bool, steel_ductile_shear: bool) -> dict:
    # ACI 318-25 Table 21.2.1 (anchor-related rows). If ACI 318-25 table extraction fails, these match 318-19 practice.
    phi_conc_tension = 0.75 if redundant else 0.65
    phi_conc_shear = 0.75
    phi_steel_tension = 0.75 if steel_ductile_tension else 0.65
    phi_steel_shear = 0.65 if steel_ductile_shear else 0.60
    return {
        "phi_conc_tension": phi_conc_tension,
        "phi_conc_shear": phi_conc_shear,
        "phi_steel_tension": phi_steel_tension,
        "phi_steel_shear": phi_steel_shear,
    }


def _ccd_Nb(lambda_factor: float, fc_psi: float, hef_in: float, is_cast_in: bool) -> float:
    # ACI CCD nominal breakout strength for a single anchor in tension, basic strength Nb (lb)
    kc = 24.0 if is_cast_in else 17.0  # post-installed conservative
    return kc * lambda_factor * math.sqrt(fc_psi) * (hef_in ** 1.5)


def _ccd_Vb(lambda_factor: float, fc_psi: float, hef_in: float) -> float:
    # Basic breakout strength in shear, Vb (lb), conservative simplification
    kc = 16.0
    return kc * lambda_factor * math.sqrt(fc_psi) * (hef_in ** 1.5)


def solve_anchors_ch17(tool_id: str, model: AnchorsCh17Inputs):
    inputs = model.model_dump()
    trace = _make_trace(tool_id, inputs)
    _add_inputs(trace, inputs)

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
    logger = _setup_logger(run_dir)

    n = int(model.anchor_count_x) * int(model.anchor_count_y)
    lambda_factor = float(model.lambda_factor)
    fc = float(model.fc_psi)
    hef = float(model.hef_in)
    dia = float(model.diameter_in)

    fam = model.anchor_family
    is_cast_in = fam == "cast_in_headed"
    phi = _anchor_phi_factors(redundant=model.redundant, steel_ductile_tension=model.steel_ductile_tension, steel_ductile_shear=model.steel_ductile_shear)

    # Steel strengths (very conservative defaults if not provided)
    fu = float(model.fu_psi or 65000.0)
    # Tensile stress area approximated by gross area (headed rod simplification)
    Ase = math.pi * (dia ** 2) / 4.0
    Nsa_lb = Ase * fu  # nominal steel tension strength
    # Shear steel nominal (conservative) = 0.6*fu*Ase
    Vsa_lb = 0.6 * fu * Ase

    # Concrete breakout basic strengths
    Nb = _ccd_Nb(lambda_factor, fc, hef, is_cast_in=is_cast_in)
    Vb = _ccd_Vb(lambda_factor, fc, hef)

    # Very conservative group/edge factors (full ACI projected area method is extensive; implemented here as conservative reductions)
    # If any edge distance < 1.5*hef, reduce to ratio (c/(1.5hef))^1.5, capped at 1.0
    cmin = min(model.edge_x_neg_in, model.edge_x_pos_in, model.edge_y_neg_in, model.edge_y_pos_in) if (model.edge_x_neg_in or model.edge_x_pos_in or model.edge_y_neg_in or model.edge_y_pos_in) else 1e9
    edge_reduction = 1.0
    if cmin < 1e8:
        edge_reduction = min(1.0, (cmin / max(1e-6, 1.5 * hef)) ** 1.5)
    # spacing reduction if spacing < 3*hef
    smin = min([x for x in [model.sx_in, model.sy_in] if x > 0] or [1e9])
    spacing_reduction = 1.0
    if smin < 1e8:
        spacing_reduction = min(1.0, (smin / max(1e-6, 3.0 * hef)) ** 1.0)

    Ncb_lb = Nb * n * edge_reduction * spacing_reduction
    Vcb_lb = Vb * n * edge_reduction * spacing_reduction

    # Pullout / pryout placeholders (conservative: do not credit unless cast-in headed)
    Np_lb = 0.0
    if is_cast_in:
        # conservative pullout for headed: 8*lambda*sqrt(fc)*A_bearing, take bearing area = 2.0*Ase
        Np_lb = 8.0 * lambda_factor * math.sqrt(fc) * (2.0 * Ase)

    Vcp_lb = 1.0 * Vcb_lb  # conservative pryout proportionality

    # Design strengths
    phiN_steel = phi["phi_steel_tension"] * Nsa_lb
    phiV_steel = phi["phi_steel_shear"] * Vsa_lb
    phiN_cb = phi["phi_conc_tension"] * Ncb_lb
    phiV_cb = phi["phi_conc_shear"] * Vcb_lb
    phiN_p = phi["phi_conc_tension"] * Np_lb if Np_lb > 0 else 0.0
    phiV_cp = phi["phi_conc_shear"] * Vcp_lb

    Nu_lb = float(model.Nu_kip) * 1000.0
    Vu_lb = float(model.Vu_kip) * 1000.0

    # Governing capacities
    N_caps = [("Steel tension", phiN_steel), ("Concrete breakout (tension)", phiN_cb)]
    if phiN_p > 0:
        N_caps.append(("Pullout (headed)", phiN_p))
    V_caps = [("Steel shear", phiV_steel), ("Concrete breakout (shear)", phiV_cb), ("Pryout", phiV_cp)]

    govN = min(N_caps, key=lambda x: x[1])
    govV = min(V_caps, key=lambda x: x[1])

    # Simple interaction (conservative): (Nu/Nn)+(Vu/Vn) <= 1.0 when both present
    interaction = 0.0
    if Nu_lb > 0 and govN[1] > 0:
        interaction += Nu_lb / govN[1]
    if Vu_lb > 0 and govV[1] > 0:
        interaction += Vu_lb / govV[1]

    okN = True if Nu_lb <= 0 else (Nu_lb <= govN[1] + 1e-9)
    okV = True if Vu_lb <= 0 else (Vu_lb <= govV[1] + 1e-9)
    okI = interaction <= 1.0 + 1e-9 if (Nu_lb > 0 and Vu_lb > 0) else True

    results = {
        "ok": True,
        "module": "anchors_ch17",
        "inputs": inputs,
        "outputs": {
            "phi_factors": phi,
            "n_anchors": n,
            "edge_reduction": edge_reduction,
            "spacing_reduction": spacing_reduction,
            "phiN_steel_kip": phiN_steel / 1000.0,
            "phiN_cb_kip": phiN_cb / 1000.0,
            "phiN_pullout_kip": phiN_p / 1000.0,
            "phiV_steel_kip": phiV_steel / 1000.0,
            "phiV_cb_kip": phiV_cb / 1000.0,
            "phiV_pryout_kip": phiV_cp / 1000.0,
            "governing_tension_mode": govN[0],
            "phiN_governing_kip": govN[1] / 1000.0,
            "governing_shear_mode": govV[0],
            "phiV_governing_kip": govV[1] / 1000.0,
            "interaction_ratio": interaction,
        },
        "summary_text": f"Anchors Ch.17: Tension {('OK' if okN else 'NG')} ({govN[0]}), Shear {('OK' if okV else 'NG')} ({govV[0]}), Interaction {('OK' if okI else 'NG')}.",
        "warnings": [
            "Concrete breakout uses conservative simplified projected-area reductions (edge/spacing). Verify for final design per ACI 318-25 Ch.17.",
            "Post-installed anchor mode strengths require manufacturer ESR/ETA tables; this initial implementation is conservative and intended as a starter database.",
        ],
    }

    export_all(trace, results, run_dir)
    return results, trace, run_dir


def solve_punching_shear(tool_id: str, model: PunchingShearInputs):
    inputs = model.model_dump()
    trace = _make_trace(tool_id, inputs)
    _add_inputs(trace, inputs)

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)

    fc = float(model.fc_psi)
    lam = float(model.lambda_factor)
    bx = float(model.column_bx_in)
    by = float(model.column_by_in)
    h = float(model.slab_thickness_in)
    d = float(model.d_in) if model.d_in is not None else 0.8 * h

    # Critical perimeter at d/2 from face; adjust for edge/corner by taking effective sides
    # Interior: b0 = 2*(bx+by+2d)
    # Edge: reduce one side -> b0 = 2*(bx+by+2d) - (by+2d) (conservative)
    # Corner: reduce two sides -> subtract (bx+2d)+(by+2d)
    b0 = 2.0 * (bx + by + 2.0 * d)
    if model.location == "edge":
        b0 -= (by + 2.0 * d)
    elif model.location == "corner":
        b0 -= (bx + 2.0 * d) + (by + 2.0 * d)
    b0 = max(b0, 1e-6)

    Vu = float(model.Vu_kip) * 1000.0

    # Nominal two-way shear strength without shear reinforcement (conservative):
    # vc = 4*lambda*sqrt(fc) (psi), Vc = vc*b0*d (lb)
    vc_psi = 4.0 * lam * math.sqrt(fc)
    Vc = vc_psi * b0 * d

    phi = 0.75
    phiVc = phi * Vc

    ratio = Vu / phiVc if phiVc > 0 else float("inf")
    ok = ratio <= 1.0 + 1e-9

    results = {
        "ok": True,
        "module": "punching_shear",
        "inputs": inputs,
        "outputs": {
            "d_in": d,
            "b0_in": b0,
            "vc_psi": vc_psi,
            "phi": phi,
            "phiVc_kip": phiVc / 1000.0,
            "Vu_kip": Vu / 1000.0,
            "utilization": ratio,
            "pass_fail": "PASS" if ok else "FAIL",
        },
        "summary_text": f"Punching shear: utilization = {ratio:.3f} ({'PASS' if ok else 'FAIL'}).",
        "warnings": [
            "This implementation uses a conservative vc = 4*lambda*sqrt(fc) without moment transfer amplification. Expand as needed for full ACI eccentricity provisions.",
        ],
    }
    export_all(trace, results, run_dir)
    return results, trace, run_dir


def solve_development_length_splices(tool_id: str, model: DevelopmentLengthSpliceInputs):
    inputs = model.model_dump()
    trace = _make_trace(tool_id, inputs)
    _add_inputs(trace, inputs)

    run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)

    # Bar diameter
    db = float(model.db_in) if model.db_in is not None else REBAR_DB.get(model.bar_size, {}).get("db_in", 0.0)
    if db <= 0:
        db = float(model.bar_size.strip("#") or 0) / 8.0  # last resort

    fc = float(model.fc_psi)
    fy = float(model.fy_psi)
    lam = float(model.lambda_factor)

    # Modifiers (conservative implementation aligned with common ACI Ch.25 practice)
    psi_t = 1.3 if model.is_top_bar else 1.0
    if model.is_epoxy:
        psi_e = 1.5 if model.epoxy_cover_lt_3db_or_spacing_lt_6db else 1.2
    else:
        psi_e = 1.0
    psi_s = 1.0  # size factor; retain 1.0 for conservative general use

    # Tension development (in)
    ld_t = (3.0/40.0) * (fy / (lam * math.sqrt(fc))) * db * psi_t * psi_e * psi_s
    ld_t = max(ld_t, 12.0)  # common minimum, conservative

    # Compression development (in) - conservative based on common ACI expression
    ld_c = 0.02 * (fy / (lam * math.sqrt(fc))) * db * 1000.0 / 1000.0  # keep scale consistent
    ld_c = max(ld_c, 8.0)

    # Splice lengths
    # Class A if As_prov/As_req >= 2.0 AND percent spliced <= 50%, else Class B.
    class_A = (model.As_provided_over_As_required >= 2.0) and (model.percent_bars_spliced <= 50.0)
    if model.calc_type == "tension_lap_splice":
        ls = (1.0 if class_A else 1.3) * ld_t
    elif model.calc_type == "compression_lap_splice":
        ls = 1.3 * ld_c
    else:
        ls = 0.0

    out = {
        "db_in": db,
        "psi_t": psi_t,
        "psi_e": psi_e,
        "psi_s": psi_s,
        "ld_tension_in": ld_t,
        "ld_compression_in": ld_c,
        "splice_class": "A" if class_A else "B",
        "ls_in": ls,
    }

    results = {
        "ok": True,
        "module": "development_length_splices",
        "inputs": inputs,
        "outputs": out,
        "summary_text": f"Ch.25 {model.calc_type}: ld_t={ld_t:.1f} in, ld_c={ld_c:.1f} in, ls={ls:.1f} in (Class {'A' if class_A else 'B'} when applicable).",
        "warnings": [
            "Development/splice equations implemented conservatively; verify project-specific modifiers and minimums against ACI 318-25 Chapter 25.",
        ],
    }
    export_all(trace, results, run_dir)
    return results, trace, run_dir

    return results, trace, run_dir


def solve(tool_id: str, module: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main solver entry used by the backend HTTP server.
    Creates run_dir, produces CalcTrace, and exports artifacts.
    Returns dict compatible with /api/solve response.
    """
    module = (module or "").strip()
    if module == "beam_flexure":
        model = BeamFlexureInputs(**inputs)
        results, trace, run_dir = solve_beam_flexure(tool_id, model)
    elif module == "slab_oneway_flexure":
        model = SlabOnewayFlexureInputs(**inputs)
        results, trace, run_dir = solve_slab_oneway_flexure(tool_id, model)
    elif module == "column_axial":
        model = ColumnAxialInputs(**inputs)
        results, trace, run_dir = solve_column_axial(tool_id, model)
    elif module == "development_length_tension":
        model = DevLengthTensionInputs(**inputs)
        results, trace, run_dir = solve_dev_length_tension(tool_id, model)
    elif module == "wall_slender":
        model = WallSlenderInputs(**inputs)
        results, trace, run_dir = solve_wall_slender(tool_id, model)
    elif module == "development_length_splices":
        model = DevelopmentLengthSpliceInputs(**inputs)
        results, trace, run_dir = solve_development_length_splices(tool_id, model)
    elif module == "anchors_ch17":
        model = AnchorsCh17Inputs(**inputs)
        results, trace, run_dir = solve_anchors_ch17(tool_id, model)
    elif module == "punching_shear":
        model = PunchingShearInputs(**inputs)
        results, trace, run_dir = solve_punching_shear(tool_id, model)
    else:
        # Placeholder modules
        trace = _make_trace(tool_id, {"module": module, **inputs})
        _add_inputs(trace, {"module": module, **inputs})
        trace.assumptions.append(TraceAssumption(id="NYI", text="This module is not yet implemented in this tool version."))
        run_dir = create_run_dir(tool_id, seed=trace.meta.input_hash)
        _setup_logger(run_dir).warning("Module not implemented: %s", module)
        results = {
            "ok": False,
            "module": module,
            "inputs": inputs,
            "outputs": {},
            "summary_text": f"Module '{module}' is not implemented in this version.",
            "warnings": [f"Module '{module}' is not implemented."],
        }
        export_all(trace, results, run_dir)

    results["run_dir"] = str(run_dir)
    results["input_hash"] = trace.meta.input_hash
    results["tool_version"] = TOOL_VERSION
    return results