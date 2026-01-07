from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .calc_trace import CalcTrace, compute_step
from .constants import EDGE_LINE_LOAD_PLF, EDGE_LOAD_OUTBOARD_IN, SIN_45
from .analysis.placement import PlacementOutcome

@dataclass(frozen=True)
class EvalResult:
    spacing_ft: float
    V_total_kip: float
    M_total_kipft: float

    depth_ft: float
    dx_ft: float
    sin_theta: float

    hanger_demand_kip: float
    diagonal_demand_kip: float

    util_hanger: float
    util_diagonal: float

    governs: str
    passes: bool

def _compute_loads(inputs: Dict[str, Any], spacing_ft: float) -> Tuple[float, float, float, float, float, float, float]:
    """
    Simplified construction-stage demand model at one bracket line.

    Tributary width = spacing (ft).

    Uniform load:
      q_total = gamma_conc*t + q_form + q_live  (psf)
      V_dist = q_total * a * s / 1000          (kip)
      M_dist = q_total * a^2 * s / (2*1000)    (kip-ft)

    Concentrated load:
      V_wheel = P
      M_wheel = P*a_out

    Edge line load:
      V_edge = w_edge * s / 1000
      M_edge = V_edge * a_out
    """
    a_ft = float(inputs["overhang_length_ft"])
    t_in = float(inputs["slab_thickness_in"])
    P = float(inputs["screed_wheel_load_kip"])
    gamma = float(inputs["concrete_unit_weight_pcf"])
    q_live = float(inputs["construction_live_load_psf"])
    q_form = float(inputs["formwork_misc_psf"])

    t_ft = t_in / 12.0
    q_conc = gamma * t_ft  # psf
    q_tot = q_conc + q_form + q_live  # psf

    V_dist = (q_tot * a_ft * spacing_ft) / 1000.0
    M_dist = (q_tot * (a_ft**2) * spacing_ft) / (2.0 * 1000.0)

    a_out_ft = a_ft + EDGE_LOAD_OUTBOARD_IN / 12.0
    V_wheel = P
    M_wheel = P * a_out_ft
    V_edge = (EDGE_LINE_LOAD_PLF * spacing_ft) / 1000.0
    M_edge = V_edge * a_out_ft
    V = V_dist + V_wheel
    M = M_dist + M_wheel
    V += V_edge
    M += M_edge
    return V, M, q_conc, q_tot, t_ft, V_edge, a_out_ft

def _compute_member_demands(V_kip: float, M_kipft: float, sin_theta: float, depth_ft: float) -> Tuple[float, float]:
    """
    Member axial demands.

    Hanger demand is kept consistent with prior behavior:
      F_h = V / sin(45°)

    Diagonal demand includes a moment-equivalent vertical component using lever arm depth_ft:
      V_M = M / depth_ft
      F_d = (V + V_M) / sin(theta)
    """
    sin_theta = max(float(sin_theta), 1e-6)
    depth_ft = max(float(depth_ft), 1e-6)
    F_h = float(V_kip) / SIN_45
    V_M = float(M_kipft) / depth_ft
    F_d = (float(V_kip) + V_M) / sin_theta
    return F_h, F_d

def evaluate_fast(inputs: Dict[str, Any], spacing_ft: float, placement: PlacementOutcome) -> EvalResult:
    V, M, *_ = _compute_loads(inputs, spacing_ft)

    if placement.feasible and bool(inputs.get("use_geometry_based_diagonal_angle", True)):
        sin_theta = placement.sin_theta
        depth_ft = placement.depth_ft
        dx_ft = placement.dx_ft
    else:
        sin_theta = SIN_45
        depth_ft = float(inputs.get("bracket_lever_arm_ft", 3.0))
        dx_ft = float("nan")

    F_h, F_d = _compute_member_demands(V, M, sin_theta=sin_theta, depth_ft=depth_ft)

    hanger_cap = float(inputs["hanger_swl_kip"])
    diag_cap = float(inputs["diagonal_swl_kip"])
    util_h = F_h / hanger_cap
    util_d = F_d / diag_cap
    passes = (F_h <= hanger_cap) and (F_d <= diag_cap)
    governs = "diagonal" if util_d >= util_h else "hanger"

    return EvalResult(
        spacing_ft=float(spacing_ft),
        V_total_kip=float(V),
        M_total_kipft=float(M),
        depth_ft=float(depth_ft),
        dx_ft=float(dx_ft),
        sin_theta=float(sin_theta),
        hanger_demand_kip=float(F_h),
        diagonal_demand_kip=float(F_d),
        util_hanger=float(util_h),
        util_diagonal=float(util_d),
        governs=governs,
        passes=bool(passes),
    )

def evaluate_with_trace(trace: CalcTrace, inputs: Dict[str, Any], spacing_ft: float, placement: PlacementOutcome) -> EvalResult:
    # Loads
    V, M, q_conc, q_tot, t_ft, V_edge, a_out_ft = _compute_loads(inputs, spacing_ft)
    a_ft = float(inputs["overhang_length_ft"])

    compute_step(
        trace,
        id="L1",
        section="Loads",
        title="Concrete self-weight pressure",
        output_symbol="q_{conc}",
        output_description="Concrete self-weight pressure",
        equation_latex=r"q_{conc} = \gamma_{conc}\,t",
        variables=[
            {"symbol": r"\gamma_{conc}", "description": "Concrete unit weight", "value": float(inputs["concrete_unit_weight_pcf"]), "units": "pcf", "source": "input:concrete_unit_weight_pcf"},
            {"symbol": "t", "description": "Slab thickness", "value": float(inputs["slab_thickness_in"]), "units": "in", "source": "input:slab_thickness_in"},
        ],
        compute_fn=lambda: q_conc,
        units="psf",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L1"}],
    )

    compute_step(
        trace,
        id="L2",
        section="Loads",
        title="Total construction-stage pressure",
        output_symbol="q_{tot}",
        output_description="Total construction-stage area load",
        equation_latex=r"q_{tot} = q_{conc} + q_{form} + q_{LL}",
        variables=[
            {"symbol": "q_{conc}", "description": "Concrete pressure", "value": q_conc, "units": "psf", "source": "step:L1"},
            {"symbol": "q_{form}", "description": "Formwork + misc", "value": float(inputs["formwork_misc_psf"]), "units": "psf", "source": "input:formwork_misc_psf"},
            {"symbol": "q_{LL}", "description": "Construction live load", "value": float(inputs["construction_live_load_psf"]), "units": "psf", "source": "input:construction_live_load_psf"},
        ],
        compute_fn=lambda: q_tot,
        units="psf",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 1},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L2"}],
    )

    compute_step(
        trace,
        id="L3",
        section="Loads",
        title="Distributed shear at one bracket line",
        output_symbol="V_{dist}",
        output_description="Distributed shear from area loads",
        equation_latex=r"V_{dist} = \frac{q_{tot}\,a\,s}{1000}",
        variables=[
            {"symbol": "q_{tot}", "description": "Total area load", "value": q_tot, "units": "psf", "source": "step:L2"},
            {"symbol": "a", "description": "Overhang length", "value": a_ft, "units": "ft", "source": "input:overhang_length_ft"},
            {"symbol": "s", "description": "Bracket spacing", "value": spacing_ft, "units": "ft", "source": "input:spacing_ft"},
        ],
        compute_fn=lambda: (q_tot * a_ft * spacing_ft) / 1000.0,
        units="kip",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L3"}],
    )

    compute_step(
        trace,
        id="L4",
        section="Loads",
        title="Distributed moment about girder line",
        output_symbol="M_{dist}",
        output_description="Distributed moment from area loads",
        equation_latex=r"M_{dist} = \frac{q_{tot}\,a^{2}\,s}{2\cdot 1000}",
        variables=[
            {"symbol": "q_{tot}", "description": "Total area load", "value": q_tot, "units": "psf", "source": "step:L2"},
            {"symbol": "a", "description": "Overhang length", "value": a_ft, "units": "ft", "source": "input:overhang_length_ft"},
            {"symbol": "s", "description": "Bracket spacing", "value": spacing_ft, "units": "ft", "source": "input:spacing_ft"},
        ],
        compute_fn=lambda: (q_tot * (a_ft**2) * spacing_ft) / (2.0 * 1000.0),
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L4"}],
    )

    compute_step(
        trace,
        id="L5",
        section="Loads",
        title="Screed wheel shear and moment",
        output_symbol=r"(V_{wheel}, M_{wheel})",
        output_description="Concentrated screed wheel effects at deck edge + 3 in",
        equation_latex=r"V_{wheel} = P,\;\;\; M_{wheel} = P\,a_{out}",
        variables=[
            {"symbol": "P", "description": "Screed wheel load", "value": float(inputs["screed_wheel_load_kip"]), "units": "kip", "source": "input:screed_wheel_load_kip"},
            {"symbol": "a_{out}", "description": "Overhang to load (deck edge + 3 in)", "value": a_out_ft, "units": "ft", "source": "constant:EDGE_LOAD_OUTBOARD_IN"},
        ],
        compute_fn=lambda: float(inputs["screed_wheel_load_kip"]),
        units="kip",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L5"}],
    )

    compute_step(
        trace,
        id="L6",
        section="Loads",
        title="Edge line load at one bracket line",
        output_symbol="V_{edge}",
        output_description="Line load along deck edge",
        equation_latex=r"V_{edge} = \dfrac{w_{edge}\,s}{1000}",
        variables=[
            {"symbol": "w_{edge}", "description": "Edge line load", "value": EDGE_LINE_LOAD_PLF, "units": "plf", "source": "constant:EDGE_LINE_LOAD_PLF"},
            {"symbol": "s", "description": "Bracket spacing", "value": spacing_ft, "units": "ft", "source": "input:spacing_ft"},
        ],
        compute_fn=lambda: V_edge,
        units="kip",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L6"}],
    )

    compute_step(
        trace,
        id="L7",
        section="Loads",
        title="Edge line load moment about girder line",
        output_symbol="M_{edge}",
        output_description="Moment from edge line load",
        equation_latex=r"M_{edge} = V_{edge}\,a_{out}",
        variables=[
            {"symbol": "V_{edge}", "description": "Edge line load per bracket", "value": V_edge, "units": "kip", "source": "step:L6"},
            {"symbol": "a_{out}", "description": "Overhang to load (deck edge + 3 in)", "value": a_out_ft, "units": "ft", "source": "constant:EDGE_LOAD_OUTBOARD_IN"},
        ],
        compute_fn=lambda: V_edge * a_out_ft,
        units="kip-ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "evaluation._compute_loads:L7"}],
    )

    # Geometry
    if placement.feasible and bool(inputs.get("use_geometry_based_diagonal_angle", True)):
        sin_theta = placement.sin_theta
        depth_ft = placement.depth_ft
        dx_ft = placement.dx_ft
        geom_note = "placement"
    else:
        sin_theta = SIN_45
        depth_ft = float(inputs.get("bracket_lever_arm_ft", 3.0))
        dx_ft = float("nan")
        geom_note = "fallback"

    compute_step(
        trace,
        id="G1",
        section="Geometry",
        title="Diagonal angle input to demand model",
        output_symbol=r"\sin(\theta)",
        output_description="Sine of diagonal angle",
        equation_latex=r"\sin(\theta) = \sin(\theta)_{source}",
        variables=[
            {"symbol": r"\sin(\theta)_{source}", "description": f"Angle source ({geom_note})", "value": sin_theta, "units": "-", "source": f"derived:{geom_note}"},
        ],
        compute_fn=lambda: sin_theta,
        units="-",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "evaluation.evaluate_with_trace:G1"}],
    )

    # Demands and checks
    F_h, F_d = _compute_member_demands(V, M, sin_theta=sin_theta, depth_ft=depth_ft)

    hanger_cap = float(inputs["hanger_swl_kip"])
    diag_cap = float(inputs["diagonal_swl_kip"])

    def _checks(_x: float):
        return [
            {"label": "Hanger SWL", "demand": F_h, "capacity": hanger_cap, "ratio": F_h / hanger_cap, "pass_fail": "PASS" if F_h <= hanger_cap else "FAIL"},
            {"label": "Diagonal SWL", "demand": F_d, "capacity": diag_cap, "ratio": F_d / diag_cap, "pass_fail": "PASS" if F_d <= diag_cap else "FAIL"},
        ]

    compute_step(
        trace,
        id="D1",
        section="Demands",
        title="Member axial demands and SWL checks",
        output_symbol=r"(F_h, F_d)",
        output_description="Hanger and diagonal axial demands",
        equation_latex=r"F_h = \frac{V}{\sin(45^\circ)},\;\;\; F_d = \frac{V + M/d}{\sin(\theta)}",
        variables=[
            {"symbol": "V", "description": "Total shear", "value": V, "units": "kip", "source": "derived:loads"},
            {"symbol": "M", "description": "Total moment", "value": M, "units": "kip-ft", "source": "derived:loads"},
            {"symbol": "d", "description": "Lever arm", "value": depth_ft, "units": "ft", "source": "derived:geometry"},
            {"symbol": r"\sin(\theta)", "description": "Diagonal angle sine", "value": sin_theta, "units": "-", "source": "step:G1"},
        ],
        compute_fn=lambda: F_d,  # store diagonal as representative numeric
        units="kip",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 3},
        references=[{"type": "derived", "ref": "evaluation._compute_member_demands:D1"}],
        checks_builder=_checks,
    )

    util_h = F_h / hanger_cap
    util_d = F_d / diag_cap
    passes = (F_h <= hanger_cap) and (F_d <= diag_cap)
    governs = "diagonal" if util_d >= util_h else "hanger"

    return EvalResult(
        spacing_ft=float(spacing_ft),
        V_total_kip=float(V),
        M_total_kipft=float(M),
        depth_ft=float(depth_ft),
        dx_ft=float(dx_ft),
        sin_theta=float(sin_theta),
        hanger_demand_kip=float(F_h),
        diagonal_demand_kip=float(F_d),
        util_hanger=float(util_h),
        util_diagonal=float(util_d),
        governs=governs,
        passes=bool(passes),
    )
