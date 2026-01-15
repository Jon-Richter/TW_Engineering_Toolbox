from __future__ import annotations

from typing import Dict, List

import math
from .steel_aisc_models import InputModel
from .steel_aisc_common import safe_div
from .aisc_shapes_db import Shape


def _principal_moments(shape: Shape, Mx: float, My: float, warnings: List[str]) -> tuple[float, float, float]:
    tan_theta = shape.tan_theta
    if tan_theta is None:
        warnings.append("Principal-axis angle not found; assuming theta=0 (w aligns with x).")
        theta = 0.0
    else:
        theta = math.atan(tan_theta)
    c = math.cos(theta)
    s = math.sin(theta)
    Mw = Mx * c + My * s
    Mz = -Mx * s + My * c
    return Mw, Mz, theta


def _interaction_single_angle(
    model: InputModel,
    shape: Shape,
    flex: Dict[str, float | str],
    ten: Dict[str, float | str],
    comp: Dict[str, float | str],
    warnings: List[str],
) -> Dict[str, float | str | list[dict[str, float | str]]]:
    """
    AISC 360-16 H2-1 (asymmetric sections) using stresses at heel/toes.
    Uses principal-axis moments and section moduli Sw/Sz at points A/B/C:
      f = P/A + Mw/Sw + Mz/Sz
      (fa/Fa) + (fbw/Fbw) + (fbz/Fbz) <= 1.0
    """
    Mw, Mz, theta = _principal_moments(shape, model.Mux_kft, model.Muy_kft, warnings)
    A = shape.A_in2
    if A <= 0:
        return {"case": "single-angle", "equation": "AISC 360-16 H2-1", "unity": math.inf, "status": "NG"}

    fa = model.Pu_k / A  # k/in^2 = ksi (compression +)
    if model.Pu_k >= 0.0:
        Pc = float(comp.get("P_design_k", 0.0))
        Fa = safe_div(Pc, A) if Pc > 0 else 0.0
    else:
        Rt = float(ten.get("R_design_k", 0.0))
        Fa = safe_div(Rt, A) if Rt > 0 else 0.0

    Md_w_kft = float(flex.get("M_design_x_kft", 0.0) or 0.0)
    Md_z_kft = float(flex.get("M_design_y_kft", 0.0) or 0.0)

    points = [
        ("A", shape.SwA_in3, shape.SzA_in3, shape.wA_in, shape.zA_in),
        ("B", shape.SwB_in3, shape.SzB_in3, shape.wB_in, shape.zB_in),
        ("C", shape.SwC_in3, shape.SzC_in3, shape.wC_in, shape.zC_in),
    ]
    point_results: list[dict[str, float | str]] = []
    max_u = 0.0
    for name, Sw, Sz, w, z in points:
        if not Sw or not Sz:
            continue
        fbw = safe_div(Mw * 12.0, Sw)  # kip-ft -> kip-in / in^3 = ksi
        fbz = safe_div(Mz * 12.0, Sz)
        Fbw = safe_div(Md_w_kft * 12.0, Sw) if Md_w_kft > 0.0 else 0.0
        Fbz = safe_div(Md_z_kft * 12.0, Sz) if Md_z_kft > 0.0 else 0.0
        u = 0.0
        if Fa > 0:
            u += abs(fa) / Fa
        if Fbw > 0:
            u += abs(fbw) / Fbw
        if Fbz > 0:
            u += abs(fbz) / Fbz
        sigma = fa + fbw + fbz
        max_u = max(max_u, u)
        point_results.append(
            {
                "point": name,
                "w_in": w,
                "z_in": z,
                "Sw_in3": Sw,
                "Sz_in3": Sz,
                "fa_ksi": fa,
                "fbw_ksi": fbw,
                "fbz_ksi": fbz,
                "Fbw_ksi": Fbw,
                "Fbz_ksi": Fbz,
                "sigma_ksi": sigma,
                "unity": u,
                "status": "OK" if u <= 1.0 + 1e-9 else "NG",
            }
        )

    if not point_results:
        warnings.append("Missing Sw/Sz values for single angle; H2-1 points could not be evaluated.")
        max_u = math.inf
    if Fa <= 0.0 or Md_w_kft <= 0.0 or Md_z_kft <= 0.0:
        warnings.append("Single-angle H2-1 uses design stresses based on Md_w/Md_z and axial capacity; check zero/blank capacities.")
    status = "OK" if max_u <= 1.0 + 1e-9 else "NG"
    return {
        "case": "single-angle",
        "equation": "AISC 360-16 H2-1",
        "theta_deg": math.degrees(theta),
        "Mw_kft": Mw,
        "Mz_kft": Mz,
        "Fa_ksi": Fa,
        "Fbw_ref_ksi": safe_div(Md_w_kft * 12.0, min([v for v in [shape.SwA_in3, shape.SwB_in3, shape.SwC_in3] if v] or [0.0])),
        "Fbz_ref_ksi": safe_div(Md_z_kft * 12.0, min([v for v in [shape.SzA_in3, shape.SzB_in3, shape.SzC_in3] if v] or [0.0])),
        "unity": max_u,
        "status": status,
        "points": point_results,
    }


def design_interaction(
    model: InputModel,
    shape: Shape,
    flex: Dict[str, float | str],
    ten: Dict[str, float | str],
    comp: Dict[str, float | str],
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | list[dict[str, float | str]]]:
    """
    Combined axial + flexure (AISC Chapter H, foundational implementation):

    Compression (Pu >= 0):
      if Pu/Pc <= 0.2:
        U = Pu/Pc + (8/9) * (Mux/Mcx + Muy/Mcy)
      else:
        U = Pu/(2Pc) + (Mux/Mcx + Muy/Mcy)

    Tension (Pu < 0):
      U = Pt/Pt_cap + Mux/Mcx + Muy/Mcy

    Notes:
    - This is a simplified H1-style interaction intended for fast screening.
    - Second-order effects (P-Δ, P-δ), B1/B2, and advanced analysis are out-of-scope.
    """
    tcode = shape.type_code.strip().upper()
    if tcode == "L":
        inter = _interaction_single_angle(model, shape, flex, ten, comp, warnings)
        trace.append(f"Interaction (single angle): U={inter.get('unity')}")
        return inter

    Mu_x = float(flex["Mux_kft"])
    Mu_y = float(flex["Muy_kft"])
    Mcx = float(flex["M_design_x_kft"])
    Mcy = float(flex["M_design_y_kft"])

    # Protect against missing capacities
    mx = safe_div(Mu_x, Mcx) if Mcx > 0 else 0.0
    my = safe_div(Mu_y, Mcy) if Mcy > 0 else 0.0

    if model.Pu_k >= 0.0:
        Pu = float(comp["Pu_compression_k"])
        Pc = float(comp["P_design_k"])
        pr = safe_div(Pu, Pc) if Pc > 0 else 0.0
        if pr <= 0.2:
            unity = pr + (8.0 / 9.0) * (mx + my)
            eqn = "H1 (foundational): Pu/Pc + (8/9)(Mx/Mcx + My/Mcy)"
        else:
            unity = pr / 2.0 + (mx + my)
            eqn = "H1 (foundational): Pu/(2Pc) + (Mx/Mcx + My/Mcy)"
        trace.append(f"Interaction (compression): pr={pr:.3f}, mx={mx:.3f}, my={my:.3f} -> U={unity:.3f}")
        return {
            "case": "compression+flexure",
            "equation": eqn,
            "Pu_k": Pu,
            "Pc_k": Pc,
            "mx": mx,
            "my": my,
            "unity": unity,
            "status": "OK" if unity <= 1.0 + 1e-9 else "NG",
        }

    # Tension interaction
    Pt = float(ten["Pu_tension_k"])
    Pt_cap = float(ten["R_design_k"])
    tr = safe_div(Pt, Pt_cap) if Pt_cap > 0 else 0.0
    unity = tr + mx + my
    trace.append(f"Interaction (tension): tr={tr:.3f}, mx={mx:.3f}, my={my:.3f} -> U={unity:.3f}")
    warnings.append("Tension+flexure interaction uses simplified linear unity in this version (AISC H provisions not fully implemented).")

    return {
        "case": "tension+flexure",
        "equation": "Simplified: Pt/Pt_cap + Mx/Mcx + My/Mcy",
        "Pt_k": Pt,
        "Pt_cap_k": Pt_cap,
        "mx": mx,
        "my": my,
        "unity": unity,
        "status": "OK" if unity <= 1.0 + 1e-9 else "NG",
    }
