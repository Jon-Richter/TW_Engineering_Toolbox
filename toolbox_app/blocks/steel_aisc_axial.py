from __future__ import annotations

import math
from typing import List, Dict

from .steel_aisc_models import InputModel, Material
from .aisc_shapes_db import Shape
from .steel_aisc_common import (
    factors_tension_yield,
    factors_tension_rupture,
    factors_compression,
    design_strength,
    ft_to_in,
    safe_div,
)


def _lambda_flange_I(shape: Shape) -> float | None:
    if shape.bf_2tf:
        return float(shape.bf_2tf)
    if shape.bf_in and shape.tf_in:
        return float(shape.bf_in) / (2.0 * float(shape.tf_in))
    return None


def _lambda_web_I(shape: Shape) -> float | None:
    if shape.h_tw:
        return float(shape.h_tw)
    if shape.h_in and shape.tw_in:
        return float(shape.h_in) / float(shape.tw_in)
    if shape.d_in and shape.tw_in and shape.tf_in:
        h = float(shape.d_in) - 2.0 * float(shape.tf_in)
        if h > 0.0:
            return h / float(shape.tw_in)
    return None


def _fcr_from_fe(fe: float, fy: float) -> float:
    if fe <= 0.0:
        return 0.0
    # AISC E3-2/E3-3 (using Fe from E4 when applicable)
    if fy / fe <= 2.25:
        return (0.658 ** (fy / fe)) * fy
    return 0.877 * fe


def _e4_fe(
    shape: Shape,
    fy: float,
    e: float,
    lcx: float,
    lcy: float,
    lcz: float,
    warnings: List[str],
) -> tuple[float | None, Dict[str, float | str]]:
    """
    AISC E4 elastic buckling stress Fe for torsional/flexural-torsional buckling.
    Returns Fe and a detail dict. Uses available properties from the AISC DB.
    """
    details: Dict[str, float | str] = {
        "Fex_ksi": 0.0,
        "Fey_ksi": 0.0,
        "Fez_ksi": 0.0,
        "Fe_ksi": 0.0,
        "equation": "",
    }

    if lcx <= 0.0 and lcy <= 0.0:
        warnings.append("E4 requires effective lengths; Lx/Ly are zero.")
        return None, details

    def _r0_calc() -> float | None:
        if shape.A_in2 <= 0.0:
            return None
        tcode = shape.type_code.strip().upper()
        symmetric = tcode in {"W", "S", "M", "HP", "HSS", "PIPE"}
        x0 = shape.x_in
        y0 = shape.y_in
        if x0 is None and tcode in {"C", "MC"} and shape.eo_in is not None:
            x0 = shape.eo_in
        if x0 is None and symmetric:
            x0 = 0.0
        if y0 is None and (symmetric or tcode in {"C", "MC", "WT", "MT", "ST"}):
            y0 = 0.0
        if x0 is None or y0 is None:
            return None
        return math.sqrt(
            float(x0) ** 2
            + float(y0) ** 2
            + (shape.Ix_in4 + shape.Iy_in4) / shape.A_in2
        )

    rx = shape.rx_in or 0.0
    ry = shape.ry_in or 0.0
    if rx <= 0.0 or ry <= 0.0:
        warnings.append("E4 requires rx/ry; missing -> skipping torsional buckling.")
        return None, details

    fex = (math.pi**2 * e) / (lcx / rx) ** 2 if lcx > 0 else 0.0
    fey = (math.pi**2 * e) / (lcy / ry) ** 2 if lcy > 0 else 0.0
    details["Fex_ksi"] = fex
    details["Fey_ksi"] = fey

    r0_calc = _r0_calc()
    missing: List[str] = []
    if not shape.J_in4:
        missing.append("J")
    if r0_calc is None:
        missing.append("ro (calc)")
    if lcz <= 0.0:
        missing.append("Lz")
    if missing:
        warnings.append(f"E4 requires {', '.join(missing)}; missing -> skipping torsional buckling.")
        return None, details

    cw = shape.Cw_in6 or 0.0
    g = e / (2.0 * (1.0 + 0.3))
    details["ro_in"] = r0_calc
    fez = ((math.pi**2 * e * cw) / (lcz**2) + g * shape.J_in4) / (shape.A_in2 * r0_calc**2)
    details["Fez_ksi"] = fez

    tcode = shape.type_code.strip().upper()
    if tcode in {"W", "S", "M", "HP"}:
        # E4-2: doubly symmetric members twisting about shear center
        denom = shape.Ix_in4 + shape.Iy_in4
        if denom <= 0.0:
            warnings.append("E4-2 requires Ix+Iy; missing -> skipping torsional buckling.")
            return None, details
        fe = ((math.pi**2 * e * cw) / (lcz**2) + g * shape.J_in4) / denom
        details["Fe_ksi"] = fe
        details["equation"] = "E4-2"
        return fe, details

    h = shape.H_const
    if not h or h <= 0.0:
        warnings.append("E4 requires H (flexural constant); missing -> skipping torsional buckling.")
        return None, details

    # E4-3 for singly symmetric shapes. For channels (x-axis symmetry), use Fex per note.
    fe_sym = fex if tcode in {"C", "MC"} else fey
    if fe_sym <= 0.0 or fez <= 0.0:
        warnings.append("E4 requires positive Fe terms; skipping torsional buckling.")
        return None, details
    ratio = 4.0 * fe_sym * fez * h / (fe_sym + fez) ** 2
    ratio = min(max(ratio, 0.0), 1.0)
    fe = (fe_sym + fez) / (2.0 * h) * (1.0 - math.sqrt(1.0 - ratio))
    details["Fe_ksi"] = fe
    details["equation"] = "E4-3"

    # Unsymmetric sections ideally use E4-4 (cubic); use E4-3 as approximation.
    if tcode in {"L", "2L"}:
        warnings.append("E4-4 (unsymmetric cubic) not implemented; using E4-3 approximation.")
    return fe, details


def _effective_width_be(
    b: float,
    lam: float,
    lam_r: float,
    fy: float,
    fcr: float,
    c1: float,
    c2: float,
) -> float:
    if fcr <= 0.0:
        return b
    limit = lam_r * math.sqrt(fy / fcr)
    if lam <= limit:
        return b
    fel = (c2 * lam_r / lam) ** 2 * fy
    ratio = math.sqrt(max(fel / fcr, 0.0))
    ratio = min(max(ratio, 0.0), 1.0)
    be = b * (1.0 - c1 * ratio) * ratio
    return max(0.0, min(be, b))


def _effective_area_e7(
    shape: Shape,
    fy: float,
    e: float,
    fcr: float,
    warnings: List[str],
) -> tuple[float, List[Dict[str, float | str]]]:
    details: List[Dict[str, float | str]] = []
    ag = shape.A_in2
    ae = ag

    tcode = shape.type_code.strip().upper()
    lam_r_flange = 0.56 * math.sqrt(e / fy)
    lam_r_web = 1.49 * math.sqrt(e / fy)
    lam_r_angle = 0.45 * math.sqrt(e / fy)
    lam_r_tee = 0.75 * math.sqrt(e / fy)
    lam_r_hss = 1.40 * math.sqrt(e / fy)

    def add_element(name: str, b: float, t: float, lam: float, lam_r: float, c1: float, c2: float, count: int) -> None:
        nonlocal ae
        if lam <= lam_r:
            return
        be = _effective_width_be(b, lam, lam_r, fy, fcr, c1, c2)
        reduction = max(0.0, (b - be) * t * count)
        ae = max(0.0, ae - reduction)
        details.append(
            {
                "element": name,
                "b_in": b,
                "t_in": t,
                "lambda": lam,
                "lambda_r": lam_r,
                "c1": c1,
                "c2": c2,
                "be_in": be,
                "reduction_in2": reduction,
            }
        )

    if tcode in {"W", "S", "M", "HP", "C", "MC"}:
        bf = shape.bf_in
        tf = shape.tf_in
        tw = shape.tw_in
        d = shape.d_in
        if bf and tf:
            lam = float(bf) / (2.0 * float(tf))
            add_element("flange", float(bf) / 2.0, float(tf), lam, lam_r_flange, 0.22, 1.49, 2)
        if tw and d and tf:
            h = float(d) - 2.0 * float(tf)
            if h > 0.0:
                lam = h / float(tw)
                add_element("web", h, float(tw), lam, lam_r_web, 0.18, 1.31, 1)
    elif tcode in {"WT", "MT", "ST"}:
        bf = shape.bf_in
        tf = shape.tf_in
        tw = shape.tw_in
        d = shape.d_in
        if bf and tf:
            lam = float(bf) / (2.0 * float(tf))
            add_element("tee flange", float(bf) / 2.0, float(tf), lam, lam_r_flange, 0.22, 1.49, 2)
        if tw and d:
            lam = float(d) / float(tw)
            add_element("tee stem", float(d), float(tw), lam, lam_r_tee, 0.22, 1.49, 1)
    elif tcode == "HSS":
        t = shape.t_des_in or shape.t_nom_in
        if shape.OD_in and t:
            # Round HSS uses direct Ae formulas.
            d_t = float(shape.OD_in) / float(t)
            if d_t <= 0.11 * e / fy:
                ae = ag
            else:
                ae = (0.038 * e / (fy * d_t) + (2.0 / 3.0)) * ag
                ae = min(ae, ag)
            details.append(
                {"element": "round wall", "D_over_t": d_t, "Ae_in2": ae, "case": "E7-6/E7-7"}
            )
        elif t and (shape.B_in or shape.b_in) and (shape.H_in or shape.h_in):
            b = float(shape.B_in or shape.b_in)
            h = float(shape.H_in or shape.h_in)
            b_clear = max(b - 3.0 * float(t), 0.0)
            h_clear = max(h - 3.0 * float(t), 0.0)
            if b_clear > 0.0:
                lam = b_clear / float(t)
                add_element("HSS wall (b)", b_clear, float(t), lam, lam_r_hss, 0.20, 1.38, 2)
            if h_clear > 0.0:
                lam = h_clear / float(t)
                add_element("HSS wall (h)", h_clear, float(t), lam, lam_r_hss, 0.20, 1.38, 2)
    elif tcode in {"L", "2L"}:
        b = shape.b_in or shape.B_in
        d = shape.d_in
        t = shape.t_in or shape.t_des_in or shape.t_nom_in
        if b and t:
            lam = float(b) / float(t)
            count = 2 if tcode == "2L" else 1
            add_element("angle leg (b)", float(b), float(t), lam, lam_r_angle, 0.22, 1.49, count)
        if d and t:
            lam = float(d) / float(t)
            count = 2 if tcode == "2L" else 1
            add_element("angle leg (d)", float(d), float(t), lam, lam_r_angle, 0.22, 1.49, count)

    if ae < ag:
        warnings.append("E7 effective area reductions applied for slender elements.")
    return ae, details


def design_tension(model: InputModel, shape: Shape, mat: Material, trace: List[str], warnings: List[str]) -> Dict[str, float | str]:
    """
    AISC Chapter D (simplified):
    - Yielding: Rn = Fy * Ag
    - Rupture:  Rn = Fu * Ae (Ae assumed equal to Ag in this tool)
    """
    Fy, Fu = mat.Fy_ksi, mat.Fu_ksi
    Ag = shape.A_in2
    Ae = Ag  # Simplified (no holes)

    Rn_y = Fy * Ag
    Rn_u = Fu * Ae

    fy = factors_tension_yield(model.design_method)
    fu = factors_tension_rupture(model.design_method)

    Ry = design_strength(model.design_method, Rn_y, fy.phi, fy.omega)
    Ru = design_strength(model.design_method, Rn_u, fu.phi, fu.omega)

    R = min(Ry, Ru)
    trace.append(f"Tension: Rn_y=Fy*Ag={Fy:.3f}*{Ag:.3f}={Rn_y:.3f} k")
    trace.append(f"Tension: Rn_u=Fu*Ae={Fu:.3f}*{Ae:.3f}={Rn_u:.3f} k")
    trace.append(f"Tension design strengths: Ry={Ry:.3f} k, Ru={Ru:.3f} k -> governs {R:.3f} k")

    Pu_t = max(0.0, -model.Pu_k)  # tension demand uses negative Pu
    ratio = safe_div(Pu_t, R) if Pu_t > 0 else 0.0

    return {
        "Pu_tension_k": Pu_t,
        "Rn_yield_k": Rn_y,
        "Rn_rupture_k": Rn_u,
        "R_design_k": R,
        "unity": ratio,
        "status": "OK" if ratio <= 1.0 + 1e-9 else "NG",
    }


def design_compression(model: InputModel, shape: Shape, mat: Material, trace: List[str], warnings: List[str]) -> Dict[str, float | str]:
    """
    AISC Chapter E:
    - Flexural buckling about principal axes (E3)
    - Torsional / flexural-torsional buckling (E4) when properties are available
    - Effective area reductions for slender elements (E7)
    """
    Fy, E = mat.Fy_ksi, mat.E_ksi
    Ag = shape.A_in2

    # Lengths
    Lx = ft_to_in(model.Lx_ft)
    Ly = ft_to_in(model.Ly_ft)
    Lz = ft_to_in(model.Lz_ft)
    Kx, Ky = model.Kx, model.Ky
    Kz = model.Kz

    tcode = shape.type_code.strip().upper()
    # Slenderness ratios (principal axes)
    if tcode == "L":
        rw = math.sqrt(shape.Iw_in4 / Ag) if shape.Iw_in4 else shape.rx_in
        rz = math.sqrt(shape.Iz_in4 / Ag) if shape.Iz_in4 else shape.ry_in
        klrx = safe_div(Kx * Lx, rw) if Lx > 0 else 0.0
        klry = safe_div(Ky * Ly, rz) if Ly > 0 else 0.0
        trace.append("Compression (angle): using principal-axis r_w and r_z from Iw/Iz.")
    else:
        klrx = safe_div(Kx * Lx, shape.rx_in) if Lx > 0 else 0.0
        klry = safe_div(Ky * Ly, shape.ry_in) if Ly > 0 else 0.0

    def Fcr_from_klr(klr: float) -> float:
        if klr <= 0.0:
            return Fy
        Fe = (math.pi**2 * E) / (klr**2)
        lam = klr
        lam_r = 4.71 * math.sqrt(E / Fy)
        if lam <= lam_r:
            return (0.658 ** (Fy / Fe)) * Fy
        return 0.877 * Fe

    Fcrx = Fcr_from_klr(klrx) if klrx > 0 else Fy
    Fcry = Fcr_from_klr(klry) if klry > 0 else Fy
    Fcr_flex = min(Fcrx, Fcry)

    # E5 single-angle effective slenderness (default to long-leg connected, planar truss case)
    if tcode == "L":
        b = shape.b_in or shape.B_in
        d = shape.d_in
        ra = max(shape.rx_in, shape.ry_in)
        rz = math.sqrt(shape.Iz_in4 / Ag) if shape.Iz_in4 else shape.ry_in
        L = max(Lx, Ly)
        if b and d and ra and rz and L > 0.0:
            b_long = max(float(b), float(d))
            b_short = min(float(b), float(d))
            if b_short > 0.0 and (b_long / b_short) < 1.7:
                l_over_ra = L / ra
                if l_over_ra <= 80.0:
                    lc_over_r = 72.0 + 0.75 * l_over_ra
                else:
                    lc_over_r = 32.0 + 1.25 * l_over_ra
                lc_over_r = max(lc_over_r, 0.95 * (L / rz))
                fcr_e5 = Fcr_from_klr(lc_over_r)
                Fcr_flex = min(Fcr_flex, fcr_e5)
                trace.append(f"Compression (E5): L/ra={l_over_ra:.3f}, Lc/r={lc_over_r:.3f}, Fcr_e5={fcr_e5:.3f} ksi")
            else:
                warnings.append("E5 single-angle provisions not applicable (leg ratio >= 1.7); using E3.")
        else:
            warnings.append("E5 single-angle inputs missing (b, d, ra, rz, L); using E3.")

    # E4 torsional / flexural-torsional buckling (when applicable)
    Lcx = Kx * Lx
    Lcy = Ky * Ly
    Lcz = Kz * Lz
    Fe_torsion, e4_details = _e4_fe(shape, Fy, E, Lcx, Lcy, Lcz, warnings)
    Fcr_torsion = _fcr_from_fe(Fe_torsion, Fy) if Fe_torsion else None

    Fcr = Fcr_flex
    if Fcr_torsion and Fcr_torsion > 0.0:
        Fcr = min(Fcr, Fcr_torsion)
        trace.append(f"Compression (E4): Fe={Fe_torsion:.3f} ksi -> Fcr_torsion={Fcr_torsion:.3f} ksi")

    # Slender elements: apply E7 effective area reductions
    Ae, e7_details = _effective_area_e7(shape, Fy, E, Fcr, warnings)
    slender_elements = [d.get("element", "") for d in e7_details if d.get("element")]

    Pn = Fcr * Ae
    fc = factors_compression(model.design_method)
    Pc = design_strength(model.design_method, Pn, fc.phi, fc.omega)

    trace.append(f"Compression: KL/rx={klrx:.3f}, KL/ry={klry:.3f}")
    trace.append(f"Compression: Fcrx={Fcrx:.3f} ksi, Fcry={Fcry:.3f} ksi -> Fcr={Fcr:.3f} ksi")
    trace.append(f"Compression: Ae={Ae:.3f} in^2 (Ag={Ag:.3f} in^2)")
    trace.append(f"Compression: Pn=Fcr*Ae={Fcr:.3f}*{Ae:.3f}={Pn:.3f} k; Pc={Pc:.3f} k")

    Pu_c = max(0.0, model.Pu_k)
    ratio = safe_div(Pu_c, Pc) if Pu_c > 0 else 0.0

    return {
        "Pu_compression_k": Pu_c,
        "KLrx": klrx,
        "KLry": klry,
        "slender_elements": slender_elements,
        "Ae_in2": Ae,
        "E4_details": e4_details,
        "E7_details": e7_details,
        "Fcr_ksi": Fcr,
        "Pn_k": Pn,
        "P_design_k": Pc,
        "unity": ratio,
        "status": "OK" if ratio <= 1.0 + 1e-9 else "NG",
    }
