from __future__ import annotations

from typing import Dict, List

from .models import InputModel, Material
from toolbox_app.blocks.aisc_shapes_db import Shape
from .design_common import factors_shear, design_strength, safe_div


def _h_tw(shape: Shape) -> float | None:
    if shape.h_tw:
        return float(shape.h_tw)
    if shape.h_in and shape.tw_in:
        return float(shape.h_in) / float(shape.tw_in)
    if shape.d_in and shape.tf_in and shape.tw_in:
        h = float(shape.d_in) - 2.0 * float(shape.tf_in)
        if h > 0.0:
            return h / float(shape.tw_in)
    return None


def _cv_from_h_tw(h_tw: float | None, Fy: float, E: float, kv: float) -> float:
    if not h_tw or h_tw <= 0.0:
        return 1.0
    limit = 1.10 * (kv * E / Fy) ** 0.5
    if h_tw <= limit:
        return 1.0
    return limit / h_tw


def _shear_area_I_major(shape: Shape) -> float:
    # Aw = h * tw (clear distance between flanges times web thickness)
    if shape.h_in and shape.tw_in:
        return shape.h_in * shape.tw_in
    if shape.h_tw and shape.tw_in:
        return shape.h_tw * shape.tw_in * shape.tw_in
    if shape.d_in and shape.tw_in and shape.tf_in:
        h = shape.d_in - 2.0 * shape.tf_in
        if h > 0:
            return h * shape.tw_in
    return shape.A_in2  # last-resort fallback (warn elsewhere)


def _shear_area_HSS_rect(shape: Shape, axis: str) -> float:
    # Very common approximation for rectangular HSS:
    # Avx ~ 2*t*H (shear parallel to B, webs are depth H)
    # Avy ~ 2*t*B
    t = shape.t_des_in or shape.t_nom_in
    H = shape.h_in or shape.H_in
    B = shape.b_in or shape.B_in
    if not (t and B and H):
        return shape.A_in2
    return 2.0 * t * (H if axis == "x" else B)


def _shear_area_round(shape: Shape) -> float:
    # Conservative/simple fallback for round HSS/pipe without full Av expression:
    # Use Av ~ 0.9*A as a practical approximation (flag in warnings)
    return 0.9 * shape.A_in2


def design_shear(model: InputModel, shape: Shape, mat: Material, trace: List[str], warnings: List[str]) -> Dict[str, float | str]:
    """
    AISC Chapter G: Vn = 0.6 Fy Av Cv using web slenderness-based Cv (no tension field action).
    """
    Fy = mat.Fy_ksi
    E = mat.E_ksi
    fs = factors_shear(model.design_method)

    tcode = shape.type_code.strip().upper()
    Cv_x = 1.0
    Cv_y = 1.0

    # Shear areas + Cv per AISC Chapter G
    if tcode in {"W", "S", "M", "HP", "C", "MC", "WT", "ST", "MT"}:
        Avx = _shear_area_I_major(shape)
        Avy = Avx
        kv = 5.34
        h_tw = _h_tw(shape)
        if tcode in {"W", "S", "M", "HP"} and h_tw is not None:
            limit = 2.24 * (E / Fy) ** 0.5
            Cv = 1.0 if h_tw <= limit else _cv_from_h_tw(h_tw, Fy, E, kv)
        else:
            Cv = _cv_from_h_tw(h_tw, Fy, E, kv)
        Cv_x = Cv_y = Cv
        Vn_x = 0.6 * Fy * Avx * Cv
        Vn_y = 0.6 * Fy * Avy * Cv
        trace.append(f"Shear Cv (G2): h/tw={h_tw}, Cv={Cv:.3f}, kv={kv}")
        if Avx == shape.A_in2:
            warnings.append("Shear area fallback used (Aw ~ A). Provide h and tw in DB for better Aw.")
    elif tcode == "HSS":
        if shape.OD_in:
            Avx = Avy = _shear_area_round(shape)
            Cv = 1.0
            Cv_x = Cv_y = Cv
            Vn_x = 0.6 * Fy * Avx * Cv
            Vn_y = 0.6 * Fy * Avy * Cv
            warnings.append("Round HSS shear uses Av≈0.9A and Cv=1.0 (G5/G2 approximation).")
        else:
            Avx = _shear_area_HSS_rect(shape, "x")
            Avy = _shear_area_HSS_rect(shape, "y")
            kv = 5.0
            t = shape.t_des_in or shape.t_nom_in
            H = shape.h_in or (shape.H_in - 2.0 * t if shape.H_in and t else None)
            B = shape.b_in or (shape.B_in - 2.0 * t if shape.B_in and t else None)
            h_tw_x = (H / t) if (H and t) else None
            h_tw_y = (B / t) if (B and t) else None
            Cvx = _cv_from_h_tw(h_tw_x, Fy, E, kv)
            Cvy = _cv_from_h_tw(h_tw_y, Fy, E, kv)
            Cv_x = Cvx
            Cv_y = Cvy
            Vn_x = 0.6 * Fy * Avx * Cvx
            Vn_y = 0.6 * Fy * Avy * Cvy
            trace.append(f"Shear Cv (G4): h/t (x)={h_tw_x}, Cvx={Cvx:.3f}, kv={kv}")
            trace.append(f"Shear Cv (G4): h/t (y)={h_tw_y}, Cvy={Cvy:.3f}, kv={kv}")
            if Avx == shape.A_in2 or Avy == shape.A_in2:
                warnings.append("HSS shear area fallback used (Av ~ A). Provide B, H, t in DB for improved Av.")
    elif tcode == "PIPE":
        Avx = Avy = _shear_area_round(shape)
        Cv = 1.0
        Cv_x = Cv_y = Cv
        Vn_x = 0.6 * Fy * Avx * Cv
        Vn_y = 0.6 * Fy * Avy * Cv
        warnings.append("Pipe shear uses Av≈0.9A and Cv=1.0 approximation.")
    elif tcode in {"L", "2L"}:
        b = shape.b_in or shape.B_in or shape.d_in
        t = shape.t_in or shape.t_des_in or shape.t_nom_in
        if b and t:
            Avx = Avy = float(b) * float(t)
            kv = 1.2
            h_tw = float(b) / float(t)
            Cv = _cv_from_h_tw(h_tw, Fy, E, kv)
            Cv_x = Cv_y = Cv
            Vn_x = 0.6 * Fy * Avx * Cv
            Vn_y = 0.6 * Fy * Avy * Cv
            trace.append(f"Angle shear Cv (G3): b/t={h_tw:.3f}, Cv={Cv:.3f}, kv={kv}")
        else:
            Avx = Avy = 0.6 * shape.A_in2
            Cv_x = Cv_y = 1.0
            Vn_x = 0.6 * Fy * Avx
            Vn_y = 0.6 * Fy * Avy
            warnings.append("Angle shear uses simplified Av approximation; missing b/t.")
    else:
        Avx = Avy = shape.A_in2
        Cv_x = Cv_y = 1.0
        Vn_x = 0.6 * Fy * Avx
        Vn_y = 0.6 * Fy * Avy
        warnings.append(f"Unknown section type '{tcode}' for shear areas; using Av=A fallback.")

    Vdx = design_strength(model.design_method, Vn_x, fs.phi, fs.omega)
    Vdy = design_strength(model.design_method, Vn_y, fs.phi, fs.omega)

    Vux = abs(model.Vux_k)
    Vuy = abs(model.Vuy_k)

    ux = safe_div(Vux, Vdx) if Vux > 0 else 0.0
    uy = safe_div(Vuy, Vdy) if Vuy > 0 else 0.0

    trace.append(f"Shear: Avx={Avx:.3f} in^2, Avy={Avy:.3f} in^2")
    trace.append(f"Shear: Vn_x=0.6FyAvx={Vn_x:.3f} k; Vn_y={Vn_y:.3f} k")
    trace.append(f"Shear design: Vdx={Vdx:.3f} k, Vdy={Vdy:.3f} k")

    return {
        "Vux_k": Vux,
        "Vuy_k": Vuy,
        "Avx_in2": Avx,
        "Avy_in2": Avy,
        "Cv_x": Cv_x,
        "Cv_y": Cv_y,
        "Vn_x_k": Vn_x,
        "Vn_y_k": Vn_y,
        "Vd_x_k": Vdx,
        "Vd_y_k": Vdy,
        "unity_x": ux,
        "unity_y": uy,
        "status_x": "OK" if ux <= 1.0 + 1e-9 else "NG",
        "status_y": "OK" if uy <= 1.0 + 1e-9 else "NG",
    }
