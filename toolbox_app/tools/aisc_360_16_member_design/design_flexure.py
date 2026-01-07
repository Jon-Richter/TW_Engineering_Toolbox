from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .models import InputModel, Material
from .shapes_db import Shape
from .design_common import (
    factors_flexure,
    design_strength,
    ft_to_in,
    kipft_to_kipin,
    kipin_to_kipft,
    safe_div,
)


def _is_doubly_symmetric_I(shape: Shape) -> bool:
    return shape.type_code.strip().upper() in {"W", "S", "M", "HP"}


def _is_rectangular_hss(shape: Shape) -> bool:
    if shape.type_code.strip().upper() != "HSS":
        return False
    label = (shape.label or "").upper().replace(" ", "")
    if not label.startswith("HSS"):
        return False
    core = label[3:]
    # Round HSS typically has one "X" (HSS6.625X0.280); rectangular has two.
    return core.count("X") != 1


def _principal_moments(shape: Shape, Mx: float, My: float, warnings: List[str]) -> tuple[float, float, float]:
    """
    Rotate geometric-axis moments (Mx, My) to principal axes (w, z).
    Uses tan(theta) from the AISC DB; theta is the angle from x to w.
    """
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


def _classify_lambda(lam: float | None, lam_p: float, lam_r: float) -> Tuple[str, float | None]:
    if lam is None:
        return "unknown", None
    if lam <= lam_p + 1e-9:
        return "compact", lam
    if lam <= lam_r + 1e-9:
        return "noncompact", lam
    return "slender", lam


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


def _lambda_b_t(shape: Shape, b: float | None, t: float | None) -> float | None:
    if shape.b_t:
        return float(shape.b_t)
    if b and t:
        return float(b) / float(t)
    return None


def _lambda_h_t(shape: Shape, h: float | None, t: float | None) -> float | None:
    if shape.h_t_des:
        return float(shape.h_t_des)
    if h and t:
        return float(h) / float(t)
    return None


def _lambda_D_t(shape: Shape, D: float | None, t: float | None) -> float | None:
    if shape.D_t:
        return float(shape.D_t)
    if D and t:
        return float(D) / float(t)
    return None


def _kc_from_h_tw(h_tw: float | None) -> float:
    if not h_tw or h_tw <= 0.0:
        return 0.76
    kc = 4.0 / math.sqrt(h_tw)
    return max(0.35, min(0.76, kc))


def _i_shape_dims(shape: Shape, warnings: List[str]) -> tuple[float, float, float, float, float] | None:
    bf = shape.bf_in
    tf = shape.tf_in
    tw = shape.tw_in
    d = shape.d_in
    if not (bf and tf and tw and d):
        warnings.append("I-shape geometry missing (bf, tf, tw, d); flexure reductions limited.")
        return None
    h = shape.h_in if shape.h_in else float(d) - 2.0 * float(tf)
    if h <= 0.0:
        warnings.append("I-shape clear web depth not positive; check geometry.")
        return None
    return float(bf), float(tf), float(tw), float(d), float(h)


def _i_shape_hc(d: float, tf: float) -> float:
    # Approximate hc as distance from centroid to inside face of compression flange.
    return max(d / 2.0 - tf, 0.0)


def _i_shape_aw(hc: float, tw: float, bfc: float, tfc: float) -> float:
    if bfc <= 0.0 or tfc <= 0.0:
        return 0.0
    return (hc * tw) / (bfc * tfc)


def _i_shape_rt(bfc: float, aw: float) -> float:
    # F4-11: rt = bfc / sqrt(12*(1+aw/6))
    return bfc / math.sqrt(12.0 * (1.0 + aw / 6.0))


def _i_shape_sxc_sxt(shape: Shape, warnings: List[str]) -> tuple[float, float]:
    """
    Estimate Sxc and Sxt about the major axis.
    For symmetric shapes, Sxc=Sxt=Sx. For tees with y provided, use Ix/y and Ix/(d-y).
    """
    Sx = shape.Sx_in3
    Ix = shape.Ix_in4
    d = shape.d_in
    y = shape.y_in
    if Ix and d and y and 0.0 < y < d:
        c_top = float(y)
        c_bot = float(d) - float(y)
        if c_top > 0.0 and c_bot > 0.0:
            S_top = Ix / c_top
            S_bot = Ix / c_bot
            Sxc = min(S_top, S_bot)
            Sxt = max(S_top, S_bot)
            return Sxc, Sxt
    if not Sx:
        warnings.append("Sx not found; using Sxc=Sxt=0.")
        return 0.0, 0.0
    return float(Sx), float(Sx)


def _angle_geometry(shape: Shape, warnings: List[str]) -> tuple[float, float, float, float, float] | None:
    b = shape.b_in or shape.B_in or shape.d_in
    d = shape.d_in or shape.b_in
    g = shape.x_in
    h = shape.y_in
    t = shape.t_in or shape.t_des_in or shape.t_nom_in
    if not (b and d and g is not None and h is not None and t):
        warnings.append("Angle geometry missing (b, d, x, y, t); cannot compute beta_w.")
        return None
    return float(b), float(d), float(g), float(h), float(t)


def _angle_beta_w(shape: Shape, warnings: List[str], moment_sign: float) -> float | None:
    tan_theta = shape.tan_theta
    if tan_theta is None:
        warnings.append("Principal-axis angle not found; cannot compute beta_w.")
        return None
    geom = _angle_geometry(shape, warnings)
    if geom is None or not shape.Iw_in4:
        warnings.append("Angle Iw not found; cannot compute beta_w.")
        return None
    b, d, g, h, t = geom
    a = math.atan(tan_theta)
    c = math.cos(a)
    s = math.sin(a)
    z0 = -s * (t / 2.0 - g) + c * (t / 2.0 - h)

    # Commentary-based polynomial for B12 (Jul 2021, Version A)
    B12 = (
        (-5.0 * c + 5.0 * s) * t**5
        + (-3.0 * b * s + 3.0 * c * d + 6.0 * c * g + 16.0 * c * h - 16.0 * g * s - 6.0 * h * s) * t**4
        + (
            2.0 * b**2 * c
            - 4.0 * b * c * h
            + 12.0 * b * g * s
            - 12.0 * c * d * h
            - 6.0 * c * g**2
            - 12.0 * c * g * h
            - 18.0 * c * h**2
            - 2.0 * d**2 * s
            + 4.0 * d * g * s
            + 18.0 * g**2 * s
            + 12.0 * g * h * s
            + 6.0 * h**2 * s
        )
        * t**3
        + (
            -2.0 * b**3 * s
            - 6.0 * b**2 * c * g
            + 6.0 * b**2 * h * s
            + 12.0 * b * c * g * h
            - 18.0 * b * g**2 * s
            - 6.0 * b * h**2 * s
            + 2.0 * c * d**3
            - 6.0 * c * d**2 * g
            + 6.0 * c * d * g**2
            + 18.0 * c * d * h**2
            + 12.0 * c * g**2 * h
            + 12.0 * c * h**3
            + 6.0 * d**2 * h * s
            - 12.0 * d * g * h * s
            - 12.0 * g**3 * s
            - 12.0 * g * h**2 * s
        )
        * t**2
        + (
            3.0 * b**4 * c
            - 12.0 * b**3 * c * h
            + 4.0 * b**3 * g * s
            + 6.0 * b**2 * c * g**2
            + 18.0 * b**2 * c * h**2
            - 12.0 * b**2 * g * h * s
            - 12.0 * b * c * g**2 * h
            - 12.0 * b * c * h**3
            + 12.0 * b * g**3 * s
            + 12.0 * b * g * h**2 * s
            - 4.0 * c * d**3 * h
            + 12.0 * c * d**2 * g * h
            - 12.0 * c * d * g**2 * h
            - 12.0 * c * d * h**3
            - 3.0 * d**4 * s
            + 12.0 * d**3 * g * s
            - 18.0 * d**2 * g**2 * s
            - 6.0 * d**2 * h**2 * s
            + 12.0 * d * g**3 * s
            + 12.0 * d * g * h**2 * s
        )
        * t
    )
    B = B12 / 12.0
    beta_w = B / float(shape.Iw_in4) - 2.0 * z0
    if moment_sign < 0.0:
        beta_w = -beta_w
    return beta_w


def _angle_leg_local_buckling(
    Fy: float,
    E: float,
    S: float,
    shape: Shape,
    warnings: List[str],
    axis_label: str,
) -> Dict[str, float | str]:
    b = shape.b_in or shape.d_in
    t = shape.t_in or shape.t_des_in or shape.t_nom_in
    if not (b and t and S):
        warnings.append(f"Angle leg local buckling ({axis_label}) missing b/t or S; using Mn=1.5My.")
        return {
            "lambda": None,
            "lambda_p": None,
            "lambda_r": None,
            "Mn_kipin": 1.5 * Fy * S,
            "case": "Missing b/t -> Mn=1.5My",
        }
    lam = float(b) / float(t)
    lam_p = 0.54 * math.sqrt(E / Fy)
    lam_r = 0.91 * math.sqrt(E / Fy)
    if lam <= lam_p:
        Mn = 1.5 * Fy * S
        case = "Compact leg -> Mn=1.5My (F10-1)"
    elif lam <= lam_r:
        Mn = Fy * S * (2.43 - 1.72 * lam * math.sqrt(Fy / E))
        case = "Noncompact leg -> Mn=Fy*Sc*(2.43-1.72*(b/t)*sqrt(Fy/E)) (F10-6)"
    else:
        Fcr = 0.71 * E / (lam**2)
        Mn = Fcr * S
        case = "Slender leg -> Mn=Fcr*Sc, Fcr=0.71E/(b/t)^2 (F10-7/F10-8)"
    return {"lambda": lam, "lambda_p": lam_p, "lambda_r": lam_r, "Mn_kipin": Mn, "case": case}


def _angle_ltb_details(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    Cb: float,
    My_kipin: float,
    moment_sign: float,
    warnings: List[str],
) -> Dict[str, float | str | None]:
    details: Dict[str, float | str | None] = {
        "Lb_in": Lb_in,
        "Cb": Cb,
        "My_kipin": My_kipin,
        "beta_w": None,
        "Mcr_kipin": None,
        "Mn_kipin": None,
        "case": "",
    }
    if Lb_in <= 0.0:
        details["Mn_kipin"] = 1.5 * My_kipin
        details["case"] = "Lb<=0 -> Mn=1.5My"
        return details

    if not shape.A_in2 or not shape.Iz_in4:
        warnings.append("Angle LTB needs A and Iz; using Mn=1.5My.")
        details["Mn_kipin"] = 1.5 * My_kipin
        details["case"] = "Missing A/Iz -> Mn=1.5My"
        return details

    rz = math.sqrt(shape.Iz_in4 / shape.A_in2) if shape.A_in2 > 0 else 0.0
    if rz <= 0.0:
        warnings.append("Angle LTB needs r_z; using Mn=1.5My.")
        details["Mn_kipin"] = 1.5 * My_kipin
        details["case"] = "Missing r_z -> Mn=1.5My"
        return details

    beta_w = _angle_beta_w(shape, warnings, moment_sign=moment_sign)
    if beta_w is None:
        details["Mn_kipin"] = 1.5 * My_kipin
        details["case"] = "Missing beta_w -> Mn=1.5My"
        return details

    t = shape.t_in or shape.t_des_in or shape.t_nom_in
    if not t:
        warnings.append("Angle LTB needs leg thickness t; using Mn=1.5My.")
        details["Mn_kipin"] = 1.5 * My_kipin
        details["case"] = "Missing t -> Mn=1.5My"
        return details

    Cb_eff = min(Cb, 1.5)
    term = 4.4 * beta_w * rz / (Lb_in * t)
    Mcr = (9.0 * E * shape.A_in2 * rz * t * Cb_eff) / (8.0 * Lb_in) * (math.sqrt(1.0 + term**2) + term)

    ratio = My_kipin / Mcr if Mcr != 0 else math.inf
    if ratio <= 1.0:
        Mn = (1.92 - 1.17 * math.sqrt(max(ratio, 0.0))) * My_kipin
        Mn = min(Mn, 1.5 * My_kipin)
        case = "F10-2: My/Mcr<=1"
    else:
        Mn = (0.92 - 0.17 * (Mcr / My_kipin)) * Mcr
        case = "F10-3: My/Mcr>1"
    details["beta_w"] = beta_w
    details["Mcr_kipin"] = Mcr
    details["Mn_kipin"] = max(Mn, 0.0)
    details["case"] = case
    return details


def _hss_rect_ltb_details(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    Cb: float,
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | None]:
    """
    AISC 360-16 F7 (rectangular/square HSS and box shapes):
      Mcr = Cb*(pi/Lb)*sqrt(E*G*J*Iy)  (elastic LTB for closed sections, Cw≈0)
      Mn = min(Mp, Mcr)
    """
    Mp = Fy * shape.Zx_in3  # kip-in
    details: Dict[str, float | str | None] = {
        "Mp_kipin": Mp,
        "Lb_in": Lb_in,
        "Cb": Cb,
        "G_ksi": None,
        "Mcr_kipin": None,
        "Fcr_ksi": None,
        "Mn_kipin": None,
        "case": "",
    }

    if Lb_in <= 0.0:
        details["Mn_kipin"] = Mp
        details["case"] = "HSS rectangular -> Lb<=0, Mn=Mp"
        trace.append("Flexure (HSS rectangular): Lb<=0 -> Mn=Mp")
        return details

    J = shape.J_in4
    Iy = shape.Iy_in4
    Sx = shape.Sx_in3
    if not (J and Iy and Sx):
        warnings.append("HSS rectangular LTB needs J, Iy, Sx; missing -> using Mn=Mp.")
        details["Mn_kipin"] = Mp
        details["case"] = "HSS rectangular -> missing J/Iy/Sx, Mn=Mp"
        return details

    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    Mcr = Cb * (math.pi / Lb_in) * math.sqrt(E * G * J * Iy)
    Fcr = Mcr / Sx
    Mn = min(Mp, Mcr)
    trace.append(
        "Flexure (HSS rectangular): Mcr = Cb*(pi/Lb)*sqrt(E*G*J*Iy) per AISC 360-16 F7."
    )
    details["G_ksi"] = G
    details["Mcr_kipin"] = Mcr
    details["Fcr_ksi"] = Fcr
    details["Mn_kipin"] = Mn
    details["case"] = "HSS rectangular LTB (AISC 360-16 F7)"
    return details


def _flange_local_buckling_I(
    Fy: float,
    E: float,
    shape: Shape,
    Sx: float,
    warnings: List[str],
) -> Dict[str, float | str | None]:
    lam = _lambda_flange_I(shape)
    lam_p = 0.38 * math.sqrt(E / Fy)
    lam_r = 1.00 * math.sqrt(E / Fy)
    case = "Unknown flange slenderness"
    if lam is None:
        warnings.append("Flange slenderness (bf/2tf) not found; using Mn=Mp for CFLB.")
        return {"lambda": None, "lambda_p": lam_p, "lambda_r": lam_r, "Mn_kipin": Fy * shape.Zx_in3, "case": case}
    if lam <= lam_p + 1e-9:
        Mn = Fy * shape.Zx_in3
        case = "Compact flange -> Mn=Mp (F3-1)"
    elif lam <= lam_r + 1e-9:
        Mp = Fy * shape.Zx_in3
        Mn = Mp - (Mp - 0.7 * Fy * Sx) * ((lam - lam_p) / (lam_r - lam_p))
        case = "Noncompact flange -> Mn per F3-1"
    else:
        kc = _kc_from_h_tw(_lambda_web_I(shape))
        Mn = 0.9 * E * kc * Sx / (lam**2)
        case = "Slender flange -> Mn=0.9EkcSx/lambda^2 (F3-2)"
    return {"lambda": lam, "lambda_p": lam_p, "lambda_r": lam_r, "Mn_kipin": Mn, "case": case}


def _flexure_I_major(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    Cb: float,
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | None]:
    Sx = shape.Sx_in3
    web_lam = _lambda_web_I(shape)
    web_p = 3.76 * math.sqrt(E / Fy)
    web_r = 5.70 * math.sqrt(E / Fy)
    web_class, _ = _classify_lambda(web_lam, web_p, web_r)

    ltb = _major_axis_LTB_details(shape, Fy, E, Lb_in, Cb, trace, warnings)
    cflb = _flange_local_buckling_I(Fy, E, shape, Sx, warnings)

    Mn_ltb = float(ltb.get("Mn_kipin") or Fy * shape.Zx_in3)
    Mn_cflb = float(cflb.get("Mn_kipin") or Fy * shape.Zx_in3)
    Mn = min(Mn_ltb, Mn_cflb)

    if web_class == "slender":
        warnings.append("Slender web detected; F5 web-slender reductions not yet applied. Using LTB+CFLB only.")

    details: Dict[str, float | str | None] = {
        "Mp_kipin": Fy * shape.Zx_in3,
        "Mn_kipin": Mn,
        "web_lambda": web_lam,
        "web_lambda_p": web_p,
        "web_lambda_r": web_r,
        "web_class": web_class,
        "flange_local_buckling": cflb,
        "ltb": ltb,
        "case": "I-shape major axis (F2 + F3)",
        "section": "F2/F3",
    }
    return details


def _flexure_I_minor(
    shape: Shape,
    Fy: float,
    E: float,
    warnings: List[str],
) -> Dict[str, float | str | None]:
    lam = _lambda_flange_I(shape)
    lam_p = 0.38 * math.sqrt(E / Fy)
    lam_r = 1.00 * math.sqrt(E / Fy)
    Mp = Fy * shape.Zy_in3
    case = "Minor-axis yielding (F6-1)"
    if lam is None:
        warnings.append("Flange slenderness (bf/2tf) not found; using Mn=Mp for minor axis.")
        return {"Mn_kipin": Mp, "lambda": None, "lambda_p": lam_p, "lambda_r": lam_r, "case": case}

    if lam <= lam_p + 1e-9:
        Mn = Mp
    elif lam <= lam_r + 1e-9:
        Mn = Mp - (Mp - 0.7 * Fy * shape.Sy_in3) * ((lam - lam_p) / (lam_r - lam_p))
        case = "Minor-axis noncompact flange -> Mn per F6-2"
    else:
        Fcr = 0.69 * E / (lam**2)
        Mn = Fcr * shape.Sy_in3
        case = "Minor-axis slender flange -> Mn=Fcr*Sy (F6-3/F6-4)"
    return {"Mn_kipin": Mn, "lambda": lam, "lambda_p": lam_p, "lambda_r": lam_r, "case": case}


def _rp_factor(mp: float, my: float, lam: float, lam_p: float, lam_r: float, iyc_over_iy: float) -> float:
    if iyc_over_iy <= 0.23:
        return 1.0
    if lam <= lam_p + 1e-9:
        return mp / my if my > 0 else 1.0
    if lam_r <= lam_p:
        return 1.0
    rp = (mp / my) - ((mp / my) - 1.0) * ((lam - lam_p) / (lam_r - lam_p))
    return min(rp, mp / my) if my > 0 else 1.0


def _flexure_I_major_F4(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    Cb: float,
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | None]:
    dims = _i_shape_dims(shape, warnings)
    if dims is None:
        return _flexure_I_major(shape, Fy, E, Lb_in, Cb, trace, warnings)
    bf, tf, tw, d, h = dims
    hc = _i_shape_hc(d, tf)
    bfc = bf
    tfc = tf
    Sxc, Sxt = _i_shape_sxc_sxt(shape, warnings)
    Mp = Fy * shape.Zx_in3
    Myc = Fy * Sxc
    Myt = Fy * Sxt

    # Web plastification factor Rpc
    lam_w = hc / tw if tw > 0.0 else 0.0
    lam_pw = 3.76 * math.sqrt(E / Fy)
    lam_rw = 5.70 * math.sqrt(E / Fy)
    Iyc = tfc * bfc**3 / 12.0
    Iyc_over_Iy = Iyc / shape.Iy_in4 if shape.Iy_in4 else 0.0
    Rpc = _rp_factor(Mp, Myc, lam_w, lam_pw, lam_rw, Iyc_over_Iy)

    Mn_cfy = Rpc * Myc
    cfy_case = "Compression flange yielding (F4-1)"

    # FL per F4-6
    sxt_over_sxc = (Sxt / Sxc) if Sxc > 0.0 else 1.0
    if sxt_over_sxc >= 0.7:
        FL = 0.7 * Fy
    else:
        FL = max(0.5 * Fy, Fy * sxt_over_sxc)

    # rt and LTB per F4-5
    aw = _i_shape_aw(hc, tw, bfc, tfc)
    rt = _i_shape_rt(bfc, aw)
    h0 = shape.h0_in or (d - tf)
    J = shape.J_in4 or 0.0
    if Iyc_over_Iy <= 0.23:
        J = 0.0
    Lp = 1.1 * rt * math.sqrt(E / Fy)
    Lr = 1.95 * rt * (E / FL) * math.sqrt(
        (J / (Sxc * h0)) + math.sqrt((J / (Sxc * h0)) ** 2 + 6.76 * (FL / E) ** 2)
    )

    if Lb_in <= Lp:
        Mn_ltb = Rpc * Myc
        ltb_case = "Lb<=Lp (F4-2)"
        Fcr = None
    elif Lb_in <= Lr:
        Mn_ltb = Cb * (Rpc * Myc - (Rpc * Myc - FL * Sxc) * ((Lb_in - Lp) / (Lr - Lp)))
        Mn_ltb = min(Mn_ltb, Rpc * Myc)
        ltb_case = "Lp<Lb<=Lr (F4-2)"
        Fcr = None
    else:
        Fcr = (Cb * math.pi**2 * E / (Lb_in / rt) ** 2) * math.sqrt(
            1.0 + 0.078 * (J / (Sxc * h0)) * (Lb_in / rt) ** 2
        )
        Mn_ltb = min(Fcr * Sxc, Rpc * Myc)
        ltb_case = "Lb>Lr (F4-3)"

    # Compression flange local buckling per F4-13/F4-14
    lam_f = bf / (2.0 * tf)
    lam_pf = 0.38 * math.sqrt(E / Fy)
    lam_rf = 1.00 * math.sqrt(E / Fy)
    if lam_f <= lam_pf + 1e-9:
        Mn_cflb = Rpc * Myc
        cflb_case = "Compact flange (F4-13 n/a)"
    elif lam_f <= lam_rf + 1e-9:
        Mn_cflb = Rpc * Myc - (Rpc * Myc - FL * Sxc) * ((lam_f - lam_pf) / (lam_rf - lam_pf))
        cflb_case = "Noncompact flange (F4-13)"
    else:
        kc = _kc_from_h_tw(h / tw if tw > 0.0 else None)
        Mn_cflb = 0.9 * E * kc * Sxc / (lam_f**2)
        cflb_case = "Slender flange (F4-14)"

    # Tension flange yielding per F4-15
    if Sxt < Sxc and Sxt > 0.0:
        Rpt = _rp_factor(Mp, Myt, lam_w, lam_pw, lam_rw, Iyc_over_Iy)
        Mn_tfy = Rpt * Myt
        tfy_case = "Tension flange yielding (F4-15)"
    else:
        Mn_tfy = math.inf
        tfy_case = "Tension flange yielding n/a"

    Mn = min(Mn_cfy, Mn_ltb, Mn_cflb, Mn_tfy)
    return {
        "section": "F4",
        "case": "Other I-shaped members with compact/noncompact webs (F4)",
        "Mp_kipin": Mp,
        "Sxc_in3": Sxc,
        "Sxt_in3": Sxt,
        "Rpc": Rpc,
        "Mn_kipin": Mn,
        "compression_flange_yielding": {"Mn_kipin": Mn_cfy, "case": cfy_case},
        "ltb": {
            "Mn_kipin": Mn_ltb,
            "case": ltb_case,
            "Lp_in": Lp,
            "Lr_in": Lr,
            "rt_in": rt,
            "Fcr_ksi": Fcr,
        },
        "flange_local_buckling": {"Mn_kipin": Mn_cflb, "case": cflb_case},
        "tension_flange_yielding": {"Mn_kipin": Mn_tfy, "case": tfy_case},
        "web_lambda": lam_w,
        "web_lambda_p": lam_pw,
        "web_lambda_r": lam_rw,
        "web_class": _classify_lambda(lam_w, lam_pw, lam_rw)[0],
    }


def _flexure_I_major_F5(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    Cb: float,
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | None]:
    dims = _i_shape_dims(shape, warnings)
    if dims is None:
        return _flexure_I_major(shape, Fy, E, Lb_in, Cb, trace, warnings)
    bf, tf, tw, d, h = dims
    hc = _i_shape_hc(d, tf)
    bfc = bf
    tfc = tf
    Sxc, Sxt = _i_shape_sxc_sxt(shape, warnings)
    Mp = Fy * shape.Zx_in3

    aw = _i_shape_aw(hc, tw, bfc, tfc)
    aw_cap = min(aw, 10.0)
    rt = _i_shape_rt(bfc, aw)
    lam_w = hc / tw if tw > 0.0 else 0.0
    Rpg = 1.0 - (aw_cap / (1200.0 + 300.0 * aw_cap)) * (
        lam_w - 5.7 * math.sqrt(E / Fy)
    )
    Rpg = min(Rpg, 1.0)
    Rpg = max(Rpg, 0.0)

    # LTB per F5-3/F5-4 with Lp/Lr
    Lp = 1.1 * rt * math.sqrt(E / Fy)
    Lr = math.pi * rt * math.sqrt(E / (0.7 * Fy))
    if Lb_in <= Lp:
        Fcr = Fy
        ltb_case = "Lb<=Lp (F5-2)"
    elif Lb_in <= Lr:
        Fcr = Cb * (Fy - 0.3 * Fy * ((Lb_in - Lp) / (Lr - Lp)))
        Fcr = min(Fcr, Fy)
        ltb_case = "Lp<Lb<=Lr (F5-3)"
    else:
        Fcr = Cb * (math.pi**2 * E / (Lb_in / rt) ** 2)
        Fcr = min(Fcr, Fy)
        ltb_case = "Lb>Lr (F5-4)"

    Mn_cfy = Rpg * Fy * Sxc
    Mn_ltb = Rpg * Fcr * Sxc

    # Compression flange local buckling per F5-8/F5-9
    lam_f = bf / (2.0 * tf)
    lam_pf = 0.38 * math.sqrt(E / Fy)
    lam_rf = 1.00 * math.sqrt(E / Fy)
    if lam_f <= lam_pf + 1e-9:
        Mn_cflb = Mn_cfy
        cflb_case = "Compact flange (F5-7 n/a)"
    elif lam_f <= lam_rf + 1e-9:
        Fcr_f = Fy - 0.3 * Fy * ((lam_f - lam_pf) / (lam_rf - lam_pf))
        Mn_cflb = Rpg * Fcr_f * Sxc
        cflb_case = "Noncompact flange (F5-8)"
    else:
        kc = _kc_from_h_tw(h / tw if tw > 0.0 else None)
        Fcr_f = 0.9 * E * kc / (lam_f**2)
        Mn_cflb = Rpg * Fcr_f * Sxc
        cflb_case = "Slender flange (F5-9)"

    # Tension flange yielding per F5-10
    if Sxt < Sxc and Sxt > 0.0:
        Mn_tfy = Fy * Sxt
        tfy_case = "Tension flange yielding (F5-10)"
    else:
        Mn_tfy = math.inf
        tfy_case = "Tension flange yielding n/a"

    Mn = min(Mn_cfy, Mn_ltb, Mn_cflb, Mn_tfy)
    return {
        "section": "F5",
        "case": "I-shaped members with slender webs (F5)",
        "Mp_kipin": Mp,
        "Sxc_in3": Sxc,
        "Sxt_in3": Sxt,
        "Rpg": Rpg,
        "Mn_kipin": Mn,
        "compression_flange_yielding": {"Mn_kipin": Mn_cfy, "case": "F5-1"},
        "ltb": {
            "Mn_kipin": Mn_ltb,
            "case": ltb_case,
            "Lp_in": Lp,
            "Lr_in": Lr,
            "rt_in": rt,
            "Fcr_ksi": Fcr,
        },
        "flange_local_buckling": {"Mn_kipin": Mn_cflb, "case": cflb_case},
        "tension_flange_yielding": {"Mn_kipin": Mn_tfy, "case": tfy_case},
        "web_lambda": lam_w,
        "web_lambda_p": 3.76 * math.sqrt(E / Fy),
        "web_lambda_r": 5.70 * math.sqrt(E / Fy),
        "web_class": "slender",
    }


def _flexure_F9(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    warnings: List[str],
) -> Dict[str, float | str | None]:
    Sx = shape.Sx_in3
    Zx = shape.Zx_in3
    Iy = shape.Iy_in4
    J = shape.J_in4
    d = shape.d_in or shape.b_in
    tw = shape.tw_in
    tcode = shape.type_code.strip().upper()

    if not (Sx and Zx):
        warnings.append("F9 requires Sx/Zx; using Mn=Fy*Zx fallback.")
        return {"Mn_kipin": Fy * Zx, "case": "F9 fallback", "section": "F9"}

    My = Fy * Sx
    if tcode == "2L":
        Mp = min(Fy * Zx, 1.5 * My)
    else:
        Mp = min(Fy * Zx, 1.6 * My)

    # LTB parameters
    Lp = 1.76 * shape.ry_in * math.sqrt(E / Fy) if shape.ry_in else 0.0
    Lr = None
    if Iy and J and Sx and J > 0.0:
        Lr = 1.95 * (E / Fy) * (Iy / Sx) * math.sqrt(2.36 * (E / Fy) * (Sx / J) + 1.0)

    Mcr = None
    if Lb_in > 0.0 and Iy and J and J > 0.0 and d:
        B = 2.3 * (float(d) / Lb_in) * math.sqrt(Iy / J)
        Mcr_t = (1.95 * E / Lb_in) * math.sqrt(Iy * J) * (B + math.sqrt(1.0 + B**2))
        Bc = -B
        Mcr_c = (1.95 * E / Lb_in) * math.sqrt(Iy * J) * (Bc + math.sqrt(1.0 + Bc**2))
        Mcr = min(Mcr_t, Mcr_c)

    # LTB strength
    if Lb_in <= 0.0 or not Lr or Lp <= 0.0:
        Mn_ltb = Mp
        ltb_case = "Lb<=0 or Lr missing (F9-6 n/a)"
    elif Lb_in <= Lp:
        Mn_ltb = Mp
        ltb_case = "Lb<=Lp (F9-6 n/a)"
    elif Lb_in <= Lr:
        Mn_ltb = Mp - (Mp - My) * ((Lb_in - Lp) / (Lr - Lp))
        ltb_case = "Lp<Lb<=Lr (F9-6)"
    else:
        Mn_ltb = Mcr if Mcr else Mp
        ltb_case = "Lb>Lr (F9-7)"

    # Flange local buckling
    bf = shape.bf_in
    tf = shape.tf_in
    Sxc = Sx
    if bf and tf:
        lam = float(bf) / (2.0 * float(tf))
        lam_pf = 0.38 * math.sqrt(E / Fy)
        lam_rf = 1.00 * math.sqrt(E / Fy)
        if lam <= lam_pf + 1e-9:
            Mn_flb = Mp
            flb_case = "Compact flange (F9-14 n/a)"
        elif lam <= lam_rf + 1e-9:
            Mn_flb = Mp - (Mp - 0.7 * Fy * Sxc) * ((lam - lam_pf) / (lam_rf - lam_pf))
            flb_case = "Noncompact flange (F9-14)"
        else:
            Mn_flb = 0.7 * E * Sxc / (lam**2)
            flb_case = "Slender flange (F9-15)"
    else:
        Mn_flb = Mp
        flb_case = "Flange slenderness missing"

    # Stem/web leg local buckling (tees)
    Mn_wlb = math.inf
    wlb_case = "Web leg buckling n/a"
    if tcode in {"WT", "MT", "ST"} and d and tw:
        lam = float(d) / float(tw)
        lam1 = 0.84 * math.sqrt(E / Fy)
        lam2 = 1.52 * math.sqrt(E / Fy)
        if lam <= lam1:
            Fcr = Fy
            wlb_case = "F9-17"
        elif lam <= lam2:
            Fcr = (1.43 - 0.515 * lam * math.sqrt(Fy / E)) * Fy
            wlb_case = "F9-18"
        else:
            Fcr = 1.52 * E / (lam**2)
            wlb_case = "F9-19"
        Mn_wlb = Fcr * Sx

    Mn = min(Mp, Mn_ltb, Mn_flb, Mn_wlb)
    return {
        "section": "F9",
        "case": "Tees / double angles in plane of symmetry (F9)",
        "Mp_kipin": Mp,
        "Mn_kipin": Mn,
        "ltb": {"Mn_kipin": Mn_ltb, "case": ltb_case, "Lp_in": Lp, "Lr_in": Lr, "Mcr_kipin": Mcr},
        "flange_local_buckling": {"Mn_kipin": Mn_flb, "case": flb_case},
        "web_leg_local_buckling": {"Mn_kipin": Mn_wlb, "case": wlb_case},
    }


def _flexure_hss_rect_axis(
    shape: Shape,
    Fy: float,
    E: float,
    axis: str,
    Lb_in: float,
    Cb: float,
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | None]:
    t = shape.t_des_in or shape.t_nom_in
    B = shape.B_in or shape.b_in
    H = shape.H_in or shape.h_in
    if not (t and B and H):
        warnings.append("HSS rectangular geometry missing; using Mn=Fy*Z for flexure.")
        Z = shape.Zx_in3 if axis == "x" else shape.Zy_in3
        return {"Mn_kipin": Fy * Z, "case": "HSS rectangular -> Mn=Fy*Z fallback"}

    if axis == "x":
        b = float(B)
        h = float(H)
        S = shape.Sx_in3
        Z = shape.Zx_in3
    else:
        b = float(H)
        h = float(B)
        S = shape.Sy_in3
        Z = shape.Zy_in3

    lam_f = _lambda_b_t(shape, b, float(t))
    lam_w = _lambda_h_t(shape, h, float(t))
    lam_pf = 1.12 * math.sqrt(E / Fy)
    lam_rf = 1.40 * math.sqrt(E / Fy)
    lam_pw = 2.42 * math.sqrt(E / Fy)
    lam_rw = 5.70 * math.sqrt(E / Fy)

    Mp = Fy * Z
    flange_class, _ = _classify_lambda(lam_f, lam_pf, lam_rf)
    web_class, _ = _classify_lambda(lam_w, lam_pw, lam_rw)

    Mn_f = Mp
    flange_case = "Flange compact -> Mn=Mp"
    if lam_f is None:
        warnings.append("HSS flange b/t not found; using Mn=Mp for flange local buckling.")
    elif flange_class == "noncompact":
        Mn_f = Mp - (Mp - Fy * S) * (3.57 * lam_f * math.sqrt(Fy / E) - 4.0)
        flange_case = "Flange noncompact -> Mn per F7-2"
    elif flange_class == "slender":
        be = 1.92 * float(t) * math.sqrt(E / Fy) * (1.0 - 0.38 / lam_f * math.sqrt(E / Fy))
        be = min(be, b)
        Se = S * (be / b)
        Mn_f = Fy * Se
        flange_case = "Flange slender -> Mn=Fy*Se (F7-3, effective width)"
        warnings.append("HSS flange slender: using effective-width approximation for Se.")

    Mn_w = Mp
    web_case = "Web compact -> Mn=Mp"
    if lam_w is None:
        warnings.append("HSS web h/t not found; using Mn=Mp for web local buckling.")
    elif web_class == "noncompact":
        Mn_w = Mp - (Mp - Fy * S) * (0.305 * lam_w * math.sqrt(Fy / E) - 0.738)
        web_case = "Web noncompact -> Mn per F7-6"
    elif web_class == "slender":
        be = 1.92 * float(t) * math.sqrt(E / Fy) * (1.0 - 0.38 / lam_w * math.sqrt(E / Fy))
        be = min(be, h)
        Se = S * (be / h)
        Mn_w = Fy * Se
        web_case = "Web slender -> Mn=Fy*Se (effective width)"
        warnings.append("HSS web slender: using effective-width approximation for Se.")

    ltb = _hss_rect_ltb_details(shape, Fy, E, Lb_in, Cb, trace, warnings)
    Mn_ltb = float(ltb.get("Mn_kipin") or Mp)
    Mn = min(Mn_f, Mn_w, Mn_ltb)

    return {
        "axis": axis,
        "Mn_kipin": Mn,
        "Mp_kipin": Mp,
        "flange_class": flange_class,
        "web_class": web_class,
        "flange_case": flange_case,
        "web_case": web_case,
        "flange_lambda": lam_f,
        "web_lambda": lam_w,
        "flange_lambda_p": lam_pf,
        "flange_lambda_r": lam_rf,
        "web_lambda_p": lam_pw,
        "web_lambda_r": lam_rw,
        "ltb": ltb,
        "case": "HSS rectangular flexure (F7)",
    }


def _flexure_hss_round(
    shape: Shape,
    Fy: float,
    E: float,
    axis: str,
    warnings: List[str],
) -> Dict[str, float | str | None]:
    t = shape.t_des_in or shape.t_nom_in
    D = shape.OD_in
    lam = _lambda_D_t(shape, D, t)
    S = shape.Sx_in3 if axis == "x" else shape.Sy_in3
    Z = shape.Zx_in3 if axis == "x" else shape.Zy_in3
    if lam is None:
        warnings.append("Round HSS D/t not found; using Mn=Fy*Z.")
        return {"Mn_kipin": Fy * Z, "lambda": None, "case": "Round HSS -> Mn=Fy*Z fallback"}
    lam_p = 0.07 * E / Fy
    lam_r = 0.31 * E / Fy
    if lam <= lam_p + 1e-9:
        Mn = Fy * Z
        case = "Round HSS compact -> Mn=Mp (F8-1)"
    elif lam <= lam_r + 1e-9:
        Mn = (0.021 * E / lam + Fy) * S
        case = "Round HSS noncompact -> Mn per F8-2"
    else:
        Fcr = 0.33 * E / lam
        Mn = Fcr * S
        case = "Round HSS slender -> Mn=Fcr*S (F8-3/F8-4)"
    return {"Mn_kipin": Mn, "lambda": lam, "lambda_p": lam_p, "lambda_r": lam_r, "case": case}


def _major_axis_LTB_details(
    shape: Shape,
    Fy: float,
    E: float,
    Lb_in: float,
    Cb: float,
    trace: List[str],
    warnings: List[str],
) -> Dict[str, float | str | None]:
    """
    AISC Chapter F2 (foundational) for doubly-symmetric I-shapes:
      - Uses Lp/Lr if available; otherwise computes per AISC expressions.
      - Elastic LTB uses Fcr expression (requires rts, J, Sx, h0).
    If required properties are missing, falls back to Mn = Fy*Zx (yielding).
    """
    Mp = Fy * shape.Zx_in3  # kip-in
    details: Dict[str, float | str | None] = {
        "Mp_kipin": Mp,
        "Lb_in": Lb_in,
        "Cb": Cb,
        "Lp_in": None,
        "Lr_in": None,
        "Fcr_ksi": None,
        "Mn_kipin": None,
        "case": "",
    }
    if Lb_in <= 0.0:
        details["Mn_kipin"] = Mp
        details["case"] = "Lb<=0 -> Mn=Mp"
        return details

    # Prefer DB Lp/Lr, else compute per AISC F2 expressions
    Lp = shape.Lp_in
    Lr = shape.Lr_in

    if Lp is None:
        Lp = 1.76 * shape.ry_in * math.sqrt(E / Fy)
        trace.append("Lp computed per AISC F2: Lp=1.76*ry*sqrt(E/Fy).")
        details["case"] = "Lp computed"
    if Lr is None:
        # Need rts, J, Sx, h0 for Lr estimate
        if not (shape.rts_in and shape.J_in4 and shape.h0_in):
            warnings.append("Lr not found and insufficient properties to estimate; using Mn=Mp (no LTB reduction).")
            details["Lp_in"] = Lp
            details["Lr_in"] = None
            details["Mn_kipin"] = Mp
            details["case"] = "Lr missing -> Mn=Mp"
            return details
        rts = shape.rts_in
        J = shape.J_in4
        h0 = shape.h0_in
        Sx = shape.Sx_in3
        term1 = J / (Sx * h0)
        # AISC-style expression structure (foundational)
        Lr = 1.95 * rts * (E / (0.7 * Fy)) * math.sqrt(term1 + math.sqrt(term1**2 + 6.76 * (0.7 * Fy / E) ** 2))
        trace.append("Lr computed per AISC F2 expression.")
        details["case"] = "Lr computed"

    Sx = shape.Sx_in3
    Mn: float

    if Lb_in <= Lp:
        Mn = Mp
        trace.append(f"Flexure (major): Lb<=Lp -> Mn=Mp={Mp:.3f} kip-in")
        details["Lp_in"] = Lp
        details["Lr_in"] = Lr
        details["Mn_kipin"] = Mn
        details["case"] = "Lb<=Lp -> Mn=Mp"
        return details

    if Lp < Lb_in <= Lr:
        Mn = Cb * (Mp - (Mp - 0.7 * Fy * Sx) * ((Lb_in - Lp) / (Lr - Lp)))
        Mn = min(Mn, Mp)
        trace.append(f"Flexure (major): Lp<Lb<=Lr -> Mn={Mn:.3f} kip-in (Cb={Cb:.3f})")
        details["Lp_in"] = Lp
        details["Lr_in"] = Lr
        details["Mn_kipin"] = Mn
        details["case"] = "Lp<Lb<=Lr -> inelastic LTB"
        return details

    # Lb > Lr: elastic LTB
    if not (shape.rts_in and shape.J_in4 and shape.h0_in):
        warnings.append("Elastic LTB requires rts, J, h0; missing -> using Mn=0.7FySx fallback.")
        Mn = 0.7 * Fy * Sx
        details["Lp_in"] = Lp
        details["Lr_in"] = Lr
        details["Mn_kipin"] = Mn
        details["case"] = "Lb>Lr -> 0.7FySx fallback"
        return details

    rts = shape.rts_in
    J = shape.J_in4
    h0 = shape.h0_in
    # Foundational elastic LTB expression form
    Fcr = Cb * (math.pi**2 * E / (Lb_in / rts) ** 2) * math.sqrt(1.0 + 0.078 * (J / (Sx * h0)) * (Lb_in / rts) ** 2)
    Mn = min(Fcr * Sx, Mp)
    trace.append(f"Flexure (major): Lb>Lr -> elastic LTB Fcr={Fcr:.3f} ksi, Mn={Mn:.3f} kip-in")
    details["Lp_in"] = Lp
    details["Lr_in"] = Lr
    details["Fcr_ksi"] = Fcr
    details["Mn_kipin"] = Mn
    details["case"] = "Lb>Lr -> elastic LTB"
    return details


def design_flexure(model: InputModel, shape: Shape, mat: Material, trace: List[str], warnings: List[str]) -> Dict[str, float | str]:
    """
    Flexure capacities about x (major) and y (minor).
    - Major axis: AISC F2 (foundational) for W/S/M/HP; AISC F8 elastic LTB for rectangular HSS; otherwise yielding-based.
    - Minor axis: yielding-based in this version.
    Local buckling reductions are out-of-scope in this version.
    """
    Fy, E = mat.Fy_ksi, mat.E_ksi
    ff = factors_flexure(model.design_method)

    Lb = ft_to_in(model.Lb_ft)
    Cb = model.Cb

    Mx_raw = model.Mux_kft
    My_raw = model.Muy_kft
    tcode = shape.type_code.strip().upper()

    # Nominal moments (kip-in)
    if tcode == "L":
        Mw_kft, Mz_kft, theta = _principal_moments(shape, Mx_raw, My_raw, warnings)
        sw_vals = [shape.SwA_in3, shape.SwB_in3, shape.SwC_in3]
        sz_vals = [shape.SzA_in3, shape.SzB_in3, shape.SzC_in3]
        sw_min = min((abs(v) for v in sw_vals if v), default=None)
        sz_min = min((abs(v) for v in sz_vals if v), default=None)
        if sw_min is None or sz_min is None:
            warnings.append("Angle principal-axis section moduli not found; using Sx/Sy fallback.")
            sw_min = abs(shape.Sx_in3)
            sz_min = abs(shape.Sy_in3)
        My_w = Fy * sw_min
        My_z = Fy * sz_min
        leg_w = _angle_leg_local_buckling(Fy, E, sw_min, shape, warnings, "w")
        leg_z = _angle_leg_local_buckling(Fy, E, sz_min, shape, warnings, "z")
        moment_sign = 1.0 if Mw_kft >= 0.0 else -1.0
        ltb_w = _angle_ltb_details(shape, Fy, E, Lb, Cb, My_w, moment_sign, warnings)
        Mn_w = min(1.5 * My_w, float(leg_w.get("Mn_kipin") or 1.5 * My_w), float(ltb_w.get("Mn_kipin") or 1.5 * My_w))
        Mn_z = min(1.5 * My_z, float(leg_z.get("Mn_kipin") or 1.5 * My_z))
        major = {
            "axis": "w",
            "theta_deg": math.degrees(theta),
            "Mw_kft": Mw_kft,
            "Mz_kft": Mz_kft,
            "Sw_min_in3": sw_min,
            "Sz_min_in3": sz_min,
            "My_w_kipin": My_w,
            "Mn_w_kipin": Mn_w,
            "Mn_z_kipin": Mn_z,
            "leg_local_buckling_w": leg_w,
            "leg_local_buckling_z": leg_z,
            "ltb_w": ltb_w,
            "case": "Single angle -> principal axes (F10)",
        }
        Mn_x = Mn_w
        Mn_y = Mn_z
    else:
        if _is_doubly_symmetric_I(shape):
            web_lam = _lambda_web_I(shape)
            web_p = 3.76 * math.sqrt(E / Fy)
            web_r = 5.70 * math.sqrt(E / Fy)
            web_class, _ = _classify_lambda(web_lam, web_p, web_r)
            if web_class == "compact":
                major = _flexure_I_major(shape, Fy, E, Lb, Cb, trace, warnings)
            elif web_class == "noncompact":
                major = _flexure_I_major_F4(shape, Fy, E, Lb, Cb, trace, warnings)
            else:
                major = _flexure_I_major_F5(shape, Fy, E, Lb, Cb, trace, warnings)
            Mn_x = float(major.get("Mn_kipin") or 0.0)
            minor = _flexure_I_minor(shape, Fy, E, warnings)
            Mn_y = float(minor.get("Mn_kipin") or 0.0)
        elif tcode in {"C", "MC"}:
            web_lam = _lambda_web_I(shape)
            web_p = 3.76 * math.sqrt(E / Fy)
            web_r = 5.70 * math.sqrt(E / Fy)
            web_class, _ = _classify_lambda(web_lam, web_p, web_r)
            if web_class == "slender":
                major = _flexure_I_major_F5(shape, Fy, E, Lb, Cb, trace, warnings)
            else:
                major = _flexure_I_major_F4(shape, Fy, E, Lb, Cb, trace, warnings)
            Mn_x = float(major.get("Mn_kipin") or 0.0)
            minor = _flexure_I_minor(shape, Fy, E, warnings)
            Mn_y = float(minor.get("Mn_kipin") or 0.0)
            warnings.append(f"{shape.type_code} flexure uses F4/F5 with symmetric assumptions for Sxc/Sxt.")
        elif tcode in {"WT", "MT", "ST", "2L"}:
            major = _flexure_F9(shape, Fy, E, Lb, warnings)
            Mn_x = float(major.get("Mn_kipin") or 0.0)
            minor = _flexure_I_minor(shape, Fy, E, warnings)
            minor["case"] = f"{shape.type_code} minor axis -> F6 fallback"
            Mn_y = float(minor.get("Mn_kipin") or 0.0)
            warnings.append(f"{shape.type_code} flexure uses F9 for major axis; minor axis uses F6 fallback.")
        elif tcode == "HSS":
            if _is_rectangular_hss(shape):
                major = _flexure_hss_rect_axis(shape, Fy, E, "x", Lb, Cb, trace, warnings)
                minor = _flexure_hss_rect_axis(shape, Fy, E, "y", Lb, Cb, trace, warnings)
                Mn_x = float(major.get("Mn_kipin") or 0.0)
                Mn_y = float(minor.get("Mn_kipin") or 0.0)
            else:
                major = _flexure_hss_round(shape, Fy, E, "x", warnings)
                minor = _flexure_hss_round(shape, Fy, E, "y", warnings)
                Mn_x = float(major.get("Mn_kipin") or 0.0)
                Mn_y = float(minor.get("Mn_kipin") or 0.0)
        else:
            # Yielding-based fallback for other shapes
            Mn_x = Fy * shape.Zx_in3
            Mn_y = Fy * shape.Zy_in3
            warnings.append(
                f"Flexure for type '{shape.type_code}' uses Mn=Fy*Z fallback; detailed AISC F provisions not implemented."
            )
            major = {
                "Mp_kipin": Mn_x,
                "Lb_in": Lb,
                "Cb": Cb,
                "Mn_kipin": Mn_x,
                "case": "Fallback -> Mn=Fy*Zx",
            }
            minor = {
                "Mn_kipin": Mn_y,
                "case": "Fallback -> Mn=Fy*Zy",
            }

    # Design strengths (convert to kip-ft for reporting)
    Md_x_kipin = design_strength(model.design_method, Mn_x, ff.phi, ff.omega)
    Md_y_kipin = design_strength(model.design_method, Mn_y, ff.phi, ff.omega)

    Md_x = kipin_to_kipft(Md_x_kipin)
    Md_y = kipin_to_kipft(Md_y_kipin)

    if tcode == "L":
        Mw = float(major.get("Mw_kft", 0.0))
        Mz = float(major.get("Mz_kft", 0.0))
        Mu_x = abs(Mw)
        Mu_y = abs(Mz)
    else:
        Mu_x = abs(Mx_raw)
        Mu_y = abs(My_raw)

    ux = safe_div(Mu_x, Md_x) if Mu_x > 0 else 0.0
    uy = safe_div(Mu_y, Md_y) if Mu_y > 0 else 0.0

    trace.append(f"Flexure: Mn_x={Mn_x:.3f} kip-in, Mn_y={Mn_y:.3f} kip-in")
    trace.append(f"Flexure design: Md_x={Md_x:.3f} kip-ft, Md_y={Md_y:.3f} kip-ft")

    result = {
        "Mx_kft_raw": Mx_raw,
        "My_kft_raw": My_raw,
        "Mux_kft": Mu_x,
        "Muy_kft": Mu_y,
        "Lb_ft": model.Lb_ft,
        "Cb": Cb,
        "Mn_x_kipin": Mn_x,
        "Mn_y_kipin": Mn_y,
        "M_design_x_kft": Md_x,
        "M_design_y_kft": Md_y,
        "unity_x": ux,
        "unity_y": uy,
        "status_x": "OK" if ux <= 1.0 + 1e-9 else "NG",
        "status_y": "OK" if uy <= 1.0 + 1e-9 else "NG",
        "major_axis": major,
        "minor_axis": {
            "Mn_kipin": Mn_y,
            "case": "Minor-axis yielding" if tcode != "L" else "Principal axis z",
        },
    }
    if tcode != "L":
        result["minor_axis"] = minor
    return result
