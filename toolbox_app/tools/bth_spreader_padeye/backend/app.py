from __future__ import annotations
import json, os, math, traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from typing import Any, Dict, Tuple

from ..models import PadeyeInputs, SpreaderInputs, SpreaderTwoWayInputs
from ..calc_trace import CalcTrace, CalcTraceMeta, TraceInput, Assumption, compute_step
from ..paths import input_hash as compute_input_hash
from ..exports import export_all
from toolbox_app.blocks.aisc_shapes_db import ShapeDatabase, Shape as AiscShape
from toolbox_app.blocks.rigging_db import get_shackles

TOOL_ID = "bth_spreader_padeye"
TOOL_VERSION = "1.2.0"
REPORT_VERSION = "1.2"
CODE_BASIS = "ASME BTH-1-2023 (Ch.3)"

TOOL_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = TOOL_DIR / "frontend" / "dist"
RUN_DIR = Path(os.environ.get("BTH_TOOL_RUN_DIR", str(Path.home() / ".local" / "share" / "EngineeringToolbox" / TOOL_ID / "runs" / "dev_run")))
RUN_DIR.mkdir(parents=True, exist_ok=True)
VALID_MODES = ("padeye","spreader","spreader_two_way")
LAST_MODE: str | None = None

_AISC_DB: ShapeDatabase | None = None
_SHAPES: Dict[str, AiscShape] | None = None
_SPREADER_SHAPES: list[Dict[str, str]] | None = None

def _load_aisc_shapes(db: ShapeDatabase) -> Dict[str, AiscShape]:
    out: Dict[str, AiscShape] = {}
    labels_by_type = db.list_labels_by_typecode()
    for labels in labels_by_type.values():
        for label in labels:
            try:
                shp = db.get_shape(label)
            except Exception:
                continue
            out[shp.label] = shp
            upper = shp.label.upper()
            if upper not in out:
                out[upper] = shp
    return out

def _search_shapes(shapes: Dict[str, AiscShape], q: str, limit: int = 50) -> list[AiscShape]:
    q2 = (q or "").strip().upper()
    if not q2:
        seen: set[str] = set()
        out: list[AiscShape] = []
        for shp in shapes.values():
            if shp.label in seen:
                continue
            seen.add(shp.label)
            out.append(shp)
            if len(out) >= limit:
                break
        return out
    hits: list[AiscShape] = []
    seen: set[str] = set()
    for k, shp in shapes.items():
        if q2 not in k.upper():
            continue
        if shp.label in seen:
            continue
        seen.add(shp.label)
        hits.append(shp)
        if len(hits) >= limit:
            break
    return hits

_ALLOWED_SPREADER_TYPES = {"W", "S", "M", "HP", "HSS", "PIPE"}

def _load_spreader_shape_options(db: ShapeDatabase) -> list[Dict[str, str]]:
    by_type = db.list_labels_by_typecode()
    items: list[Dict[str, str]] = []
    def _add(labels: list[str], group: str) -> None:
        for label in labels:
            items.append({"label": label, "type": group})
    for t in ("W", "S", "M", "HP"):
        _add(by_type.get(t, []), t)
    hss_labels = by_type.get("HSS", [])
    hss_rect, hss_round = db.partition_hss_labels(hss_labels)
    _add(hss_rect, "HSS Rect/Square")
    _add(hss_round, "HSS Round")
    _add(by_type.get("PIPE", []), "PIPE")
    return items

def _get_aisc_db() -> ShapeDatabase:
    global _AISC_DB
    if _AISC_DB is None:
        _AISC_DB = ShapeDatabase()
    return _AISC_DB

def _get_shapes() -> Dict[str, AiscShape]:
    global _SHAPES
    if _SHAPES is None:
        _SHAPES = _load_aisc_shapes(_get_aisc_db())
    return _SHAPES

def _get_spreader_shapes() -> list[Dict[str, str]]:
    global _SPREADER_SHAPES
    if _SPREADER_SHAPES is None:
        _SPREADER_SHAPES = _load_spreader_shape_options(_get_aisc_db())
    return _SPREADER_SHAPES

def _load_beam_solver():
    import importlib
    return importlib.import_module("toolbox_app.blocks.1D_beam_solver")


def _units_map(k: str) -> str:
    m = {
        "P":"kip","theta_deg":"deg","beta_deg":"deg","H":"in","h":"in","a1":"in","Wb":"in","Wb1":"in","t":"in","Dh":"in","Dp":"in","R":"in","tcheek":"in","ex":"in","ey":"in",
        "weld_type":"", "weld_group":"", "weld_size_16":"1/16 in", "weld_exx_ksi":"ksi",
        "Fy":"ksi","Fu":"ksi","Nd":"-","design_category":"-","impact_factor":"-",
        "shape":"", "span_L_ft":"ft","Lb_ft":"ft","KL_ft":"ft","ey":"in","mx_includes_total":"-","Cb":"-","V_kip":"kip","P_kip":"kip","Mx_app_kipft":"kip-ft","My_app_kipft":"kip-ft",
        "include_self_weight":"-","Cmx":"-","Cmy":"-","braced_against_twist":"-","weld_check":"-","weld_size_in":"in","weld_length_in":"in","weld_exx_ksi":"ksi",
        "length_ft":"ft","padeye_edge_ft":"ft","padeye_height_in":"in","sling_angle_deg":"deg","point_loads":"", "mode":""
    }
    return m.get(k,"")

def _design_factor_for_category(category: str) -> float:
    cat = (category or "").upper()
    return {"A": 2.0, "B": 3.0, "C": 6.0}.get(cat, 6.0)

def _apply_design_category(inp):
    return inp.model_copy(update={"Nd": _design_factor_for_category(inp.design_category)})

def _mode_from_query(qs: Dict[str, Any]) -> str | None:
    mode = (qs.get("mode") or [None])[0]
    return mode if mode in VALID_MODES else None

def _mode_dir(mode: str | None) -> Path:
    return RUN_DIR / mode if mode else RUN_DIR

def _base_trace(inputs: Dict[str, Any]) -> CalcTrace:
    ih = compute_input_hash(inputs)
    meta = CalcTraceMeta(
        tool_id=TOOL_ID,
        tool_version=TOOL_VERSION,
        report_version=REPORT_VERSION,
        timestamp=str(__import__('datetime').datetime.now().isoformat(timespec='seconds')),
        units_system=inputs.get("units_system","US"),
        code_basis=CODE_BASIS,
        input_hash=ih,
        app_build=None,
    )
    t_inputs = [TraceInput(id=k, label=k, value=v, units=_units_map(k), source="user") for k,v in inputs.items()]
    assumptions = [
        Assumption(id="A1", text="BTH design factor Nd applied as an ASD-style divisor on nominal capacity (allowable = nominal/Nd)."),
        Assumption(id="A2", text="Boss/cheek plates contribute fully to effective thickness at the hole for bearing and tear-out (composite action assumed by welding)."),
        Assumption(id="A3", text="Spreader bar self-weight moment assumed simply-supported uniform load: M_sw = w L^2 / 8 applied about strong axis."),
        Assumption(id="A4", text="Shear stress in members computed conservatively using average shear V/A when web area is not explicitly provided by the database."),
        Assumption(id="A5", text="Padeye base section treated as a rectangular section Wb × t for section moduli and torsion constant (Saint-Venant approximation)."),
    ]
    return CalcTrace(meta=meta, inputs=t_inputs, assumptions=assumptions, steps=[], tables={}, figures=[], summary={})

def _collect_checks(trace: CalcTrace) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for step in trace.steps:
        if not step.checks:
            continue
        for chk in step.checks:
            out.append(
                {
                    "step_id": step.id,
                    "section": step.section,
                    "title": step.title,
                    "label": chk.label,
                    "demand": chk.demand,
                    "capacity": chk.capacity,
                    "ratio": chk.ratio,
                    "pass_fail": chk.pass_fail,
                }
            )
    return out

def _solve_padeye(inp: PadeyeInputs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inp = _apply_design_category(inp)
    d = inp.model_dump()
    trace = _base_trace(d)
    weld_warnings: list[str] = []

    P = inp.P * inp.impact_factor

    Px = compute_step(
        trace, "P-060", "Padeye Loads", "Resolve load component Px",
        "P_x", "In-plane x-component of applied load",
        "P_x = P · cos(θ) · cos(β)",
        variables=[
            dict(symbol="P", description="Applied resultant load incl. impact", value=P, units="kip", source="input:P"),
            dict(symbol="θ", description="In-plane angle", value=inp.theta_deg, units="deg", source="input:theta_deg"),
            dict(symbol="β", description="Out-of-plane angle", value=inp.beta_deg, units="deg", source="input:beta_deg"),
        ],
        compute_fn=lambda v: v["P"]*math.cos(math.radians(v["θ"])) * math.cos(math.radians(v["β"])),
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 load component resolution (padeye global)")]
    )

    Py = compute_step(
        trace, "P-061", "Padeye Loads", "Resolve load component Py",
        "P_y", "In-plane y-component of applied load",
        "P_y = P · sin(θ) · cos(β)",
        variables=[
            dict(symbol="P", description="Applied resultant load incl. impact", value=P, units="kip", source="input:P"),
            dict(symbol="θ", description="In-plane angle", value=inp.theta_deg, units="deg", source="input:theta_deg"),
            dict(symbol="β", description="Out-of-plane angle", value=inp.beta_deg, units="deg", source="input:beta_deg"),
        ],
        compute_fn=lambda v: v["P"]*math.sin(math.radians(v["θ"])) * math.cos(math.radians(v["β"])),
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 load component resolution (padeye global)")]
    )

    Pz = compute_step(
        trace, "P-062", "Padeye Loads", "Resolve load component Pz",
        "P_z", "Out-of-plane component of applied load",
        "P_z = P · sin(β)",
        variables=[
            dict(symbol="P", description="Applied resultant load incl. impact", value=P, units="kip", source="input:P"),
            dict(symbol="β", description="Out-of-plane angle", value=inp.beta_deg, units="deg", source="input:beta_deg"),
        ],
        compute_fn=lambda v: v["P"]*math.sin(math.radians(v["β"])),
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 load component resolution (padeye global)")]
    )


    def _padeye_width_at_hole(Wb: float, a1: float, R: float, H: float, h: float, e_z: float) -> float:
        if Wb <= 0:
            return 0.0
        if R <= 0:
            return Wb

        y_center = H - R
        x_left = -Wb / 2.0
        x_right = Wb / 2.0
        R2 = R * R

        def _tangent_point(x_edge: float) -> tuple[float, float] | None:
            dx = x_edge - e_z
            dy = a1 - y_center
            d2 = dx * dx + dy * dy
            if d2 <= R2:
                return None
            sqrt_term = math.sqrt(max(d2 - R2, 0.0))
            coeff1 = R2 / d2
            coeff2 = R * sqrt_term / d2
            px = -dy
            py = dx
            tx1 = e_z + coeff1 * dx + coeff2 * px
            ty1 = y_center + coeff1 * dy + coeff2 * py
            tx2 = e_z + coeff1 * dx - coeff2 * px
            ty2 = y_center + coeff1 * dy - coeff2 * py
            return max([(tx1, ty1), (tx2, ty2)], key=lambda c: c[1])

        def _edge_x(x_edge: float, side_sign: float) -> float:
            if h <= a1 + 1e-9:
                return x_edge
            tangent = _tangent_point(x_edge)
            if tangent is not None:
                tx, ty = tangent
                if h <= ty + 1e-9 and abs(ty - a1) > 1e-9:
                    t = (h - a1) / (ty - a1)
                    return x_edge + t * (tx - x_edge)
            dy = h - y_center
            if abs(dy) <= R:
                dx = math.sqrt(max(R2 - dy * dy, 0.0))
                return e_z + side_sign * dx
            return x_edge

        left_x = _edge_x(x_left, -1.0)
        right_x = _edge_x(x_right, 1.0)
        width = right_x - left_x
        return width if width > 0 else Wb

    def _max_weld_eq_stress(v: Dict[str, float]) -> float:
        Aw = v["A_w"]
        if Aw <= 0:
            return 0.0
        x_vals = (-v["W_b"] / 2.0, v["W_b"] / 2.0)
        z_vals = (-v["t"] / 2.0, v["t"] / 2.0)
        max_eq = 0.0
        for x in x_vals:
            for z in z_vals:
                sigma = v["P_y"] / Aw
                if v["I_xw"] > 0:
                    sigma += (v["M_x"] * z) / v["I_xw"]
                if v["I_zw"] > 0:
                    sigma += (v["M_z"] * x) / v["I_zw"]
                tau_x = v["P_x"] / Aw
                tau_z = v["P_z"] / Aw
                if v["J_w"] > 0:
                    tau_x += (-v["T"] * z) / v["J_w"]
                    tau_z += (v["T"] * x) / v["J_w"]
                tau = math.sqrt(tau_x**2 + tau_z**2)
                eq = math.sqrt(sigma**2 + tau**2)
                if eq > max_eq:
                    max_eq = eq
        return max_eq

    e_z = compute_step(
        trace, "P-062A", "Padeye Geometry", "Hole offset from base centerline",
        "e_z", "Horizontal offset from base centerline to hole center",
        "e_z = W_b/2 - W_b1",
        variables=[
            dict(symbol="W_b", description="Base width", value=inp.Wb, units="in", source="input:Wb"),
            dict(symbol="W_b1", description="Hole center to edge distance", value=inp.Wb1, units="in", source="input:Wb1"),
        ],
        compute_fn=lambda v: v["W_b"]/2.0 - v["W_b1"],
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:HoleOffsetFromWb1")]
    )

    R_edge = compute_step(
        trace, "P-062B", "Padeye Geometry", "Top edge distance to hole center",
        "R_edge", "Vertical distance from top edge to hole center",
        "R_edge = H - h",
        variables=[
            dict(symbol="H", description="Padeye height", value=inp.H, units="in", source="input:H"),
            dict(symbol="h", description="Height to hole center", value=inp.h, units="in", source="input:h"),
        ],
        compute_fn=lambda v: max(v["H"] - v["h"], 0.0),
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:TopEdgeDistance")]
    )

    w_hole = compute_step(
        trace, "P-062C", "Padeye Geometry", "Width at hole (derived)",
        "w", "Width at hole derived from base + top radius geometry",
        "w = f(W_b, a1, R, H, h, e_z)",
        variables=[
            dict(symbol="W_b", description="Base width", value=inp.Wb, units="in", source="input:Wb"),
            dict(symbol="a1", description="Straight vertical corner height", value=inp.a1, units="in", source="input:a1"),
            dict(symbol="R", description="Top radius", value=inp.R, units="in", source="input:R"),
            dict(symbol="H", description="Padeye height", value=inp.H, units="in", source="input:H"),
            dict(symbol="h", description="Height to hole center", value=inp.h, units="in", source="input:h"),
            dict(symbol="e_z", description="Hole offset", value=e_z, units="in", source="step:P-062A"),
        ],
        compute_fn=lambda v: _padeye_width_at_hole(v["W_b"], v["a1"], v["R"], v["H"], v["h"], v["e_z"]),
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:WidthAtHoleFromGeometry")]
    )

    V = compute_step(
        trace, "P-063", "Padeye Base Actions", "Resultant base shear",
        "V", "Resultant shear at base from Px and Pz",
        "V = √(P_x^2 + P_z^2)",
        variables=[
            dict(symbol="P_x", description="In-plane x component", value=Px, units="kip", source="step:P-060"),
            dict(symbol="P_z", description="Out-of-plane component", value=Pz, units="kip", source="step:P-062"),
        ],
        compute_fn=lambda v: math.sqrt(v["P_x"]**2 + v["P_z"]**2),
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 padeye base resultant shear (global)")]
    )

    Mz = compute_step(
        trace, "P-064", "Padeye Base Actions", "Strong-axis base moment",
        "M_z", "Base bending moment including hole offset",
        "M_z = P_x · h - P_y · e_z",
        variables=[
            dict(symbol="P_x", description="In-plane x component", value=Px, units="kip", source="step:P-060"),
            dict(symbol="P_y", description="In-plane y component", value=Py, units="kip", source="step:P-061"),
            dict(symbol="h", description="Height to hole center", value=inp.h, units="in", source="input:h"),
            dict(symbol="e_z", description="Hole offset", value=e_z, units="in", source="step:P-062A"),
        ],
        compute_fn=lambda v: v["P_x"]*v["h"] - v["P_y"]*v["e_z"],
        units="kip-in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=3),
        references=[dict(type="derived", ref="backend._solve_padeye:StrongAxisMomentWithOffset")]
    )

    Mx = compute_step(
        trace, "P-065", "Padeye Base Actions", "Weak-axis base moment (out-of-plane)",
        "M_x", "Base weak-axis bending moment from Pz and eccentricity",
        "M_x = P_z · (h + e_y)",
        variables=[
            dict(symbol="P_z", description="Out-of-plane component", value=Pz, units="kip", source="step:P-062"),
            dict(symbol="h", description="Height to hole center", value=inp.h, units="in", source="input:h"),
            dict(symbol="e_y", description="Vertical eccentricity", value=inp.ey, units="in", source="input:ey"),
        ],
        compute_fn=lambda v: v["P_z"]*(v["h"]+v["e_y"]),
        units="kip-in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=3),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 padeye out-of-plane bending (global)")]
    )

    T = compute_step(
        trace, "P-066", "Padeye Base Actions", "Torsion at base",
        "T", "Torsional moment from out-of-plane load eccentricity",
        "T = P_z · e_x",
        variables=[
            dict(symbol="P_z", description="Out-of-plane component", value=Pz, units="kip", source="step:P-062"),
            dict(symbol="e_x", description="Eccentricity to torsion arm", value=inp.ex, units="in", source="input:ex"),
        ],
        compute_fn=lambda v: v["P_z"]*v["e_x"],
        units="kip-in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=3),
        references=[dict(type="derived", ref="backend._solve_padeye:TorsionFromEccentricity")]
    )

    A = compute_step(
        trace, "P-070", "Padeye Base Section", "Base area",
        "A", "Cross-sectional area at base",
        "A = W_b · t",
        variables=[
            dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
            dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
        ],
        compute_fn=lambda v: v["W_b"]*v["t"],
        units="in^2",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:BaseArea")]
    )

    Sz = compute_step(
        trace, "P-071", "Padeye Base Section", "Strong-axis section modulus",
        "S_z", "Section modulus about strong axis (rectangular)",
        "S_z = t · W_b^2 / 6",
        variables=[
            dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
            dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
        ],
        compute_fn=lambda v: v["t"]*(v["W_b"]**2)/6.0,
        units="in^3",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:RectSectionModulusStrong")]
    )

    Sx = compute_step(
        trace, "P-072", "Padeye Base Section", "Weak-axis section modulus",
        "S_x", "Section modulus about weak axis (rectangular)",
        "S_x = W_b · t^2 / 6",
        variables=[
            dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
            dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
        ],
        compute_fn=lambda v: v["W_b"]*(v["t"]**2)/6.0,
        units="in^3",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:RectSectionModulusWeak")]
    )

    J = compute_step(
        trace, "P-073", "Padeye Base Section", "Torsion constant (approx.)",
        "J", "Saint-Venant torsion constant (rectangular approx.)",
        "J ≈ (1/3) · W_b · t^3",
        variables=[
            dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
            dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
        ],
        compute_fn=lambda v: (1.0/3.0)*v["W_b"]*(v["t"]**3),
        units="in^4",
        rounding_rule=dict(rule="sigfigs", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:TorsionConstantRectApprox")]
    )

    sigma = compute_step(
        trace, "P-080", "Padeye Base Stresses", "Normal stress at base (combined)",
        "σ", "Combined normal stress from axial + bending",
        "σ = P_y/A + M_z/S_z + M_x/S_x",
        variables=[
            dict(symbol="P_y", description="Axial component", value=Py, units="kip", source="step:P-061"),
            dict(symbol="A", description="Area", value=A, units="in^2", source="step:P-070"),
            dict(symbol="M_z", description="Strong-axis moment", value=Mz, units="kip-in", source="step:P-064"),
            dict(symbol="S_z", description="Strong-axis section modulus", value=Sz, units="in^3", source="step:P-071"),
            dict(symbol="M_x", description="Weak-axis moment", value=Mx, units="kip-in", source="step:P-065"),
            dict(symbol="S_x", description="Weak-axis section modulus", value=Sx, units="in^3", source="step:P-072"),
        ],
        compute_fn=lambda v: (v["P_y"]/v["A"]) + (v["M_z"]/v["S_z"]) + (v["M_x"]/v["S_x"]),
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 combined normal stress at padeye base")]
    )

    tau_v = compute_step(
        trace, "P-081", "Padeye Base Stresses", "Average shear stress at base",
        "τ_V", "Average shear stress from base shear",
        "τ_V = V/A",
        variables=[
            dict(symbol="V", description="Resultant shear", value=V, units="kip", source="step:P-063"),
            dict(symbol="A", description="Area", value=A, units="in^2", source="step:P-070"),
        ],
        compute_fn=lambda v: v["V"]/v["A"],
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:AvgShearStress")]
    )

    c = max(inp.Wb, inp.t)/2.0
    tau_t = compute_step(
        trace, "P-082", "Padeye Base Stresses", "Torsional shear stress (approx.)",
        "τ_T", "Shear stress from torsion (approx.)",
        "τ_T = T · c / J",
        variables=[
            dict(symbol="T", description="Torsional moment", value=T, units="kip-in", source="step:P-066"),
            dict(symbol="c", description="Outer fiber distance (approx.)", value=c, units="in", source="derived:c"),
            dict(symbol="J", description="Torsion constant", value=J, units="in^4", source="step:P-073"),
        ],
        compute_fn=lambda v: v["T"]*v["c"]/v["J"] if v["J"]>0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:TorsionalShearApprox")]
    )

    tau = compute_step(
        trace, "P-083", "Padeye Base Stresses", "Total shear stress (conservative sum)",
        "τ", "Total shear stress (conservative)",
        "τ = τ_V + τ_T",
        variables=[
            dict(symbol="τ_V", description="Average shear stress", value=tau_v, units="ksi", source="step:P-081"),
            dict(symbol="τ_T", description="Torsional shear stress", value=tau_t, units="ksi", source="step:P-082"),
        ],
        compute_fn=lambda v: v["τ_V"]+v["τ_T"],
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:ConservativeShearSum")]
    )

    seq = compute_step(
        trace, "P-084", "Padeye Base Stresses", "Von Mises equivalent stress",
        "σ_eq", "Equivalent stress for combined normal and shear",
        "σ_eq = √(σ^2 + 3τ^2)",
        variables=[
            dict(symbol="σ", description="Normal stress", value=sigma, units="ksi", source="step:P-080"),
            dict(symbol="τ", description="Shear stress", value=tau, units="ksi", source="step:P-083"),
        ],
        compute_fn=lambda v: math.sqrt(v["σ"]**2 + 3.0*(v["τ"]**2)),
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 combined stress (equivalent stress check)")]
    )

    Fallow = compute_step(
        trace, "P-085", "Padeye Base Stresses", "Allowable equivalent stress",
        "F_allow", "Allowable equivalent stress using Nd",
        "F_allow = F_y / N_d",
        variables=[
            dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        ],
        compute_fn=lambda v: v["F_y"]/v["N_d"],
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 design factor method (allowable stress form)")]
    )

    def chk_eq(_u,_r):
        ratio = seq / Fallow if Fallow>0 else 1e9
        return [dict(label="Combined Stress at Base of Padeye", demand=seq, capacity=Fallow, ratio=ratio, pass_fail="PASS" if ratio<=1 else "FAIL")]

    Ueq = compute_step(
        trace, "P-086", "Padeye Base Stresses", "Utilization (equivalent stress)",
        "U", "Utilization ratio (equivalent stress / allowable)",
        "U = σ_eq / F_allow",
        variables=[
            dict(symbol="σ_eq", description="Equivalent stress", value=seq, units="ksi", source="step:P-084"),
            dict(symbol="F_allow", description="Allowable equivalent stress", value=Fallow, units="ksi", source="step:P-085"),
        ],
        compute_fn=lambda v: v["σ_eq"]/v["F_allow"],
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:UtilizationEqStress")],
        checks_builder=chk_eq
    )

    f_t = compute_step(
        trace, "P-087", "Padeye Base Stresses", "Axial stress at base",
        "f_t", "Axial stress magnitude at base",
        "f_t = abs(P_y) / A",
        variables=[
            dict(symbol="P_y", description="Axial component", value=Py, units="kip", source="step:P-061"),
            dict(symbol="A", description="Area", value=A, units="in^2", source="step:P-070"),
        ],
        compute_fn=lambda v: abs(v["P_y"]) / v["A"] if v["A"] > 0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:AxialStressBase")]
    )

    f_b_in = compute_step(
        trace, "P-088", "Padeye Base Stresses", "In-plane bending stress at base",
        "f_b_in", "Strong-axis bending stress magnitude",
        "f_b_in = abs(M_z) / S_z",
        variables=[
            dict(symbol="M_z", description="Strong-axis moment", value=Mz, units="kip-in", source="step:P-064"),
            dict(symbol="S_z", description="Strong-axis section modulus", value=Sz, units="in^3", source="step:P-071"),
        ],
        compute_fn=lambda v: abs(v["M_z"]) / v["S_z"] if v["S_z"] > 0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:StrongAxisBendingStressBase")]
    )

    f_b_out = compute_step(
        trace, "P-089", "Padeye Base Stresses", "Out-of-plane bending stress at base",
        "f_b_out", "Weak-axis bending stress magnitude",
        "f_b_out = abs(M_x) / S_x",
        variables=[
            dict(symbol="M_x", description="Weak-axis moment", value=Mx, units="kip-in", source="step:P-065"),
            dict(symbol="S_x", description="Weak-axis section modulus", value=Sx, units="in^3", source="step:P-072"),
        ],
        compute_fn=lambda v: abs(v["M_x"]) / v["S_x"] if v["S_x"] > 0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:WeakAxisBendingStressBase")]
    )

    Fv_allow = compute_step(
        trace, "P-090", "Padeye Base Stresses", "Allowable shear stress at base",
        "F_v", "Allowable shear stress",
        "F_v = 0.6 * F_y / N_d",
        variables=[
            dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        ],
        compute_fn=lambda v: 0.6 * v["F_y"] / v["N_d"],
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:ShearAllowableBase")]
    )

    compute_step(
        trace, "P-091", "Padeye Base Stresses", "Utilization: base shear",
        "U_v", "Shear utilization at base",
        "U_v = f_v / F_v",
        variables=[
            dict(symbol="f_v", description="Total shear stress", value=tau, units="ksi", source="step:P-083"),
            dict(symbol="F_v", description="Allowable shear stress", value=Fv_allow, units="ksi", source="step:P-090"),
        ],
        compute_fn=lambda v: v["f_v"] / v["F_v"] if v["F_v"] > 0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:ShearUtilizationBase")],
        checks_builder=lambda _u,_r: [dict(label="Shear at Base of Padeye", demand=tau, capacity=Fv_allow, ratio=(tau/Fv_allow if Fv_allow>0 else 1e9), pass_fail="PASS" if tau <= Fv_allow else "FAIL")]
    )

    compute_step(
        trace, "P-092", "Padeye Base Stresses", "Utilization: in-plane bending",
        "U_b_in", "In-plane bending utilization at base",
        "U_b_in = f_b_in / F_allow",
        variables=[
            dict(symbol="f_b_in", description="In-plane bending stress", value=f_b_in, units="ksi", source="step:P-088"),
            dict(symbol="F_allow", description="Allowable stress", value=Fallow, units="ksi", source="step:P-085"),
        ],
        compute_fn=lambda v: v["f_b_in"] / v["F_allow"] if v["F_allow"] > 0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:InPlaneBendingUtilizationBase")],
        checks_builder=lambda _u,_r: [dict(label="In-Plane Bending at Base of Padeye", demand=f_b_in, capacity=Fallow, ratio=(f_b_in/Fallow if Fallow>0 else 1e9), pass_fail="PASS" if f_b_in <= Fallow else "FAIL")]
    )

    compute_step(
        trace, "P-093", "Padeye Base Stresses", "Utilization: out-of-plane bending",
        "U_b_out", "Out-of-plane bending utilization at base",
        "U_b_out = f_b_out / F_allow",
        variables=[
            dict(symbol="f_b_out", description="Out-of-plane bending stress", value=f_b_out, units="ksi", source="step:P-089"),
            dict(symbol="F_allow", description="Allowable stress", value=Fallow, units="ksi", source="step:P-085"),
        ],
        compute_fn=lambda v: v["f_b_out"] / v["F_allow"] if v["F_allow"] > 0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:OutOfPlaneBendingUtilizationBase")],
        checks_builder=lambda _u,_r: [dict(label="Out-of-plane Bending at Base of Padeye", demand=f_b_out, capacity=Fallow, ratio=(f_b_out/Fallow if Fallow>0 else 1e9), pass_fail="PASS" if f_b_out <= Fallow else "FAIL")]
    )

    compute_step(
        trace, "P-094", "Padeye Base Stresses", "Utilization: base tension",
        "U_t", "Tension utilization at base",
        "U_t = f_t / F_allow",
        variables=[
            dict(symbol="f_t", description="Axial stress", value=f_t, units="ksi", source="step:P-087"),
            dict(symbol="F_allow", description="Allowable stress", value=Fallow, units="ksi", source="step:P-085"),
        ],
        compute_fn=lambda v: v["f_t"] / v["F_allow"] if v["F_allow"] > 0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:TensionUtilizationBase")],
        checks_builder=lambda _u,_r: [dict(label="Tension at Base of Padeye", demand=f_t, capacity=Fallow, ratio=(f_t/Fallow if Fallow>0 else 1e9), pass_fail="PASS" if f_t <= Fallow else "FAIL")]
    )

    # Weld group check at base (elastic method)
    if inp.weld_type == "CJP":
        weld_warnings.append("CJP weld assumed to meet or exceed base metal capacity; weld group check skipped.")
    else:
        weld_size_in = compute_step(
            trace, "P-095", "Padeye Weld", "Weld size (leg) from 16ths",
            "w_weld", "Weld leg size",
            "w_weld = w_16 / 16",
            variables=[dict(symbol="w_16", description="Weld size in 16ths", value=inp.weld_size_16, units="1/16 in", source="input:weld_size_16")],
            compute_fn=lambda v: v["w_16"] / 16.0,
            units="in",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldSizeFromSixteenths")]
        )

        if inp.weld_type == "Fillet":
            te = compute_step(
                trace, "P-096", "Padeye Weld", "Effective throat (fillet)",
                "t_e", "Effective throat",
                "t_e = 0.707 * w_weld",
                variables=[dict(symbol="w_weld", description="Fillet weld leg size", value=weld_size_in, units="in", source="step:P-095")],
                compute_fn=lambda v: 0.707 * v["w_weld"],
                units="in",
                rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
                references=[dict(type="code", ref="AWS fillet weld effective throat definition (0.707w)")]
            )
        elif inp.weld_type == "PJP 60° Bevel":
            te = compute_step(
                trace, "P-096", "Padeye Weld", "Effective throat (PJP 60°)",
                "t_e", "Effective throat",
                "t_e = t",
                variables=[dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t")],
                compute_fn=lambda v: v["t"],
                units="in",
                rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
                references=[dict(type="derived", ref="backend._solve_padeye:PJP60EffectiveThroat")]
            )
        else:
            te = compute_step(
                trace, "P-096", "Padeye Weld", "Effective throat (PJP 45°)",
                "t_e", "Effective throat",
                "t_e = t - 1/8",
                variables=[dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t")],
                compute_fn=lambda v: max(v["t"] - 0.125, 0.0),
                units="in",
                rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
                references=[dict(type="derived", ref="backend._solve_padeye:PJP45EffectiveThroat")]
            )

        L_w = compute_step(
            trace, "P-097", "Padeye Weld", "Total weld length",
            "L_w", "Total weld length (weld group)",
            "L_w = 2 * (W_b + t) for all-around, else 2 * W_b",
            variables=[
                dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
                dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
                dict(symbol="group", description="Weld group", value=inp.weld_group, units="", source="input:weld_group"),
            ],
            compute_fn=lambda v: 2.0 * (v["W_b"] + v["t"]) if v["group"] == "All Around" else 2.0 * v["W_b"],
            units="in",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldGroupLength")]
        )

        Aw = compute_step(
            trace, "P-098", "Padeye Weld", "Effective weld throat area",
            "A_w", "Effective weld throat area",
            "A_w = t_e * L_w",
            variables=[
                dict(symbol="t_e", description="Effective throat", value=te, units="in", source="step:P-096"),
                dict(symbol="L_w", description="Weld length", value=L_w, units="in", source="step:P-097"),
            ],
            compute_fn=lambda v: v["t_e"] * v["L_w"],
            units="in^2",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldArea")]
        )

        Ix_w = compute_step(
            trace, "P-099", "Padeye Weld", "Weld group inertia about x",
            "I_xw", "Weld group inertia about x-axis",
            "I_xw = t_e * (W_b * t^2 / 2 + t^3 / 6) for all-around, else t_e * W_b * t^2 / 2",
            variables=[
                dict(symbol="t_e", description="Effective throat", value=te, units="in", source="step:P-096"),
                dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
                dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
                dict(symbol="group", description="Weld group", value=inp.weld_group, units="", source="input:weld_group"),
            ],
            compute_fn=lambda v: v["t_e"] * ((v["W_b"] * (v["t"]**2) / 2.0 + (v["t"]**3) / 6.0) if v["group"] == "All Around" else (v["W_b"] * (v["t"]**2) / 2.0)),
            units="in^4",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldInertiaX")]
        )

        Iz_w = compute_step(
            trace, "P-100", "Padeye Weld", "Weld group inertia about z",
            "I_zw", "Weld group inertia about z-axis",
            "I_zw = t_e * (W_b^3 / 6 + t * W_b^2 / 2) for all-around, else t_e * W_b^3 / 6",
            variables=[
                dict(symbol="t_e", description="Effective throat", value=te, units="in", source="step:P-096"),
                dict(symbol="W_b", description="Plate width at base", value=inp.Wb, units="in", source="input:Wb"),
                dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
                dict(symbol="group", description="Weld group", value=inp.weld_group, units="", source="input:weld_group"),
            ],
            compute_fn=lambda v: v["t_e"] * ((v["W_b"]**3 / 6.0 + v["t"] * (v["W_b"]**2) / 2.0) if v["group"] == "All Around" else (v["W_b"]**3 / 6.0)),
            units="in^4",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldInertiaZ")]
        )

        J_w = compute_step(
            trace, "P-101", "Padeye Weld", "Weld group polar inertia",
            "J_w", "Weld group polar inertia (Ix + Iz)",
            "J_w = I_xw + I_zw",
            variables=[
                dict(symbol="I_xw", description="Weld group Ix", value=Ix_w, units="in^4", source="step:P-099"),
                dict(symbol="I_zw", description="Weld group Iz", value=Iz_w, units="in^4", source="step:P-100"),
            ],
            compute_fn=lambda v: v["I_xw"] + v["I_zw"],
            units="in^4",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldPolarInertia")]
        )

        Fw_allow = compute_step(
            trace, "P-102", "Padeye Weld", "Allowable weld stress",
            "F_w", "Allowable weld throat stress",
            "F_w = 0.6 * E_xx / N_d",
            variables=[
                dict(symbol="E_xx", description="Weld metal tensile strength", value=inp.weld_exx_ksi, units="ksi", source="input:weld_exx_ksi"),
                dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
            ],
            compute_fn=lambda v: 0.6 * v["E_xx"] / v["N_d"],
            units="ksi",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
            references=[dict(type="derived", ref="backend._solve_padeye:WeldAllowableStress")]
        )

        if Aw <= 0 or Ix_w <= 0 or Iz_w <= 0 or J_w <= 0:
            weld_warnings.append("Weld group geometry produced zero throat area or inertia; weld check skipped.")
        else:
            f_eq_weld = compute_step(
                trace, "P-103", "Padeye Weld", "Max weld group equivalent stress",
                "f_eq", "Max combined weld stress (elastic method)",
                "f_eq = max sqrt(å^2 + ç^2) at weld group corners",
                variables=[
                    dict(symbol="P_x", description="Shear component (x)", value=Px, units="kip", source="step:P-060"),
                    dict(symbol="P_y", description="Axial component (y)", value=Py, units="kip", source="step:P-061"),
                    dict(symbol="P_z", description="Shear component (z)", value=Pz, units="kip", source="step:P-062"),
                    dict(symbol="M_x", description="Moment about x", value=Mx, units="kip-in", source="step:P-065"),
                    dict(symbol="M_z", description="Moment about z", value=Mz, units="kip-in", source="step:P-064"),
                    dict(symbol="T", description="Torsion", value=T, units="kip-in", source="step:P-066"),
                    dict(symbol="A_w", description="Weld throat area", value=Aw, units="in^2", source="step:P-098"),
                    dict(symbol="I_xw", description="Weld group Ix", value=Ix_w, units="in^4", source="step:P-099"),
                    dict(symbol="I_zw", description="Weld group Iz", value=Iz_w, units="in^4", source="step:P-100"),
                    dict(symbol="J_w", description="Weld group polar inertia", value=J_w, units="in^4", source="step:P-101"),
                    dict(symbol="W_b", description="Base width", value=inp.Wb, units="in", source="input:Wb"),
                    dict(symbol="t", description="Plate thickness", value=inp.t, units="in", source="input:t"),
                ],
                compute_fn=lambda v: _max_weld_eq_stress(v),
                units="ksi",
                rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
                references=[dict(type="derived", ref="backend._solve_padeye:WeldElasticMethod")]
            )

            compute_step(
                trace, "P-104", "Padeye Weld", "Weld utilization",
                "U_w", "Weld utilization ratio",
                "U_w = f_eq / F_w",
                variables=[
                    dict(symbol="f_eq", description="Max weld stress", value=f_eq_weld, units="ksi", source="step:P-103"),
                    dict(symbol="F_w", description="Allowable weld stress", value=Fw_allow, units="ksi", source="step:P-102"),
                ],
                compute_fn=lambda v: v["f_eq"] / v["F_w"] if v["F_w"] > 0 else 1e9,
                units="-",
                rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
                references=[dict(type="derived", ref="backend._solve_padeye:WeldUtilization")],
                checks_builder=lambda _u,_r: [dict(label="Weld Group Combined Stress", demand=f_eq_weld, capacity=Fw_allow, ratio=(f_eq_weld/Fw_allow if Fw_allow>0 else 1e9), pass_fail="PASS" if f_eq_weld <= Fw_allow else "FAIL")]
            )

    # Hole region checks (boss plates included at hole, aligned to BTH sheet Pt/Pb/Pv)
    t_eff = inp.t + inp.tcheek

    be = compute_step(
        trace, "P-108", "Padeye Hole", "Plate edge width",
        "b_e", "Actual plate edge width at pinhole",
        "b_e = (w - D_h) / 2",
        variables=[
            dict(symbol="w", description="Width at hole", value=w_hole, units="in", source="step:P-062C"),
            dict(symbol="D_h", description="Hole diameter", value=inp.Dh, units="in", source="input:Dh"),
        ],
        compute_fn=lambda v: max((v["w"]-v["D_h"])/2.0, 0.0),
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="BTH sheet: be=(w-Dh)/2")]
    )

    beff = compute_step(
        trace, "P-109", "Padeye Hole", "Effective plate edge width",
        "b_eff", "Effective plate edge width per BTH sheet",
        "b_eff = min(b_e, 4 t_eff, b_e · 0.6 · (F_u/F_y) · sqrt(D_h/b_e))",
        variables=[
            dict(symbol="b_e", description="Plate edge width", value=be, units="in", source="step:P-108"),
            dict(symbol="t_eff", description="Effective thickness at hole", value=t_eff, units="in", source="input:t+tcheek"),
            dict(symbol="F_u", description="Ultimate strength", value=inp.Fu, units="ksi", source="input:Fu"),
            dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
            dict(symbol="D_h", description="Hole diameter", value=inp.Dh, units="in", source="input:Dh"),
        ],
        compute_fn=lambda v: min(
            v["b_e"],
            4.0*v["t_eff"],
            v["b_e"]*0.6*(v["F_u"]/v["F_y"])*math.sqrt(v["D_h"]/v["b_e"]) if v["b_e"]>0 else 0.0,
        ),
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="BTH sheet: beff=min(be,4t,be*0.6*(Fu/Fy)*sqrt(Dh/be))")]
    )

    A_n = compute_step(
        trace, "P-109A", "Padeye Hole", "Net area through hole",
        "A_n", "Net area at hole line (effective thickness)",
        "A_n = (w - D_h) · t_eff",
        variables=[
            dict(symbol="w", description="Width at hole", value=w_hole, units="in", source="step:P-062C"),
            dict(symbol="D_h", description="Hole diameter", value=inp.Dh, units="in", source="input:Dh"),
            dict(symbol="t_eff", description="Effective thickness at hole", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: (v["w"]-v["D_h"])*v["t_eff"],
        units="in^2",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 §3-3.3 net section at hole (implemented)")],
    )

    Cr = compute_step(
        trace, "P-110", "Padeye Hole", "Cr reduction factor",
        "C_r", "Net-section reduction factor",
        "C_r = 1 if D_p/D_h > 0.9 else 1 - 0.275 · sqrt(1 - (D_p/D_h)^2)",
        variables=[
            dict(symbol="D_p", description="Pin diameter", value=inp.Dp, units="in", source="input:Dp"),
            dict(symbol="D_h", description="Hole diameter", value=inp.Dh, units="in", source="input:Dh"),
        ],
        compute_fn=lambda v: 1.0 if (v["D_p"]/v["D_h"])>0.9 else (1.0 - 0.275*math.sqrt(max(0.0, 1.0 - (v["D_p"]/v["D_h"])**2))),
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="BTH sheet: Cr factor based on Dp/Dh")]
    )

    Pt = compute_step(
        trace, "P-111", "Padeye Hole", "Allowable tensile strength through pinhole",
        "P_t", "Allowable net-section tensile strength",
        "P_t = (F_u/(1.2·N_d)) · C_r · 2 · b_eff · t_eff",
        variables=[
            dict(symbol="F_u", description="Ultimate strength", value=inp.Fu, units="ksi", source="input:Fu"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
            dict(symbol="C_r", description="Reduction factor", value=Cr, units="-", source="step:P-110"),
            dict(symbol="b_eff", description="Effective edge width", value=beff, units="in", source="step:P-109"),
            dict(symbol="t_eff", description="Effective thickness", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: (v["F_u"]/(1.2*v["N_d"])) * v["C_r"] * 2.0 * v["b_eff"] * v["t_eff"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=3),
        references=[dict(type="derived", ref="BTH sheet: Pt (net tension)")]
    )

    Pb = compute_step(
        trace, "P-112", "Padeye Hole", "Allowable single-plane fracture strength",
        "P_b", "Allowable single-plane fracture strength",
        "P_b = (F_u/(1.2·N_d)) · C_r · (1.13·(R_edge - D_h/2) + 0.92·b_e/(1 + b_e/D_h)) · t_eff",
        variables=[
            dict(symbol="F_u", description="Ultimate strength", value=inp.Fu, units="ksi", source="input:Fu"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
            dict(symbol="C_r", description="Reduction factor", value=Cr, units="-", source="step:P-110"),
            dict(symbol="R_edge", description="Top edge distance", value=R_edge, units="in", source="step:P-062B"),
            dict(symbol="D_h", description="Hole diameter", value=inp.Dh, units="in", source="input:Dh"),
            dict(symbol="b_e", description="Edge width", value=be, units="in", source="step:P-108"),
            dict(symbol="t_eff", description="Effective thickness", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: (v["F_u"]/(1.2*v["N_d"])) * v["C_r"] * (
            1.13*(v["R_edge"] - v["D_h"]/2.0) + (0.92*v["b_e"]) / (1.0 + v["b_e"]/v["D_h"]) if v["D_h"]>0 else 0.0
        ) * v["t_eff"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=3),
        references=[dict(type="derived", ref="BTH sheet: Pb (single-plane fracture)")]
    )

    Z = compute_step(
        trace, "P-113", "Padeye Hole", "Shear plane length (Z)",
        "Z", "Shear plane length for splitting failure",
        "Z = R_edge - D_h/2",
        variables=[
            dict(symbol="R_edge", description="Top edge distance", value=R_edge, units="in", source="step:P-062B"),
            dict(symbol="D_h", description="Hole diameter", value=inp.Dh, units="in", source="input:Dh"),
        ],
        compute_fn=lambda v: max(v["R_edge"] - v["D_h"]/2.0, 0.0),
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="note", ref="BTH sheet uses Z from 2008 provisions; tool uses edge distance R-Dh/2")]
    )

    Zprime = compute_step(
        trace, "P-114", "Padeye Hole", "Reduced shear plane length (Z')",
        "Z'", "Reduced shear plane length for curved padeye",
        "Z' = 0 (flat padeye)",
        variables=[],
        compute_fn=lambda _v: 0.0,
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="note", ref="Curved padeye reduction not modeled; Z' assumed 0.")]
    )

    Pv = compute_step(
        trace, "P-115", "Padeye Hole", "Allowable double-plane shear strength",
        "P_v", "Allowable splitting shear strength",
        "P_v = 0.7·F_u/(1.2·N_d) · 2 · (Z - Z') · t_eff",
        variables=[
            dict(symbol="F_u", description="Ultimate strength", value=inp.Fu, units="ksi", source="input:Fu"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
            dict(symbol="Z", description="Shear plane length", value=Z, units="in", source="step:P-113"),
            dict(symbol="Z'", description="Reduced length", value=Zprime, units="in", source="step:P-114"),
            dict(symbol="t_eff", description="Effective thickness", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: 0.7*(v["F_u"]/(1.2*v["N_d"])) * 2.0 * max(v["Z"] - v["Z'"], 0.0) * v["t_eff"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=3),
        references=[dict(type="derived", ref="BTH sheet: Pv (splitting shear)")]
    )

    Pdem = P

    def chk_hole(label, cap):
        ratio = Pdem/cap if cap>0 else 1e9
        return [dict(label=label, demand=Pdem, capacity=cap, ratio=ratio, pass_fail="PASS" if ratio<=1 else "FAIL")]

    compute_step(
        trace, "P-116", "Padeye Hole", "Utilization: Pt",
        "U_pt", "Utilization ratio for Pt",
        "U_pt = P / P_t",
        variables=[
            dict(symbol="P", description="Hole region demand", value=Pdem, units="kip", source="input:P"),
            dict(symbol="P_t", description="Allowable tensile strength", value=Pt, units="kip", source="step:P-111"),
        ],
        compute_fn=lambda v: v["P"]/v["P_t"] if v["P_t"]>0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="BTH sheet: Pt utilization")],
        checks_builder=lambda _u,_r: chk_hole("Allowable Tensile Strength Through Pin Hole, Pt", Pt)
    )

    compute_step(
        trace, "P-117", "Padeye Hole", "Utilization: Pb",
        "U_pb", "Utilization ratio for Pb",
        "U_pb = P / P_b",
        variables=[
            dict(symbol="P", description="Hole region demand", value=Pdem, units="kip", source="input:P"),
            dict(symbol="P_b", description="Allowable fracture strength", value=Pb, units="kip", source="step:P-112"),
        ],
        compute_fn=lambda v: v["P"]/v["P_b"] if v["P_b"]>0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="BTH sheet: Pb utilization")],
        checks_builder=lambda _u,_r: chk_hole("Allowable Single Plane Fracture Strength, Pb", Pb)
    )

    compute_step(
        trace, "P-118", "Padeye Hole", "Utilization: Pv",
        "U_pv", "Utilization ratio for Pv",
        "U_pv = P / P_v",
        variables=[
            dict(symbol="P", description="Hole region demand", value=Pdem, units="kip", source="input:P"),
            dict(symbol="P_v", description="Allowable shear strength", value=Pv, units="kip", source="step:P-115"),
        ],
        compute_fn=lambda v: v["P"]/v["P_v"] if v["P_v"]>0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="BTH sheet: Pv utilization")],
        checks_builder=lambda _u,_r: chk_hole("Allowable Double Plane Shear Strength, Pv", Pv)
    )

    fp = compute_step(
        trace, "P-119", "Padeye Hole", "Pin bearing stress",
        "f_p", "Bearing stress between pin and plate",
        "f_p = P / (D_p · t_eff)",
        variables=[
            dict(symbol="P", description="Hole region demand", value=Pdem, units="kip", source="input:P"),
            dict(symbol="D_p", description="Pin diameter", value=inp.Dp, units="in", source="input:Dp"),
            dict(symbol="t_eff", description="Effective thickness", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: v["P"]/(v["D_p"]*v["t_eff"]) if (v["D_p"]>0 and v["t_eff"]>0) else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 §3-3.3.4 bearing stress definition")]
    )

    Fp_allow = compute_step(
        trace, "P-120", "Padeye Hole", "Allowable pin bearing stress",
        "F_p", "Allowable bearing stress per BTH",
        "F_p = 1.25 · F_y / N_d",
        variables=[
            dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        ],
        compute_fn=lambda v: 1.25*v["F_y"]/v["N_d"],
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 §3-3.3.4 eq. (3-55)")],
    )

    compute_step(
        trace, "P-121", "Padeye Hole", "Utilization: pin bearing",
        "U_pbear", "Utilization ratio for pin bearing",
        "U_pbear = f_p / F_p",
        variables=[
            dict(symbol="f_p", description="Bearing stress", value=fp, units="ksi", source="step:P-119"),
            dict(symbol="F_p", description="Allowable bearing stress", value=Fp_allow, units="ksi", source="step:P-120"),
        ],
        compute_fn=lambda v: v["f_p"]/v["F_p"] if v["F_p"]>0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 §3-3.3.4 bearing check")],
        checks_builder=lambda _u,_r: [dict(label="Pin Bearing Stress", demand=fp, capacity=Fp_allow, ratio=(fp/Fp_allow if Fp_allow>0 else 1e9), pass_fail="PASS" if fp<=Fp_allow else "FAIL")]
    )

    A_gross_hole = compute_step(
        trace, "P-122", "Padeye Hole", "Gross area at hole section",
        "A_h", "Gross area at hole section (effective thickness)",
        "A_h = w · t_eff",
        variables=[
            dict(symbol="w", description="Width at hole", value=w_hole, units="in", source="step:P-062C"),
            dict(symbol="t_eff", description="Effective thickness at hole", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: v["w"]*v["t_eff"],
        units="in^2",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:GrossAreaHole")]
    )

    Sz_hole = compute_step(
        trace, "P-123", "Padeye Hole", "Hole section modulus (strong axis)",
        "S_zh", "Section modulus about strong axis at hole",
        "S_zh = t_eff · w^2 / 6",
        variables=[
            dict(symbol="t_eff", description="Effective thickness at hole", value=t_eff, units="in", source="input:t+tcheek"),
            dict(symbol="w", description="Width at hole", value=w_hole, units="in", source="step:P-062C"),
        ],
        compute_fn=lambda v: v["t_eff"]*(v["w"]**2)/6.0,
        units="in^3",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:RectSectionModulusStrongHole")]
    )

    Sx_hole = compute_step(
        trace, "P-124", "Padeye Hole", "Hole section modulus (weak axis)",
        "S_xh", "Section modulus about weak axis at hole",
        "S_xh = w · t_eff^2 / 6",
        variables=[
            dict(symbol="w", description="Width at hole", value=w_hole, units="in", source="step:P-062C"),
            dict(symbol="t_eff", description="Effective thickness at hole", value=t_eff, units="in", source="input:t+tcheek"),
        ],
        compute_fn=lambda v: v["w"]*(v["t_eff"]**2)/6.0,
        units="in^3",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:RectSectionModulusWeakHole")]
    )

    sigma_gross_hole = compute_step(
        trace, "P-125", "Padeye Hole", "Gross normal stress at hole section",
        "σ_h", "Gross normal stress at hole section (axial + bending)",
        "σ_h = P_y/A_h + M_z/S_zh + M_x/S_xh",
        variables=[
            dict(symbol="P_y", description="Axial component", value=Py, units="kip", source="step:P-061"),
            dict(symbol="A_h", description="Hole gross area", value=A_gross_hole, units="in^2", source="step:P-122"),
            dict(symbol="M_z", description="Strong-axis moment", value=Mz, units="kip-in", source="step:P-064"),
            dict(symbol="S_zh", description="Strong-axis section modulus", value=Sz_hole, units="in^3", source="step:P-123"),
            dict(symbol="M_x", description="Weak-axis moment", value=Mx, units="kip-in", source="step:P-065"),
            dict(symbol="S_xh", description="Weak-axis section modulus", value=Sx_hole, units="in^3", source="step:P-124"),
        ],
        compute_fn=lambda v: (v["P_y"]/v["A_h"]) + (v["M_z"]/v["S_zh"]) + (v["M_x"]/v["S_xh"]),
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:GrossStressHole")]
    )

    sigma_local_net = compute_step(
        trace, "P-126", "Padeye Hole", "Local net-section stress",
        "σ_net", "Local net-section normal stress from pin load",
        "σ_net = P / A_n",
        variables=[
            dict(symbol="P", description="Hole region demand", value=Pdem, units="kip", source="input:P"),
            dict(symbol="A_n", description="Net area", value=A_n, units="in^2", source="step:P-109A"),
        ],
        compute_fn=lambda v: v["P"]/v["A_n"] if v["A_n"]>0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_padeye:NetStressHole")]
    )


    ratios=[]
    for s in trace.steps:
        if s.checks:
            for c in s.checks:
                ratios.append((c.ratio, s.id, c.label))
    ratios.sort(reverse=True, key=lambda x:x[0])
    gov = ratios[0] if ratios else (0.0,"","")

    trace.summary = {
        "governing": {"ratio": gov[0], "step_id": gov[1], "check": gov[2]},
        "note": "Padeye includes base combined stress (incl. out-of-plane), weld group elastic method (when applicable), and hole-region rupture/bearing/tear-out with boss plates included in effective thickness."
    }
    trace.tables = {"hole": {"t_eff": t_eff, "be": be, "beff": beff, "Cr": Cr, "Pt": Pt, "Pb": Pb, "Pv": Pv}}

    results = {
        "key_outputs": {
            "Px": {"value": Px, "units":"kip"},
            "Py": {"value": Py, "units":"kip"},
            "Pz": {"value": Pz, "units":"kip"},
            "governing_ratio": {"value": gov[0], "units":"-"},
            "governing_step": {"value": gov[1], "units":""},
            "governing_check": {"value": gov[2], "units":""},
        },
        "tables": trace.tables,
        "checks": _collect_checks(trace),
        "warnings": weld_warnings
    }
    return trace.to_dict(), results

def _solve_spreader_two_way(inp: SpreaderTwoWayInputs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inp = _apply_design_category(inp)
    d = inp.model_dump()
    trace = _base_trace(d)
    warnings: list[str] = []

    shapes = _get_shapes()
    shp = shapes.get(inp.shape)
    if shp is None:
        raise ValueError(f"Shape not found in bundled AISC database: {inp.shape}")
    shape_type = (shp.type_code or "").strip().upper()
    if shape_type not in _ALLOWED_SPREADER_TYPES:
        raise ValueError(f"Shape type {shape_type} is not supported for spreader analysis.")

    w_kipft = (shp.W_lbft or 0.0) / 1000.0
    depth_in = shp.d_in or shp.H_in or shp.OD_in or shp.B_in or 0.0

    x_left = inp.padeye_edge_ft
    x_right = inp.length_ft - inp.padeye_edge_ft
    support_spacing = x_right - x_left

    beam_solver = _load_beam_solver()
    point_loads = [{"x_ft": pl.x_ft, "P_kip": pl.P_kip} for pl in inp.point_loads]
    beam = beam_solver.solve_simple_beam(
        total_length_ft=inp.length_ft,
        support_positions_ft=[x_left, x_right],
        point_loads_kip=point_loads,
        w_kipft=w_kipft,
    )

    def _reaction_at(x_target: float) -> float:
        reactions = beam.get("reactions_kip", [])
        if not reactions:
            return 0.0
        closest = min(reactions, key=lambda r: abs(r["x_ft"] - x_target))
        return float(closest["reaction_kip"])

    R_left = _reaction_at(x_left)
    R_right = _reaction_at(x_right)

    total_load = sum(pl.P_kip for pl in inp.point_loads) + w_kipft * inp.length_ft
    if total_load <= 0:
        warnings.append("Total vertical load is zero; CG and sling geometry assumed at midspan.")
        x_cg = inp.length_ft / 2.0
    else:
        moment = sum(pl.P_kip * pl.x_ft for pl in inp.point_loads) + (w_kipft * inp.length_ft) * (inp.length_ft / 2.0)
        x_cg = moment / total_load

    d_left = abs(x_cg - x_left)
    d_right = abs(x_right - x_cg)
    d_long = max(d_left, d_right)
    if d_long <= 1e-9:
        warnings.append("Load CG aligns with padeye; sling geometry assumed vertical.")

    sling_angle_rad = math.radians(inp.sling_angle_deg)
    hook_height = d_long * math.tan(sling_angle_rad) if d_long > 0 else 0.0

    def _angle_from_horizontal(dist: float) -> float:
        if dist <= 0 and hook_height <= 0:
            return 90.0
        return math.degrees(math.atan2(hook_height, dist if dist > 0 else 1e-9))

    angle_left = _angle_from_horizontal(d_left)
    angle_right = _angle_from_horizontal(d_right)

    sling_len_left = math.hypot(hook_height, d_left)
    sling_len_right = math.hypot(hook_height, d_right)

    def _sling_tension(R: float, angle_deg: float, label: str) -> tuple[float, float]:
        if R <= 0:
            warnings.append(f"{label} padeye vertical reaction is non-positive; sling tension set to 0.")
            return 0.0, 0.0
        sin_a = math.sin(math.radians(angle_deg))
        if sin_a <= 1e-6:
            warnings.append(f"{label} sling angle too shallow; tension set to 0.")
            return 0.0, 0.0
        T = R / sin_a
        H = T * math.cos(math.radians(angle_deg))
        return T, H

    T_left, H_left = _sling_tension(R_left, angle_left, "Left")
    T_right, H_right = _sling_tension(R_right, angle_right, "Right")

    axial = 0.5 * (abs(H_left) + abs(H_right))
    Mecc = axial * (depth_in / 2.0 + inp.padeye_height_in) / 12.0 if depth_in > 0 else 0.0

    max_shear = float(beam.get("max_shear_kip", 0.0))
    max_moment = float(beam.get("max_moment_kipft", 0.0))
    max_moment_total = max_moment + Mecc

    compute_step(
        trace, "T-010", "Spreader Two-way", "Support spacing",
        "L_s", "Distance between padeyes",
        "L_s = L - 2 a",
        variables=[
            dict(symbol="L", description="Total beam length", value=inp.length_ft, units="ft", source="input:length_ft"),
            dict(symbol="a", description="Edge to padeye distance", value=inp.padeye_edge_ft, units="ft", source="input:padeye_edge_ft"),
        ],
        compute_fn=lambda v: v["L"] - 2.0 * v["a"],
        units="ft",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader_two_way:SupportSpacing")]
    )

    compute_step(
        trace, "T-011", "Spreader Two-way", "Total vertical load",
        "W", "Total vertical load including self-weight",
        "W = sum(P_i) + w L",
        variables=[
            dict(symbol="sum(P_i)", description="Total point loads", value=sum(pl.P_kip for pl in inp.point_loads), units="kip", source="input:point_loads"),
            dict(symbol="w", description="Self-weight per length", value=w_kipft, units="kip/ft", source="db:aisc.W_lbft"),
            dict(symbol="L", description="Total beam length", value=inp.length_ft, units="ft", source="input:length_ft"),
        ],
        compute_fn=lambda v: v["sum(P_i)"] + v["w"] * v["L"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader_two_way:TotalLoad")]
    )

    compute_step(
        trace, "T-012", "Spreader Two-way", "Load centroid",
        "x_cg", "Resultant load location from left end",
        "x_cg = sum(P_i x_i) / W",
        variables=[
            dict(symbol="sum(P_i x_i)", description="First moment of loads", value=(sum(pl.P_kip * pl.x_ft for pl in inp.point_loads) + (w_kipft * inp.length_ft) * (inp.length_ft / 2.0)), units="kip-ft", source="derived"),
            dict(symbol="W", description="Total load", value=total_load if total_load > 0 else 1.0, units="kip", source="step:T-011"),
        ],
        compute_fn=lambda v: v["sum(P_i x_i)"] / v["W"] if total_load > 0 else inp.length_ft / 2.0,
        units="ft",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader_two_way:LoadCentroid")]
    )

    compute_step(
        trace, "T-013", "Spreader Two-way", "Padeye reactions",
        "R", "Vertical reactions at padeyes",
        "R = beam_solver(...)",
        variables=[
            dict(symbol="R_left", description="Left reaction", value=R_left, units="kip", source="beam_solver"),
            dict(symbol="R_right", description="Right reaction", value=R_right, units="kip", source="beam_solver"),
        ],
        compute_fn=lambda v: v["R_left"] + v["R_right"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="note", ref="Reactions computed via beam_solver with supports at padeyes.")],
    )

    compute_step(
        trace, "T-014", "Spreader Two-way", "Hook height from sling angle",
        "h", "Vertical rise from padeye to hook",
        "h = d_long * tan(α_min)",
        variables=[
            dict(symbol="d_long", description="Long sling horizontal distance", value=d_long, units="ft", source="derived"),
            dict(symbol="α_min", description="Minimum sling angle", value=inp.sling_angle_deg, units="deg", source="input:sling_angle_deg"),
        ],
        compute_fn=lambda v: v["d_long"] * math.tan(math.radians(v["α_min"])) if v["d_long"] > 0 else 0.0,
        units="ft",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader_two_way:HookHeight")]
    )

    compute_step(
        trace, "T-015", "Spreader Two-way", "Sling tensions and axial compression",
        "T", "Sling tensions and horizontal components",
        "T = R / sin(α)",
        variables=[
            dict(symbol="T_left", description="Left sling tension", value=T_left, units="kip", source="derived"),
            dict(symbol="T_right", description="Right sling tension", value=T_right, units="kip", source="derived"),
            dict(symbol="H_left", description="Left horizontal component", value=H_left, units="kip", source="derived"),
            dict(symbol="H_right", description="Right horizontal component", value=H_right, units="kip", source="derived"),
        ],
        compute_fn=lambda v: v["T_left"] + v["T_right"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader_two_way:SlingTensionFromAngles")]
    )

    trace.summary = {
        "note": "Two-way spreader uses a simply supported beam model with overhangs at padeyes, vertical reactions from beam_solver, sling geometry based on CG alignment, and axial compression from sling horizontal components."
    }
    trace.tables = {
        "two_way": {
            "beam_length_ft": inp.length_ft,
            "padeye_edge_ft": inp.padeye_edge_ft,
            "padeye_spacing_ft": support_spacing,
            "shape": inp.shape,
            "shape_depth_in": depth_in,
            "self_weight_kipft": w_kipft,
            "point_loads": [pl.model_dump() for pl in inp.point_loads],
            "beam_solver": beam,
        }
    }

    results = {
        "key_outputs": {
            "support_spacing": {"value": support_spacing, "units":"ft"},
            "total_load": {"value": total_load, "units":"kip"},
            "cg_x": {"value": x_cg, "units":"ft"},
            "R_left": {"value": R_left, "units":"kip"},
            "R_right": {"value": R_right, "units":"kip"},
            "sling_angle_left": {"value": angle_left, "units":"deg"},
            "sling_angle_right": {"value": angle_right, "units":"deg"},
            "sling_length_left": {"value": sling_len_left, "units":"ft"},
            "sling_length_right": {"value": sling_len_right, "units":"ft"},
            "sling_tension_left": {"value": T_left, "units":"kip"},
            "sling_tension_right": {"value": T_right, "units":"kip"},
            "axial_compression": {"value": axial, "units":"kip"},
            "max_shear": {"value": max_shear, "units":"kip"},
            "max_moment": {"value": max_moment, "units":"kip-ft"},
            "max_moment_total": {"value": max_moment_total, "units":"kip-ft"},
        },
        "tables": trace.tables,
        "checks": [],
        "warnings": warnings,
    }
    return trace.to_dict(), results

def _solve_spreader(inp: SpreaderInputs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inp = _apply_design_category(inp)
    d = inp.model_dump()
    trace = _base_trace(d)

    shapes = _get_shapes()
    shp = shapes.get(inp.shape)
    if shp is None:
        raise ValueError(f"Shape not found in bundled AISC database: {inp.shape}")
    shape_type = (shp.type_code or "").strip().upper()
    if shape_type == "L":
        raise ValueError("Single angles (L) are not supported for spreader checks.")

    depth_in = shp.d_in or shp.H_in or shp.OD_in or shp.B_in or 0.0

    d_shape = compute_step(
        trace, "S-005", "Spreader Geometry", "Section depth used for eccentricity",
        "d", "Section depth used for eccentric axial moment",
        "d = d_section",
        variables=[dict(symbol="d_section", description="Section depth (AISC)", value=depth_in, units="in", source="db:aisc.depth")],
        compute_fn=lambda v: v["d_section"],
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader:SectionDepthFallback")]
    )

    e_total = compute_step(
        trace, "S-006", "Spreader Geometry", "Total eccentricity to axial load",
        "e", "Total eccentricity from centroid",
        "e = e_y + d/2",
        variables=[
            dict(symbol="e_y", description="Top padeye height", value=inp.ey, units="in", source="input:ey"),
            dict(symbol="d", description="Section depth", value=d_shape, units="in", source="step:S-005"),
        ],
        compute_fn=lambda v: v["e_y"] + v["d"] / 2.0,
        units="in",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader:TotalEccentricity")]
    )

    L = inp.span_L_ft
    w_kip_per_ft = ((shp.W_lbft or 0.0)/1000.0) if inp.include_self_weight else 0.0

    Msw = compute_step(
        trace, "S-010", "Spreader Loads", "Self-weight moment (strong axis)",
        "M_sw", "Midspan moment from uniform self-weight (simply supported)",
        "M_sw = w · L^2 / 8",
        variables=[
            dict(symbol="w", description="Member self-weight per length", value=w_kip_per_ft, units="kip/ft", source="db:aisc.W_lbft"),
            dict(symbol="L", description="Span between supports", value=L, units="ft", source="input:span_L_ft"),
        ],
        compute_fn=lambda v: v["w"]*(v["L"]**2)/8.0,
        units="kip-ft",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader:SelfWeightMomentSS")]
    )

    Mecc = compute_step(
        trace, "S-011A", "Spreader Loads", "Eccentric axial load moment",
        "M_e", "Moment from axial load eccentricity",
        "M_e = P ú e / 12",
        variables=[
            dict(symbol="P", description="Axial compression", value=inp.P_kip, units="kip", source="input:P_kip"),
            dict(symbol="e", description="Total eccentricity", value=e_total, units="in", source="step:S-006"),
        ],
        compute_fn=lambda v: (v["P"] * v["e"]) / 12.0,
        units="kip-ft",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader:EccentricAxialMoment")]
    )

    if inp.mx_includes_total:
        Mx_tot = compute_step(
            trace, "S-011", "Spreader Loads", "Total strong-axis moment",
            "M_x", "Total strong-axis moment (user-provided)",
            "M_x = M_x_app",
            variables=[
                dict(symbol="M_x_app", description="Applied strong-axis end moment", value=inp.Mx_app_kipft, units="kip-ft", source="input:Mx_app_kipft"),
            ],
            compute_fn=lambda v: v["M_x_app"],
            units="kip-ft",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
            references=[dict(type="derived", ref="backend._solve_spreader:TotalStrongAxisMomentUserProvided")]
        )
    else:
        Mx_tot = compute_step(
            trace, "S-011", "Spreader Loads", "Total strong-axis moment",
            "M_x", "Total strong-axis moment = applied + self-weight + eccentric axial",
            "M_x = M_x_app + M_sw + M_e",
            variables=[
                dict(symbol="M_x_app", description="Applied strong-axis end moment", value=inp.Mx_app_kipft, units="kip-ft", source="input:Mx_app_kipft"),
                dict(symbol="M_sw", description="Self-weight moment", value=Msw, units="kip-ft", source="step:S-010"),
                dict(symbol="M_e", description="Eccentric axial moment", value=Mecc, units="kip-ft", source="step:S-011A"),
            ],
            compute_fn=lambda v: v["M_x_app"]+v["M_sw"]+v["M_e"],
            units="kip-ft",
            rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
            references=[dict(type="derived", ref="backend._solve_spreader:TotalStrongAxisMomentWithEccentricity")]
        )

    My_tot = compute_step(
        trace, "S-012", "Spreader Loads", "Total weak-axis moment",
        "M_y", "Total weak-axis moment (applied)",
        "M_y = M_y_app",
        variables=[dict(symbol="M_y_app", description="Applied weak-axis end moment", value=inp.My_app_kipft, units="kip-ft", source="input:My_app_kipft")],
        compute_fn=lambda v: v["M_y_app"],
        units="kip-ft",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader:TotalWeakAxisMoment")]
    )

    fbx = compute_step(
        trace, "S-020", "Spreader Stresses", "Strong-axis bending stress",
        "f_bx", "Bending stress about strong axis",
        "f_bx = M_x · 12 / S_x",
        variables=[
            dict(symbol="M_x", description="Total strong-axis moment", value=Mx_tot, units="kip-ft", source="step:S-011"),
            dict(symbol="S_x", description="Strong-axis elastic section modulus", value=shp.Sx_in3, units="in^3", source="db:aisc.Sx_in3"),
        ],
        compute_fn=lambda v: (v["M_x"]*12.0)/v["S_x"] if v["S_x"]>0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 flexure stress definition")]
    )

    fby = compute_step(
        trace, "S-021", "Spreader Stresses", "Weak-axis bending stress",
        "f_by", "Bending stress about weak axis",
        "f_by = M_y · 12 / S_y",
        variables=[
            dict(symbol="M_y", description="Total weak-axis moment", value=My_tot, units="kip-ft", source="step:S-012"),
            dict(symbol="S_y", description="Weak-axis elastic section modulus", value=shp.Sy_in3, units="in^3", source="db:aisc.Sy_in3"),
        ],
        compute_fn=lambda v: (v["M_y"]*12.0)/v["S_y"] if v["S_y"]>0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 flexure stress definition")]
    )

    fa = compute_step(
        trace, "S-022", "Spreader Stresses", "Axial compressive stress",
        "f_a", "Axial stress from compression",
        "f_a = P / A",
        variables=[
            dict(symbol="P", description="Axial compression", value=inp.P_kip, units="kip", source="input:P_kip"),
            dict(symbol="A", description="Area", value=shp.A_in2, units="in^2", source="db:aisc.A_in2"),
        ],
        compute_fn=lambda v: v["P"]/v["A"] if v["A"]>0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 axial stress definition")]
    )

    fv = compute_step(
        trace, "S-023", "Spreader Stresses", "Average shear stress (conservative)",
        "f_v", "Average shear stress (V/A)",
        "f_v = V / A",
        variables=[
            dict(symbol="V", description="Shear force", value=inp.V_kip, units="kip", source="input:V_kip"),
            dict(symbol="A", description="Area", value=shp.A_in2, units="in^2", source="db:aisc.A_in2"),
        ],
        compute_fn=lambda v: v["V"]/v["A"] if v["A"]>0 else 0.0,
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
        references=[dict(type="derived", ref="backend._solve_spreader:AvgShearConservative")]
    )


    # --- Allowable bending stress about strong axis per ASME BTH-1-2023 (Ch.3, eqs. (3-16) and (3-17)) ---
    # Note: r_T is approximated as r_y of the full section; see Assumptions & Limitations in report.
    Af = compute_step(
    trace, "S-026", "Spreader Allowables", "Compression flange area (approx.)",
    "A_f", "Compression flange area (rectangular flange approximation)",
    "A_f = b_f · t_f",
    variables=[
        dict(symbol="b_f", description="Flange width", value=shp.bf_in or 0.0, units="in", source="db:aisc.bf_in"),
        dict(symbol="t_f", description="Flange thickness", value=shp.tf_in or 0.0, units="in", source="db:aisc.tf_in"),
    ],
    compute_fn=lambda v: v["b_f"]*v["t_f"],
    units="in^2",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="derived", ref="backend._solve_spreader:FlangeAreaApprox")]
    )

    rT = compute_step(
    trace, "S-027", "Spreader Allowables", "r_T approximation for LTB",
    "r_T", "Radius of gyration for compression flange + 1/3 web area (approximated)",
    "r_T ≈ r_y",
    variables=[dict(symbol="r_y", description="Minor-axis radius of gyration", value=shp.ry_in, units="in", source="db:aisc.ry_in")],
    compute_fn=lambda v: v["r_y"],
    units="in",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="note", ref="BTH-1 defines r_T as compression flange + 1/3 web area about axis in plane of web; tool uses conservative approximation r_T≈r_y (see report assumptions).")]
    )

    CLTB = compute_step(
    trace, "S-028", "Spreader Allowables", "CLTB factor",
    "C_LTB", "Lateral-torsional buckling modifier",
    "C_LTB = 1.00 (braced) or min(1.00, 2.00 · (E I_y)/(G J) · (b_f/H)^2 + 0.275)",
    variables=[
        dict(symbol="braced", description="Braced against twist at ends of unbraced length", value=1 if inp.braced_against_twist else 0, units="-", source="input:braced_against_twist"),
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="I_y", description="Minor-axis moment of inertia", value=shp.Iy_in4, units="in^4", source="db:aisc.Iy_in4"),
        dict(symbol="G", description="Shear modulus", value=11200.0, units="ksi", source="assumption:G=11200ksi"),
        dict(symbol="J", description="Torsional constant", value=shp.J_in4 or 0.0, units="in^4", source="db:aisc.J_in4"),
        dict(symbol="H", description="H parameter from AISC database", value=shp.H_const or 0.0, units="in", source="db:aisc.H"),
        dict(symbol="b_f", description="Flange width", value=shp.bf_in or 0.0, units="in", source="db:aisc.bf_in"),
    ],
    compute_fn=lambda v: 1.0 if v["braced"]==1 else min(1.0, (2.0*((v["E"]*v["I_y"])/(v["G"]*v["J"])) * ((v["b_f"]/v["H"])**2) + 0.275) if (v["J"]>0 and v["H"]>0) else 1.0),
    units="-",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=4),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-16) definition of C_LTB")],
    )

    Fb16 = compute_step(
    trace, "S-029", "Spreader Allowables", "Allowable bending stress (eq. 3-16)",
    "F_b,16", "Allowable bending stress from elastic LTB form",
    "F_b,16 = C_LTB · C_b · π^2 E / (N_d · (L_b/r_T)^2)",
    variables=[
        dict(symbol="C_LTB", description="LTB modifier", value=CLTB, units="-", source="step:S-028"),
        dict(symbol="C_b", description="Bending coefficient", value=inp.Cb, units="-", source="input:Cb"),
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        dict(symbol="L_b", description="Unbraced length", value=inp.Lb_ft*12.0, units="in", source="input:Lb_ft"),
        dict(symbol="r_T", description="r_T approximation", value=rT, units="in", source="step:S-027"),
    ],
    compute_fn=lambda v: (v["C_LTB"]*v["C_b"]*(math.pi**2)*v["E"]/(v["N_d"]*((v["L_b"]/v["r_T"])**2))) if v["r_T"]>0 else 0.0,
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-16)")],
    )

    Fb17 = compute_step(
    trace, "S-030", "Spreader Allowables", "Allowable bending stress (eq. 3-17)",
    "F_b,17", "Allowable bending stress from eq. (3-17) flange-based form",
    "F_b,17 = min(F_y/N_d, 0.66 · E · C_b / (N_d · (L_b · d / A_f)))",
    variables=[
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="C_b", description="Bending coefficient", value=inp.Cb, units="-", source="input:Cb"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        dict(symbol="L_b", description="Unbraced length", value=inp.Lb_ft*12.0, units="in", source="input:Lb_ft"),
        dict(symbol="d", description="Section depth", value=shp.d_in or 0.0, units="in", source="db:aisc.d_in"),
        dict(symbol="A_f", description="Compression flange area (approx.)", value=Af, units="in^2", source="step:S-026"),
    ],
    compute_fn=lambda v: min(v["F_y"]/v["N_d"], (0.66*v["E"]*v["C_b"]/(v["N_d"]*((v["L_b"]*v["d"]/v["A_f"]) if v["A_f"]>0 else 1e99)))),
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-17) (applicability per text)")]
    )

    Fb_allow_i = compute_step(
    trace, "S-031", "Spreader Allowables", "Allowable strong-axis bending stress (I/tee form)",
    "F_bx,i", "Allowable strong-axis bending stress from eqs. (3-16) and (3-17)",
    "F_bx,i = min(F_y/N_d, max(F_b,16, F_b,17))",
    variables=[
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        dict(symbol="F_b,16", description="Allowable stress from eq. (3-16)", value=Fb16, units="ksi", source="step:S-029"),
        dict(symbol="F_b,17", description="Allowable stress from eq. (3-17)", value=Fb17, units="ksi", source="step:S-030"),
    ],
    compute_fn=lambda v: min(v["F_y"]/v["N_d"], max(v["F_b,16"], v["F_b,17"])),
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 I-shape major-axis compression bending, eqs. (3-16) and (3-17)")]
    )

    shape_family = "I"
    if shape_type in {"C", "MC"}:
        shape_family = "CHANNEL"
    elif shape_type in {"WT", "MT", "ST"}:
        shape_family = "TEE"
    elif shape_type in {"HSS", "PIPE", "BOX"}:
        shape_family = "BOX"
    elif shape_type in {"2L"}:
        shape_family = "DOUBLE_ANGLE"
    elif shape_type not in {"W", "S", "M", "HP"}:
        shape_family = "OTHER"

    Fb_allow = compute_step(
    trace, "S-032", "Spreader Allowables", "Allowable strong-axis bending stress, Fbx",
    "F_bx", "Allowable strong-axis bending stress per BTH",
    "F_bx = {I/tee: F_bx,i} {channel: min(F_y/N_d, F_b,17)} {others: F_y/N_d}",
    variables=[
        dict(symbol="shape", description="Shape family", value=shape_family, units="-", source="db:aisc.Type"),
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        dict(symbol="F_bx,i", description="I/tee allowable", value=Fb_allow_i, units="ksi", source="step:S-031"),
        dict(symbol="F_b,17", description="Eq. (3-17) allowable", value=Fb17, units="ksi", source="step:S-030"),
    ],
    compute_fn=lambda v: (
        v["F_bx,i"] if v["shape"] in {"I", "TEE"} else
        min(v["F_y"]/v["N_d"], v["F_b,17"]) if v["shape"] == "CHANNEL" else
        (v["F_y"]/v["N_d"])
    ),
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 bending allowables by section type")]
    )

    Fby_allow = compute_step(
    trace, "S-033", "Spreader Allowables", "Allowable weak-axis bending stress",
    "F_by", "Allowable weak-axis bending stress (yield-controlled)",
    "F_by = F_y / N_d",
    variables=[
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
    ],
    compute_fn=lambda v: v["F_y"]/v["N_d"],
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-9) with Nd for bending (weak-axis assumed yield-controlled)")]
    )

    # --- Compression allowable Fa directly per BTH-1 (eqs. (3-4) and (3-5)) ---
    KLx = inp.KL_ft*12.0
    Cc = compute_step(
    trace, "S-040", "Spreader Allowables", "Cc limit",
    "C_c", "Transition slenderness parameter",
    "C_c = √(2 · π^2 · E / F_y)",
    variables=[
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
    ],
    compute_fn=lambda v: math.sqrt(2.0*(math.pi**2)*v["E"]/v["F_y"]),
    units="-",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-4)")]
    )

    KLr_min = compute_step(
    trace, "S-041", "Spreader Allowables", "Governing KL/r (min r)",
    "KLr", "Governing slenderness ratio using r_min",
    "KLr = KL / r_min",
    variables=[
        dict(symbol="KL", description="Effective length", value=KLx, units="in", source="input:KL_ft"),
        dict(symbol="r_min", description="Minimum radius of gyration", value=min(shp.rx_in, shp.ry_in), units="in", source="db:aisc"),
    ],
    compute_fn=lambda v: v["KL"]/v["r_min"] if v["r_min"]>0 else 1e99,
    units="-",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="derived", ref="backend._solve_spreader:KLrMin")]
    )

    Fa_allow = compute_step(
    trace, "S-042", "Spreader Allowables", "Allowable axial compression stress",
    "F_a", "Allowable axial compressive stress per BTH-1",
    "F_a = (1 - (KLr)^2/(2 C_c^2)) · (F_y/N_d) / (5/3 + 3·KLr/(8 C_c) - KLr^3/(8 C_c^3))  (for KLr < C_c); else F_a = π^2 E / (1.1 · N_d · KLr^2)",
    variables=[
        dict(symbol="KLr", description="Slenderness ratio", value=KLr_min, units="-", source="step:S-041"),
        dict(symbol="C_c", description="Transition parameter", value=Cc, units="-", source="step:S-040"),
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
    ],
    compute_fn=lambda v: (
        ((1.0 - (v["KLr"]**2)/(2.0*(v["C_c"]**2))) * (v["F_y"]/v["N_d"]) / ((5.0/3.0) + (3.0*v["KLr"]/(8.0*v["C_c"])) - ((v["KLr"]**3)/(8.0*(v["C_c"]**3)))))
        if v["KLr"] < v["C_c"] else
        ((math.pi**2)*v["E"]/(1.1*v["N_d"]*(v["KLr"]**2)))
    ),
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eqs. (3-4) and (3-5)")]
    )

    # Euler stresses divided by design factor for interaction equations (3-29) and (3-31)
    Fex_prime = compute_step(
    trace, "S-043", "Spreader Combined", "Fex' (Euler stress / Nd)",
    "F_ex'", "Euler buckling stress about x-axis divided by Nd",
    "F_ex' = π^2 E / (N_d · (KL/r_x)^2)",
    variables=[
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        dict(symbol="KL", description="Effective length", value=KLx, units="in", source="input:KL_ft"),
        dict(symbol="r_x", description="Radius of gyration about x", value=shp.rx_in, units="in", source="db:aisc.rx_in"),
    ],
    compute_fn=lambda v: (math.pi**2)*v["E"]/(v["N_d"]*((v["KL"]/v["r_x"])**2)) if v["r_x"]>0 else 0.0,
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 symbols for eq. (3-29): Fe' definition")]
    )

    Fey_prime = compute_step(
    trace, "S-044", "Spreader Combined", "Fey' (Euler stress / Nd)",
    "F_ey'", "Euler buckling stress about y-axis divided by Nd",
    "F_ey' = π^2 E / (N_d · (KL/r_y)^2)",
    variables=[
        dict(symbol="E", description="Elastic modulus", value=29000.0, units="ksi", source="assumption:E=29000ksi"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        dict(symbol="KL", description="Effective length", value=KLx, units="in", source="input:KL_ft"),
        dict(symbol="r_y", description="Radius of gyration about y", value=shp.ry_in, units="in", source="db:aisc.ry_in"),
    ],
    compute_fn=lambda v: (math.pi**2)*v["E"]/(v["N_d"]*((v["KL"]/v["r_y"])**2)) if v["r_y"]>0 else 0.0,
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 symbols for eq. (3-29): Fey' definition")]
    )

    # Shear allowable (kept as conservative form; BTH provides shear allowables in Ch.3)
    Fv_allow = compute_step(
    trace, "S-045", "Spreader Allowables", "Allowable shear stress",
    "F_v", "Allowable shear stress",
    "F_v = 0.6 F_y / N_d",
    variables=[
        dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
        dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
    ],
    compute_fn=lambda v: 0.6*v["F_y"]/v["N_d"],
    units="ksi",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
    references=[dict(type="derived", ref="backend._solve_spreader:ShearAllowableConservative")]
    )

    # --- Combined axial compression + biaxial bending per BTH-1 (eqs. (3-29) and (3-31)) ---
    U329 = compute_step(
        trace, "S-050", "Spreader Combined", "Interaction utilization (eq. 3-29)",
        "U_3-29", "Combined axial compression + biaxial bending utilization",
    "U = f_a/F_a + C_mx·f_bx/((1 - f_a/F_ex')·F_bx) + C_my·f_by/((1 - f_a/F_ey')·F_by) ≤ 1.0",
    variables=[
        dict(symbol="f_a", description="Axial compressive stress", value=fa, units="ksi", source="step:S-022"),
        dict(symbol="F_a", description="Allowable compressive stress", value=Fa_allow, units="ksi", source="step:S-042"),
        dict(symbol="C_mx", description="Cmx factor", value=inp.Cmx, units="-", source="input:Cmx"),
        dict(symbol="f_bx", description="Strong-axis bending stress", value=fbx, units="ksi", source="step:S-020"),
        dict(symbol="F_bx", description="Allowable strong-axis bending stress", value=Fb_allow, units="ksi", source="step:S-032"),
        dict(symbol="F_ex'", description="Euler stress / Nd about x", value=Fex_prime, units="ksi", source="step:S-043"),
        dict(symbol="C_my", description="Cmy factor", value=inp.Cmy, units="-", source="input:Cmy"),
        dict(symbol="f_by", description="Weak-axis bending stress", value=fby, units="ksi", source="step:S-021"),
        dict(symbol="F_by", description="Allowable weak-axis bending stress", value=Fby_allow, units="ksi", source="step:S-033"),
        dict(symbol="F_ey'", description="Euler stress / Nd about y", value=Fey_prime, units="ksi", source="step:S-044"),
    ],
    compute_fn=lambda v: (
        (v["f_a"]/v["F_a"] if v["F_a"]>0 else 0.0) +
        (v["C_mx"]*v["f_bx"]/(((1.0 - (v["f_a"]/v["F_ex'"]) )*v["F_bx"]) if (v["F_ex'"]>0 and (1.0 - v["f_a"]/v["F_ex'"])>0 and v["F_bx"]>0) else 1e99)) +
        (v["C_my"]*v["f_by"]/(((1.0 - (v["f_a"]/v["F_ey'"]) )*v["F_by"]) if (v["F_ey'"]>0 and (1.0 - v["f_a"]/v["F_ey'"])>0 and v["F_by"]>0) else 1e99))
    ),
    units="-",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-29)")],
        checks_builder=lambda u,r: [dict(label="Eq. (3-29) interaction", demand=u, capacity=1.0, ratio=u, pass_fail="PASS" if u<=1.0 else "FAIL")]
    )

    U330 = compute_step(
        trace, "S-051", "Spreader Combined", "Interaction utilization (eq. 3-30)",
        "U_3-30", "Combined axial + biaxial bending utilization (second condition)",
        "U = f_a/(F_y/N_d) + f_bx/F_bx + f_by/F_by ≤ 1.0",
        variables=[
            dict(symbol="f_a", description="Axial compressive stress", value=fa, units="ksi", source="step:S-022"),
            dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
            dict(symbol="f_bx", description="Strong-axis bending stress", value=fbx, units="ksi", source="step:S-020"),
            dict(symbol="F_bx", description="Allowable strong-axis bending stress", value=Fb_allow, units="ksi", source="step:S-032"),
            dict(symbol="f_by", description="Weak-axis bending stress", value=fby, units="ksi", source="step:S-021"),
            dict(symbol="F_by", description="Allowable weak-axis bending stress", value=Fby_allow, units="ksi", source="step:S-033"),
        ],
        compute_fn=lambda v: (
            (v["f_a"]/(v["F_y"]/v["N_d"]) if v["F_y"]>0 else 0.0) +
            (v["f_bx"]/v["F_bx"] if v["F_bx"]>0 else 0.0) +
            (v["f_by"]/v["F_by"] if v["F_by"]>0 else 0.0)
        ),
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-30)")],
        checks_builder=lambda u,r: [dict(label="Eq. (3-30) interaction", demand=u, capacity=1.0, ratio=u, pass_fail="PASS" if u<=1.0 else "FAIL")]
    )

    U331 = compute_step(
        trace, "S-052", "Spreader Combined", "Alternate interaction (eq. 3-31) when fa/Fa ≤ 0.15",
        "U_3-31", "Simplified combined axial + biaxial bending utilization",
        "U = f_a/F_a + f_bx/F_bx + f_by/F_by ≤ 1.0",
        variables=[
        dict(symbol="f_a", description="Axial compressive stress", value=fa, units="ksi", source="step:S-022"),
        dict(symbol="F_a", description="Allowable compressive stress", value=Fa_allow, units="ksi", source="step:S-042"),
        dict(symbol="f_bx", description="Strong-axis bending stress", value=fbx, units="ksi", source="step:S-020"),
        dict(symbol="F_bx", description="Allowable strong-axis bending stress", value=Fb_allow, units="ksi", source="step:S-032"),
        dict(symbol="f_by", description="Weak-axis bending stress", value=fby, units="ksi", source="step:S-021"),
        dict(symbol="F_by", description="Allowable weak-axis bending stress", value=Fby_allow, units="ksi", source="step:S-033"),
    ],
    compute_fn=lambda v: (v["f_a"]/v["F_a"] if v["F_a"]>0 else 0.0) + (v["f_bx"]/v["F_bx"] if v["F_bx"]>0 else 0.0) + (v["f_by"]/v["F_by"] if v["F_by"]>0 else 0.0),
    units="-",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-31) (per text applicability when f_a/F_a ≤ 0.15)")],
        checks_builder=lambda u,r: [dict(label="Eq. (3-31) alternate", demand=u, capacity=1.0, ratio=u, pass_fail="PASS" if u<=1.0 else "FAIL")]
    )

    U_combo = compute_step(
        trace, "S-053", "Spreader Combined", "Governing combined-stress utilization",
        "U", "Governing combined axial + biaxial bending utilization per BTH",
        "U = U_3-31 (if f_a/F_a ≤ 0.15) else max(U_3-29, U_3-30)",
        variables=[
            dict(symbol="f_a", description="Axial compressive stress", value=fa, units="ksi", source="step:S-022"),
            dict(symbol="F_a", description="Allowable compressive stress", value=Fa_allow, units="ksi", source="step:S-042"),
            dict(symbol="U_3-29", description="Eq. (3-29) utilization", value=U329, units="-", source="step:S-050"),
            dict(symbol="U_3-30", description="Eq. (3-30) utilization", value=U330, units="-", source="step:S-051"),
            dict(symbol="U_3-31", description="Eq. (3-31) utilization", value=U331, units="-", source="step:S-052"),
        ],
        compute_fn=lambda v: v["U_3-31"] if (v["F_a"]>0 and (v["f_a"]/v["F_a"])<=0.15) else max(v["U_3-29"], v["U_3-30"]),
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 para 3-2.4(a) applicability statement for eq. (3-31)")],
        checks_builder=lambda u,r: [dict(label="Governing combined stress", demand=u, capacity=1.0, ratio=u, pass_fail="PASS" if u<=1.0 else "FAIL")]
    )

    Ushear = compute_step(
        trace, "S-054", "Spreader Combined", "Shear utilization",
        "U_v", "Shear utilization ratio",
        "U_v = f_v / F_v",
    variables=[
        dict(symbol="f_v", description="Shear stress", value=fv, units="ksi", source="step:S-023"),
        dict(symbol="F_v", description="Allowable shear stress", value=Fv_allow, units="ksi", source="step:S-045"),
    ],
    compute_fn=lambda v: v["f_v"]/v["F_v"] if v["F_v"]>0 else 0.0,
    units="-",
    rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 shear check (Nd form)")],
        checks_builder=lambda _u,_r: [dict(label="Shear", demand=fv, capacity=Fv_allow, ratio=(fv/Fv_allow if Fv_allow>0 else 1e9), pass_fail="PASS" if fv<=Fv_allow else "FAIL")]
    )

    fn_comb = compute_step(
        trace, "S-055", "Spreader Combined", "Combined normal stress (abs sum)",
        "f_n", "Combined normal stress from axial + bending",
        "f_n = |f_a| + |f_bx| + |f_by|",
        variables=[
            dict(symbol="f_a", description="Axial stress", value=fa, units="ksi", source="step:S-022"),
            dict(symbol="f_bx", description="Strong-axis bending stress", value=fbx, units="ksi", source="step:S-020"),
            dict(symbol="f_by", description="Weak-axis bending stress", value=fby, units="ksi", source="step:S-021"),
        ],
        compute_fn=lambda v: abs(v["f_a"]) + abs(v["f_bx"]) + abs(v["f_by"]),
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="derived", ref="backend._solve_spreader:CombinedNormalAbs")]
    )

    U337 = compute_step(
        trace, "S-056", "Spreader Combined", "Combined normal + shear stress (eq. 3-37)",
        "U_3-37", "Combined normal + shear utilization",
        "U = sqrt(f_n^2 + 3 f_v^2) / (F_y/N_d)",
        variables=[
            dict(symbol="f_n", description="Combined normal stress", value=fn_comb, units="ksi", source="step:S-055"),
            dict(symbol="f_v", description="Shear stress", value=fv, units="ksi", source="step:S-023"),
            dict(symbol="F_y", description="Yield strength", value=inp.Fy, units="ksi", source="input:Fy"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        ],
        compute_fn=lambda v: math.sqrt(v["f_n"]**2 + 3.0*(v["f_v"]**2)) / (v["F_y"]/v["N_d"]) if v["F_y"]>0 else 1e9,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 eq. (3-37)")],
        checks_builder=lambda u,r: [dict(label="Eq. (3-37) normal + shear", demand=u, capacity=1.0, ratio=u, pass_fail="PASS" if u<=1.0 else "FAIL")]
    )

    # --- Optional weld sizing check (fillet weld, direct shear + axial only) ---
    if inp.weld_check and inp.weld_length_in > 0:
        F_w_allow = compute_step(
        trace, "S-060", "Spreader Weld", "Allowable weld throat shear stress (derived)",
        "F_w", "Allowable weld throat shear stress (conservative)",
        "F_w = 0.6 · E_xx / N_d",
        variables=[
            dict(symbol="E_xx", description="Weld metal tensile strength", value=inp.weld_exx_ksi, units="ksi", source="input:weld_exx_ksi"),
            dict(symbol="N_d", description="Design factor", value=inp.Nd, units="-", source="input:Nd"),
        ],
        compute_fn=lambda v: 0.6*v["E_xx"]/v["N_d"],
        units="ksi",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="derived", ref="exports.weld_allowable_shear:0.6Exx/Nd (conservative, verify against project welding spec)")]
        )

        Aw = compute_step(
        trace, "S-061", "Spreader Weld", "Effective weld throat area",
        "A_w", "Effective weld throat area",
        "A_w = 0.707 · w · L_eff",
        variables=[
            dict(symbol="w", description="Fillet weld leg size", value=inp.weld_size_in, units="in", source="input:weld_size_in"),
            dict(symbol="L_eff", description="Total effective weld length", value=inp.weld_length_in, units="in", source="input:weld_length_in"),
        ],
        compute_fn=lambda v: 0.707*v["w"]*v["L_eff"],
        units="in^2",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="code", ref="ASME BTH-1-2023 Ch.3 fillet weld throat definition (see para. 3-3.4.3)")]
        )

        Rw = compute_step(
        trace, "S-062", "Spreader Weld", "Weld capacity (direct shear + axial)",
        "R_w", "Allowable weld resultant force capacity (direct stress method)",
        "R_w = F_w · A_w",
        variables=[
            dict(symbol="F_w", description="Allowable weld throat shear stress", value=F_w_allow, units="ksi", source="step:S-060"),
            dict(symbol="A_w", description="Effective throat area", value=Aw, units="in^2", source="step:S-061"),
        ],
        compute_fn=lambda v: v["F_w"]*v["A_w"],
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="derived", ref="backend._solve_spreader:FilletWeldCapacityDirect")]
        )

        Pweld = compute_step(
        trace, "S-063", "Spreader Weld", "Resultant weld demand",
        "P_w", "Resultant weld demand (direct shear + axial only)",
        "P_w = √(P^2 + V^2)",
        variables=[
            dict(symbol="P", description="Axial compression (demand on weld)", value=inp.P_kip, units="kip", source="input:P_kip"),
            dict(symbol="V", description="Shear (demand on weld)", value=inp.V_kip, units="kip", source="input:V_kip"),
        ],
        compute_fn=lambda v: math.sqrt(v["P"]**2 + v["V"]**2),
        units="kip",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="derived", ref="backend._solve_spreader:WeldResultantDemand")]
        )

        Uw = compute_step(
        trace, "S-064", "Spreader Weld", "Weld utilization",
        "U_w", "Weld utilization ratio",
        "U_w = P_w / R_w",
        variables=[
            dict(symbol="P_w", description="Weld demand", value=Pweld, units="kip", source="step:S-063"),
            dict(symbol="R_w", description="Weld capacity", value=Rw, units="kip", source="step:S-062"),
        ],
        compute_fn=lambda v: v["P_w"]/v["R_w"] if v["R_w"]>0 else 1e99,
        units="-",
        rounding_rule=dict(rule="decimals", decimals_or_sigfigs=5),
        references=[dict(type="derived", ref="backend._solve_spreader:WeldUtilization")],
        checks_builder=lambda _u,_r: [dict(label="Weld (direct shear+axial)", demand=Pweld, capacity=Rw, ratio=Uw, pass_fail="PASS" if Uw<=1.0 else "FAIL")]
        )

    ratios=[]
    for s in trace.steps:
        if s.checks:
            for c in s.checks:
                ratios.append((c.ratio, s.id, c.label))
    ratios.sort(reverse=True, key=lambda x:x[0])
    gov = ratios[0] if ratios else (0.0,"","")

    trace.summary = {
        "shape": inp.shape,
        "governing": {"ratio": gov[0], "step_id": gov[1], "check": gov[2]},
        "note": "Spreader includes self-weight moment, ASME BTH-1-2023 lateral-torsional buckling allowables (eqs. 3-16/3-17), combined axial compression + biaxial bending interaction (eqs. 3-29/3-31), compression allowables (eqs. 3-4/3-5), shear check, and optional weld sizing (direct shear+axial)."
    }
    trace.tables = {"shape_props": {
        "name": shp.label, "type": shp.type_code, "A": shp.A_in2, "W_lbft": shp.W_lbft, "d": shp.d_in, "bf": shp.bf_in, "tf": shp.tf_in, "tw": shp.tw_in, "Sx": shp.Sx_in3, "Sy": shp.Sy_in3, "Ix": shp.Ix_in4, "Iy": shp.Iy_in4, "J": shp.J_in4, "H": shp.H_const, "rx": shp.rx_in, "ry": shp.ry_in
    }}

    results = {
        "key_outputs": {
            "M_sw": {"value": Msw, "units":"kip-ft"},
            "Mx_total": {"value": Mx_tot, "units":"kip-ft"},
            "fbx": {"value": fbx, "units":"ksi"},
            "fby": {"value": fby, "units":"ksi"},
            "fa": {"value": fa, "units":"ksi"},
            "U_3-29": {"value": U329, "units":"-"},
            "U_3-31": {"value": U331, "units":"-"},
            "U_combined": {"value": U_combo, "units":"-"},
            "U_shear": {"value": Ushear, "units":"-"},
            "governing_ratio": {"value": gov[0], "units":"-"},
            "governing_step": {"value": gov[1], "units":""},
            "governing_check": {"value": gov[2], "units":""},
        },
        "tables": trace.tables,
        "checks": _collect_checks(trace),
        "warnings": []
    }
    return trace.to_dict(), results

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        p = urlparse(path).path
        if p.startswith("/api/"):
            return str(FRONTEND_DIST / "index.html")
        if p == "/" or p == "":
            return str(FRONTEND_DIST / "index.html")
        candidate = FRONTEND_DIST / p.lstrip("/")
        if candidate.exists() and candidate.is_file():
            return str(candidate)
        return str(FRONTEND_DIST / "index.html")

    def _send_json(self, obj: Any, code: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            if u.path == "/api/health":
                return self._send_json({"ok": True, "tool": TOOL_ID, "version": TOOL_VERSION})
            if u.path == "/api/shackles":
                return self._send_json({"ok": True, "items": get_shackles()})
            if u.path == "/api/report.html":
                qs = parse_qs(u.query or "")
                mode = _mode_from_query(qs) or LAST_MODE
                if mode is None:
                    return self._send_json({"ok": False, "error": "No report available yet. Run Solve first."}, 404)
                p = _mode_dir(mode) / "report.html"
                if not p.exists():
                    return self._send_json({"ok": False, "error": f"No report.html yet for mode: {mode}. Run Solve first."}, 404)
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path.startswith("/api/download/"):
                tail = u.path.split("/api/download/",1)[1]
                qs = parse_qs(u.query or "")
                mode = None
                name = tail
                if "/" in tail:
                    maybe_mode, name = tail.split("/",1)
                    if maybe_mode in VALID_MODES:
                        mode = maybe_mode
                if mode is None:
                    mode = _mode_from_query(qs) or LAST_MODE
                p = _mode_dir(mode) / name
                if not p.exists():
                    return self._send_json({"ok": False, "error": f"File not found: {name}"}, 404)
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type","application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path == "/api/shapes":
                qs = parse_qs(u.query or "")
                q = (qs.get("q") or [""])[0]
                limit = int((qs.get("limit") or ["50"])[0])
                hits = _search_shapes(_get_shapes(), q, limit=limit)
                return self._send_json({"ok": True, "items": [{"name": s.label, "type": s.type_code, "W_lbft": s.W_lbft} for s in hits]})
            if u.path == "/api/spreader_shapes":
                return self._send_json({"ok": True, "items": _get_spreader_shapes()})
            return super().do_GET()
        except Exception as e:
            return self._send_json({"ok": False, "error": str(e)}, 500)

    def do_POST(self):
        try:
            u = urlparse(self.path)
            length = int(self.headers.get("Content-Length","0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}

            if u.path == "/api/optimize_theta":
                if payload.get("mode") not in ("padeye", None):
                    return self._send_json({"ok": False, "error": "mode must be padeye"}, 400)
                base = dict(payload)
                base["mode"] = "padeye"
                best = None
                best_results = None
                for theta in range(0, 91):
                    base["theta_deg"] = theta
                    try:
                        inp = PadeyeInputs(**base)
                        _trace, results = _solve_padeye(inp)
                        ratio = results["key_outputs"]["governing_ratio"]["value"]
                        if ratio is None or not math.isfinite(float(ratio)):
                            continue
                        if best is None or ratio > best["governing_ratio"]:
                            best = {"theta_deg": theta, "governing_ratio": float(ratio)}
                            best_results = results
                    except Exception:
                        continue
                if best is None or best_results is None:
                    return self._send_json({"ok": False, "error": "No valid theta results found."}, 400)
                return self._send_json({"ok": True, "best": best, "results": best_results})

            if u.path == "/api/optimize_shape":
                if payload.get("mode") not in ("spreader", None):
                    return self._send_json({"ok": False, "error": "mode must be spreader"}, 400)
                base = dict(payload)
                base["mode"] = "spreader"
                unique_shapes = {}
                for shp in _get_shapes().values():
                    unique_shapes[shp.label] = shp

                best = None
                best_results = None
                checked = 0
                passed = 0
                for shp in unique_shapes.values():
                    shape_type = (shp.type_code or "").strip().upper()
                    if shape_type == "L":
                        continue
                    if shape_type not in _ALLOWED_SPREADER_TYPES:
                        continue
                    checked += 1
                    try:
                        base["shape"] = shp.label
                        inp = SpreaderInputs(**base)
                        _trace, results = _solve_spreader(inp)
                        ratio = results["key_outputs"]["governing_ratio"]["value"]
                        if ratio is None or not math.isfinite(float(ratio)):
                            continue
                        if float(ratio) > 1.0:
                            continue
                        passed += 1
                        weight = float(shp.W_lbft or 0.0)
                        if best is None or weight < best["weight_lbft"]:
                            best = {
                                "shape": shp.label,
                                "weight_lbft": weight,
                                "governing_ratio": float(ratio),
                            }
                            best_results = results
                    except Exception:
                        continue

                if best is None or best_results is None:
                    return self._send_json(
                        {"ok": False, "error": "No shapes passed the checks for the given inputs."},
                        400,
                    )
                return self._send_json(
                    {"ok": True, "best": best, "results": best_results, "checked": checked, "passed": passed}
                )

            if u.path != "/api/solve":
                return self._send_json({"ok": False, "error": "Not found"}, 404)
            mode = payload.get("mode")
            if mode == "padeye":
                inp = PadeyeInputs(**payload)
                trace, results = _solve_padeye(inp)
            elif mode == "spreader":
                inp = SpreaderInputs(**payload)
                trace, results = _solve_spreader(inp)
            elif mode == "spreader_two_way":
                inp = SpreaderTwoWayInputs(**payload)
                trace, results = _solve_spreader_two_way(inp)
            else:
                return self._send_json({"ok": False, "error": "mode must be padeye, spreader, or spreader_two_way"}, 400)

            global LAST_MODE
            LAST_MODE = mode
            files = export_all(_mode_dir(mode), trace, results)
            results["artifacts"] = {Path(v).name: f"/api/download/{mode}/{Path(v).name}" for v in files.values()}
            return self._send_json({"ok": True, "results": results})
        except Exception as e:
            tb = traceback.format_exc()
            (RUN_DIR/"run.log").write_text(tb, encoding="utf-8")
            return self._send_json({"ok": False, "error": str(e), "detail": tb.splitlines()[-1]}, 500)

def main():
    port = int(os.environ.get("BTH_TOOL_PORT","0") or "0")
    os.chdir(str(FRONTEND_DIST))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    actual_port = httpd.server_address[1]
    try:
        (RUN_DIR/"server_port.txt").write_text(str(actual_port), encoding="utf-8")
    except Exception:
        pass
    httpd.serve_forever()

if __name__ == "__main__":
    main()
