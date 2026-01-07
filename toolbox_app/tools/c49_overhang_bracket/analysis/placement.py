from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..calc_trace import CalcTrace, compute_step
from ..constants import EDGE_LOAD_OUTBOARD_IN, TOP_DIAG_END_OFFSET_IN
from .girder_outline import GirderGeometry
from .geometry import (
    Segment,
    rect_polygon,
    poly_intersects_poly,
    segment_intersects_polygon,
    point_in_polygon,
)

@dataclass(frozen=True)
class PlacementOutcome:
    feasible: bool
    message: str
    # Geometry results (ft). Coordinate system:
    # x=0 at exterior web face; +x outboard
    # y=0 at top of girder; +y downward
    y_top_ft: float
    x_top_ft: float
    x_diag_top_ft: float
    y_bot_ft: float
    x_bot_ft: float
    depth_ft: float
    dx_ft: float
    theta_deg: float
    sin_theta: float
    bearing_face: str
    governing_clearance_in: float
    bracket_depth_in: float
    top_offset_ft: float
    bottom_offset_ft: float

def solve_best_placement(trace: CalcTrace, inputs: Dict[str, Any], girder_geom: GirderGeometry) -> PlacementOutcome:
    """
    Placement screening and selection.

    Constraints:
      - Top member / vertical member envelopes clear the girder polygon.
      - Bottom bearing pad is on a tagged vertical face.
      - Diagonal centerline segment remains outside the girder polygon (screening).

    Selection:
      - Prefer web face bearing.
      - Prefer deeper bracket.
      - Prefer smaller horizontal offset.
    """
    # Inputs
    soffit_offset_in = float(inputs["deck_soffit_offset_in"])
    plywood_in = float(inputs["plywood_thickness_in"])
    fourby_in = float(inputs["fourbyfour_thickness_in"])
    nailer_in = float(inputs["twobysix_thickness_in"])
    overhang_ft = float(inputs["overhang_length_ft"])

    max_depth_in = float(inputs["max_bracket_depth_in"])
    min_depth_in = float(inputs["min_bracket_depth_in"])
    clearance_in = float(inputs["clearance_in"])

    top_member_height_in = float(inputs["top_member_height_in"])
    vertical_member_width_in = float(inputs["vertical_member_width_in"])
    diagonal_thickness_in = float(inputs["diagonal_envelope_thickness_in"])  # currently not used in centerline screening
    bottom_pad_height_in = float(inputs["bottom_pad_height_in"])
    top_hanger_edge_clear_in = float(inputs["top_hanger_edge_clear_in"])

    # Girder geometry is passed as a typed object.
    poly = list(girder_geom.outline_poly_ft)
    faces: List[Segment] = list(girder_geom.bearing_faces)

    # P1: stack thickness
    t_stack_in = compute_step(
        trace,
        id="P1",
        section="Placement",
        title="Decking stack thickness to locate bracket top",
        output_symbol="t_{stack}",
        output_description="Total forming/decking stack thickness",
        equation_latex=r"t_{stack} = t_{ply} + t_{4x4} + t_{2x6}",
        variables=[
            {"symbol": "t_{ply}", "description": "Plywood thickness", "value": plywood_in, "units": "in", "source": "input:plywood_thickness_in"},
            {"symbol": "t_{4x4}", "description": "4x4 thickness (actual)", "value": fourby_in, "units": "in", "source": "input:fourbyfour_thickness_in"},
            {"symbol": "t_{2x6}", "description": "2x6 thickness (flat, actual)", "value": nailer_in, "units": "in", "source": "input:twobysix_thickness_in"},
        ],
        compute_fn=lambda: plywood_in + fourby_in + nailer_in,
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:P1"}],
    )

    # P2: y_top
    y_top_in = compute_step(
        trace,
        id="P2",
        section="Placement",
        title="Top-of-bracket elevation from girder top",
        output_symbol="y_{top}",
        output_description="Vertical location of top bracket reference line (down from girder top)",
        equation_latex=r"y_{top} = y_{soffit} + t_{stack}",
        variables=[
            {"symbol": "y_{soffit}", "description": "Deck soffit offset from girder top", "value": soffit_offset_in, "units": "in", "source": "input:deck_soffit_offset_in"},
            {"symbol": "t_{stack}", "description": "Decking stack thickness", "value": t_stack_in, "units": "in", "source": "step:P1"},
        ],
        compute_fn=lambda: soffit_offset_in + t_stack_in,
        units="in",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 2},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:P2"}],
    )
    y_top_ft = y_top_in / 12.0

    # P3: place the hanger line outboard of the *girder outline* at the top flange.
    # This keeps all bracket members outside the concrete outline (no interference).
    x_edge_top = max(x for x, y in poly if abs(y - 0.0) < 1e-9)
    clearance_ft = clearance_in / 12.0
    b_vert_ft = vertical_member_width_in / 12.0

    x_top_ft = compute_step(
        trace,
        id="P3",
        section="Placement",
        title="Top hanger line x-location (outboard of top flange edge)",
        output_symbol="x_{top}",
        output_description="Horizontal location of the hanger line from exterior web face",
        equation_latex=r"x_{top} = x_{edge} + c + \dfrac{b_{vert}}{2}",
        variables=[
            {"symbol": "x_{edge}", "description": "Outboard top-flange edge (envelope)", "value": x_edge_top, "units": "ft", "source": "db:girder_outline"},
            {"symbol": "c", "description": "Clearance", "value": clearance_ft, "units": "ft", "source": "input:clearance_in"},
            {"symbol": "b_{vert}", "description": "Vertical member envelope width", "value": b_vert_ft, "units": "ft", "source": "input:vertical_member_width_in"},
        ],
        compute_fn=lambda: x_edge_top + clearance_ft + b_vert_ft / 2.0,
        units="ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "assumption", "ref": "Assumption A7: hanger line placed outboard of top flange edge"}],
    )

    edge_out_ft = EDGE_LOAD_OUTBOARD_IN / 12.0
    diag_end_offset_ft = TOP_DIAG_END_OFFSET_IN / 12.0
    x_diag_top_ft = compute_step(
        trace,
        id="P3b",
        section="Placement",
        title="Diagonal connection x-location on top member",
        output_symbol="x_{diag}",
        output_description="Horizontal location of diagonal-to-top-member connection",
        equation_latex=r"x_{diag} = x_{edge} + a + e_{out} - e_{diag}",
        variables=[
            {"symbol": "x_{edge}", "description": "Outboard top-flange edge (envelope)", "value": x_edge_top, "units": "ft", "source": "db:girder_outline"},
            {"symbol": "a", "description": "Overhang length", "value": overhang_ft, "units": "ft", "source": "input:overhang_length_ft"},
            {"symbol": "e_{out}", "description": "Outboard load offset", "value": edge_out_ft, "units": "ft", "source": "constant:EDGE_LOAD_OUTBOARD_IN"},
            {"symbol": "e_{diag}", "description": "Diagonal offset from top member end", "value": diag_end_offset_ft, "units": "ft", "source": "constant:TOP_DIAG_END_OFFSET_IN"},
        ],
        compute_fn=lambda: x_edge_top + overhang_ft + edge_out_ft - diag_end_offset_ft,
        units="ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:P3b"}],
    )

    top_offset_ft = compute_step(
        trace,
        id="P3c",
        section="Placement",
        title="Top horizontal offset (hanger line to diagonal connection)",
        output_symbol=r"\Delta x_{top}",
        output_description="Horizontal offset between hanger line and diagonal connection",
        equation_latex=r"\Delta x_{top} = x_{diag} - x_{top}",
        variables=[
            {"symbol": "x_{diag}", "description": "Diagonal connection x-location", "value": x_diag_top_ft, "units": "ft", "source": "step:P3b"},
            {"symbol": "x_{top}", "description": "Hanger line x-location", "value": x_top_ft, "units": "ft", "source": "step:P3"},
        ],
        compute_fn=lambda: x_diag_top_ft - x_top_ft,
        units="ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:P3c"}],
    )

    if x_diag_top_ft <= x_top_ft + 1e-6:
        return PlacementOutcome(
            feasible=False,
            message="Diagonal connection falls inboard of hanger line for fixed top offset.",
            y_top_ft=y_top_ft,
            x_top_ft=x_top_ft,
            x_diag_top_ft=x_diag_top_ft,
            y_bot_ft=float("nan"),
            x_bot_ft=float("nan"),
            depth_ft=float("nan"),
            dx_ft=float("nan"),
            theta_deg=float("nan"),
            sin_theta=float("nan"),
            bearing_face="",
            governing_clearance_in=float(inputs.get("clearance_in", 0.0)),
            bracket_depth_in=float("nan"),
            top_offset_ft=top_offset_ft,
            bottom_offset_ft=float("nan"),
        )

    # Member envelopes (rectangles)
    top_h_ft = top_member_height_in / 12.0
    extra_up_ft = top_hanger_edge_clear_in / 12.0

    top_rect = rect_polygon(
        x_top_ft - b_vert_ft / 2.0,
        y_top_ft - top_h_ft / 2.0,
        x_top_ft + b_vert_ft / 2.0,
        y_top_ft + top_h_ft / 2.0,
    )
    vert_rect = rect_polygon(
        x_top_ft - b_vert_ft / 2.0,
        -extra_up_ft,
        x_top_ft + b_vert_ft / 2.0,
        y_top_ft,
    )

    top_ok = not poly_intersects_poly(top_rect, poly)
    vert_ok = not poly_intersects_poly(vert_rect, poly)

    compute_step(
        trace,
        id="P4",
        section="Placement",
        title="Top member interference check",
        output_symbol="I_{top}",
        output_description="Indicator (1=clear, 0=interferes) for top member vs girder outline",
        equation_latex=r"I_{top} = \mathrm{no\_intersection}(R_{top}, \mathcal{P})",
        variables=[
            {"symbol": "R_{top}", "description": "Top member rectangle envelope", "value": "rect", "units": "-", "source": "derived:placement"},
            {"symbol": r"\mathcal{P}", "description": "Girder outline polygon", "value": "poly", "units": "-", "source": "db:txdot_outline"},
        ],
        compute_fn=lambda: 1.0 if top_ok else 0.0,
        units="-",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 0},
        references=[{"type": "derived", "ref": "geometry.poly_intersects_poly"}],
        checks_builder=lambda un: [{"label": "Top member clears girder", "demand": 1.0 - un, "capacity": 0.0, "ratio": (0.0 if un == 1 else 1.0), "pass_fail": ("PASS" if un == 1 else "FAIL")}],
    )

    compute_step(
        trace,
        id="P5",
        section="Placement",
        title="Vertical/hanger member interference check",
        output_symbol="I_{vert}",
        output_description="Indicator (1=clear, 0=interferes) for vertical member vs girder outline",
        equation_latex=r"I_{vert} = \mathrm{no\_intersection}(R_{vert}, \mathcal{P})",
        variables=[
            {"symbol": "R_{vert}", "description": "Vertical member rectangle envelope", "value": "rect", "units": "-", "source": "derived:placement"},
            {"symbol": r"\mathcal{P}", "description": "Girder outline polygon", "value": "poly", "units": "-", "source": "db:txdot_outline"},
        ],
        compute_fn=lambda: 1.0 if vert_ok else 0.0,
        units="-",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 0},
        references=[{"type": "derived", "ref": "geometry.poly_intersects_poly"}],
        checks_builder=lambda un: [{"label": "Vertical member clears girder", "demand": 1.0 - un, "capacity": 0.0, "ratio": (0.0 if un == 1 else 1.0), "pass_fail": ("PASS" if un == 1 else "FAIL")}],
    )

    if not top_ok or not vert_ok:
        return PlacementOutcome(
            feasible=False,
            message="Top/vertical member interferes with girder outline at computed top location.",
            y_top_ft=y_top_ft,
            x_top_ft=x_top_ft,
            x_diag_top_ft=x_diag_top_ft,
            y_bot_ft=float("nan"),
            x_bot_ft=float("nan"),
            depth_ft=float("nan"),
            dx_ft=float("nan"),
            theta_deg=float("nan"),
            sin_theta=float("nan"),
            bearing_face="",
            governing_clearance_in=float(inputs.get("clearance_in", 0.0)),
            bracket_depth_in=float("nan"),
            top_offset_ft=top_offset_ft,
            bottom_offset_ft=float("nan"),
        )

    # Bottom scan
    pad_h_ft = bottom_pad_height_in / 12.0
    step_in = 0.5

    def diagonal_clear(p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        # Allow boundary touch at the bearing face by shifting p2 slightly outboard.
        eps = 1e-4
        p2s = (p2[0] + eps, p2[1])
        if segment_intersects_polygon(p1, p2s, poly):
            return False
        for t in (0.2, 0.4, 0.6, 0.8):
            x = p1[0] + t * (p2s[0] - p1[0])
            y = p1[1] + t * (p2s[1] - p1[1])
            if point_in_polygon((x, y), poly):
                return False
        return True

    best: Optional[PlacementOutcome] = None
    best_score = -1e99

    for face in faces:
        if not face.is_vertical:
            continue

        x_face = face.x_const
        y_min = max(face.y_min, y_top_ft + min_depth_in / 12.0)
        y_max = min(face.y_max, y_top_ft + max_depth_in / 12.0)
        if y_max <= y_min + 1e-9:
            continue

        n_steps = int(math.floor((y_max - y_min) * 12.0 / step_in)) + 1
        for k in range(n_steps):
            y_bot = y_min + (k * step_in) / 12.0

            pad_ymin = y_bot - pad_h_ft / 2.0
            pad_ymax = y_bot + pad_h_ft / 2.0
            if pad_ymin < face.y_min - 1e-9 or pad_ymax > face.y_max + 1e-9:
                continue

            # Bearing pad envelope: assumed entirely *outboard* of the vertical bearing face.
            # To avoid false rejections from boundary contact, shift the member centerline slightly outboard.
            eps_ft = (1.0 / 16.0) / 12.0  # 1/16-in numerical offset
            x_bearing = x_face + eps_ft

            pad_thk_ft = (0.25) / 12.0  # 1/4-in assumed pad thickness for screening
            pad_rect = rect_polygon(x_bearing, pad_ymin, x_bearing + pad_thk_ft, pad_ymax)
            if poly_intersects_poly(pad_rect, poly):
                continue

            p_top = (x_diag_top_ft, y_top_ft)
            p_bot = (x_bearing, y_bot)
            if not diagonal_clear(p_top, p_bot):
                continue

            depth_ft = y_bot - y_top_ft
            dx_ft = max(abs(x_diag_top_ft - x_bearing), 1e-6)
            bottom_offset_ft = abs(x_top_ft - x_bearing)
            theta = math.degrees(math.atan2(depth_ft, dx_ft))
            sin_theta = math.sin(math.radians(theta))

            web_bonus = 10.0 if face.tag == "web" else 0.0
            score = web_bonus + (depth_ft * 12.0) * 1.0 - (dx_ft * 12.0) * 0.75 + (theta) * 0.25

            if score > best_score:
                best_score = score
                best = PlacementOutcome(
                    feasible=True,
                    message="Placement feasible.",
                    y_top_ft=y_top_ft,
                    x_top_ft=x_top_ft,
                    x_diag_top_ft=x_diag_top_ft,
                    y_bot_ft=y_bot,
                    x_bot_ft=x_face,
                    depth_ft=depth_ft,
                    dx_ft=dx_ft,
                    theta_deg=theta,
                    sin_theta=sin_theta,
                    bearing_face=face.tag,
                    governing_clearance_in=float(clearance_in),
                    bracket_depth_in=float(depth_ft * 12.0),
                    top_offset_ft=top_offset_ft,
                    bottom_offset_ft=bottom_offset_ft,
                )

    if best is None:
        compute_step(
            trace,
            id="P6",
            section="Placement",
            title="Bottom bearing feasibility",
            output_symbol="I_{bear}",
            output_description="Indicator (1=found feasible bottom bearing point, 0=none)",
            equation_latex=r"I_{bear} = \mathrm{exists\_feasible}(B \in \mathcal{F})",
            variables=[
                {"symbol": r"\mathcal{F}", "description": "Eligible vertical bearing faces", "value": [f.tag for f in faces], "units": "-", "source": "db:txdot_outline"},
                {"symbol": r"d_{max}", "description": "Max bracket depth", "value": max_depth_in, "units": "in", "source": "input:max_bracket_depth_in"},
            ],
            compute_fn=lambda: 0.0,
            units="-",
            rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 0},
            references=[{"type": "derived", "ref": "placement.solve_best_placement:scan"}],
            checks_builder=lambda un: [{"label": "Feasible bottom bearing point found", "demand": 1.0 - un, "capacity": 0.0, "ratio": (0.0 if un == 1 else 1.0), "pass_fail": "FAIL"}],
        )
        return PlacementOutcome(
            feasible=False,
            message="No feasible bottom bearing point found on eligible vertical faces within depth/clearance limits.",
            y_top_ft=y_top_ft,
            x_top_ft=x_top_ft,
            y_bot_ft=float("nan"),
            x_bot_ft=float("nan"),
            depth_ft=float("nan"),
            dx_ft=float("nan"),
            theta_deg=float("nan"),
            sin_theta=float("nan"),
            bearing_face="",
            governing_clearance_in=float(inputs.get("clearance_in", 0.0)),
            bracket_depth_in=float("nan"),
        )

    # Report selected geometry
    compute_step(
        trace,
        id="P6",
        section="Placement",
        title="Selected bracket depth",
        output_symbol="d",
        output_description="Bracket depth between top line and bottom bearing point",
        equation_latex=r"d = y_{bot} - y_{top}",
        variables=[
            {"symbol": r"y_{bot}", "description": "Bottom bearing point elevation", "value": best.y_bot_ft, "units": "ft", "source": "derived:placement"},
            {"symbol": r"y_{top}", "description": "Top line elevation", "value": best.y_top_ft, "units": "ft", "source": "step:P2"},
        ],
        compute_fn=lambda: best.depth_ft,
        units="ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:selected"}],
    )

    compute_step(
        trace,
        id="P7",
        section="Placement",
        title="Horizontal offset from diagonal connection to bottom bearing face",
        output_symbol=r"\Delta x",
        output_description="Horizontal offset between diagonal top connection and bottom bearing face",
        equation_latex=r"\Delta x = \left|x_{diag} - x_{bot}\right|",
        variables=[
            {"symbol": r"x_{diag}", "description": "Diagonal top x-location", "value": best.x_diag_top_ft, "units": "ft", "source": "step:P3b"},
            {"symbol": r"x_{bot}", "description": "Bottom face x-location", "value": best.x_bot_ft, "units": "ft", "source": "derived:placement"},
        ],
        compute_fn=lambda: best.dx_ft,
        units="ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:selected"}],
    )

    compute_step(
        trace,
        id="P7b",
        section="Placement",
        title="Bottom horizontal offset (hanger line to bearing face)",
        output_symbol=r"\Delta x_{bot}",
        output_description="Horizontal offset between hanger line and bottom bearing face",
        equation_latex=r"\Delta x_{bot} = \left|x_{top} - x_{bot}\right|",
        variables=[
            {"symbol": r"x_{top}", "description": "Hanger line x-location", "value": best.x_top_ft, "units": "ft", "source": "step:P3"},
            {"symbol": r"x_{bot}", "description": "Bottom face x-location", "value": best.x_bot_ft, "units": "ft", "source": "derived:placement"},
        ],
        compute_fn=lambda: abs(best.x_top_ft - best.x_bot_ft),
        units="ft",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:P7b"}],
    )

    compute_step(
        trace,
        id="P8",
        section="Placement",
        title="Diagonal angle sine from geometry",
        output_symbol=r"\sin\theta",
        output_description="Sine of diagonal angle (computed from depth and offset)",
        equation_latex=r"\sin\theta = \dfrac{d}{\sqrt{d^2 + (\Delta x)^2}}",
        variables=[
            {"symbol": r"d", "description": "Bracket depth", "value": best.depth_ft, "units": "ft", "source": "step:P6"},
            {"symbol": r"\Delta x", "description": "Horizontal offset", "value": best.dx_ft, "units": "ft", "source": "step:P7"},
        ],
        compute_fn=lambda: best.sin_theta,
        units="-",
        rounding_rule={"rule": "decimals", "decimals_or_sigfigs": 4},
        references=[{"type": "derived", "ref": "placement.solve_best_placement:P8"}],
        checks_builder=lambda un: [{
            "label": "Depth ≤ max depth",
            "demand": best.depth_ft * 12.0,
            "capacity": max_depth_in,
            "ratio": (best.depth_ft * 12.0 / max_depth_in if max_depth_in > 0 else 999.0),
            "pass_fail": ("PASS" if best.depth_ft * 12.0 <= max_depth_in + 1e-9 else "FAIL"),
        }],
    )

    return best


def solve_best_placement_fast(inputs: Dict[str, Any], girder_geom: GirderGeometry) -> PlacementOutcome:
    """Fast placement solve without CalcTrace side effects.

    This is intended for *interactive UI preview* where the section diagram needs to
    update on every input change.

    The logic matches solve_best_placement(...):
      - locate top line from soffit + decking stack
      - place hanger line outboard of top flange edge by clearance
      - enforce member envelope interference screening
      - scan eligible vertical bearing faces for feasible bottom bearing point

    Returns PlacementOutcome.
    """

    # Inputs
    soffit_offset_in = float(inputs.get("deck_soffit_offset_in", 0.0))
    plywood_in = float(inputs.get("plywood_thickness_in", 0.75))
    fourby_in = float(inputs.get("fourbyfour_thickness_in", 3.5))
    nailer_in = float(inputs.get("twobysix_thickness_in", 1.5))
    overhang_ft = float(inputs.get("overhang_length_ft", 6.0))

    max_depth_in = float(inputs.get("max_bracket_depth_in", 50.0))
    min_depth_in = float(inputs.get("min_bracket_depth_in", 12.0))
    clearance_in = float(inputs.get("clearance_in", 0.25))

    top_member_height_in = float(inputs.get("top_member_height_in", 3.0))
    vertical_member_width_in = float(inputs.get("vertical_member_width_in", 2.0))
    bottom_pad_height_in = float(inputs.get("bottom_pad_height_in", 2.0))
    top_hanger_edge_clear_in = float(inputs.get("top_hanger_edge_clear_in", 0.0))

    poly = list(girder_geom.outline_poly_ft)
    faces: List[Segment] = list(girder_geom.bearing_faces)

    # Stack thickness and top line
    t_stack_in = plywood_in + fourby_in + nailer_in
    y_top_in = soffit_offset_in + t_stack_in

    y_top_ft = y_top_in / 12.0

    # Top flange edge (outboard)
    x_edge_top = max(x for x, y in poly if abs(y - 0.0) < 1e-9)

    clearance_ft = clearance_in / 12.0
    b_vert_ft = vertical_member_width_in / 12.0

    x_top_ft = x_edge_top + clearance_ft + b_vert_ft / 2.0

    edge_out_ft = EDGE_LOAD_OUTBOARD_IN / 12.0
    diag_end_offset_ft = TOP_DIAG_END_OFFSET_IN / 12.0
    x_diag_top_ft = x_edge_top + overhang_ft + edge_out_ft - diag_end_offset_ft
    top_offset_ft = x_diag_top_ft - x_top_ft

    if x_diag_top_ft <= x_top_ft + 1e-6:
        return PlacementOutcome(
            feasible=False,
            message="Diagonal connection falls inboard of hanger line for fixed top offset.",
            y_top_ft=y_top_ft,
            x_top_ft=x_top_ft,
            x_diag_top_ft=x_diag_top_ft,
            y_bot_ft=float("nan"),
            x_bot_ft=float("nan"),
            depth_ft=float("nan"),
            dx_ft=float("nan"),
            theta_deg=float("nan"),
            sin_theta=float("nan"),
            bearing_face="",
            governing_clearance_in=float(clearance_in),
            bracket_depth_in=float("nan"),
            top_offset_ft=top_offset_ft,
            bottom_offset_ft=float("nan"),
        )

    # Member envelopes
    top_h_ft = top_member_height_in / 12.0
    extra_up_ft = top_hanger_edge_clear_in / 12.0

    top_rect = rect_polygon(
        x_top_ft - b_vert_ft / 2.0,
        y_top_ft - top_h_ft / 2.0,
        x_top_ft + b_vert_ft / 2.0,
        y_top_ft + top_h_ft / 2.0,
    )
    vert_rect = rect_polygon(
        x_top_ft - b_vert_ft / 2.0,
        -extra_up_ft,
        x_top_ft + b_vert_ft / 2.0,
        y_top_ft,
    )

    top_ok = not poly_intersects_poly(top_rect, poly)
    vert_ok = not poly_intersects_poly(vert_rect, poly)

    if not top_ok or not vert_ok:
        return PlacementOutcome(
            feasible=False,
            message="Top/vertical member interferes with girder outline at computed top location.",
            y_top_ft=y_top_ft,
            x_top_ft=x_top_ft,
            x_diag_top_ft=x_diag_top_ft,
            y_bot_ft=float("nan"),
            x_bot_ft=float("nan"),
            depth_ft=float("nan"),
            dx_ft=float("nan"),
            theta_deg=float("nan"),
            sin_theta=float("nan"),
            bearing_face="",
            governing_clearance_in=float(clearance_in),
            bracket_depth_in=float("nan"),
            top_offset_ft=top_offset_ft,
            bottom_offset_ft=float("nan"),
        )

    # Bottom scan
    pad_h_ft = bottom_pad_height_in / 12.0
    step_in = 0.5

    def diagonal_clear(p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        eps = 1e-4
        p2s = (p2[0] + eps, p2[1])
        if segment_intersects_polygon(p1, p2s, poly):
            return False
        for t in (0.2, 0.4, 0.6, 0.8):
            x = p1[0] + t * (p2s[0] - p1[0])
            y = p1[1] + t * (p2s[1] - p1[1])
            if point_in_polygon((x, y), poly):
                return False
        return True

    best: Optional[PlacementOutcome] = None
    best_score = -1e99

    for face in faces:
        if not face.is_vertical:
            continue
        x_face = face.x_const
        y_min = max(face.y_min, y_top_ft + min_depth_in / 12.0)
        y_max = min(face.y_max, y_top_ft + max_depth_in / 12.0)
        if y_max <= y_min + 1e-9:
            continue

        n_steps = int(math.floor((y_max - y_min) * 12.0 / step_in)) + 1
        for k in range(n_steps):
            y_bot = y_min + (k * step_in) / 12.0

            pad_ymin = y_bot - pad_h_ft / 2.0
            pad_ymax = y_bot + pad_h_ft / 2.0
            if pad_ymin < face.y_min - 1e-9 or pad_ymax > face.y_max + 1e-9:
                continue

            eps_ft = (1.0 / 16.0) / 12.0
            x_bearing = x_face + eps_ft

            pad_thk_ft = (0.25) / 12.0
            pad_rect = rect_polygon(x_bearing, pad_ymin, x_bearing + pad_thk_ft, pad_ymax)
            if poly_intersects_poly(pad_rect, poly):
                continue

            p_top = (x_diag_top_ft, y_top_ft)
            p_bot = (x_bearing, y_bot)
            if not diagonal_clear(p_top, p_bot):
                continue

            depth_ft = y_bot - y_top_ft
            dx_ft = max(abs(x_diag_top_ft - x_bearing), 1e-6)
            bottom_offset_ft = abs(x_top_ft - x_bearing)
            theta = math.degrees(math.atan2(depth_ft, dx_ft))
            sin_theta = math.sin(math.radians(theta))

            web_bonus = 10.0 if face.tag == "web" else 0.0
            score = web_bonus + (depth_ft * 12.0) * 1.0 - (dx_ft * 12.0) * 0.75 + (theta) * 0.25

            if score > best_score:
                best_score = score
                best = PlacementOutcome(
                    feasible=True,
                    message="Placement feasible.",
                    y_top_ft=y_top_ft,
                    x_top_ft=x_top_ft,
                    x_diag_top_ft=x_diag_top_ft,
                    y_bot_ft=y_bot,
                    x_bot_ft=x_face,
                    depth_ft=depth_ft,
                    dx_ft=dx_ft,
                    theta_deg=theta,
                    sin_theta=sin_theta,
                    bearing_face=face.tag,
                    governing_clearance_in=float(clearance_in),
                    bracket_depth_in=float(depth_ft * 12.0),
                    top_offset_ft=top_offset_ft,
                    bottom_offset_ft=bottom_offset_ft,
                )

    if best is None:
        return PlacementOutcome(
            feasible=False,
            message="No feasible bottom bearing point found on eligible vertical faces within depth/clearance limits.",
            y_top_ft=y_top_ft,
            x_top_ft=x_top_ft,
            x_diag_top_ft=x_diag_top_ft,
            y_bot_ft=float("nan"),
            x_bot_ft=float("nan"),
            depth_ft=float("nan"),
            dx_ft=float("nan"),
            theta_deg=float("nan"),
            sin_theta=float("nan"),
            bearing_face="",
            governing_clearance_in=float(clearance_in),
            bracket_depth_in=float("nan"),
            top_offset_ft=top_offset_ft,
            bottom_offset_ft=float("nan"),
        )

    return best
