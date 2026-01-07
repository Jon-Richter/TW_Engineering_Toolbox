from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .geometry import Segment

Point = Tuple[float, float]  # (x_ft, y_ft)


@dataclass(frozen=True)
class GirderGeometry:
    """2D geometry used for constructability screening and UI diagram.

    The geometry is expressed in the *section plane* for the exterior (overhang) side.

    Coordinate convention (consistent across the tool):
      - y = 0 at top of girder, +y downward
      - x = 0 at exterior face of web, +x outboard (toward overhang)

    outline_poly_ft:
      - Closed polygon (clockwise) representing the concrete region for x >= 0.

    bearing_faces:
      - Tagged boundary segments that are eligible for bottom bearing contact.
        (Compression-only horizontal reaction on a vertical surface.)
    """

    outline_poly_ft: List[Point]
    bearing_faces: List[Segment]


def build_a21_exterior_half_outline(profile: dict) -> GirderGeometry:
    """Build an exterior-half outline from TxDOT Standard Sheet A-21 dimensions.

    Uses A-21 parameters:
      - D (overall depth)
      - B, C, E, F (as labeled on the A-21 sheet)

    Uses fixed A-21 callouts:
      - Web thickness = 7 in (used only to derive flange half-width offsets)
      - Bottom flange width = 32 in (fixed)
      - Top flange edge thickness = 3.5 in
      - Top haunch horizontal offset = 2 in (from web face)
      - Bottom haunch horizontal offset = 3 in (from web face)
      - Bottom haunch vertical segments = 4.75 in and 3 in
      - Top haunch vertical segment = 2 in

    Curved fillets/chamfers are represented as straight segments, because A-21 does not
    provide radii for these transitions. The resulting outline is *dimensionally exact*
    at the controlling points and suitable for interference screening and scaled UI drawing.

    Returns:
      GirderGeometry with:
        - outline polygon (ft)
        - vertical bearing faces (web, bottom flange side)
    """

    # A-21 parameters (in)
    D_in = float(profile["depth_in"])
    B_in = float(profile["B_in"])
    C_in = float(profile["C_in"])
    E_in = float(profile["E_in"])
    F_in = float(profile["F_in"])

    tfw_in = float(profile["top_flange_width_in"])  # 36 or 42
    bfw_in = float(profile["bottom_flange_width_in"])  # 32
    web_in = float(profile["web_thickness_in"])  # 7

    # Fixed callouts (in)
    t_top_edge_in = 3.5
    x_top_haunch_in = 2.0
    y_top_haunch_vert_in = 2.0

    x_bot_haunch_in = 3.0
    y_bot_seg1_in = 4.75
    y_bot_seg2_in = 3.0

    # Derived half-widths from web face (in)
    # x=0 at exterior web face; outboard top flange edge:
    x_edge_top_in = max(0.0, (tfw_in - web_in) / 2.0)
    # outboard bottom flange edge:
    x_edge_bot_in = max(0.0, (bfw_in - web_in) / 2.0)

    # Sanity: The A-21 table provides C such that x_edge_top_in = x_top_haunch_in + C
    # but we do not enforce; we just build from the provided values.

    # Key y-levels (in)
    y1 = t_top_edge_in
    y2 = t_top_edge_in + E_in
    y3 = y2 + y_top_haunch_vert_in
    y4 = y3 + B_in
    y5 = y4 + y_bot_seg1_in
    y6 = y5 + y_bot_seg2_in
    y7 = D_in

    # Numeric closure check: y7 should equal D_in
    # (This holds for A-21 table values; we do not raise if user overrides D_in.)

    # Key x positions (in)
    x0 = 0.0
    xA = x_edge_top_in
    xB = x_top_haunch_in
    xK = x_bot_haunch_in
    xE = x_edge_bot_in

    # Convert to feet
    def ft(x_in: float) -> float:
        return x_in / 12.0

    # Exterior-half polygon (clockwise)
    # Start at top outboard edge, trace perimeter, then close along the cut line x=0.
    poly_in: List[Tuple[float, float]] = [
        (xA, 0.0),        # top outboard edge
        (xA, y1),         # down to underside at outboard edge
        (xB, y2),         # underside slope to haunch face
        (xB, y3),         # down haunch face
        (x0, y3),         # inboard to web face (straight approximation of fillet)
        (x0, y4),         # down web face
        (xK, y5),         # steep bottom haunch
        (xE, y6),         # shallow haunch to bottom flange side
        (xE, y7),         # down bottom flange side
        (x0, y7),         # bottom back to web face
        (x0, 0.0),        # up along cut line
    ]

    poly_ft: List[Point] = [(ft(x), ft(y)) for x, y in poly_in]

    # Eligible bearing faces (vertical only)
    # Web face is vertical from y3 to y4 at x=0.
    faces: List[Segment] = [
        Segment((ft(x0), ft(y3)), (ft(x0), ft(y4)), tag="web"),
        Segment((ft(xE), ft(y6)), (ft(xE), ft(y7)), tag="bottom_flange_side"),
    ]

    return GirderGeometry(outline_poly_ft=poly_ft, bearing_faces=faces)


def build_rectilinear_exterior_half_outline(profile: dict) -> GirderGeometry:
    """Legacy rectilinear outline builder (kept for backwards compatibility).

    The UI and placement solver now use build_a21_exterior_half_outline for TxDOT girders.
    """

    depth_in = float(profile["depth_in"])
    tw_in = float(profile["web_thickness_in"])
    tfw_in = float(profile["top_flange_width_in"])
    bfw_in = float(profile["bottom_flange_width_in"])
    tft_in = float(profile.get("top_flange_thickness_in", 3.5))
    bft_in = float(profile.get("bottom_flange_thickness_in", 8.75))

    D = depth_in / 12.0
    tw = tw_in / 12.0
    tft = tft_in / 12.0
    bft = bft_in / 12.0

    x_top = max(0.0, (tfw_in - tw_in) / 2.0) / 12.0
    x_bot = max(0.0, (bfw_in - tw_in) / 2.0) / 12.0

    y_web_top = tft
    y_web_bot = max(y_web_top, D - bft)

    poly: List[Point] = [
        (x_top, 0.0),
        (x_top, y_web_top),
        (0.0, y_web_top),
        (0.0, y_web_bot),
        (x_bot, y_web_bot),
        (x_bot, D),
        (0.0, D),
        (0.0, 0.0),
    ]

    faces: List[Segment] = [
        Segment((0.0, y_web_top), (0.0, y_web_bot), tag="web"),
        Segment((x_bot, y_web_bot), (x_bot, D), tag="bottom_flange_side"),
    ]
    return GirderGeometry(outline_poly_ft=poly, bearing_faces=faces)
