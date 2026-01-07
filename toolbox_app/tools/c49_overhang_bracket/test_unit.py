from __future__ import annotations

from .analysis.girder_outline import build_rectilinear_exterior_half_outline
from .analysis.placement import solve_best_placement
from .calc_trace import CalcTrace, TraceMeta
from .db.txdot_girders import get_txdot_profile
from .paths import compute_input_hash


def _trace() -> CalcTrace:
    meta = TraceMeta(
        tool_id="c49_overhang_bracket",
        tool_version="test",
        report_version="test",
        timestamp="2000-01-01T00:00:00",
        units_system="US",
        input_hash="testhash",
    )
    return CalcTrace(meta=meta)


def test_input_hash_deterministic() -> None:
    a = {"b": 2.0, "a": 1.0}
    b = {"a": 1.0, "b": 2.0}
    assert compute_input_hash(a) == compute_input_hash(b)


def test_txdot_profile_basic() -> None:
    p = get_txdot_profile("TX54")
    assert p.depth_in == 54.0
    assert p.bottom_flange_width_in == 30.0


def test_outline_geometry() -> None:
    p = get_txdot_profile("TX54")
    g = build_rectilinear_exterior_half_outline(p.as_dict())
    assert len(g.outline_poly_ft) >= 6
    assert any(f.tag == "web" for f in g.bearing_faces)
    assert any(f.tag == "bottom_flange_side" for f in g.bearing_faces)


def test_placement_feasible_default_envelopes() -> None:
    inputs = {
        "deck_soffit_offset_in": 0.0,
        "plywood_thickness_in": 0.75,
        "fourbyfour_thickness_in": 3.5,
        "twobysix_thickness_in": 1.5,
        "max_bracket_depth_in": 50.0,
        "min_bracket_depth_in": 12.0,
        "clearance_in": 0.25,
        "top_member_height_in": 3.0,
        "vertical_member_width_in": 2.0,
        "diagonal_envelope_thickness_in": 2.0,
        "bottom_pad_height_in": 2.0,
        "top_hanger_edge_clear_in": 0.0,
    }
    tr = _trace()
    p = get_txdot_profile("TX54")
    g = build_rectilinear_exterior_half_outline(p.as_dict())
    out = solve_best_placement(tr, inputs, g)
    assert out.feasible is True
    assert out.depth_ft > 0.0
    assert out.sin_theta > 0.0
