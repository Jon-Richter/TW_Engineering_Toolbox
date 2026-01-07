# C49 Overhang Bracket (TxDOT) — Toolbox Plugin

Tool ID: `c49_overhang_bracket`  
Purpose: Construction-stage overhang bracket placement screening + bracket spacing optimization, with an audit-grade calc package export (HTML/PDF/Excel/JSON/Mathcad handoff).

## Install

Unzip this folder into:

`toolbox_app/tools/c49_overhang_bracket/`

The host “Engineering Toolbox” app will discover the tool via `__init__.py` exporting `TOOL`.

## Run outputs (authoritative output location)

All outputs are written to:

`%LOCALAPPDATA%\EngineeringToolbox\c49_overhang_bracket\runs\YYYYMMDD_HHMMSS_<hash>\`

Example:

`C:\Users\<you>\AppData\Local\EngineeringToolbox\c49_overhang_bracket\runs\20260101_101530_a1b2c3d4\`

Files produced on every run:
- `report.html` (authoritative calc package; offline)
- `report.pdf` (minimal printable summary; full detail remains in HTML)
- `calc_trace.json` (full CalcTrace object)
- `results.json` (key outputs)
- `results.xlsx` (Inputs / Assumptions / Calcs / Tables / Results)
- `mathcad_inputs.csv` + `mathcad_steps.json` (minimum viable handoff)
- `run.log`

## Inputs of interest

- `girder_type`: `TX28, TX34, TX40, TX46, TX54, TX62, TX70`
- `overhang_length_ft`, `slab_thickness_in`, `screed_wheel_load_kip`
- Member SWL: `hanger_swl_kip`, `diagonal_swl_kip`
- Placement/deck stack:
  - `deck_soffit_offset_in`
  - `plywood_thickness_in` (default 0.75 in)
  - `fourbyfour_thickness_in` (default 3.50 in)
  - `twobysix_thickness_in` (default 1.50 in)
- Constructability:
  - `max_bracket_depth_in` (default 50 in)
  - `clearance_in` (default 0.25 in)

### Girder profile “exactness” and overrides

The shipped TxDOT library uses TxDOT standard sheet A-21 for:
- Overall girder depth `D`
- Bottom flange width `B`
- Section area and weight metadata

For constructability screening (interference + valid bearing face), the tool needs web/flange thicknesses to form an outline. These are **defaulted** and can be overridden:
- `girder_web_thickness_in`
- `girder_top_flange_thickness_in`
- `girder_bottom_flange_thickness_in`
- `girder_top_flange_width_in`
- `girder_bottom_flange_width_in`
- `girder_depth_in`

If you require a fabrication-accurate outline (fillets/chamfers), use overrides consistent with the project’s standard details.

## Placement logic (C49 constructability)

The tool enforces:
- Top of bracket is located from the deck soffit plus the stack: **3/4 ply + 4x4 + 2x6 flat nailer**.
- The only positive connection is the **top hanger**.
- Bottom reaction is purely horizontal; therefore it must bear on a **vertical** girder surface:
  - exterior web face, or
  - exterior side of bottom flange (if applicable and reachable).
- No member envelope may intersect the girder outline, within the specified `clearance_in`.

Among feasible placements, the tool selects the **deepest feasible bracket** (up to 50 in), with preference to web bearing.

## Verification cases (sanity checks)

Case 1 (defaults):
- `girder_type=TX54`, `overhang_length_ft=6`, `slab_thickness_in=8.5`, `screed_wheel_load_kip=2`
Expected:
- Run completes (`ok=True`)
- Outputs exist in run directory

Case 2:
- `girder_type=TX34`, `overhang_length_ft=4`, `screed_wheel_load_kip=3.5`, `girder_web_thickness_in=6.5`
Expected:
- Run completes (`ok=True`)
- Outputs exist

Tests:
- `pytest -q test_unit.py` (fast unit tests)
- `pytest -q tests_smoke.py` (required end-to-end smoke tests)

## Assumptions & limitations (high-level)

- Rectilinear cross-section model; fillets/chamfers ignored (covered by `clearance_in`).
- Demand model is a simplified construction-stage tributary-strip model; it is intended for bracket-spacing screening and should be reviewed against project-specific temporary works requirements.

## Source notes

- TxDOT standard sheet A-21 “Girder Dimensions” (depth and bottom flange width).
