# Beam Analysis (Continuous Beam FEM)

## Summary
Multi-span (property-segment) beam analysis using an Euler–Bernoulli beam finite element model. Supports:
- Variable `E` and `I` per span
- User-defined joint restraints (vertical and rotational)
- Optional internal hinges at interior joints
- Distributed loads (linearly varying), point loads, point moments
- Shear / moment / deflection diagrams
- Exports: Excel, Mathcad handoff, JSON

## Units and sign conventions
**Kinematics**
- `v` positive upward
- `θ` positive counterclockwise (CCW)

**User load input signs**
- Distributed `w` positive downward
- Point `P` positive downward
- Moment `M` positive clockwise

**Imperial mode**
- Span length: ft
- `E`: psi
- `I`: in^4
- `w`: lb/ft
- `P`: lb
- `M`: ft-lb
- Outputs: `V` (lb), `M` (ft-lb), `v` (in)

**SI mode**
- Span length: m
- `E`: Pa
- `I`: m^4
- `w`: kN/m
- `P`: kN
- `M`: kN-m
- Outputs: `V` (kN), `M` (kN-m), `v` (m)

Internally everything is converted to **in, lbf, in-lbf, psi, in^4**.

## Hard-coded assumptions
- Euler–Bernoulli beam theory (no shear deformation)
- Linear elastic
- Small deflection (geometric linear)
- Vertical loads only (no axial, no torsion)
- Constant `E` and `I` within each span
- Internal hinges allowed only at interior physical joints (span boundaries)

## User-controlled inputs vs fixed
User-controlled:
- Number of spans
- Per-span `L`, `E`, `I`
- Per-joint vertical and rotational restraints
- Per-joint internal hinge (interior only)
- Loads and locations (global x)
- Mesh max element length (accuracy control)

Fixed by the tool:
- FEM formulation and sign conventions
- Consistent load integration via Gauss quadrature

## Out of scope (intentional)
- Shear deformation (Timoshenko)
- Nonlinear behavior (cracking, plasticity)
- Moving loads / influence lines
- Springs / foundation / semi-rigid supports
- Releases at arbitrary non-joint locations
- Code design checks / combinations

## Outputs
Files are written to:
`%LOCALAPPDATA%\EngineeringToolbox\beam_analysis\<project_name>\<timestamp>\`

Exports:
- `beam_results.xlsx`
- `beam_results_mathcad.txt`
- `beam_results.json`
- `beam_analysis.log`

## Threading / UI
The tool launches a Qt window, so `RUNS_ON_UI_THREAD = True`. The **solve** is dispatched to Qt's global thread pool to keep the UI responsive.


## Freeze mitigation
This tool schedules window creation with `QTimer.singleShot(0, QApplication, ...)` so the toolbox `Run` action returns immediately even if first-time imports are slow. If your toolbox still shows "Running...", check the log file in the output directory for any import errors.
