# AISC 360-16 Steel Member Design (Tool)

## Purpose
This tool evaluates a selected AISC steel section for:
- Strong-axis flexure (Mx)
- Weak-axis flexure (My)
- Strong-axis shear (Vx)
- Weak-axis shear (Vy)
- Axial tension
- Axial compression
- Combined axial + flexure interaction (screening-level)

It supports LRFD and ASD toggling and uses the AISC Shapes Database (v16) for section properties.

## Installation
1. Copy this folder into:
   `toolbox_app/tools/aisc360_16_member_design/`

2. Place your AISC Shapes CSV here:
   `toolbox_app/tools/aisc360_16_member_design/data/aisc_shapes_database_v16.csv`

   Alternative override location (user-writable):
   `%LOCALAPPDATA%\\EngineeringToolbox\\databases\\aisc\\aisc_shapes_database_v16.csv`

## Inputs (user-controlled)
### Design
- Design method: LRFD / ASD
- Material grade preset, or custom:
  - Fy (ksi), Fu (ksi), E (ksi)

### Section selection
- Exactly one section must be selected from one of the family dropdowns:
  - W/HP, S/M, C/MC, WT/MT/ST, L, 2L, HSS (rect/round), PIPE
- The tool then pulls all available section properties from the DB.

### Geometry / bracing
- Buckling lengths: Lx, Ly (ft)
- Effective length factors: Kx, Ky
- Major-axis unbraced length: Lb (ft)
- Cb (dimensionless)

### Demands
- Pu (kips): compression positive, tension negative
- Mux, Muy (kip-ft)
- Vux, Vuy (kips)

## Units used internally
- Length: inches
- Area: in^2
- Section properties: in^3, in^4, in^6
- Stress: ksi
- Forces: kips
- Moments: kip-in (internally), reported as kip-ft

## Outputs
All outputs are written to:
`%LOCALAPPDATA%\\EngineeringToolbox\\tools\\aisc360_16_member_design\\runs\\<timestamp>\\`

Files produced:
- `aisc360_design_results.xlsx`
- `mathcad_handoff.json`
- `mathcad_assignments.txt`
- `trace.txt`
- `warnings.txt` (if any)

## What assumptions are hard-coded?
1. **Tension net section**: Ae is assumed equal to Ag in this tool. Net-section rupture reductions and shear lag are not implemented.
2. **Shear buckling**: shear coefficient Cv is assumed 1.0. Web shear buckling checks and Cv reductions are out-of-scope.
3. **Flexure local buckling**: compact/noncompact/slender element classification and local buckling reductions are out-of-scope (yielding-based Mn is used as baseline).
4. **Major-axis LTB**: implemented for **W/S/M/HP** shapes (AISC F2) and **rectangular HSS** (AISC F8 elastic LTB; assumes Cw≈0 and G = E/(2*(1+0.3))). Other shapes use Mn = Fy*Zx with warning.
5. **Compression buckling**: flexural buckling about principal axes (AISC E3 foundational). Torsional and flexural-torsional buckling is not implemented; tool warns for L/C/WT/etc.
6. **Interaction**: simplified H1-style screening equations; second-order effects (P-Δ/P-δ), B1/B2, and advanced analysis are out-of-scope.

## What is intentionally out of scope?
- Element slenderness classification and local buckling reductions (Chapter B / detailed Chapter F/G reductions)
- Web yielding/crippling, bearing stiffeners, concentrated loads
- Detailed torsional/flexural-torsional column buckling (open, singly-symmetric sections)
- Lateral-torsional buckling for round HSS and pipes
- Detailed HSS/pipe shear area per all AISC nuances (tool uses approximations with warnings if geometry is missing)
- Second-order analysis effects and stability amplification factors

## Notes on UI (tabs / families)
The input schema includes `json_schema_extra={"ui_tab": "..."}`
hints per family. If your toolbox schema-to-widgets layer supports tabbing/grouping,
it can use these hints to keep the UI clean.
If not supported, users will still see grouped fields and can select exactly one section.
