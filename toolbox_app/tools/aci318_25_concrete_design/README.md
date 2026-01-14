# ACI 318-25 Concrete Design Tool (Offline)

Tool ID: `aci318_25_concrete_design`  
Code basis: **ACI 318-25**  
Units: **US customary** (in, psi, kip, kip-ft)

This tool is a drop-in plugin for the Engineering Toolbox desktop app. It launches a local backend server on `127.0.0.1` and embeds a React UI in QtWebEngine. Every solve generates an **audit-grade calculation package** (HTML report + Excel + JSON + Mathcad handoff files) from a deterministic `CalcTrace` object.

## Installation

Unzip this folder into:

`toolbox_app/tools/aci318_25_concrete_design/`

The host app will discover the tool via `__init__.py` exporting `TOOL`.

## Output directory

All outputs/logs are written under:

**Windows (required):**
`%LOCALAPPDATA%\EngineeringToolbox\aci318_25_concrete_design\runs\YYYYMMDD_HHMMSS_<short_hash>\`

**Non-Windows fallback (for tests):**
`~/.local/share/EngineeringToolbox/aci318_25_concrete_design/runs/...`

Example (Windows):

`C:\Users\<you>\AppData\Local\EngineeringToolbox\aci318_25_concrete_design\runs\20260112_153012_1a2b3c4d\`

Each solve writes:
- `report.html` (authoritative calc package)
- `calc_trace.json` (full CalcTrace object)
- `results.json` (key results)
- `results.xlsx` (Inputs / Assumptions / Calcs / Tables)
- `mathcad_inputs.csv` + `mathcad_steps.json` (minimum viable handoff)

## UI + API architecture

### Wrapper (PySide6 / QtWebEngine)
- `tool.py` starts the backend server as a subprocess (non-blocking) and opens a `QWebEngineView` to `http://127.0.0.1:<port>/`.
- It returns immediately with:
  `{
    "ok": True,
    "status": "launched",
    "url": "...",
    "run_dir": "..."
  }`

### Backend server
Implemented using Python stdlib `ThreadingHTTPServer` for maximum offline reliability (no FastAPI/Flask dependency at runtime).

Endpoints:
- `GET  /api/health`
- `POST /api/solve` (runs solver + writes artifacts)
- `GET  /api/report.html`
- `GET  /api/download/<file_name>`

### Frontend (React)
Prebuilt static assets are under `frontend/dist/` and served by the backend. No Node is required at runtime.

UI features:
- Tabbed modules
- Inputs with units
- Solve/Run
- Governing summary + key outputs
- Embedded calc package viewer (iframe)
- Export/download buttons (HTML report, Excel, CalcTrace JSON, Mathcad CSV)

## Implemented calculation modules (v0.3.0)

Implemented:
- **Beams**: Rectangular flexure with user-defined reinforcement layers (multiple layers; tension and compression allowed). If no layers are provided, the tool also supports a legacy “required As” design path (singly reinforced).
- **Slabs**: One-way slab strip flexure (12-in strip).
- **Columns**: Concentric axial compression (short column) per Eq. (22.4.2.2) and Table 22.4.2.1.
- **Development & Splices**:
  - Straight bar development length in tension per Table 25.4.2.3, with factors per Table 25.4.2.5 and minimum per 25.4.2.1(b).
  - Tension lap splice lengths (Class A/B) per Table 25.5.2.1.

- **Walls (Slender)**:
  - Combined Pu + in-plane + out-of-plane end moments.
  - Second-order effects via ACI 318-25 moment magnification for nonsway members (6.6.4.5), with k from Table 11.5.3.2.
  - Effective stiffness uses Eq. (6.6.4.4.4c) with wall inertia factors from Table 6.6.3.1.1(a).
  - Section strength via strain compatibility using Chapter 22 stress block and φ per Table 21.2.2.
  - Current limitation: checks are performed independently about each principal axis (biaxial interaction not yet implemented).

Placeholders (not yet implemented in this version):
- Anchors

These return `ok=false` with a clear “not implemented” message and still produce a calc package noting NYI.

## Calc package structure & reproducibility guarantees

All intermediate values are recorded as `CalcStep` objects. Every step includes:
1. Full symbolic equation (`equation_latex`)
2. Full numeric substitution (`substitution_latex`)
3. Variable list table with sources (`input:<id>`, `step:<id>`, etc.)
4. Unrounded and rounded results with explicit rounding rule
5. Reference (ACI clause/table or derived method label)

All exports (HTML/Excel/JSON/Mathcad) are rendered from the same `CalcTrace` object.

## Verification cases (expected key outputs)

### Case 1 — Beam flexure (rectangular, singly reinforced)
Inputs (US):
- b = 12 in, h = 24 in
- cover = 1.5 in, stirrup dia = 0.375 in, bar dia = 1.0 in
- f'c = 4000 psi, fy = 60000 psi, Es = 29,000,000 psi
- Mu = 180 kip-ft

Expected key outputs (approx):
- φ ≈ 0.90 (tension-controlled)
- As,prov(required) ≈ 1.9835 in²
- φMn ≈ 200.00 kip-ft
- Status: PASS

### Case 2 — Tension development length
Inputs (US):
- Bar size #5 (db = 0.625 in)
- f'c = 4000 psi, fy = 60000 psi
- Normalweight (λ = 1.0), not top bar, uncoated
- Cover≥db and spacing≥2db (Table 25.4.2.3 first-row selection)

Expected key outputs (approx):
- ℓd ≈ 37.50 in (minimum 12 in not governing)
- Lap splice Class A ≈ 37.50 in; Class B ≈ 48.75 in

## Runtime dependencies

- Python 3.13 (host)
- PySide6 + QtWebEngine (host provides)
- `pydantic`
- `openpyxl`

No internet. No Node required at runtime.

## Running smoke tests (developer)

From the tool directory:

`python tests_smoke.py`

This runs two representative cases and asserts that `report.html`, `calc_trace.json`, `results.xlsx`, and `mathcad_inputs.csv` are created.
