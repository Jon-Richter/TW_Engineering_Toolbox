# bth_spreader_padeye — ASME BTH-1-2023 Spreader Bar + Padeye

## Install
Unzip into:
`toolbox_app/tools/bth_spreader_padeye/`

Tool discovery requires:
`toolbox_app/tools/bth_spreader_padeye/__init__.py` exporting `TOOL`.

## Runtime
- Python 3.13 in host application.
- No Node required at runtime.
- Backend is a local `http.server` on 127.0.0.1 with an ephemeral port.

## Frontend (Vite + React)
UI source lives under `frontend/` and builds into `frontend/dist/` (served by the backend).

From `toolbox_app/tools/bth_spreader_padeye/frontend/`:
- `npm install`
- `npm run dev` (local dev server)
- `npm run build` (updates `frontend/dist/` for runtime)

Tip: when running `npm run dev`, set `VITE_API_BASE` to the backend URL (e.g. `http://127.0.0.1:<port>`) so API calls hit the tool server.

## How it runs
When launched from Engineering Toolbox:
1. Creates run directory at:
   `%LOCALAPPDATA%\EngineeringToolbox\bth_spreader_padeye\runs\YYYYMMDD_HHMMSS_<hash>\`
2. Starts backend server in a subprocess.
3. Opens embedded web UI in QtWebEngine (fallback: system browser).

## Endpoints
- `GET /api/health`
- `POST /api/solve` (JSON body)
- `GET /api/report.html`
- `GET /api/download/<file_name>`
- `GET /api/shapes?q=<text>&limit=<n>`

## Exports (saved into run_dir on every solve)
- `report.html` (authoritative calc package)
- `design.xlsx` (Inputs / Assumptions / Calcs / Tables)
- `calc_trace.json`
- `results.json`
- `mathcad_inputs.csv`
- `mathcad_steps.json`

## Notes
- The calculation package is rendered directly from `calc_trace.json` and includes, for each step:
  equation, numeric substitution, variable table, unrounded+rounded results, and reference label.
- “Print / Save PDF” uses the browser/Qt print capability on `report.html`.

## Assumptions & Limitations
1. This implementation applies `Nd` as an ASD-style divisor (allowable = nominal / Nd).
2. Spreader self-weight moment uses `M = w L^2 / 8` and is applied to strong axis only.
3. Spreader LTB allowable is a conservative elastic form using `ry` and `Lb`; replace with exact BTH flexure expressions if you want strict parity with your legacy sheet.
4. Padeye hole checks implemented as conservative net-rupture, bearing, and tear-out using `Fu` and clear distance `Lc = R - Dh/2`.
5. Padeye base section modeled as rectangle `Wb × t` for section moduli and torsion constant.
