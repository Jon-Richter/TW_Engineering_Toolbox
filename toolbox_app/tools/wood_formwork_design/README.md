# wood_formwork_design - React + Python Tool

This tool provides a React + Vite frontend backed by a local Python API for:
- ACI 347R-14 lateral pressure
- NDS-adjusted member checks (plywood, studs, walers)
- Segment-by-elevation utilization checks
- CSV + PDF report export

## Layout and assets
- Frontend: `toolbox_app/tools/wood_formwork_design/frontend`
- Backend API: `toolbox_app/tools/wood_formwork_design/backend`
- NDS data: `toolbox_app/blocks/NDS_SUPP_db.py`

## Local development
1) Start the backend server:
   ```bash
   python toolbox_app/tools/wood_formwork_design/backend/app.py --host 127.0.0.1 --port 8000
   ```
2) Start the frontend:
   ```bash
   npm run dev --workspace=toolbox_app/tools/wood_formwork_design/frontend
   ```

## Build frontend for distribution
```bash
npm run build --workspace=toolbox_app/tools/wood_formwork_design/frontend
```

## Quick verification
```bash
python -c "from toolbox_app.tools.wood_formwork_design.backend import engine; print(engine.solve(engine.default_inputs())['ok'])"
```
