# Engineering Toolbox (Template)

This is a **template** for a locally-run Python "toolbox" application intended for internal engineering teams.

Design goals:
- No hosting costs; runs locally.
- Shared code folder can be **read-only** (SharePoint/Teams synced).
- Clear boundaries between UI, compute core, and integrations (Excel/Mathcad).
- Simple plugin model for adding tools.

## What you get in this template
- A Qt desktop shell (v2) with:
  - Tool list (left)
  - Tool description + "Run" panel (right)
  - Schema-driven inputs (Pydantic) + output area
  - Background runner with progress + cancel (cooperative)
- A plugin interface (`ToolBase`) and loader (`discover_tools`)
- Two example tools:
  - `sample_beam_design` (simple calc + report)
  - `sample_excel_roundtrip` (writes/reads an .xlsx via openpyxl)
- A SharePoint-friendly runtime model:
  - Code is read-only
  - Outputs/logs go to `%LOCALAPPDATA%\EngineeringToolbox\`

## Quick start (developer workstation)
1. Create a Python environment (choose one), **Python 3.13** (QtWebEngine is not available, so web tools open in the default browser):
   - Conda: `conda env create -f environment.yml`
   - Pip/venv: see `docs/SETUP_PIP_VENV.md`
2. Activate environment:
   - Conda: `conda activate eng_toolbox`
3. Run:
   - `python -m toolbox_app`

## Quick start (end users, read-only SharePoint folder)
See `docs/SETUP_END_USERS.md`.
To ship a single self-contained folder, run `scripts/vendorize_deps.bat` and include the `vendor/` directory.

## Adding a new tool
See `docs/ADDING_TOOLS.md`.
