# wood_formwork_design - Dash Tool (Core Runner)

This tool uses the toolbox core `DashRunner` to embed a local Dash app in a Qt `QWebEngineView` and adds a toolbar for:
- Capture results (JSON snapshot)
- Export to Excel (`.xlsx`)
- Mathcad handoff (JSON + CSV + assignments TXT)
- Save report HTML

## Integration approach
The core runner starts the Dash app in a local subprocess on an auto-selected free port, then embeds the page via Qt WebEngine.

## What you still need to add
The tool expects the Dash entry script at `assets/formwork_design.py`.
- Put your Dash entry script into `assets/` (next to the NDS CSV).
- If the entry filename differs, update `__init__.py` to point at the new entry script.

The NDS database CSV was included and is already placed in `assets/`.

## Dependencies
The original app’s requirements (from `assets/requirements.txt`) are:
- dash
- pandas
- numpy
- reportlab

Additional requirement for the embedded UI:
- `PySide6-QtWebEngine` (not available on Python 3.13; in that case the tool opens in the default browser)

Optional (for Excel export):
- `openpyxl` (used by the core runner export utility)

## Run / verify in your repo
1) Ensure your Dash entry script exists at: `toolbox_app/tools/wood_formwork_design/assets/formwork_design.py`
2) Install Python deps (as needed):
   ```bash
   pip install -r toolbox_app/tools/wood_formwork_design/assets/requirements.txt
   pip install openpyxl
   ```
3) Quick verification:
   ```bash
   python -c "import toolbox_app.tools.wood_formwork_design as t; print(t.TOOL.meta)"
   ```

## Capture limitations (and the “better” option)
By default, **Capture results** uses an in-page JavaScript fallback that collects:
- Values from `<input>`, `<select>`, `<textarea>`
- A truncated `document.body.innerText` snapshot (first 200k chars)

If you want a higher-fidelity JSON payload (recommended), add a JS hook in your Dash app that populates:
- `window.__TOOLBOX_RESULTS__ = <your structured results JSON>`; or
- `window.toolboxGetResults = async () => <your results JSON>`

The runner will automatically prefer these hooks when present.

## Where exports are written
All outputs go to a user-writable directory:
`%LOCALAPPDATA%\EngineeringToolbox\tools\wood_formwork_design\...`

This avoids issues with read-only / synced code folders (SharePoint/OneDrive).
