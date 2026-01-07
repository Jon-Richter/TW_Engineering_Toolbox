# tower_crane_foundation (Tower Crane Foundation Design)

This tool uses the toolbox core `StaticWebRunner` to embed a **Vite + React production build** in the Engineering Toolbox using **Qt WebEngine**. Runtime requires **no Node.js**.

## Folder contents

- `assets/`  
  The Vite production build files (served locally).
- `__init__.py`  
  Exports `TOOL = <runner instance>` for toolbox discovery.

## Runtime dependencies (minimal)

- Qt binding with WebEngine (optional):
  - `PySide6-QtWebEngine` is not available on Python 3.13; in that case the tool opens in the default browser.
- `openpyxl` (Excel export)

If your toolbox already ships with PySide6 and openpyxl, **no new pip installs** are needed.

## Where files are written

All outputs are written under (Windows):
- `%LOCALAPPDATA%\EngineeringToolbox\tower_crane_foundation\exports\`

Nothing is written into the tool folder (important for SharePoint/OneDrive read-only sync).

## How results capture works

Toolbar action: **Capture results**

The wrapper tries, in order:
1. `window.__TOOLBOX_RESULTS__`
2. `window.getToolboxResults()` (if defined)
3. `localStorage` keys: `toolbox_results`, `tower_crane_foundation_results`, `results`, `tcf_results`
4. A DOM element with `[data-toolbox-results]` or `#toolbox-results` containing JSON text

If none are present, capture will fail with a clear message.  
**Recommended integration**: in your React app, set:

```js
window.__TOOLBOX_RESULTS__ = { ...yourResultsObject };
```

(or expose `window.getToolboxResults = () => ...`)

## Export outputs

- **Export to Excel (.xlsx)**  
  Generic workbook with:
  - `Summary` sheet (flattened path/value)
  - `JSON` sheet (full payload)

- **Mathcad handoff (JSON/CSV/TXT)**  
  Writes:
  - `handoff.json` (full payload)
  - `handoff.csv` (flattened key/value)
  - `assignments.txt` (sanitized scalar assignments, `var := value`)

- **Save report HTML**  
  Saves the current rendered page HTML as `report_*.html`.

## Quick verification command

From the repo root:

```bash
python -c "import toolbox_app.tools.tower_crane_foundation as t; print(t.TOOL.meta)"
```

## Notes / limitations

- Requires Qt WebEngine. If it is missing, the tool import will raise an informative error.
- The runner hosts the `assets/` build via a local HTTP server bound to `127.0.0.1` on a free port.
- If your React app uses client-side routing (History API), it should still work for a single-entry build; deep links may require additional server routing (not enabled in this minimal server).
