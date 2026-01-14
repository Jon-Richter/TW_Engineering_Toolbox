# Guying Catenary Web Tool (Embedded HTML)

## Dependencies
This tool requires Qt WebEngine.

If you use pip, ensure:
- pyside6>=6.8,<6.11
- (if needed) pyside6-addons>=6.8,<6.11

Excel export uses openpyxl.
Mathcad handoff is file-based (CSV/JSON/TXT). For direct automation, add pywin32 and implement COM in integrations.

## How to use
1. Open the tool from the Engineering Toolbox.
2. Solve inside the embedded HTML UI.
3. Click **Capture results** (optional; exports will auto-capture).
4. Click **Export to Excel** or **Mathcad handoff**.

## Mathcad usage
- Read the generated CSV/Excel in Mathcad Prime (e.g., Excel Component or file read).
- Or copy-paste assignments from the generated TXT file.
