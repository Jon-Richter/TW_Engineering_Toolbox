# AISC Design Pack (Excel)

## What this tool does
- Copies the master `.xlsm` workbook from the toolbox `assets/` folder to a **user-local working directory**.
- Opens the local copy in Microsoft Excel (via COM automation).

This avoids SharePoint/Teams file locking and ensures each run is isolated per user.

## Output location
Working copies are created under:
`%LOCALAPPDATA%\EngineeringToolbox\excel_runs\AISC_Design_Pack\`

## Requirements
- Windows + Microsoft Excel installed
- Python package:
  - `pywin32` (for Excel COM automation)

Install:
```powershell
python -m pip install pywin32
```

## Notes / limitations
- Macro security prompts and group policies are controlled by your organization. This tool does not bypass security controls.
- This tool does not yet auto-populate cells or extract results. It is intentionally a launcher + run-isolation wrapper.
  If you want deeper integration, define named ranges for inputs/outputs and we can add:
  - write inputs
  - trigger calculation/macro
  - read outputs
  - export a standardized results sheet/PDF
