# Wood Formwork Design – Dash App (ACI 347R-14 + ASD Checks)

## Features
- **ACI 347R-14** lateral pressure (inch‑pound) with hydrostatic cap and minimum-pressure check.
- **Wall height segmentation** with non‑uniform pressure diagram `p(z)=min(w·z, p_cap)` and **member-by-elevation** utilization checks.
- **NDS dimension-lumber presets** (species/grade) with simplified adjustment factors (CD, CM, Ct, Cf) to auto-fill stud/waler allowables.
- **Tie demand cases**: interior, edge, corner, and top/bottom edge tributary reductions.
- **Export**:
  - CSV: full per-segment check table
  - PDF: inputs + summaries + truncated segment table (download CSV for full detail)

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python formwork_design.py
```

## Engineering Notes / Limitations
- One-way idealization with simply supported members.
- Walers are checked using a conservative uniform line-load approximation (not discrete stud reactions).
- Connection, bearing, combined actions, and stability checks are not included.
- **Verify NDS preset values** against your current NDS Supplement and project requirements before use.


## NDS database
Place `NDS_Supplement_2018_Tables_4A_4B.csv` in the same folder as the application (next to `formwork_design.py`). The app reads it at startup to populate species/grade options and base design values.

## Waler shear check
Waler shear demand is evaluated at a section located one member depth, d, from each tie (support) using the selected waler depth (e.g., 2x6 -> d=5.5 in) under the uniform line-load approximation.
