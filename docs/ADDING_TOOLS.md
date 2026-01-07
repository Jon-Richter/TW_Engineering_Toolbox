# Adding tools to the toolbox

## Conceptual boundaries (to minimize interface conflicts)
- `toolbox_app/gui/` : UI shell only (Qt). Should not contain engineering logic.
- `toolbox_app/core/` : shared primitives (tool interface, logging, paths).
- `toolbox_app/tools/` : tool plugins (one folder per tool).
- `toolbox_app/integrations/` : external app interfaces (Excel COM, Mathcad, file I/O). Keep these separate from tool business logic.

Rule of thumb:
- Tools call integrations; UI does not.
- Tools return simple JSON-like outputs; UI renders them.

## Tool interface (required)

This template now supports **typed inputs** via Pydantic (recommended).
Each tool is a module under `toolbox_app/tools/<tool_id>/` and must export:
- `TOOL` : an instance implementing:
  - `meta` (ToolMeta)
  - `default_inputs() -> dict`
  - `run(inputs: dict) -> dict`

See `toolbox_app/tools/sample_beam_design/__init__.py` for a minimal example.

## Naming and versioning
- `meta.id` must match the folder name and be unique.
- Version should follow SemVer (`major.minor.patch`).
- Category groups tools in the UI.

## Inputs/outputs conventions
To keep things stable:
- Inputs should be JSON-serializable where possible (strings, numbers, booleans).
- Tool code should parse/validate inputs internally.
- Outputs should be primitive types and file paths as strings.
- Put large artifacts (xlsx/pdf) into `%LOCALAPPDATA%\EngineeringToolbox\outputs\` and return the file path.

## Limitations of this template (intentionally)
- Input widgets are basic `QLineEdit`. Complex inputs (tables, dropdowns, file pickers) can be added later.
- No persistent per-tool settings UI yet (but a settings.json path exists).
- No concurrency controls yet (for long-running calcs); add a worker thread when needed.

## Where the interfaces are
- Plugin boundary: `toolbox_app/core/tool_base.py`
- Plugin discovery: `toolbox_app/core/loader.py`
- User-writable paths: `toolbox_app/core/paths.py`
- UI entry point: `toolbox_app/gui/main.py`
- External apps: `toolbox_app/integrations/` (add Excel/Mathcad modules here)

## Adding Excel COM (pywin32) without affecting all tools
Put Excel automation in `toolbox_app/integrations/excel_com.py` and import it only inside tools that need it (lazy import). This keeps memory/startup low.

## Adding Mathcad integration
Mathcad automation varies by your Prime version and deployment. Keep it behind an integration module:
- `toolbox_app/integrations/mathcad.py`
Tools should not directly call subprocess/COM scattered across the codebase—keep a single integration boundary.


## Recommended: Pydantic InputModel (typed inputs)
Define an `InputModel` on your tool class (a `pydantic.BaseModel`). The UI will auto-generate widgets based on types:
- `float` -> numeric spin box
- `int` -> integer spin box
- `bool` -> checkbox
- file/path fields (name contains `path`/`file`/`template`) -> browse control
- otherwise -> text box

Example:

```python
from pydantic import BaseModel, Field

class Inputs(BaseModel):
    span_ft: float = Field(20, ge=0.1, description="Beam span (ft)")
    w_plf: float = Field(1000, ge=0, description="Uniform load (plf)")

class MyTool:
    InputModel = Inputs
    ...
```

Validation errors are shown to the user before the tool runs.
