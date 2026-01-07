# Limitations and extension points

## Current limitations (v2)
- Input UI is now schema-driven for common field types; complex grids/tables still require customization. Production tools typically need:
  - numeric parsing with units
  - dropdowns for discrete choices
  - file pickers for templates
  - tables/grids for multiple load cases
- Tools run off the UI thread with a basic Cancel mechanism. True cooperative cancellation requires tools to check the cancel flag in `run_with_context`.
- No built-in permissions/role model (assumes SharePoint handles edit rights).
- No auto-updater (assumes SharePoint version folders + launcher).

## Extension points (recommended next steps)
1. Add a small "input schema" system using Pydantic so tools can declare fields + types, enabling:
   - numeric validation
   - defaults
   - units display
   - better UI widgets
2. Add a background worker (Qt `QThread` or `QtConcurrent`) so UI stays responsive.
3. Add a report module that produces:
   - PDF output
   - HTML summaries
4. Add integrations:
   - Excel COM for live Excel interaction
   - Mathcad handoff (template population / inputs file generation)
