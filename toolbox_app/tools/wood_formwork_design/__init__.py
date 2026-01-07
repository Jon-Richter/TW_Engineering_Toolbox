from __future__ import annotations

from pathlib import Path

from toolbox_app.core.runners import DashRunner
from toolbox_app.core.tool_base import ToolMeta


META = ToolMeta(
    id="wood_formwork_design",
    name="Wood Formwork Design",
    category="Concrete",
    version="1.0.0",
    description="Dash-based wood formwork design (ACI 347R-14 + ASD checks) embedded in Qt WebEngine.",
)


TOOL = DashRunner(
    meta=META,
    entry_script="assets/formwork_design.py",
    port="auto",
    tool_dir=Path(__file__).resolve().parent,
    window_title="Wood Formwork Design (Dash)",
)
