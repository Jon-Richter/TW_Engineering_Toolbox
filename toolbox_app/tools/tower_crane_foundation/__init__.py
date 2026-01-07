from __future__ import annotations

from pathlib import Path

from toolbox_app.core.runners import StaticWebRunner
from toolbox_app.core.tool_base import ToolMeta


META = ToolMeta(
    id="tower_crane_foundation",
    name="Tower Crane Foundation Design",
    category="Cranes",
    version="1.1",
    description="Tower crane spread footing foundation sizing and checks (embedded Vite+React UI).",
)


TOOL = StaticWebRunner(
    meta=META,
    index_file="assets/index.html",
    tool_dir=Path(__file__).resolve().parent,
    window_title="Tower Crane Foundation Design",
)
