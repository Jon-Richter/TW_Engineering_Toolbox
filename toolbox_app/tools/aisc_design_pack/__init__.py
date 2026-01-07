from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from toolbox_app.core.tool_base import ToolMeta
from .excel_launcher import AISCDesignPackLauncher


class Inputs(BaseModel):
    job_tag: str = Field("", description="Optional tag appended to the working copy filename (e.g., job number).")
    visible: bool = Field(True, description="Open Excel visibly (recommended).")


@dataclass
class AISCDesignPackTool:
    RUNS_ON_UI_THREAD = True
    meta: ToolMeta = ToolMeta(
        id="aisc_design_pack",
        name="AISC Design Pack 14th Edition",
        category="Steel",
        version="1.0.0",
        description="Launches the department AISC Design Pack workbook (.xlsm) as a user-local working copy to avoid SharePoint locks.",
    )

    InputModel = Inputs

    def default_inputs(self) -> Dict[str, Any]:
        return Inputs().model_dump()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        asset = Path(__file__).parent / "assets" / "AISC 14th Edition Design Pack.xlsm"
        launcher = AISCDesignPackLauncher(asset_path=asset)
        return launcher.run(inputs)


TOOL = AISCDesignPackTool()
