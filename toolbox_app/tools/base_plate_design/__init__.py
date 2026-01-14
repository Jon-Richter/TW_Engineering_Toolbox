from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from PySide6.QtCore import QTimer

from toolbox_app.core.tool_base import ToolMeta
from .web_tool import launch_web_tool


@dataclass
class BasePlateDesignTool:
    """
    UI-launching tool wrapper.

    IMPORTANT:
    The main toolbox runs tools in a background thread. Qt UI (and especially Qt WebEngine)
    must be created on the GUI thread. We therefore schedule launch_web_tool() onto the GUI thread.
    """
    meta: ToolMeta = ToolMeta(
        id="base_plate_design",
        name="Base Plate Design",
        category="Structural Analysis",
        version="1.0.1",
        description="Embedded HTML tool (identical UI) with Excel export and Mathcad handoff.",
    )

    # No toolkit-side inputs; embedded UI owns inputs.
    InputModel = None

    # Hint to future improvements (optional)
    RUNS_ON_UI_THREAD = True

    def default_inputs(self) -> Dict[str, Any]:
        return {}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Post to GUI thread and return immediately (avoid WebEngine creation on worker thread)
        QTimer.singleShot(0, launch_web_tool)
        return {"status": "Launching embedded HTML UI window..."}


TOOL = BasePlateDesignTool()
