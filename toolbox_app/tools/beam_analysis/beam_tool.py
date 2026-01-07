from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from PySide6 import QtCore, QtWidgets

from .logging_utils import get_local_appdata_dir, configure_file_logger


# Redundant but intentional: some host frameworks look for this flag on the implementation module
# in addition to the package __init__.py.
RUNS_ON_UI_THREAD = True


class LaunchInputs(BaseModel):
    project_name: str = Field(
        "BeamAnalysis",
        description="Used to name the output folder under %LOCALAPPDATA%\\EngineeringToolbox\\beam_analysis\\.",
    )
    open_new_window: bool = Field(True, description="Launch a new window on each run.")


class BeamAnalysisTool:
    """Tool entry point: launches the Beam Analysis window.

    Important:
    - The host should run this tool on the UI thread. Even so, we schedule the window creation
      via QTimer on the QApplication thread to guarantee the `run()` call returns immediately.
    - Heavy computation is performed from within the window on a thread pool.
    """

    InputModel = LaunchInputs
    RUNS_ON_UI_THREAD = True

    def __init__(self, meta: Any):
        self.meta = meta
        self._windows: list[Any] = []
        self._logger = None

    def default_inputs(self) -> Dict[str, Any]:
        return {"project_name": "BeamAnalysis", "open_new_window": True}

    def _launch_window(self, out_dir: str) -> None:
        # Lazy import to avoid blocking the host UI during plugin load or tool start.
        try:
            from .beam_ui import BeamAnalysisWindow  # local import by design
            win = BeamAnalysisWindow(output_dir=Path(out_dir))
        except Exception as e:
            if self._logger:
                self._logger.exception("Beam Analysis UI failed to launch: %s", e)
            QtWidgets.QMessageBox.critical(None, "Beam Analysis - Launch failed", str(e))
            return

        win.show()
        self._windows.append(win)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = self.InputModel(**(inputs or {}))
        tool_id = getattr(self.meta, "id", "beam_analysis")
        base_dir = get_local_appdata_dir(tool_id)

        session = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir / data.project_name / session
        out_dir.mkdir(parents=True, exist_ok=True)

        logger = configure_file_logger("beam_analysis", out_dir / "beam_analysis.log")
        self._logger = logger
        logger.info("Launching Beam Analysis UI. Output dir: %s", out_dir)

        app = QtWidgets.QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication instance not found. The toolbox host must create QApplication before launching tools.")

        # Schedule on the QApplication (main/UI) thread and return immediately.
        QtCore.QTimer.singleShot(0, lambda: self._launch_window(str(out_dir)))

        return {"status": "ui_launch_scheduled", "output_dir": str(out_dir)}
