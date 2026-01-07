from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from toolbox_app.core.paths import user_data_dir


def _copy_to_working_dir(asset_path: Path, job_tag: Optional[str] = None) -> Path:
    """
    Copies the master workbook (stored in assets, possibly read-only from SharePoint)
    into a user-writable working directory.

    Returns the path to the local working copy.
    """
    out_dir = user_data_dir() / "excel_runs" / "AISC_Design_Pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{job_tag}" if job_tag else ""
    dest = out_dir / f"AISC_Design_Pack{tag}_{ts}.xlsm"

    shutil.copy2(asset_path, dest)
    return dest


def open_in_excel(path: Path, visible: bool = True) -> str:
    """
    Opens the given workbook in Excel using COM automation (Windows only).
    This does not modify the SharePoint master; it opens the user-local working copy.

    Uses pywin32 for COM automation when available; falls back to shell open if COM fails.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import win32com.client  # type: ignore
        import pywintypes  # type: ignore
    except Exception:
        os.startfile(str(path))
        return "shell"

    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Workbooks.Open(str(path))
        if visible:
            try:
                excel.Visible = True
            except pywintypes.com_error:
                # Visibility can fail on some setups; workbook is still opened.
                pass
        return "com"
    except Exception:
        os.startfile(str(path))
        return "shell"


@dataclass
class AISCDesignPackLauncher:
    """
    Launcher-style tool: copies the .xlsm to a local working directory and opens it in Excel.
    """
    asset_path: Path

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        job_tag = str(inputs.get("job_tag") or "").strip() or None
        visible = bool(inputs.get("visible", True))

        local_path = _copy_to_working_dir(self.asset_path, job_tag=job_tag)
        method = open_in_excel(local_path, visible=visible)

        return {
            "status": "Opened Excel workbook (local working copy).",
            "working_copy": str(local_path),
            "note": (
                "A new timestamped copy is created per run in the user-local folder to avoid SharePoint locks."
                if method == "com"
                else "Opened via shell fallback (COM automation unavailable or failed)."
            ),
        }
