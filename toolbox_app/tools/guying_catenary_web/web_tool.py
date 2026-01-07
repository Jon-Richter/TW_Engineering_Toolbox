from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Qt WebEngine (requires PySide6 QtWebEngine components)
from PySide6.QtWebEngineWidgets import QWebEngineView

from toolbox_app.core.paths import user_data_dir
from .exports import export_mathcad_handoff, export_to_excel

ASSETS_DIR = Path(__file__).parent / "assets"
HTML_PATH = ASSETS_DIR / "Catenary_Guying_v3.html"

# JS capture: returns JSON string (never blocks Python)
_JS_CAPTURE = r"""
(() => {
  try {
    const ok = (typeof lastResult !== "undefined" && lastResult);

    const baseType = document.getElementById("baseType") ? document.getElementById("baseType").value : null;
    const ropeSize = document.getElementById("ropeSize") ? document.getElementById("ropeSize").value : null;

    const ids = ["H","y1","x1","q","A","E","EI","y2","x2","wCable","breakStrength","T0","T0_2"];
    const inputs = {};
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      const v = (el.value ?? "").toString().trim();
      inputs[id] = v;
    });

    const reportArea = document.getElementById("reportArea");
    const report_html = reportArea ? (reportArea.innerHTML || "") : "";

    return JSON.stringify({
      ok: true,
      has_result: !!ok,
      baseType,
      ropeSize,
      inputs,
      lastResult: ok ? lastResult : null,
      report_html
    });
  } catch (e) {
    return JSON.stringify({ ok: false, error: String(e) });
  }
})()
"""


class WebToolWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Catenary Guying Tool")
        self.resize(1300, 850)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Header / actions
        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self.status = QLabel("Loading embedded tool…")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_capture = QPushButton("Capture results")
        self.btn_excel = QPushButton("Export to Excel")
        self.btn_mathcad = QPushButton("Mathcad handoff")
        self.btn_save_report = QPushButton("Save report HTML")

        self.btn_capture.clicked.connect(self.capture)
        self.btn_excel.clicked.connect(self.export_excel)
        self.btn_mathcad.clicked.connect(self.export_mathcad)
        self.btn_save_report.clicked.connect(self.save_report)

        h.addWidget(self.status, 1)
        h.addWidget(self.btn_capture)
        h.addWidget(self.btn_excel)
        h.addWidget(self.btn_mathcad)
        h.addWidget(self.btn_save_report)

        # Browser
        self.view = QWebEngineView()
        self.view.loadFinished.connect(self._on_loaded)

        # Log/output
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setFixedHeight(150)

        root.addWidget(header)
        root.addWidget(self.view, 1)
        root.addWidget(QLabel("Exports / log"))
        root.addWidget(self.out)

        self._last_payload: Optional[Dict[str, Any]] = None
        self._page_loaded: bool = False

        # Disable buttons until page loads
        self._set_actions_enabled(False)

        # Load HTML
        if not HTML_PATH.exists():
            self.status.setText(f"Missing HTML: {HTML_PATH}")
            self._append(f"ERROR: HTML file not found at: {HTML_PATH}")
        else:
            self.view.setUrl(QUrl.fromLocalFile(str(HTML_PATH.resolve())))

    def _append(self, msg: str) -> None:
        self.out.append(msg)

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.btn_capture.setEnabled(enabled)
        self.btn_excel.setEnabled(enabled)
        self.btn_mathcad.setEnabled(enabled)
        self.btn_save_report.setEnabled(enabled)

    def _on_loaded(self, ok: bool) -> None:
        self._page_loaded = bool(ok)
        if not ok:
            self.status.setText("Failed to load embedded HTML.")
            self._append("ERROR: WebEngine loadFinished returned False.")
            self._set_actions_enabled(False)
            return

        self.status.setText("Loaded. Solve inside the embedded tool, then Capture/Export.")
        self._append("Embedded HTML loaded.")
        self._set_actions_enabled(True)

    def _ensure_loaded(self) -> bool:
        if not self._page_loaded:
            QMessageBox.information(self, "Not ready", "The embedded tool is still loading. Try again in a moment.")
            return False
        return True

    def _capture_async(self, on_payload) -> None:
        """
        Runs the JS capture and calls on_payload(payload_dict) or on_payload(None).
        This is fully async (no loops, no waiting).
        """
        if not self._ensure_loaded():
            on_payload(None)
            return

        def _cb(result: Any) -> None:
            if not result:
                on_payload(None)
                return
            try:
                payload = json.loads(result)
            except Exception:
                on_payload(None)
                return

            # JS may return ok:false
            if isinstance(payload, dict) and payload.get("ok") is False:
                err = payload.get("error") or "Unknown JS error"
                self._append(f"Capture JS error: {err}")
                on_payload(None)
                return

            on_payload(payload)

        self.view.page().runJavaScript(_JS_CAPTURE, _cb)

    def capture(self) -> None:
        def _after(payload: Optional[Dict[str, Any]]) -> None:
            if not payload:
                QMessageBox.warning(self, "Capture failed", "Could not capture data from the embedded page.")
                return

            self._last_payload = payload

            if not payload.get("has_result"):
                self.status.setText("Captured inputs only (no solved result yet).")
                self._append("Captured inputs only. Solve inside the embedded UI, then capture again.")
                return

            self.status.setText("Captured solved results.")
            self._append("Captured solved results and inputs.")

        self._capture_async(_after)

    def export_excel(self) -> None:
        def _after(payload: Optional[Dict[str, Any]]) -> None:
            if not payload:
                QMessageBox.warning(self, "Capture failed", "Could not capture data from the embedded page.")
                return

            self._last_payload = payload

            if not payload.get("has_result"):
                QMessageBox.information(self, "No solved result", "Solve the model inside the embedded tool first.")
                return

            out_dir = user_data_dir() / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            default_path = out_dir / "catenary_guying_export.xlsx"

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Excel export",
                str(default_path),
                "Excel Workbook (*.xlsx)",
            )
            if not path:
                return

            p = export_to_excel(Path(path), payload)
            self._append(f"Excel export written: {p}")
            self.status.setText("Excel export complete.")

        # Always recapture on export (so exports match current page state)
        self._capture_async(_after)

    def export_mathcad(self) -> None:
        def _after(payload: Optional[Dict[str, Any]]) -> None:
            if not payload:
                QMessageBox.warning(self, "Capture failed", "Could not capture data from the embedded page.")
                return

            self._last_payload = payload

            if not payload.get("has_result"):
                QMessageBox.information(self, "No solved result", "Solve the model inside the embedded tool first.")
                return

            out_dir = user_data_dir() / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select output folder for Mathcad handoff",
                str(out_dir),
            )
            if not folder:
                return

            paths = export_mathcad_handoff(Path(folder), payload)
            for p in paths:
                self._append(f"Mathcad handoff written: {p}")
            self.status.setText("Mathcad handoff files created.")

        self._capture_async(_after)

    def save_report(self) -> None:
        def _after(payload: Optional[Dict[str, Any]]) -> None:
            if not payload:
                QMessageBox.warning(self, "Save failed", "Could not capture report HTML from the embedded page.")
                return

            html = payload.get("report_html") or ""
            if not str(html).strip():
                QMessageBox.information(
                    self,
                    "No report generated",
                    "Click the tool's 'Generate report' button first, then try again.",
                )
                return

            out_dir = user_data_dir() / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            default_path = out_dir / "catenary_guying_report.html"

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save report HTML",
                str(default_path),
                "HTML (*.html)",
            )
            if not path:
                return

            Path(path).write_text(html, encoding="utf-8")
            self._append(f"Report HTML saved: {path}")
            self.status.setText("Report HTML saved.")

        self._capture_async(_after)


_window_ref: Optional[WebToolWindow] = None


def launch_web_tool() -> None:
    """
    Opens a separate window. Keeps a module-level reference so the window is not garbage-collected.
    """
    global _window_ref
    if _window_ref is None:
        _window_ref = WebToolWindow()
    _window_ref.show()
