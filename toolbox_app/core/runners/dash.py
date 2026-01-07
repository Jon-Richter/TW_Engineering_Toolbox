from __future__ import annotations

import csv
import json
import os
import site
import sys
import inspect
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QProcess, QTimer, Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QToolBar

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover
    Workbook = None  # type: ignore

from toolbox_app.core.paths import user_data_dir
from toolbox_app.core.tool_base import ToolMeta


JS_CAPTURE = r"""
(async () => {
  try {
    if (window.toolboxGetResults) {
      const payload = await window.toolboxGetResults();
      return { ok: true, payload };
    }
    if (window.__TOOLBOX_RESULTS__ !== undefined) {
      return { ok: true, payload: window.__TOOLBOX_RESULTS__ };
    }

    const inputs = Array.from(document.querySelectorAll('input, select, textarea')).map(el => {
      const tag = (el.tagName || '').toLowerCase();
      const type = el.type || null;
      const obj = {
        id: el.id || null,
        name: el.name || null,
        tag,
        type,
        value: (el.value !== undefined) ? el.value : null
      };
      if (type === 'checkbox' || type === 'radio') {
        obj.checked = !!el.checked;
      }
      return obj;
    });

    const text = document.body ? (document.body.innerText || '') : '';
    const payload = { inputs, text: text.slice(0, 200000) };
    return { ok: true, payload };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
})();
"""


def _guess_tool_dir() -> Path:
    core_root = Path(__file__).resolve().parents[1]
    for frame_info in inspect.stack():
        mod = inspect.getmodule(frame_info.frame)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        p = Path(mod_file).resolve()
        if core_root in p.parents:
            continue
        return p.parent
    return Path.cwd()


def _tool_export_dir(tool_id: str) -> Path:
    p = user_data_dir() / "tools" / tool_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _timestamp_slug(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def _safe_key(s: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in s.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "value"


def _flatten(obj: Any, prefix: str = "", max_depth: int = 6) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _walk(o: Any, p: str, depth: int) -> None:
        if depth > max_depth:
            out[p or "value"] = json.dumps(o, default=str)
            return
        if isinstance(o, dict):
            if not o:
                out[p or "value"] = ""
                return
            for k, v in o.items():
                kk = _safe_key(str(k))
                _walk(v, f"{p}.{kk}" if p else kk, depth + 1)
            return
        if isinstance(o, (list, tuple)):
            if not o:
                out[p or "value"] = ""
                return
            for i, v in enumerate(o):
                _walk(v, f"{p}[{i}]" if p else f"[{i}]", depth + 1)
            return
        out[p or "value"] = o

    _walk(obj, prefix, 0)
    return out


def _write_json(payload: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path


def _export_excel(payload: Any, out_path: Path) -> Path:
    if Workbook is None:
        raise RuntimeError("openpyxl is required for Excel export but is not installed.")

    wb = Workbook()
    ws_meta = wb.active
    ws_meta.title = "meta"
    ws_flat = wb.create_sheet("flat")

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    ws_meta.append(["key", "value"])
    for k, v in sorted(_flatten(meta).items()):
        ws_meta.append([k, v])

    ws_flat.append(["key", "value"])
    for k, v in sorted(_flatten(payload).items()):
        ws_flat.append([k, v])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


def _mathcad_handoff(payload: Any, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "handoff.json"
    csv_path = out_dir / "handoff.csv"
    txt_path = out_dir / "assignments.txt"

    _write_json(payload, json_path)

    flat = _flatten(payload)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in sorted(flat.items(), key=lambda kv: kv[0]):
            if isinstance(v, (dict, list, tuple)):
                v = json.dumps(v, default=str)
            w.writerow([k, v])

    lines = []
    for k in sorted(flat.keys()):
        v = flat[k]
        key = _safe_key(k)
        if v is None:
            rhs = "0"
        elif isinstance(v, bool):
            rhs = "1" if v else "0"
        elif isinstance(v, (int, float)):
            rhs = str(v)
        else:
            s = str(v).replace('"', '""')
            rhs = f"\"{s}\""
        lines.append(f"{key}:={rhs}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "assignments": txt_path}


class DashRunner:
    RUNS_ON_UI_THREAD = True
    InputModel = None

    def __init__(
        self,
        meta: ToolMeta,
        entry_script: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int | str = 0,
        python_executable: str | None = None,
        tool_dir: str | Path | None = None,
        window_title: str | None = None,
        open_in_browser: bool | None = None,
    ) -> None:
        self.meta = meta
        self._entry_script = str(entry_script)
        self._host = host
        self._port = port
        self._python_executable = python_executable
        self._tool_dir = Path(tool_dir) if tool_dir is not None else _guess_tool_dir()
        self._window_title = window_title or f"{meta.name} (Dash)"
        self._open_in_browser = open_in_browser
        self._window: Optional[_DashToolWindow] = None
        self._external: Optional[_DashExternalLauncher] = None

    def _use_external(self) -> bool:
        if self._open_in_browser is True:
            return True
        if self._open_in_browser is False:
            return False
        return QWebEngineView is None

    def default_inputs(self) -> dict[str, Any]:
        return {}

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._use_external():
            if self._external is None or not self._external.is_running():
                self._external = _DashExternalLauncher(
                    tool_id=self.meta.id,
                    tool_name=self.meta.name,
                    tool_dir=self._tool_dir,
                    entry_script=self._entry_script,
                    host=self._host,
                    port=self._port,
                    python_executable=self._python_executable,
                )
                self._external.start()
            return {"status": "ok", "launched": True, "mode": "external_browser"}

        if self._window is None or not self._window.isVisible():
            self._window = _DashToolWindow(
                tool_id=self.meta.id,
                title=self._window_title,
                tool_dir=self._tool_dir,
                entry_script=self._entry_script,
                host=self._host,
                port=self._port,
                python_executable=self._python_executable,
            )
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        return {"status": "ok", "launched": True, "mode": "embedded"}


class _DashExternalLauncher:
    def __init__(
        self,
        *,
        tool_id: str,
        tool_name: str,
        tool_dir: Path,
        entry_script: str,
        host: str,
        port: int | str,
        python_executable: Optional[str],
    ) -> None:
        self._tool_id = tool_id
        self._tool_name = tool_name
        self._tool_dir = tool_dir
        self._entry_script = entry_script
        self._host = host
        self._port = port
        self._python_executable = python_executable or sys.executable

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("Qt application not initialized.")

        self._export_dir = _tool_export_dir(tool_id)
        self._proc_output_log = self._export_dir / "server_output.log"
        self._proc_output_tail = ""
        self._server_url: Optional[str] = None
        self._opened = False

        self._process = QProcess(app)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_proc_output)
        self._process.finished.connect(self._on_proc_finished)
        self._process.errorOccurred.connect(self._on_proc_error)

        self._poll_timer = QTimer(app)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_for_server_json)

        app.aboutToQuit.connect(self.stop)  # type: ignore[attr-defined]

    def is_running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    def start(self) -> None:
        self._start_dash()

    def stop(self) -> None:
        self._poll_timer.stop()
        if self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(1500):
                self._process.kill()
                self._process.waitForFinished(1500)

    def _show_startup_error(self, title: str, message: str) -> None:
        QMessageBox.critical(None, title, message)

    def _open_browser(self) -> None:
        if self._server_url and not self._opened:
            webbrowser.open(self._server_url)
            self._opened = True
            QMessageBox.information(
                None,
                f"{self._tool_name} (Browser Mode)",
                "Opened in your default browser.\n\n"
                "Capture/export from the Toolbox is not available in external browser mode.",
            )

    def _start_dash(self) -> None:
        entry_path = Path(self._entry_script)
        if not entry_path.is_absolute():
            entry_path = self._tool_dir / self._entry_script
        app_path = entry_path.resolve()

        if not app_path.exists():
            self._show_startup_error(
                "Missing Dash app",
                "Dash entry script not found:\n"
                f"{app_path}\n\n"
                "Place your Dash entry script in the assets folder or update the tool config.",
            )
            return

        runner = Path(__file__).resolve().parent / "dash_subprocess.py"
        if not runner.exists():
            self._show_startup_error("Missing runner", f"dash_subprocess.py not found at:\n{runner}")
            return

        for name in ("server.json", "server_error.txt"):
            try:
                (self._export_dir / name).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            self._proc_output_log.unlink(missing_ok=True)
        except Exception:
            pass

        env = self._process.processEnvironment()
        env.insert("TOOLBOX_EXPORT_DIR", str(self._export_dir))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONNOUSERSITE", "0")
        try:
            existing = env.value("PYTHONPATH")
            paths = [p for p in existing.split(os.pathsep) if p] if existing else []

            repo_root = None
            if len(self._tool_dir.parents) >= 3:
                repo_root = self._tool_dir.parents[2]
            if repo_root:
                repo_root_str = str(repo_root)
                if repo_root_str not in paths:
                    paths.insert(0, repo_root_str)

            user_site = site.getusersitepackages()
            if user_site and user_site not in paths:
                paths.insert(0, user_site)

            if paths:
                env.insert("PYTHONPATH", os.pathsep.join(paths))
        except Exception:
            pass
        self._process.setProcessEnvironment(env)

        self._process.setWorkingDirectory(str(app_path.parent))

        port = self._port
        if isinstance(port, str):
            port = 0 if port.strip().lower() == "auto" else int(port)

        args = [
            str(runner),
            "--app",
            str(app_path),
            "--host",
            str(self._host),
            "--port",
            str(int(port)),
            "--export-dir",
            str(self._export_dir),
        ]
        self._process.start(os.fspath(Path(self._python_executable)), args)
        self._poll_timer.start()

    @Slot()
    def _on_proc_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            try:
                self._proc_output_log.parent.mkdir(parents=True, exist_ok=True)
                with self._proc_output_log.open("a", encoding="utf-8") as fh:
                    fh.write(data)
            except Exception:
                pass
            self._proc_output_tail = (self._proc_output_tail + data)[-4000:]
        if "TOOLBOX_SERVER" in data and self._server_url is None:
            for line in data.splitlines():
                if line.startswith("TOOLBOX_SERVER "):
                    try:
                        info = json.loads(line[len("TOOLBOX_SERVER ") :].strip())
                        self._server_url = f"http://{info['host']}:{info['port']}/"
                        self._open_browser()
                    except Exception:
                        pass

    @Slot(QProcess.ProcessError)
    def _on_proc_error(self, err: QProcess.ProcessError) -> None:
        msg = (
            "Dash server process error.\n\n"
            f"Error: {err}\n\n"
            f"Output log:\n{self._proc_output_log}"
        )
        self._show_startup_error("Dash server error", msg)

    @Slot(int, QProcess.ExitStatus)
    def _on_proc_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        err_path = self._export_dir / "server_error.txt"
        if code != 0 and err_path.exists():
            try:
                details = err_path.read_text(encoding="utf-8")[:4000]
            except Exception:
                details = f"Details were written to:\n{err_path}"
            self._show_startup_error(
                "Dash server failed",
                f"The Dash server exited with code {code}.\n\n{details}",
            )
        elif code != 0:
            self._show_startup_error(
                "Dash server failed",
                "Dash server exited without an error report.\n\n"
                f"Exit code: {code}\n"
                f"Output log:\n{self._proc_output_log}\n\n"
                f"Last output:\n{self._proc_output_tail}",
            )

    @Slot()
    def _poll_for_server_json(self) -> None:
        if self._server_url is not None:
            self._poll_timer.stop()
            return
        if self._process.state() == QProcess.NotRunning:
            if self._proc_output_tail:
                self._show_startup_error(
                    "Dash server stopped",
                    "The Dash server stopped before it could be reached.\n\n"
                    f"Output log:\n{self._proc_output_log}\n\n"
                    f"Last output:\n{self._proc_output_tail}",
                )
            return
        p = self._export_dir / "server.json"
        if not p.exists():
            return
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            self._server_url = f"http://{info['host']}:{info['port']}/"
            self._open_browser()
            self._poll_timer.stop()
        except Exception:
            return


class _DashToolWindow(QMainWindow):
    def __init__(
        self,
        *,
        tool_id: str,
        title: str,
        tool_dir: Path,
        entry_script: str,
        host: str,
        port: int | str,
        python_executable: Optional[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._tool_id = tool_id
        self._tool_dir = tool_dir
        self._entry_script = entry_script
        self._host = host
        self._port = port
        self._python_executable = python_executable or sys.executable

        self._export_dir = _tool_export_dir(tool_id)
        self._last_capture: Optional[dict[str, Any]] = None
        self._proc_output_log = self._export_dir / "server_output.log"
        self._proc_output_tail = ""

        self.setWindowTitle(title)

        if QWebEngineView is None:
            QMessageBox.critical(
                self,
                "Missing Qt WebEngine",
                "PySide6 QtWebEngine is not available. Install it and restart.\n\n"
                "Typical pip package: PySide6-QtWebEngine",
            )
            raise RuntimeError("QtWebEngine not available")

        self.web = QWebEngineView(self)
        self.setCentralWidget(self.web)

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_proc_output)
        self._process.finished.connect(self._on_proc_finished)
        self._process.errorOccurred.connect(self._on_proc_error)

        self._server_url: Optional[str] = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(250)
        self._poll_timer.timeout.connect(self._poll_for_server_json)

        self._build_toolbar()
        self._start_dash()

    def _show_startup_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        if QWebEngineView is not None:
            safe = (
                message.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            self.web.setHtml(f"<h3>{title}</h3><pre>{safe}</pre>")

    def _build_toolbar(self) -> None:
        tb = QToolBar("Toolbox", self)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_capture = QAction("Capture results", self)
        act_capture.triggered.connect(self.capture_results)
        tb.addAction(act_capture)

        act_xlsx = QAction("Export to Excel (.xlsx)", self)
        act_xlsx.triggered.connect(self.export_to_excel)
        tb.addAction(act_xlsx)

        act_mc = QAction("Mathcad handoff", self)
        act_mc.triggered.connect(self.export_mathcad_handoff)
        tb.addAction(act_mc)

        act_html = QAction("Save report HTML", self)
        act_html.triggered.connect(self.save_report_html)
        tb.addAction(act_html)

        tb.addSeparator()

        act_stop = QAction("Stop server", self)
        act_stop.triggered.connect(self.stop_server)
        tb.addAction(act_stop)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_server()
        super().closeEvent(event)

    def _start_dash(self) -> None:
        entry_path = Path(self._entry_script)
        if not entry_path.is_absolute():
            entry_path = self._tool_dir / self._entry_script
        app_path = entry_path.resolve()

        if not app_path.exists():
            self._show_startup_error(
                "Missing Dash app",
                "Dash entry script not found:\n"
                f"{app_path}\n\n"
                "Place your Dash entry script in the assets folder or update the tool config.",
            )
            return

        runner = Path(__file__).resolve().parent / "dash_subprocess.py"
        if not runner.exists():
            self._show_startup_error("Missing runner", f"dash_subprocess.py not found at:\n{runner}")
            return

        for name in ("server.json", "server_error.txt"):
            try:
                (self._export_dir / name).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            self._proc_output_log.unlink(missing_ok=True)
        except Exception:
            pass

        env = self._process.processEnvironment()
        env.insert("TOOLBOX_EXPORT_DIR", str(self._export_dir))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONNOUSERSITE", "0")
        try:
            existing = env.value("PYTHONPATH")
            paths = [p for p in existing.split(os.pathsep) if p] if existing else []

            repo_root = None
            if len(self._tool_dir.parents) >= 3:
                repo_root = self._tool_dir.parents[2]
            if repo_root:
                repo_root_str = str(repo_root)
                if repo_root_str not in paths:
                    paths.insert(0, repo_root_str)

            user_site = site.getusersitepackages()
            if user_site and user_site not in paths:
                paths.insert(0, user_site)

            if paths:
                env.insert("PYTHONPATH", os.pathsep.join(paths))
        except Exception:
            pass
        self._process.setProcessEnvironment(env)

        self._process.setWorkingDirectory(str(app_path.parent))

        port = self._port
        if isinstance(port, str):
            port = 0 if port.strip().lower() == "auto" else int(port)

        args = [
            str(runner),
            "--app",
            str(app_path),
            "--host",
            str(self._host),
            "--port",
            str(int(port)),
            "--export-dir",
            str(self._export_dir),
        ]
        self._process.start(os.fspath(Path(self._python_executable)), args)
        self._poll_timer.start()

    @Slot()
    def stop_server(self) -> None:
        self._poll_timer.stop()
        if self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(1500):
                self._process.kill()
                self._process.waitForFinished(1500)

    @Slot()
    def _on_proc_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            try:
                self._proc_output_log.parent.mkdir(parents=True, exist_ok=True)
                with self._proc_output_log.open("a", encoding="utf-8") as fh:
                    fh.write(data)
            except Exception:
                pass
            self._proc_output_tail = (self._proc_output_tail + data)[-4000:]
        if "TOOLBOX_SERVER" in data and self._server_url is None:
            for line in data.splitlines():
                if line.startswith("TOOLBOX_SERVER "):
                    try:
                        info = json.loads(line[len("TOOLBOX_SERVER ") :].strip())
                        self._server_url = f"http://{info['host']}:{info['port']}/"
                        self.web.load(self._server_url)
                    except Exception:
                        pass

    @Slot(QProcess.ProcessError)
    def _on_proc_error(self, err: QProcess.ProcessError) -> None:
        msg = (
            "Dash server process error.\n\n"
            f"Error: {err}\n\n"
            f"Output log:\n{self._proc_output_log}"
        )
        self._show_startup_error("Dash server error", msg)

    @Slot(int, QProcess.ExitStatus)
    def _on_proc_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        err_path = self._export_dir / "server_error.txt"
        if code != 0 and err_path.exists():
            try:
                details = err_path.read_text(encoding="utf-8")[:4000]
            except Exception:
                details = f"Details were written to:\n{err_path}"
            self._show_startup_error(
                "Dash server failed",
                f"The Dash server exited with code {code}.\n\n{details}",
            )
        elif code != 0:
            self._show_startup_error(
                "Dash server failed",
                "Dash server exited without an error report.\n\n"
                f"Exit code: {code}\n"
                f"Output log:\n{self._proc_output_log}\n\n"
                f"Last output:\n{self._proc_output_tail}",
            )

    @Slot()
    def _poll_for_server_json(self) -> None:
        if self._server_url is not None:
            self._poll_timer.stop()
            return
        if self._process.state() == QProcess.NotRunning:
            if self._proc_output_tail:
                self._show_startup_error(
                    "Dash server stopped",
                    "The Dash server stopped before it could be reached.\n\n"
                    f"Output log:\n{self._proc_output_log}\n\n"
                    f"Last output:\n{self._proc_output_tail}",
                )
            return
        p = self._export_dir / "server.json"
        if not p.exists():
            return
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            self._server_url = f"http://{info['host']}:{info['port']}/"
            self.web.load(self._server_url)
            self._poll_timer.stop()
        except Exception:
            return

    @Slot()
    def capture_results(self) -> None:
        def _cb(result: Any) -> None:
            if not isinstance(result, dict) or not result.get("ok"):
                err = result.get("error") if isinstance(result, dict) else "Unknown error"
                QMessageBox.warning(self, "Capture failed", f"Could not capture results.\n\n{err}")
                return

            payload = result.get("payload")
            capture = {
                "meta": {
                    "captured_at": _timestamp_slug(),
                    "url": self._server_url,
                    "tool_id": self._tool_id,
                },
                "payload": payload,
            }
            self._last_capture = capture
            out_path = self._export_dir / f"capture_{_timestamp_slug()}.json"
            _write_json(capture, out_path)
            QMessageBox.information(self, "Captured", f"Captured results to:\n{out_path}")

        self.web.page().runJavaScript(JS_CAPTURE, _cb)

    @Slot()
    def export_to_excel(self) -> None:
        if not self._last_capture:
            QMessageBox.information(self, "No capture", "Run 'Capture results' first.")
            return
        out_path = self._export_dir / f"export_{_timestamp_slug()}.xlsx"
        try:
            _export_excel(self._last_capture, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Excel export failed", str(e))
            return
        QMessageBox.information(self, "Excel exported", f"Wrote:\n{out_path}")

    @Slot()
    def export_mathcad_handoff(self) -> None:
        if not self._last_capture:
            QMessageBox.information(self, "No capture", "Run 'Capture results' first.")
            return
        out_dir = self._export_dir / f"mathcad_{_timestamp_slug()}"
        try:
            paths = _mathcad_handoff(self._last_capture, out_dir)
        except Exception as e:
            QMessageBox.critical(self, "Mathcad handoff failed", str(e))
            return
        QMessageBox.information(
            self,
            "Mathcad handoff written",
            "Wrote:\n" + "\n".join(str(p) for p in paths.values()),
        )

    @Slot()
    def save_report_html(self) -> None:
        out_path = self._export_dir / f"report_{_timestamp_slug()}.html"

        def _cb(html: str) -> None:
            try:
                out_path.write_text(html, encoding="utf-8")
            except Exception as e:
                QMessageBox.critical(self, "Save failed", str(e))
                return
            QMessageBox.information(self, "Saved", f"Wrote:\n{out_path}")

        self.web.page().toHtml(_cb)
