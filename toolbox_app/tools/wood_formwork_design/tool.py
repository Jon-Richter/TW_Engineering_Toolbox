from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from .paths import create_run_dir, server_state_dir

try:
    from toolbox_app.core.meta import ToolMeta  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class ToolMeta:
        id: str
        name: str
        category: str
        version: str
        description: str


class _ServerManager:
    def __init__(self, tool_root: Path, tool_id: str):
        # Always keep an absolute path to avoid cwd-relative duplication when spawning.
        self.tool_root = tool_root.resolve()
        self.tool_id = tool_id
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.log_file = None

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def url(self) -> str:
        if self.port is None:
            return "http://127.0.0.1:0/"
        return f"http://127.0.0.1:{self.port}/"

    def is_healthy(self) -> bool:
        if self.port is None:
            return False
        try:
            with urlopen(f"http://127.0.0.1:{self.port}/api/health", timeout=0.25) as r:
                return r.status == 200
        except Exception:
            return False

    def is_listening(self, timeout: float = 0.05) -> bool:
        if self.port is None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", int(self.port)), timeout=timeout):
                return True
        except OSError:
            return False

    def start(self) -> None:
        if self.process and self.is_healthy():
            return

        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass

        self.port = self._find_free_port()
        app_py = (self.tool_root / "backend" / "app.py").resolve()
        if not app_py.exists():
            raise FileNotFoundError(f"Backend app not found: {app_py}")

        sdir = server_state_dir(self.tool_id)
        sdir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(sdir / "server.log", "a", encoding="utf-8")

        cmd = [sys.executable, str(app_py), "--host", "127.0.0.1", "--port", str(self.port)]

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.tool_root),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def stop(self) -> None:
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass
        try:
            if self.log_file:
                self.log_file.flush()
                self.log_file.close()
        except Exception:
            pass
        self.process = None
        self.port = None


class WoodFormworkDesignTool:
    RUNS_ON_UI_THREAD = True

    def __init__(self):
        self.meta = ToolMeta(
            id="wood_formwork_design",
            name="Wood Formwork Design",
            category="Concrete",
            version="1.1.0",
            description="Wood formwork design per ACI 347R-14 with NDS-adjusted allowables and segment checks (React UI + Python backend).",
        )
        self.InputModel = None
        self._server: Optional[_ServerManager] = None
        self._view = None
        self._startup_timer = None

    def default_inputs(self) -> dict:
        return {}

    def _ensure_server(self) -> _ServerManager:
        if self._server is None:
            tool_root = Path(__file__).resolve().parent
            self._server = _ServerManager(tool_root=tool_root, tool_id=self.meta.id)
            atexit.register(self._server.stop)
        return self._server

    def run(self, inputs: dict) -> dict:
        launch_run_dir = create_run_dir(self.meta.id, inputs or {"seed": str(time.time())})

        server = self._ensure_server()
        server.start()

        url = server.url()
        startup_timeout_s = float(os.environ.get("TOOLBOX_TOOL_STARTUP_TIMEOUT_S", "20.0"))
        t0 = time.time()

        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtCore import QTimer
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            view = QWebEngineView()
            view.setWindowTitle(self.meta.name)
            view.resize(1200, 860)

            def show_startup_message(message: str) -> None:
                view.setHtml(
                    f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{self.meta.name}</title>
    <style>
      body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
      code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
      .muted {{ color: #666; }}
    </style>
  </head>
  <body>
    <h2>{self.meta.name}</h2>
    <p>{message}</p>
    <p class="muted">URL: <code>{url}</code></p>
  </body>
</html>
""",
                )

            if server.is_listening():
                view.load(QUrl(url))
            else:
                show_startup_message("Starting backend…")

                deadline = t0 + startup_timeout_s
                timer = QTimer(view)
                timer.setInterval(150)

                def tick() -> None:
                    if server.is_listening():
                        timer.stop()
                        view.load(QUrl(url))
                        return

                    if server.process and server.process.poll() is not None:
                        timer.stop()
                        log_path = server_state_dir(self.meta.id) / "server.log"
                        show_startup_message(
                            f"Backend failed to start. Check <code>{log_path}</code> and retry opening the tool."
                        )
                        return

                    if time.time() >= deadline:
                        timer.stop()
                        log_path = server_state_dir(self.meta.id) / "server.log"
                        show_startup_message(
                            f"Backend is still starting. Wait a few seconds, then refresh. If it never loads, check <code>{log_path}</code>."
                        )

                timer.timeout.connect(tick)  # type: ignore[attr-defined]
                timer.start()
                self._startup_timer = timer

            view.show()

            try:
                view.destroyed.connect(lambda *_: server.stop())  # type: ignore
            except Exception:
                pass

            self._view = view
        except Exception:
            # No embedded web view available: ensure the server is at least listening
            # before returning the URL to any host app.
            while (time.time() - t0) < startup_timeout_s:
                if server.is_listening():
                    break
                if server.process and server.process.poll() is not None:
                    break
                time.sleep(0.1)

        return {
            "ok": True,
            "status": "launched",
            "url": url,
            "run_dir": str(launch_run_dir),
            "server_healthy": server.is_healthy(),
        }


TOOL = WoodFormworkDesignTool()
