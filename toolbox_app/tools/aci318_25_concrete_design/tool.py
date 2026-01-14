\
from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen

from .paths import create_run_dir, server_state_dir

# ToolMeta fallback (host app may provide its own ToolMeta)
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
        self.tool_root = tool_root
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
                return (r.status == 200)
        except Exception:
            return False

    def start(self) -> None:
        if self.process and self.is_healthy():
            return

        # best effort: clean up any prior process handle
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass

        self.port = self._find_free_port()

        app_py = self.tool_root / "backend" / "app.py"
        if not app_py.exists():
            raise FileNotFoundError(f"Backend app not found: {app_py}")

        # logs must go to LOCALAPPDATA tool dir
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


class ACI318ConcreteDesignTool:
    RUNS_ON_UI_THREAD = True

    def __init__(self):
        self.meta = ToolMeta(
            id="aci318_25_concrete_design",
            name="ACI 318-25 Concrete Design",
            category="Concrete",
            version="0.4.0",
            description="Offline ACI 318-25 concrete design tool (React UI + Python backend) producing audit-grade calc packages.",
        )
        self.InputModel = None
        self._server: Optional[_ServerManager] = None
        self._view = None

    def default_inputs(self) -> dict:
        return {}

    def _ensure_server(self) -> _ServerManager:
        if self._server is None:
            tool_root = Path(__file__).resolve().parent
            self._server = _ServerManager(tool_root=tool_root, tool_id=self.meta.id)
            atexit.register(self._server.stop)
        return self._server

    def run(self, inputs: dict) -> dict:
        # Create a run_dir for this launch (individual solves create their own run_dir)
        launch_run_dir = create_run_dir(self.meta.id, seed=str(time.time()))

        server = self._ensure_server()
        server.start()

        # brief readiness check (non-blocking policy)
        t0 = time.time()
        healthy = False
        while time.time() - t0 < 2.0:
            if server.is_healthy():
                healthy = True
                break
            time.sleep(0.1)

        url = server.url()

        # Open QtWebEngine view
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                # Host app should have one; if not, create for standalone dev.
                app = QApplication(sys.argv)

            view = QWebEngineView()
            view.setWindowTitle(self.meta.name)
            view.resize(1200, 800)
            view.load(QUrl(url))
            view.show()

            # Best-effort shutdown when view destroyed
            try:
                view.destroyed.connect(lambda *_: server.stop())  # type: ignore
            except Exception:
                pass

            self._view = view
        except Exception:
            # If QtWebEngine isn't available, still return launch info.
            pass

        return {
            "ok": True,
            "status": "launched",
            "url": url,
            "run_dir": str(launch_run_dir),
            "server_healthy": healthy,
        }


TOOL = ACI318ConcreteDesignTool()
