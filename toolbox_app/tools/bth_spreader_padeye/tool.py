from __future__ import annotations
import json, os, sys, time, socket, subprocess, threading
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field

from .paths import create_run_dir

TOOL_ID = "bth_spreader_padeye"
TOOL_VERSION = "1.2.0"
RUNS_ON_UI_THREAD = True

try:
    from toolbox_app.core.meta import ToolMeta  # type: ignore
except Exception:
    @dataclass
    class ToolMeta:
        id: str
        name: str
        category: str
        version: str
        description: str

try:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
except Exception:
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    QUrl = None  # type: ignore[assignment]
    QMainWindow = None  # type: ignore[assignment]
    QWidget = None  # type: ignore[assignment]
    QVBoxLayout = None  # type: ignore[assignment]

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None  # type: ignore[assignment]

def _find_free_port() -> int:
    s=socket.socket()
    s.bind(("127.0.0.1",0))
    p=s.getsockname()[1]
    s.close()
    return p

def _healthcheck(base: str, timeout_s: float = 1.5) -> bool:
    import urllib.request
    end=time.time()+timeout_s
    while time.time()<end:
        try:
            with urllib.request.urlopen(base+"/api/health", timeout=0.5) as r:
                return r.status==200
        except Exception:
            time.sleep(0.1)
    return False

class _Tool:
    RUNS_ON_UI_THREAD = True
    meta = ToolMeta(
        id=TOOL_ID,
        name="BTH-1 Spreader Bar + Padeye",
        category="Lifting",
        version=TOOL_VERSION,
        description="Design steel spreader bars and padeyes per ASME BTH-1-2023 with audit-grade calc package.",
    )
    class InputModel(BaseModel):
        mode: Literal["padeye", "spreader", "spreader_two_way"] = Field(
            "padeye",
            title="Mode",
            description="Select the module to open in the full UI.",
        )

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._base_url: str | None = None
        self._view = None
        self._window: _WebWindow | None = None
        self._log_fp = None
        self._last_run_dir: str | None = None

    def default_inputs(self) -> dict:
        return {
            "mode":"padeye",
        }

    def _ensure_server(self, run_dir: str, port: int | None = None) -> str:
        if self._proc and self._proc.poll() is not None:
            self._proc = None
            self._base_url = None
        if self._proc and self._base_url:
            return self._base_url

        port = port or _find_free_port()
        root_dir = Path(__file__).resolve().parents[3]
        env=os.environ.copy()
        env["BTH_TOOL_PORT"]=str(port)
        env["BTH_TOOL_RUN_DIR"]=run_dir
        env["PYTHONPATH"]=str(root_dir) + os.pathsep + env.get("PYTHONPATH","")

        mod="toolbox_app.tools.bth_spreader_padeye.backend.app"
        cmd=[sys.executable, "-m", mod]
        log_path = os.path.join(run_dir, "server.log")
        try:
            Path(log_path).write_text("Launching backend process...\n", encoding="utf-8")
            self._log_fp = open(log_path, "a", encoding="utf-8")
        except Exception:
            self._log_fp = None

        try:
            self._proc=subprocess.Popen(
                cmd,
                env=env,
                cwd=str(root_dir),
                stdout=self._log_fp or subprocess.DEVNULL,
                stderr=subprocess.STDOUT if self._log_fp else subprocess.DEVNULL,
            )
        except Exception:
            app_py=os.path.join(os.path.dirname(__file__), "backend", "app.py")
            self._proc=subprocess.Popen(
                [sys.executable, app_py],
                env=env,
                cwd=str(root_dir),
                stdout=self._log_fp or subprocess.DEVNULL,
                stderr=subprocess.STDOUT if self._log_fp else subprocess.DEVNULL,
            )

        self._base_url=f"http://127.0.0.1:{port}"
        return self._base_url

    def _build_loader_html(self, base_url: str, mode: str) -> tuple[str, str]:
        base = base_url.rstrip("/")
        target_url = f"{base}/?mode={mode}"
        loader_html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{self.meta.name}</title>
    <style>
      body {{
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
        background: #0b1220;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
      }}
      .card {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
      }}
      .title {{
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 0.02em;
      }}
      .sub {{
        margin-top: 6px;
        font-size: 12px;
        color: #94a3b8;
      }}
      .bar {{
        margin: 16px auto 0;
        height: 3px;
        width: 140px;
        background: linear-gradient(90deg, #38bdf8, #f59e0b);
        border-radius: 999px;
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="title">Starting {self.meta.name}...</div>
      <div class="sub">Waiting for the local analysis server.</div>
      <div class="bar"></div>
    </div>
    <script>
      const base = {json.dumps(base)};
      const target = {json.dumps(target_url)};
      async function ping() {{
        try {{
          const res = await fetch(base + "/api/health", {{ cache: "no-store" }});
          if (res.ok) {{
            window.location.replace(target);
            return;
          }}
        }} catch (err) {{}}
        setTimeout(ping, 400);
      }}
      ping();
    </script>
  </body>
</html>"""
        return loader_html, target_url

    def _open_window(self, base: str, mode: str) -> None:
        loader_html, target_url = self._build_loader_html(base, mode)
        try:
            if QWebEngineView is None:
                import webbrowser
                webbrowser.open(target_url)
                return
            if self._window is None or not self._window.isVisible():
                self._window = _WebWindow(self.meta.name)
            self._window.load(loader_html, base)
            self._window.show()
            self._window.raise_()
            self._window.activateWindow()
        except Exception as e:
            try:
                if self._last_run_dir:
                    Path(self._last_run_dir, "launcher.log").write_text(
                        f"Window launch failed: {e}\n", encoding="utf-8"
                    )
            except Exception:
                pass
            import webbrowser
            webbrowser.open(target_url)

    def run(self, inputs: dict) -> dict:
        run_dir=str(create_run_dir(TOOL_ID, inputs))
        self._last_run_dir = run_dir
        mode = str(inputs.get("mode") or "padeye")
        if mode not in ("padeye", "spreader", "spreader_two_way"):
            mode = "padeye"
        port = _find_free_port()
        base = f"http://127.0.0.1:{port}"
        self._base_url = base

        def _start_server() -> None:
            try:
                self._ensure_server(run_dir, port)
            except Exception as e:
                try:
                    Path(run_dir, "launcher.log").write_text(
                        f"Backend launch failed: {e}\n", encoding="utf-8"
                    )
                except Exception:
                    pass

        threading.Thread(target=_start_server, daemon=True).start()
        try:
            Path(run_dir, "launcher.log").write_text(
                f"Launcher started. base_url={base}\n", encoding="utf-8"
            )
        except Exception:
            pass
        self._open_window(base, mode)
        return {"ok": True, "status":"launched", "url": base + "/", "run_dir": run_dir, "server_log": os.path.join(run_dir, "server.log"), "launcher_log": os.path.join(run_dir, "launcher.log")}

TOOL=_Tool()


if QMainWindow is not None:
    class _WebWindow(QMainWindow):
        def __init__(self, title: str) -> None:
            super().__init__()
            self.setWindowTitle(title)
            self.resize(1200, 850)
            self._web = QWebEngineView()
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._web)
            self.setCentralWidget(central)

        def load(self, loader_html: str, base_url: str) -> None:
            self._web.setHtml(loader_html, QUrl(base_url.rstrip("/") + "/"))
else:
    class _WebWindow:
        def __init__(self, title: str) -> None:
            raise RuntimeError("Qt runtime not available; cannot open embedded window.")

        def load(self, loader_html: str, base_url: str) -> None:
            return None
