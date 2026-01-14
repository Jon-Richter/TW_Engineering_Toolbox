\
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure tool root is importable when backend app runs as a script
_TOOL_ROOT = _Path(__file__).resolve().parents[1]
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))


import argparse
import json
import mimetypes
import os
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, unquote

from models import SolveRequest
from backend.solver import solve as solve_engine, TOOL_VERSION


def _tool_root() -> Path:
    # .../toolbox_app/tools/<tool_id>/backend/app.py -> tool root is parents[2]
    return Path(__file__).resolve().parents[1]


def _frontend_dist() -> Path:
    return _tool_root() / "frontend" / "dist"


STATE: Dict[str, Any] = {
    "latest_run_dir": None,  # str
}


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _serve_file(handler: BaseHTTPRequestHandler, path: Path, content_type: Optional[str] = None) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND, "File not found")
        return
    data = path.read_bytes()
    if content_type is None:
        content_type, _ = mimetypes.guess_type(str(path))
    if content_type is None:
        content_type = "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class ToolHandler(BaseHTTPRequestHandler):
    server_version = "EngineeringToolboxACI318/0.1"

    def log_message(self, format: str, *args) -> None:
        # Quiet stdout logging; host app may capture separately.
        return

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/health":
                _json_response(self, 200, {"ok": True, "tool_version": TOOL_VERSION})
                return

            if path == "/api/report.html":
                run_dir = STATE.get("latest_run_dir")
                if not run_dir:
                    self.send_error(HTTPStatus.NOT_FOUND, "No report available yet. Run a solve first.")
                    return
                report = Path(run_dir) / "report.html"
                _serve_file(self, report, content_type="text/html; charset=utf-8")
                return

            if path.startswith("/api/download/"):
                run_dir = STATE.get("latest_run_dir")
                if not run_dir:
                    self.send_error(HTTPStatus.NOT_FOUND, "No artifacts available yet. Run a solve first.")
                    return
                file_name = unquote(path.split("/api/download/", 1)[1])
                # basic safety: prevent path traversal
                if (".." in file_name) or file_name.startswith(("/", "\\")):
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid file name")
                    return
                target = Path(run_dir) / file_name
                _serve_file(self, target)
                return

            # Static frontend
            dist = _frontend_dist()
            if path == "/" or path == "":
                _serve_file(self, dist / "index.html", content_type="text/html; charset=utf-8")
                return

            # Try serve exact file
            rel = path.lstrip("/")
            candidate = dist / rel
            if candidate.exists() and candidate.is_file():
                _serve_file(self, candidate)
                return

            # SPA fallback to index.html
            _serve_file(self, dist / "index.html", content_type="text/html; charset=utf-8")
        except Exception as e:
            _json_response(self, 500, {"ok": False, "error": str(e), "traceback": traceback.format_exc()})

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/solve":
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw.decode("utf-8") if raw else "{}")
                except Exception:
                    _json_response(self, 400, {"ok": False, "error": "Invalid JSON body"})
                    return

                # Validate base request
                try:
                    req = SolveRequest(**data)
                except Exception as e:
                    _json_response(self, 422, {"ok": False, "error": "Validation error", "details": str(e)})
                    return

                tool_id = _tool_root().name
                try:
                    results = solve_engine(tool_id, req.module, req.inputs)
                except Exception as e:
                    _json_response(self, 500, {"ok": False, "error": str(e), "traceback": traceback.format_exc()})
                    return

                STATE["latest_run_dir"] = results.get("run_dir")
                # Provide stable download names
                results["download"] = {
                    "report_html": "/api/report.html",
                    "excel": "/api/download/results.xlsx",
                    "calc_trace": "/api/download/calc_trace.json",
                    "mathcad_inputs": "/api/download/mathcad_inputs.csv",
                }
                _json_response(self, 200, results)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as e:
            _json_response(self, 500, {"ok": False, "error": str(e), "traceback": traceback.format_exc()})


def run_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ToolHandler)
    return server


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ACI 318-25 Concrete Design Tool backend server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    server = run_server(args.host, args.port)
    actual_port = server.server_address[1]
    # Print port for wrapper discovery (stdout)
    print(actual_port, flush=True)

    try:
        server.serve_forever(poll_interval=0.25)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())