from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import pandas as pd

# Ensure tool root is importable when backend app runs as a script.
_TOOL_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[4]
for root in (_REPO_ROOT, _TOOL_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from backend import engine  # noqa: E402
from paths import create_run_dir  # noqa: E402


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _frontend_dist() -> Path:
    return _tool_root() / "frontend" / "dist"


STATE: Dict[str, Any] = {"latest_run_dir": None}


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
    server_version = "EngineeringToolboxFormwork/0.1"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/health":
                _json_response(self, 200, {"ok": True, "tool_version": engine.TOOL_VERSION})
                return

            if path == "/api/config":
                _json_response(self, 200, engine.get_config())
                return

            if path.startswith("/api/download/"):
                run_dir = STATE.get("latest_run_dir")
                if not run_dir:
                    self.send_error(HTTPStatus.NOT_FOUND, "No artifacts available yet. Run a solve first.")
                    return
                file_name = unquote(path.split("/api/download/", 1)[1])
                if (".." in file_name) or file_name.startswith(("/", "\\")):
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid file name")
                    return
                target = Path(run_dir) / file_name
                _serve_file(self, target)
                return

            dist = _frontend_dist()
            if path == "/" or path == "":
                _serve_file(self, dist / "index.html", content_type="text/html; charset=utf-8")
                return

            rel = path.lstrip("/")
            candidate = dist / rel
            if candidate.exists() and candidate.is_file():
                _serve_file(self, candidate)
                return

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

                try:
                    results = engine.solve(data)
                except Exception as e:
                    _json_response(self, 500, {"ok": False, "error": str(e), "traceback": traceback.format_exc()})
                    return

                run_dir = create_run_dir(engine.TOOL_ID, data)
                (run_dir / "inputs.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
                (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

                seg_df = pd.DataFrame(results.get("segment_table") or [])
                seg_csv = run_dir / "segment_checks.csv"
                seg_df.to_csv(seg_csv, index=False)

                report_payload = engine.build_report_payload(
                    inputs=results.get("inputs", {}),
                    pressure_summary=results.get("pressure_summary", {}),
                    util_summary=results.get("util_summary", {}),
                    segment_table=results.get("segment_table", []),
                )
                report_pdf = engine.build_pdf_bytes(report_payload)
                (run_dir / "report.pdf").write_bytes(report_pdf)

                STATE["latest_run_dir"] = str(run_dir)
                results["run_dir"] = str(run_dir)
                results["download"] = {
                    "segment_csv": "/api/download/segment_checks.csv",
                    "report_pdf": "/api/download/report.pdf",
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
    parser = argparse.ArgumentParser(description="Wood Formwork Design backend server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    server = run_server(args.host, args.port)
    actual_port = server.server_address[1]
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
