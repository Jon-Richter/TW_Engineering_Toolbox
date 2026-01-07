from __future__ import annotations

import argparse
import json
import os
import runpy
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _write_server_info(export_dir: Path, host: str, port: int) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "host": host,
        "port": port,
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (export_dir / "server.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"TOOLBOX_SERVER {json.dumps(info)}", flush=True)


def _find_dash_app(ns: dict[str, Any]) -> Any:
    candidate = ns.get("app") or ns.get("APP") or ns.get("dash_app") or ns.get("DASH_APP")
    if candidate is not None:
        return candidate

    try:
        from dash import Dash  # type: ignore
    except Exception:
        return None

    for v in ns.values():
        if isinstance(v, Dash):
            return v
    return None


def _prepare_sys_path(app_path: Path) -> None:
    runner_dir = Path(__file__).resolve().parent
    app_dir = app_path.parent.resolve()
    runner_norm = os.path.normcase(os.path.abspath(str(runner_dir)))
    app_norm = os.path.normcase(os.path.abspath(str(app_dir)))

    new_paths: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            entry_norm = os.path.normcase(os.path.abspath(entry))
        except Exception:
            entry_norm = entry
        if entry_norm == runner_norm:
            continue
        if entry_norm == app_norm:
            continue
        new_paths.append(entry)

    new_paths.insert(0, str(app_dir))
    sys.path[:] = new_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="Path to Dash entry script (e.g., app.py)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = auto-pick free port")
    ap.add_argument("--export-dir", required=True, help="User-writable directory for server.json, logs, exports")
    args = ap.parse_args()

    app_path = Path(args.app).resolve()
    export_dir = Path(args.export_dir).resolve()

    if not app_path.exists():
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "server_error.txt").write_text(
            f"Dash entry script not found: {app_path}\n", encoding="utf-8"
        )
        print(f"ERROR Dash entry script not found: {app_path}", file=sys.stderr, flush=True)
        return 2

    host = str(args.host)
    port = int(args.port) if int(args.port) != 0 else _pick_free_port(host)

    os.chdir(str(app_path.parent))
    os.environ.setdefault("TOOLBOX_EXPORT_DIR", str(export_dir))
    os.environ.setdefault("TOOLBOX_HOST", host)
    os.environ.setdefault("TOOLBOX_PORT", str(port))

    _write_server_info(export_dir, host, port)
    _prepare_sys_path(app_path)

    try:
        ns = runpy.run_path(str(app_path), run_name="toolbox_dash_app")
        dash_app = _find_dash_app(ns)
        if dash_app is None:
            raise RuntimeError(
                "No Dash app instance found in the entry script. "
                "Expected a global variable named 'app' (dash.Dash)."
            )

        if hasattr(dash_app, "run"):
            dash_app.run(host=host, port=port, debug=False, use_reloader=False)
        else:
            dash_app.run_server(host=host, port=port, debug=False, use_reloader=False)

        return 0
    except Exception:
        export_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (export_dir / "server_error.txt").write_text(tb, encoding="utf-8")
        print(tb, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
