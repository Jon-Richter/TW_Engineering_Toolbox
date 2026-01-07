from __future__ import annotations

import importlib
from pathlib import Path


def _check_path(label: str, path: Path) -> bool:
    ok = path.exists()
    status = "OK" if ok else "MISSING"
    print(f"[{status}] {label}: {path}")
    return ok


def _check_import(label: str, module_name: str, attr: str | None = None) -> bool:
    try:
        mod = importlib.import_module(module_name)
        if attr:
            getattr(mod, attr)
        print(f"[OK] import {label}")
        return True
    except Exception as e:
        print(f"[WARN] import {label} failed: {e}")
        return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ok = True

    ok &= _check_path(
        "Dash runner subprocess",
        root / "toolbox_app" / "core" / "runners" / "dash_subprocess.py",
    )
    ok &= _check_path(
        "Wood formwork Dash entry",
        root / "toolbox_app" / "tools" / "wood_formwork_design" / "assets" / "formwork_design.py",
    )
    ok &= _check_path(
        "Tower crane static index",
        root / "toolbox_app" / "tools" / "tower_crane_foundation" / "assets" / "index.html",
    )

    _check_import("PySide6", "PySide6")
    _check_import("PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineWidgets", "QWebEngineView")
    _check_import("openpyxl", "openpyxl")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
