from __future__ import annotations

"""
Build a self-contained zipapp (.pyz) for the Engineering Toolbox without modifying source.

Outputs (in dist/):
- EngineeringToolbox.pyz
- Launch_Toolbox.bat
- run_extracted.py
- README.txt
- build_manifest.json
"""

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipapp
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
PYZ_NAME = "EngineeringToolbox.pyz"
APP_NAME = "Engineering Toolbox"


SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "dist",
    ".venv",
    "venv",
}


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    return False


def _copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if _should_skip(rel):
            continue
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return None


def _read_version() -> str:
    # Prefer toolbox_app.__version__ if it exists and imports cleanly.
    try:
        sys.path.insert(0, str(ROOT))
        import toolbox_app  # type: ignore

        ver = getattr(toolbox_app, "__version__", None)
        if isinstance(ver, str) and ver.strip():
            return ver.strip()
    except Exception:
        pass
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)

    # Fallback: read pyproject.toml
    if tomllib:
        try:
            data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
            ver = data.get("project", {}).get("version")
            if isinstance(ver, str) and ver.strip():
                return ver.strip()
        except Exception:
            pass

    return "0.0.0-dev"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_pyz(staging: Path, pyz_path: Path) -> None:
    # Ensure target parent exists.
    pyz_path.parent.mkdir(parents=True, exist_ok=True)
    zipapp.create_archive(
        source=staging,
        target=pyz_path,
        main="toolbox_app.__main__:main",
        compressed=True,
    )


def _render_launcher() -> str:
    return r"""@echo off
setlocal
set "HERE=%~dp0"
set "RUNNER=%HERE%run_extracted.py"

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Please install Python 3.11+ and try again.
  exit /b 1
)

python "%RUNNER%" %*
endlocal & exit /b %errorlevel%
"""


def _render_run_extracted(manifest_name: str) -> str:
    return f'''from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


APP_FOLDER_NAME = "EngineeringToolbox"
PYZ_NAME = "{PYZ_NAME}"
MANIFEST_NAME = "{manifest_name}"


def _base_data_dir() -> Path:
    env_dir = os.environ.get("ENGINEERING_TOOLBOX_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local)
    home = Path.home()
    return home / "AppData" / "Local"


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {{}}


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    for member in zf.namelist():
        member_path = dest / member
        if not member_path.resolve().is_relative_to(dest.resolve()):
            raise RuntimeError(f"Unsafe path in archive: {{member}}")
    zf.extractall(dest)


def _discover_entrypoint(app_dir: Path) -> tuple[str, str | Path]:
    # 1) toolbox_app/__main__.py => module run
    if (app_dir / "toolbox_app" / "__main__.py").exists():
        return ("module", "toolbox_app")

    # 2) pyproject scripts
    pyproject = app_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # type: ignore

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            scripts = data.get("project", {{}}).get("scripts", {{}})
            if isinstance(scripts, dict):
                for name, target in scripts.items():
                    if isinstance(target, str) and ":" in target:
                        return ("module_func", target)
        except Exception:
            pass

    # 3) heuristic filenames
    candidates = []
    for rel in ("main.py", "app.py", "run.py"):
        for base in (app_dir, app_dir / "toolbox_app"):
            cand = base / rel
            if cand.exists():
                candidates.append(cand)
    if candidates:
        return ("script", candidates[0])

    return ("error", "No entry point found. Checked toolbox_app/__main__.py and main.py/app.py/run.py")


def _env_with_vendor(app_dir: Path) -> dict:
    env = os.environ.copy()
    use_vendor = env.get("TOOLBOX_USE_VENDOR", "1").lower()
    if use_vendor in ("0", "false", "no", "off"):
        return env
    vendor_dir = app_dir / "vendor"
    if vendor_dir.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(vendor_dir) + (os.pathsep + existing if existing else "")
    return env


def _run_app(app_dir: Path, argv: list[str]) -> int:
    mode, target = _discover_entrypoint(app_dir)
    if mode == "module":
        cmd = [sys.executable, "-m", str(target)]
    elif mode == "module_func":
        # module:function
        mod, func = str(target).split(":", 1)
        stub = (
            "import importlib; "
            "m=importlib.import_module(\\"" + mod + "\\"); "
            "getattr(m, \\"" + func + "\\")()"
        )
        cmd = [sys.executable, "-c", stub]
    elif mode == "script":
        cmd = [sys.executable, str(target)]
    else:
        sys.stderr.write(str(target) + "\\n")
        return 2

    env = _env_with_vendor(app_dir)
    proc = subprocess.run(cmd + argv, cwd=str(app_dir), env=env)
    return proc.returncode


def main() -> int:
    here = Path(__file__).resolve().parent
    manifest = _load_manifest(here / MANIFEST_NAME)
    version = manifest.get("version", "0.0.0-dev")
    expected_sha = manifest.get("sha256")

    pyz_path = here / PYZ_NAME
    if not pyz_path.exists():
        sys.stderr.write(f"Missing {{PYZ_NAME}} next to run_extracted.py\\n")
        return 2
    actual_sha = None
    try:
        import hashlib

        h = hashlib.sha256()
        with pyz_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
    except Exception:
        pass

    base_dir = _base_data_dir() / APP_FOLDER_NAME / "deployed" / version
    app_dir = base_dir / "app"
    deployed_manifest = base_dir / MANIFEST_NAME

    needs_extract = True
    if app_dir.exists() and deployed_manifest.exists():
        old = _load_manifest(deployed_manifest)
        if expected_sha and old.get("sha256") == expected_sha and actual_sha == expected_sha:
            needs_extract = False

    if needs_extract:
        app_dir.mkdir(parents=True, exist_ok=True)
        # Clean existing content
        if any(app_dir.iterdir()):
            shutil.rmtree(app_dir)
            app_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(pyz_path, "r") as zf:
            _safe_extract(zf, app_dir)
        if manifest:
            deployed_manifest.parent.mkdir(parents=True, exist_ok=True)
            deployed_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return _run_app(app_dir, sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
'''


def _render_readme() -> str:
    return f"""Engineering Toolbox – Zipapp Distribution

Build:
  python scripts/build_dist_pyz.py

What to upload/share:
  Upload/sync the entire dist/ folder (including {PYZ_NAME} and the launcher).

How to run (recommended):
  Run Launch_Toolbox.bat

Where it deploys locally:
  %LOCALAPPDATA%\\EngineeringToolbox\\deployed\\<version>\\app\\
  (override base with ENGINEERING_TOOLBOX_DATA_DIR env var)

Note:
  Launch_Toolbox.bat runs extracted mode so relative-path assets keep working.
"""


def _write_manifest(pyz_path: Path, version: str) -> dict:
    manifest = {
        "app_name": APP_NAME,
        "build_timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python_version_used": sys.version,
        "version": version,
        "sha256": _sha256(pyz_path),
    }
    commit = _git_commit()
    if commit:
        manifest["git_commit"] = commit
    manifest_path = DIST_DIR / "build_manifest.json"
    _write_file(manifest_path, json.dumps(manifest, indent=2))
    return manifest


def build() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    version = _read_version()

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir) / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        _copy_tree(ROOT, staging)
        pyz_path = DIST_DIR / PYZ_NAME
        _create_pyz(staging, pyz_path)

    manifest = _write_manifest(DIST_DIR / PYZ_NAME, version)

    _write_file(DIST_DIR / "Launch_Toolbox.bat", _render_launcher())
    _write_file(DIST_DIR / "run_extracted.py", _render_run_extracted("build_manifest.json"))
    _write_file(DIST_DIR / "README.txt", _render_readme())

    # Echo a brief summary
    print(f"Built {PYZ_NAME}")
    print(f"Version: {manifest.get('version')}")
    print(f"SHA256: {manifest.get('sha256')}")


if __name__ == "__main__":
    build()
