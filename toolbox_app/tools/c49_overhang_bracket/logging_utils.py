from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

LOGURU_AVAILABLE = False
try:
    from loguru import logger as _logger  # type: ignore

    LOGURU_AVAILABLE = True
except Exception:  # pragma: no cover
    import logging

    _logger = logging.getLogger(__name__)  # type: ignore


def get_run_logger(run_dir: Path, tool_id: str, input_hash: Optional[str] = None) -> Tuple[Any, Optional[int]]:
    """Create a run-scoped logger writing to <run_dir>/run.log.

    - In the host app, `loguru` is used and already configured at the application level.
      We add a *run-only* sink filtered to this tool/run.
    - In environments without loguru, fall back to stdlib logging.

    Returns:
      (logger, sink_id) where sink_id is only used for loguru and may be None.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    if LOGURU_AVAILABLE:
        bound = _logger.bind(tool_id=tool_id, run_dir=str(run_dir), input_hash=input_hash or "")
        sink_id = _logger.add(
            str(log_path),
            level="DEBUG",
            enqueue=False,
            backtrace=True,
            diagnose=False,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[tool_id]} | {message}",
            filter=lambda r: r["extra"].get("tool_id") == tool_id and r["extra"].get("run_dir") == str(run_dir),
        )
        return bound, int(sink_id)

    # --- Fallback: stdlib logging (minimal)
    import logging  # pragma: no cover

    logger = logging.getLogger(f"{tool_id}:{run_dir}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers):
        fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger, None


def remove_run_logger_sink(sink_id: Optional[int]) -> None:
    """Remove a loguru sink created by get_run_logger (no-op for None)."""
    if sink_id is None:
        return
    if LOGURU_AVAILABLE:
        try:
            _logger.remove(sink_id)
        except Exception:
            # Never allow logging cleanup to break tool execution
            pass
