from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from loguru import logger

@dataclass
class RunResult:
    ok: bool
    output: Dict[str, Any]
    error: Optional[str] = None

class RunnerSignals(QObject):
    finished = Signal(object)  # RunResult
    progress = Signal(int)     # 0-100
    status = Signal(str)

class ToolRunnable(QRunnable):
    def __init__(
        self,
        fn: Callable[[Dict[str, Any], Callable[[int], None], Callable[[str], None], Callable[[], bool]], Dict[str, Any]],
        inputs: Dict[str, Any],
        cancel_flag: Callable[[], bool],
    ) -> None:
        super().__init__()
        self.fn = fn
        self.inputs = inputs
        self.signals = RunnerSignals()
        self._cancel_flag = cancel_flag

    @Slot()
    def run(self) -> None:
        try:
            def prog(p: int) -> None:
                self.signals.progress.emit(int(max(0, min(100, p))))
            def stat(s: str) -> None:
                self.signals.status.emit(str(s))
            out = self.fn(self.inputs, prog, stat, self._cancel_flag)
            self.signals.finished.emit(RunResult(ok=True, output=out))
        except Exception as e:
            logger.exception(e)
            self.signals.finished.emit(RunResult(ok=False, output={}, error=str(e)))

class ToolRunner:
    """
    Runs tools off the UI thread using QThreadPool.

    Tools can optionally accept progress callbacks by defining `run_with_context`.
    Signature:
      run_with_context(inputs, progress_cb, status_cb, is_cancelled_cb) -> dict
    Otherwise we call `run(inputs)`.
    """
    def __init__(self) -> None:
        self.pool = QThreadPool.globalInstance()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def start(self, tool: Any, inputs: Dict[str, Any]) -> ToolRunnable:
        self._cancelled = False
        if hasattr(tool, "run_with_context"):
            fn = getattr(tool, "run_with_context")
        else:
            # wrap plain run()
            def fn(inputs_: Dict[str, Any], _p, _s, _c) -> Dict[str, Any]:
                return tool.run(inputs_)
        r = ToolRunnable(fn=fn, inputs=inputs, cancel_flag=self.is_cancelled)
        self.pool.start(r)
        return r
