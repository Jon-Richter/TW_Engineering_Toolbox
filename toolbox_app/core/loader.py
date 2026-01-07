from __future__ import annotations
import importlib
import pkgutil
from typing import List
from loguru import logger
from .tool_base import ToolBase

TOOLS_PKG = "toolbox_app.tools"

def discover_tools() -> List[ToolBase]:
    """
    Discovers tools under toolbox_app.tools.* that expose `TOOL` at module level.
    Pattern:
      toolbox_app/tools/my_tool/__init__.py defines TOOL = MyTool()
    """
    tools: List[ToolBase] = []
    pkg = importlib.import_module(TOOLS_PKG)
    for m in pkgutil.iter_modules(pkg.__path__):
        mod_name = f"{TOOLS_PKG}.{m.name}"
        try:
            mod = importlib.import_module(mod_name)
            tool = getattr(mod, "TOOL", None)
            if tool is None:
                logger.warning(f"Module {mod_name} has no TOOL export; skipping.")
                continue
            tools.append(tool)
        except Exception as e:
            logger.exception(f"Failed loading tool {mod_name}: {e}")
    # stable ordering
    tools.sort(key=lambda t: (t.meta.category.lower(), t.meta.name.lower()))
    return tools
