from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Type

from pydantic import BaseModel

@dataclass(frozen=True)
class ToolMeta:
    id: str
    name: str
    category: str
    version: str
    description: str

class ToolBase(Protocol):
    """
    Tool contract (v2).

    Preferred: tools declare an `InputModel` (Pydantic) for typed inputs.
      - UI can auto-generate appropriate widgets
      - validation errors are user-friendly

    Backward compatible: tools may omit InputModel and implement default_inputs().
    """
    meta: ToolMeta
    InputModel: Optional[Type[BaseModel]]

    def default_inputs(self) -> Dict[str, Any]:
        ...

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...
