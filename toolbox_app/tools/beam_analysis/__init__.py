"""Beam Analysis Tool plugin for Engineering Toolbox (PySide6).

Exports:
  - TOOL: an instance of BeamAnalysisTool
  - RUNS_ON_UI_THREAD = True (tool launches its own Qt window)
"""
from __future__ import annotations

try:
    # Prefer the host app's ToolMeta if present
    from toolbox_app.models import ToolMeta  # type: ignore
except Exception:  # pragma: no cover
    try:
        from toolbox_app.core import ToolMeta  # type: ignore
    except Exception:  # pragma: no cover
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ToolMeta:  # fallback: host typically only needs attribute access
            id: str
            name: str
            category: str
            version: str
            description: str

from .beam_tool import BeamAnalysisTool

RUNS_ON_UI_THREAD = True

TOOL = BeamAnalysisTool(
    ToolMeta(
        id="beam_analysis",
        name="Beam Analysis Tool",
        category="Structural Analysis",
        version="1.0.0",
        description="Multi-span Euler–Bernoulli beam analysis with variable EI, supports, and common loads. Plots shear, moment, and deflection; exports to Excel/Mathcad.",
    )
)
