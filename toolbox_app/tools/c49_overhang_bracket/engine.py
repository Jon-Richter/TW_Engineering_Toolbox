"""
Legacy facade retained for backward compatibility with earlier iterations.

New workflow:
- solve_best_placement(...) -> PlacementOutcome
- solve_optimal_spacing(..., placement=...) -> SolveOutcome
"""
from __future__ import annotations

from typing import Any, Dict

from .calc_trace import CalcTrace
from .analysis.placement import PlacementOutcome, solve_best_placement
from .evaluation import EvalResult, evaluate_with_trace, evaluate_fast
from .solver import solve_optimal_spacing, SolveOutcome

__all__ = [
    "CalcTrace",
    "PlacementOutcome",
    "solve_best_placement",
    "EvalResult",
    "evaluate_fast",
    "evaluate_with_trace",
    "solve_optimal_spacing",
    "SolveOutcome",
]
