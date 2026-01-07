from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .calc_trace import CalcTrace
from .analysis.placement import PlacementOutcome
from .evaluation import EvalResult, evaluate_fast, evaluate_with_trace

@dataclass(frozen=True)
class SolveOutcome:
    optimal_spacing_ft: float
    final: Optional[EvalResult]
    feasible: bool
    message: str

def solve_optimal_spacing(trace: CalcTrace, inputs: Dict[str, Any], placement: PlacementOutcome) -> SolveOutcome:
    """
    Scan bracket spacing between min/max bounds and return the largest spacing that passes SWL checks.
    """
    min_s = float(inputs["min_spacing_ft"])
    max_s = float(inputs["max_spacing_ft"])
    step = float(inputs["spacing_step_ft"])

    if step <= 0:
        raise ValueError("spacing_step_ft must be > 0.")

    # Deterministic scan from max down to min (find first passing = optimal)
    n = int(round((max_s - min_s) / step))
    spacings = [max_s - i * step for i in range(n + 1)]
    spacings = [s for s in spacings if s >= min_s - 1e-9]

    best_spacing = None
    best_eval = None

    for s in spacings:
        res = evaluate_fast(inputs, s, placement=placement)
        if res.passes:
            best_spacing = s
            best_eval = res
            break

    if best_spacing is None:
        return SolveOutcome(
            optimal_spacing_ft=float("nan"),
            final=None,
            feasible=False,
            message="No feasible spacing found within bounds.",
        )

    # Produce full trace for governing spacing
    final = evaluate_with_trace(trace, inputs, best_spacing, placement=placement)

    return SolveOutcome(
        optimal_spacing_ft=float(best_spacing),
        final=final,
        feasible=True,
        message="Feasible spacing found.",
    )
