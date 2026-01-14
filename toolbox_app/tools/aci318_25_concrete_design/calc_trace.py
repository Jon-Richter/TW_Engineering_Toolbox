\
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field, ConfigDict


class CalcVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    description: str
    value: float
    units: str
    source: str  # input:<id> | db:<name> | step:<step_id> | assumption:<id>


class CalcReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str  # code/table/note/derived
    ref: str


class CalcCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    demand: float
    capacity: float
    ratio: float
    pass_fail: str


class CalcValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    units: str


class Rounding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str  # "decimals" or "sigfigs" or "none"
    decimals_or_sigfigs: int


class CalcStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    section: str
    title: str

    output_symbol: str
    output_description: str

    equation_latex: str
    substitution_latex: str

    variables: List[CalcVariable]

    result_unrounded: CalcValue
    rounding: Rounding
    result_rounded: CalcValue

    references: List[CalcReference]

    checks: Optional[List[CalcCheck]] = None
    warnings: Optional[List[str]] = None


class TraceMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_id: str
    tool_version: str
    report_version: str
    timestamp: str
    units_system: str
    code_basis: Optional[str] = None
    input_hash: str
    app_build: Optional[str] = None


class TraceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    value: Union[float, str, int, bool]
    units: str
    source: str  # user/default/extracted
    notes: Optional[str] = None


class TraceAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    governing_checks: List[Dict[str, Any]] = Field(default_factory=list)
    controlling_step_ids: List[str] = Field(default_factory=list)
    key_outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class CalcTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: TraceMeta
    inputs: List[TraceInput] = Field(default_factory=list)
    assumptions: List[TraceAssumption] = Field(default_factory=list)
    steps: List[CalcStep] = Field(default_factory=list)
    tables: Dict[str, Any] = Field(default_factory=dict)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    summary: TraceSummary = Field(default_factory=TraceSummary)

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def _format_num(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return str(value)
    # Use a compact representation with up to 12 significant digits.
    return f"{value:.12g}"


def _latex_value_with_units(value: float, units: str) -> str:
    u = units.strip()
    if u == "":
        return _format_num(value)
    # Use \mathrm for units to keep LaTeX-ish formatting without KaTeX requirement.
    return f"{_format_num(value)}\\,\\mathrm{{{u}}}"


def apply_rounding(value: float, rule: str, n: int) -> float:
    if rule == "none":
        return float(value)
    if rule == "decimals":
        return float(round(value, int(n)))
    if rule == "sigfigs":
        if value == 0:
            return 0.0
        sign = -1.0 if value < 0 else 1.0
        v = abs(float(value))
        exp = math.floor(math.log10(v))
        factor = 10 ** (n - 1 - exp)
        return sign * round(v * factor) / factor
    raise ValueError(f"Unknown rounding rule: {rule!r}")


def compute_step(
    trace: CalcTrace,
    *,
    id: str,
    section: str,
    title: str,
    output_symbol: str,
    output_description: str,
    equation_latex: str,
    variables: Sequence[Dict[str, Any]],
    compute_fn: Callable[[], float],
    units: str,
    rounding_rule: Dict[str, Any],
    references: Sequence[Dict[str, str]],
    checks_builder: Optional[Callable[[float], List[Dict[str, Any]]]] = None,
    warnings: Optional[List[str]] = None,
) -> float:
    """
    Required computation wrapper that:
    - Computes unrounded result
    - Applies rounding rule
    - Auto-generates substitution_latex by replacing symbols with numeric values + units
    - Appends CalcStep to trace
    - Returns rounded value
    """
    # Validate required args
    if not id or not section or not title:
        raise ValueError("compute_step requires non-empty id/section/title")
    if not output_symbol or not output_description:
        raise ValueError("compute_step requires output_symbol/output_description")
    if not equation_latex:
        raise ValueError("compute_step requires equation_latex")
    if not variables:
        raise ValueError("compute_step requires variables (non-empty)")
    if not references:
        raise ValueError("compute_step requires references (non-empty)")

    var_models: List[CalcVariable] = []
    for v in variables:
        if "symbol" not in v or "value" not in v or "units" not in v or "description" not in v or "source" not in v:
            raise ValueError(f"Variable missing required fields: {v}")
        var_models.append(CalcVariable(**v))

    # Compute
    unrounded = float(compute_fn())

    # Rounding
    rule = rounding_rule.get("rule", "none")
    n = int(rounding_rule.get("decimals_or_sigfigs", 6))
    rounded = apply_rounding(unrounded, rule, n)

    # Substitution generation
    sub = equation_latex
    # Replace longer symbols first to reduce partial overlap issues
    for vm in sorted(var_models, key=lambda x: len(x.symbol), reverse=True):
        sub = sub.replace(vm.symbol, _latex_value_with_units(vm.value, vm.units))

    step_checks = None
    if checks_builder is not None:
        raw_checks = checks_builder(rounded)
        step_checks = [CalcCheck(**c) for c in raw_checks]

    step = CalcStep(
        id=id,
        section=section,
        title=title,
        output_symbol=output_symbol,
        output_description=output_description,
        equation_latex=equation_latex,
        substitution_latex=sub,
        variables=var_models,
        result_unrounded=CalcValue(value=unrounded, units=units),
        rounding=Rounding(rule=rule, decimals_or_sigfigs=n),
        result_rounded=CalcValue(value=rounded, units=units),
        references=[CalcReference(**r) for r in references],
        checks=step_checks,
        warnings=warnings,
    )
    trace.steps.append(step)
    return rounded
