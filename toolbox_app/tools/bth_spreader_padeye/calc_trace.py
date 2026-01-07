from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Literal
import math

@dataclass
class TraceVar:
    symbol: str
    description: str
    value: Any
    units: str
    source: str

@dataclass
class ResultVal:
    value: Any
    units: str

@dataclass
class Rounding:
    rule: str
    decimals_or_sigfigs: int

@dataclass
class Reference:
    type: Literal["code","table","note","derived"]
    ref: str

@dataclass
class CheckItem:
    label: str
    demand: float
    capacity: float
    ratio: float
    pass_fail: str

@dataclass
class CalcStep:
    id: str
    section: str
    title: str
    output_symbol: str
    output_description: str
    equation_latex: str
    substitution_latex: str
    variables: List[TraceVar]
    result_unrounded: ResultVal
    rounding: Rounding
    result_rounded: ResultVal
    references: List[Reference]
    checks: Optional[List[CheckItem]] = None
    warnings: Optional[List[str]] = None

@dataclass
class CalcTraceMeta:
    tool_id: str
    tool_version: str
    report_version: str
    timestamp: str
    units_system: str
    code_basis: Optional[str]
    input_hash: str
    app_build: Optional[str] = None

@dataclass
class TraceInput:
    id: str
    label: str
    value: Any
    units: str
    source: str
    notes: str = ""

@dataclass
class Assumption:
    id: str
    text: str

@dataclass
class CalcTrace:
    meta: CalcTraceMeta
    inputs: List[TraceInput]
    assumptions: List[Assumption]
    steps: List[CalcStep]
    tables: Dict[str, Any]
    figures: List[Dict[str, Any]]
    summary: Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

def _apply_rounding(x: float, rule: str, n: int) -> float:
    if rule == "decimals":
        return round(x, n)
    if rule == "sigfigs":
        if x == 0: return 0.0
        p = int(math.floor(math.log10(abs(x))))
        return round(x, n - p - 1)
    raise ValueError(f"Unknown rounding rule: {rule}")

def _substitute(eqn: str, vars_: List[TraceVar]) -> str:
    sub = eqn
    for v in sorted(vars_, key=lambda a: -len(a.symbol)):
        rep = f"({_fmt(float(v.value))}\\,{v.units})" if isinstance(v.value,(int,float)) else f"({v.value}\\,{v.units})"
        sub = sub.replace(v.symbol, rep)
    return sub

def compute_step(
    trace: CalcTrace,
    id: str,
    section: str,
    title: str,
    output_symbol: str,
    output_description: str,
    equation_latex: str,
    variables: List[Dict[str, Any]],
    compute_fn: Callable[[Dict[str, Any]], float],
    units: str,
    rounding_rule: Dict[str, Any],
    references: List[Dict[str, str]],
    checks_builder: Optional[Callable[[float, float], List[Dict[str, Any]]]] = None,
    warnings: Optional[List[str]] = None,
) -> float:
    if not id or not equation_latex or not units:
        raise ValueError("compute_step requires id, equation_latex, units")

    tv = [TraceVar(**v) for v in variables]
    varmap = {v["symbol"]: v["value"] for v in variables}
    unrounded = float(compute_fn(varmap))

    rule = rounding_rule.get("rule","decimals")
    n = int(rounding_rule.get("decimals_or_sigfigs", 3))
    rounded = float(_apply_rounding(unrounded, rule, n))

    step = CalcStep(
        id=id, section=section, title=title,
        output_symbol=output_symbol, output_description=output_description,
        equation_latex=equation_latex,
        substitution_latex=_substitute(equation_latex, tv),
        variables=tv,
        result_unrounded=ResultVal(value=unrounded, units=units),
        rounding=Rounding(rule=rule, decimals_or_sigfigs=n),
        result_rounded=ResultVal(value=rounded, units=units),
        references=[Reference(type=r["type"], ref=r["ref"]) for r in references],
        checks=None, warnings=warnings
    )
    if checks_builder is not None:
        step.checks = [CheckItem(**c) for c in checks_builder(unrounded, rounded)]
    trace.steps.append(step)
    return rounded
