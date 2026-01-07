from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class TraceMeta:
    tool_id: str
    tool_version: str
    report_version: str
    timestamp: str
    units_system: str
    input_hash: str
    code_basis: Optional[str] = None
    app_build: Optional[str] = None


@dataclass(frozen=True)
class TraceInput:
    id: str
    label: str
    value: Any
    units: str
    source: str  # user/default/extracted
    notes: str = ""


@dataclass(frozen=True)
class Assumption:
    id: str
    text: str


@dataclass(frozen=True)
class CalcVar:
    symbol: str
    description: str
    value: Any
    units: str
    source: str


@dataclass(frozen=True)
class CalcResult:
    value: float
    units: str


@dataclass(frozen=True)
class Rounding:
    rule: str  # "decimals" | "sigfigs"
    decimals_or_sigfigs: int


@dataclass(frozen=True)
class Reference:
    type: str  # "code" | "table" | "note" | "derived"
    ref: str


@dataclass(frozen=True)
class CheckResult:
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
    variables: List[CalcVar]
    result_unrounded: CalcResult
    rounding: Rounding
    result_rounded: CalcResult
    references: List[Reference]
    checks: List[CheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CalcTrace:
    """Authoritative, fully reproducible calculation record.

    IMPORTANT: All exports (HTML/PDF/Excel/JSON/Mathcad handoff) must be rendered
    from this object. Do not write ad-hoc report logic that bypasses CalcTrace.
    """

    meta: TraceMeta
    inputs: List[TraceInput] = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    steps: List[CalcStep] = field(default_factory=list)
    tables: Dict[str, Any] = field(default_factory=dict)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        tool_id: str,
        tool_version: str,
        inputs: Dict[str, Any],
        units_system: str = "imperial",
        report_version: str = "1.0",
        input_hash: Optional[str] = None,
        code_basis: Optional[str] = None,
        app_build: Optional[str] = None,
        input_sources: Optional[Dict[str, str]] = None,
        input_labels: Optional[Dict[str, str]] = None,
        input_units: Optional[Dict[str, str]] = None,
    ) -> "CalcTrace":
        """Create a new CalcTrace with deterministic input_hash and input listing.

        input_sources can optionally provide per-input source tags (user/default/extracted).
        """

        if input_hash is None:
            input_hash = compute_input_hash(inputs)

        meta = TraceMeta(
            tool_id=str(tool_id),
            tool_version=str(tool_version),
            report_version=str(report_version),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            units_system=str(units_system),
            input_hash=str(input_hash),
            code_basis=code_basis,
            app_build=app_build,
        )

        src = input_sources or {}
        lbl = input_labels or {}
        unt = input_units or {}

        trace_inputs: List[TraceInput] = []
        for k in sorted(inputs.keys()):
            v = inputs[k]
            trace_inputs.append(
                TraceInput(
                    id=str(k),
                    label=str(lbl.get(k, _default_label(k))),
                    value=v,
                    units=str(unt.get(k, _infer_units(k))),
                    source=str(src.get(k, "user")),
                    notes="",
                )
            )

        return cls(meta=meta, inputs=trace_inputs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def trace_to_dict(trace: CalcTrace) -> Dict[str, Any]:
    """Compatibility helper used by export/render modules."""
    return trace.to_dict()


def dump_trace_json(trace: CalcTrace, path: str) -> None:
    Path(path).write_text(
        json.dumps(trace_to_dict(trace), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _default_label(key: str) -> str:
    return key.replace("_", " ")


def _infer_units(key: str) -> str:
    # light-weight heuristic, keeps reports readable without hard-coding every field
    if key.endswith("_ft"):
        return "ft"
    if key.endswith("_in"):
        return "in"
    if key.endswith("_kip"):
        return "kip"
    if key.endswith("_kipft"):
        return "kip-ft"
    if key.endswith("_pcf"):
        return "pcf"
    if key.endswith("_psf"):
        return "psf"
    if key.endswith("_deg"):
        return "deg"
    if key.endswith("_pct"):
        return "%"
    return "-"


def compute_input_hash(inputs: Dict[str, Any]) -> str:
    """Deterministic hash computed from normalized, sorted inputs."""

    norm: Dict[str, Any] = {}
    for k in sorted(inputs.keys()):
        v = inputs[k]
        if isinstance(v, float):
            norm[k] = float(f"{v:.12g}")
        else:
            norm[k] = v
    payload = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _format_value_units(value: Any, units: str) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}\\,\\mathrm{{{units}}}" if units and units != "-" else f"{value:g}"
    return f"{value}\\,\\mathrm{{{units}}}" if units and units != "-" else str(value)


def _apply_rounding(x: float, rounding: Rounding) -> float:
    if rounding.rule == "decimals":
        return round(float(x), int(rounding.decimals_or_sigfigs))
    if rounding.rule == "sigfigs":
        sig = int(rounding.decimals_or_sigfigs)
        if x == 0:
            return 0.0
        import math

        return round(float(x), sig - 1 - int(math.floor(math.log10(abs(float(x))))))
    raise ValueError(f"Unsupported rounding rule: {rounding.rule}")


def compute_step(
    trace: CalcTrace,
    *,
    id: str,
    section: str,
    title: str,
    output_symbol: str,
    output_description: str,
    equation_latex: str,
    variables: List[Dict[str, Any]],
    compute_fn: Callable[[], float],
    units: str,
    rounding_rule: Dict[str, Any],
    references: List[Dict[str, Any]],
    checks_builder: Optional[Callable[[float], List[Dict[str, Any]]]] = None,
) -> float:
    """Compute one step and append to CalcTrace, enforcing completeness."""

    if not id or not section or not title:
        raise ValueError("compute_step requires non-empty id/section/title.")

    var_objs: List[CalcVar] = []
    for v in variables:
        for req in ("symbol", "description", "value", "units", "source"):
            if req not in v:
                raise ValueError(f"Variable missing '{req}' in step {id}.")
        var_objs.append(
            CalcVar(
                symbol=str(v["symbol"]),
                description=str(v["description"]),
                value=v["value"],
                units=str(v["units"]),
                source=str(v["source"]),
            )
        )

    unrounded = float(compute_fn())
    rounding = Rounding(rule=str(rounding_rule["rule"]), decimals_or_sigfigs=int(rounding_rule["decimals_or_sigfigs"]))
    rounded = _apply_rounding(unrounded, rounding)

    # Build substitution latex by replacing each variable symbol with value+units (best-effort).
    substitution = equation_latex
    for v in var_objs:
        substitution = substitution.replace(v.symbol, _format_value_units(v.value, v.units))

    ref_objs: List[Reference] = []
    for r in references:
        if "type" not in r or "ref" not in r:
            raise ValueError(f"Reference missing type/ref in step {id}.")
        ref_objs.append(Reference(type=str(r["type"]), ref=str(r["ref"])))

    checks: List[CheckResult] = []
    if checks_builder:
        for c in checks_builder(unrounded):
            checks.append(
                CheckResult(
                    label=str(c["label"]),
                    demand=float(c["demand"]),
                    capacity=float(c["capacity"]),
                    ratio=float(c["ratio"]),
                    pass_fail=str(c["pass_fail"]),
                )
            )

    trace.steps.append(
        CalcStep(
            id=id,
            section=section,
            title=title,
            output_symbol=output_symbol,
            output_description=output_description,
            equation_latex=equation_latex,
            substitution_latex=substitution,
            variables=var_objs,
            result_unrounded=CalcResult(value=unrounded, units=units),
            rounding=rounding,
            result_rounded=CalcResult(value=rounded, units=units),
            references=ref_objs,
            checks=checks,
            warnings=[],
        )
    )

    return rounded
