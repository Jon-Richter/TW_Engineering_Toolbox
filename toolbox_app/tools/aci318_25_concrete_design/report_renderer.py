\
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Optional

from calc_trace import CalcTrace


def _h(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def render_report_html(trace: CalcTrace, results: Dict[str, Any]) -> str:
    meta = trace.meta
    title = f"{meta.tool_id} — Calculation Package"
    css = r"""
    :root{
      --fg:#111;
      --muted:#555;
      --border:#cfcfcf;
      --bg:#fff;
      --box:#f6f6f6;
      --pass:#0b6b0b;
      --fail:#b00020;
      --warn:#8a5a00;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    html,body{background:var(--bg); color:var(--fg); font-family:var(--sans); margin:0; padding:0;}
    .page{max-width:1100px; margin:24px auto; padding:0 18px 36px;}
    h1{font-size:20px; margin:0 0 6px;}
    .meta{color:var(--muted); font-size:12px; margin:0 0 18px;}
    h2{font-size:16px; margin:22px 0 10px; border-bottom:1px solid var(--border); padding-bottom:6px;}
    h3{font-size:14px; margin:16px 0 8px;}
    table{border-collapse:collapse; width:100%; font-size:12px;}
    th,td{border:1px solid var(--border); padding:6px 8px; vertical-align:top;}
    th{background:#f1f1f1; text-align:left;}
    .box{border:1px solid var(--border); background:var(--box); padding:10px 12px; margin:10px 0;}
    .eq{font-family:var(--mono); font-size:12px; white-space:pre-wrap; word-break:break-word;}
    .small{font-size:12px; color:var(--muted);}
    .step{page-break-inside:avoid; margin:0 0 16px;}
    .step-header{display:flex; gap:10px; align-items:baseline; flex-wrap:wrap;}
    .step-id{font-family:var(--mono); color:var(--muted);}
    .result{font-weight:600;}
    .tag{display:inline-block; font-size:11px; padding:2px 8px; border-radius:10px; border:1px solid var(--border); background:#fff;}
    .tag.pass{border-color:rgba(11,107,11,.35); color:var(--pass);}
    .tag.fail{border-color:rgba(176,0,32,.35); color:var(--fail);}
    .tag.warn{border-color:rgba(138,90,0,.35); color:var(--warn);}
    ul{margin:6px 0 0 18px; padding:0;}
    @media print{
      .page{max-width:none; margin:0; padding:0 10mm;}
      a{color:inherit; text-decoration:none;}
    }
    """
    # Header
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'/>")
    parts.append(f"<title>{_h(title)}</title>")
    parts.append("<style>"+css+"</style></head><body>")
    parts.append("<div class='page'>")
    parts.append(f"<h1>{_h(title)}</h1>")
    parts.append(
        "<div class='meta'>"
        f"Tool version: {_h(meta.tool_version)} | Report version: {_h(meta.report_version)} | "
        f"Timestamp: {_h(meta.timestamp)} | Units: {_h(meta.units_system)} | "
        f"Code basis: {_h(meta.code_basis or '')} | Input hash: {_h(meta.input_hash)}"
        "</div>"
    )

    # Summary
    parts.append("<h2>Summary</h2>")
    summary = trace.summary.model_dump(mode="json")
    parts.append("<div class='box'>")
    parts.append("<div class='small'>Key outputs</div>")
    parts.append("<pre class='eq'>"+_h(results.get("summary_text",""))+"</pre>")
    if summary.get("warnings"):
        parts.append("<div class='small'>Warnings</div><ul>")
        for w in summary["warnings"]:
            parts.append(f"<li class='small'>{_h(w)}</li>")
        parts.append("</ul>")
    parts.append("</div>")

    # Inputs
    parts.append("<h2>Inputs</h2>")
    parts.append("<table><thead><tr><th>ID</th><th>Label</th><th>Value</th><th>Units</th><th>Source</th><th>Notes</th></tr></thead><tbody>")
    for inp in trace.inputs:
        parts.append(
            "<tr>"
            f"<td>{_h(inp.id)}</td>"
            f"<td>{_h(inp.label)}</td>"
            f"<td>{_h(inp.value)}</td>"
            f"<td>{_h(inp.units)}</td>"
            f"<td>{_h(inp.source)}</td>"
            f"<td>{_h(inp.notes or '')}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    # Assumptions
    parts.append("<h2>Assumptions &amp; Limitations</h2>")
    if trace.assumptions:
        parts.append("<ul>")
        for a in trace.assumptions:
            parts.append(f"<li>{_h(a.id)} — {_h(a.text)}</li>")
        parts.append("</ul>")
    else:
        parts.append("<div class='small'>None recorded.</div>")

    # Steps
    parts.append("<h2>Calculations</h2>")
    for st in trace.steps:
        parts.append("<div class='step'>")
        parts.append("<div class='step-header'>")
        parts.append(f"<div class='step-id'>{_h(st.id)}</div>")
        parts.append(f"<div><strong>{_h(st.section)}</strong> — {_h(st.title)}</div>")
        parts.append("</div>")
        parts.append("<div class='box'>")
        parts.append(f"<div class='small'>Output</div><div class='result'>{_h(st.output_symbol)} — {_h(st.output_description)} = {_h(st.result_rounded.value)} {_h(st.result_rounded.units)}</div>")
        parts.append("<hr style='border:none;border-top:1px solid var(--border); margin:10px 0;'/>")
        parts.append("<div class='small'>Symbolic equation</div>")
        parts.append(f"<pre class='eq'>{_h(st.equation_latex)}</pre>")
        parts.append("<div class='small'>Numeric substitution</div>")
        parts.append(f"<pre class='eq'>{_h(st.substitution_latex)}</pre>")
        parts.append("<div class='small'>Variables</div>")
        parts.append("<table><thead><tr><th>Symbol</th><th>Description</th><th>Value</th><th>Units</th><th>Source</th></tr></thead><tbody>")
        for v in st.variables:
            parts.append(
                "<tr>"
                f"<td>{_h(v.symbol)}</td>"
                f"<td>{_h(v.description)}</td>"
                f"<td>{_h(v.value)}</td>"
                f"<td>{_h(v.units)}</td>"
                f"<td>{_h(v.source)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
        parts.append("<div class='small'>Result</div>")
        parts.append(
            "<div class='eq'>"
            f"Unrounded: {_h(st.result_unrounded.value)} {_h(st.result_unrounded.units)}<br/>"
            f"Rounding: {_h(st.rounding.rule)} ({_h(st.rounding.decimals_or_sigfigs)})<br/>"
            f"Rounded used downstream: {_h(st.result_rounded.value)} {_h(st.result_rounded.units)}"
            "</div>"
        )
        parts.append("<div class='small'>References</div><ul>")
        for r in st.references:
            parts.append(f"<li class='small'>{_h(r.type)} — {_h(r.ref)}</li>")
        parts.append("</ul>")

        # Checks
        if st.checks:
            parts.append("<div class='small'>Checks</div>")
            parts.append("<table><thead><tr><th>Label</th><th>Demand</th><th>Capacity</th><th>Ratio</th><th>Status</th></tr></thead><tbody>")
            for c in st.checks:
                tag_class = "pass" if c.pass_fail.upper() == "PASS" else "fail"
                parts.append(
                    "<tr>"
                    f"<td>{_h(c.label)}</td>"
                    f"<td>{_h(c.demand)}</td>"
                    f"<td>{_h(c.capacity)}</td>"
                    f"<td>{_h(c.ratio)}</td>"
                    f"<td><span class='tag {tag_class}'>{_h(c.pass_fail)}</span></td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")

        # Warnings
        if st.warnings:
            parts.append("<div class='small'>Warnings</div><ul>")
            for w in st.warnings:
                parts.append(f"<li class='small'><span class='tag warn'>WARN</span> {_h(w)}</li>")
            parts.append("</ul>")

        parts.append("</div></div>")  # box, step

    # Mathcad handoff
    parts.append("<h2>Mathcad Handoff</h2>")
    parts.append("<div class='box'>")
    parts.append("<div class='small'>Reproducibility</div>")
    parts.append("<div class='small'>The exported <strong>mathcad_inputs.csv</strong> contains the user inputs plus key derived values. "
                 "The exported <strong>mathcad_steps.json</strong> contains an ordered list of all CalcSteps including equations, substitutions, and references. "
                 "A Mathcad worksheet can be reconstructed by importing the CSV for variables and then recreating each step in order using the JSON file.</div>")
    parts.append("</div>")

    # Footer
    parts.append("<div class='meta'>")
    parts.append(f"Generated by { _h(meta.tool_id) } v{ _h(meta.tool_version) } | Input hash: { _h(meta.input_hash) }")
    parts.append("</div>")

    parts.append("</div></body></html>")
    return "".join(parts)
