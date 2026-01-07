from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict

from .calc_trace import CalcTrace

def _h(s: Any) -> str:
    return html.escape(str(s))

def render_report_html(trace: CalcTrace) -> str:
    meta = trace.meta
    ts = meta.timestamp

    css = """
    @page { size: letter; margin: 0.6in; }
    body { font-family: Arial, Helvetica, sans-serif; font-size: 10.5pt; color: #111; }
    h1 { font-size: 16pt; margin: 0 0 6px 0; }
    h2 { font-size: 12.5pt; margin: 16px 0 6px 0; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
    h3 { font-size: 11pt; margin: 12px 0 4px 0; }
    .meta { font-size: 9pt; color: #333; }
    .box { border: 1px solid #999; padding: 8px; margin: 6px 0; }
    .eq { font-family: "Courier New", monospace; background: #f7f7f7; padding: 6px; white-space: pre-wrap; }
    table { border-collapse: collapse; width: 100%; margin: 6px 0 10px 0; }
    th, td { border: 1px solid #bbb; padding: 4px 6px; vertical-align: top; }
    th { background: #f1f1f1; text-align: left; }
    .pass { color: #0a6; font-weight: bold; }
    .fail { color: #b00; font-weight: bold; }
    .footer { position: fixed; bottom: 0; left: 0; right: 0; font-size: 8pt; color: #444; }
    .footer .inner { border-top: 1px solid #ccc; padding-top: 4px; }
    """

    html_parts = []
    html_parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    html_parts.append(f"<title>C49 Overhang Bracket Calcs - { _h(meta.input_hash) }</title>")
    html_parts.append(f"<style>{css}</style></head><body>")
    html_parts.append("<div class='footer'><div class='inner'>"
                      f"Tool: {_h(meta.tool_id)} v{_h(meta.tool_version)} | Input hash: {_h(meta.input_hash)} | Generated: {_h(ts)}"
                      "</div></div>")

    html_parts.append("<h1>C49 Overhang Bracket - Calculation Package</h1>")
    html_parts.append("<div class='meta'>"
                      f"<div><b>Tool ID:</b> {_h(meta.tool_id)}</div>"
                      f"<div><b>Tool Version:</b> {_h(meta.tool_version)}</div>"
                      f"<div><b>Report Version:</b> {_h(meta.report_version)}</div>"
                      f"<div><b>Timestamp:</b> {_h(ts)}</div>"
                      f"<div><b>Units System:</b> {_h(meta.units_system)}</div>"
                      f"<div><b>Input Hash:</b> {_h(meta.input_hash)}</div>"
                      "</div>")
    html_parts.append("<div class='box'>"
                      "This report includes step-by-step calculations with explicit substitutions that can be "
                      "independently reproduced to verify results."
                      "</div>")

    # Inputs
    html_parts.append("<h2>Inputs</h2>")
    html_parts.append("<table><tr><th>ID</th><th>Label</th><th>Value</th><th>Units</th><th>Source</th><th>Notes</th></tr>")
    for i in trace.inputs:
        html_parts.append(
            f"<tr><td>{_h(i.id)}</td><td>{_h(i.label)}</td><td>{_h(i.value)}</td><td>{_h(i.units)}</td><td>{_h(i.source)}</td><td>{_h(i.notes)}</td></tr>"
        )
    html_parts.append("</table>")

    # Assumptions
    html_parts.append("<h2>Assumptions & Limitations</h2>")
    if trace.assumptions:
        html_parts.append("<ul>")
        for a in trace.assumptions:
            html_parts.append(f"<li><b>{_h(a.id)}</b>: {_h(a.text)}</li>")
        html_parts.append("</ul>")
    else:
        html_parts.append("<div class='box'>None.</div>")

    # Steps
    html_parts.append("<h2>Calculations</h2>")
    for s in trace.steps:
        html_parts.append(f"<h3>{_h(s.id)} — {_h(s.title)}</h3>")
        html_parts.append("<div class='box'>")
        html_parts.append(f"<div><b>Output:</b> {_h(s.output_symbol)} — {_h(s.output_description)}</div>")
        html_parts.append("<div class='eq'><b>Equation</b>\n" + _h(s.equation_latex) + "</div>")
        html_parts.append("<div class='eq'><b>Substitution</b>\n" + _h(s.substitution_latex) + "</div>")

        # Variable table
        html_parts.append("<table><tr><th>Symbol</th><th>Description</th><th>Value</th><th>Units</th><th>Source</th></tr>")
        for v in s.variables:
            html_parts.append(
                f"<tr><td>{_h(v.symbol)}</td><td>{_h(v.description)}</td><td>{_h(v.value)}</td><td>{_h(v.units)}</td><td>{_h(v.source)}</td></tr>"
            )
        html_parts.append("</table>")

        html_parts.append(
            f"<div><b>Result (unrounded):</b> {_h(s.result_unrounded.value)} {_h(s.result_unrounded.units)}</div>"
        )
        html_parts.append(
            f"<div><b>Rounding:</b> {_h(s.rounding.rule)} ({_h(s.rounding.decimals_or_sigfigs)})</div>"
        )
        html_parts.append(
            f"<div><b>Result (used):</b> {_h(s.result_rounded.value)} {_h(s.result_rounded.units)}</div>"
        )

        if s.checks:
            html_parts.append("<div style='margin-top:6px;'><b>Checks</b></div>")
            html_parts.append("<table><tr><th>Label</th><th>Demand</th><th>Capacity</th><th>Ratio</th><th>Status</th></tr>")
            for c in s.checks:
                cls = "pass" if c.pass_fail.upper() == "PASS" else "fail"
                html_parts.append(
                    f"<tr><td>{_h(c.label)}</td><td>{_h(c.demand)}</td><td>{_h(c.capacity)}</td><td>{_h(c.ratio)}</td><td class='{cls}'>{_h(c.pass_fail)}</td></tr>"
                )
            html_parts.append("</table>")

        if s.references:
            html_parts.append("<div style='margin-top:6px;'><b>References</b>: " +
                              ", ".join([_h(f"{r.type}: {r.ref}") for r in s.references]) + "</div>")

        html_parts.append("</div>")

    # Summary
    html_parts.append("<h2>Summary</h2>")
    if trace.summary:
        html_parts.append("<div class='box'><pre class='eq'>" + _h(trace.summary) + "</pre></div>")
    else:
        html_parts.append("<div class='box'>No summary provided.</div>")

    # Mathcad Handoff
    html_parts.append("<h2>Mathcad Handoff</h2>")
    html_parts.append("<div class='box'>"
                      "<div>Outputs include:</div>"
                      "<ul>"
                      "<li><b>mathcad_inputs.csv</b>: inputs and key derived values.</li>"
                      "<li><b>mathcad_steps.json</b>: each CalcStep with equation, substitution, variables, and results.</li>"
                      "</ul>"
                      "<div>Reproduction approach: import the CSV as global variables, then replay each step using the substitution lines.</div>"
                      "</div>")

    html_parts.append("</body></html>")
    return "".join(html_parts)
