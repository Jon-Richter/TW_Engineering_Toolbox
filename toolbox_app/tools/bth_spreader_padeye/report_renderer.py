from __future__ import annotations
import html, json
from typing import Any, Dict

def _e(x: Any) -> str:
    return html.escape(str(x))

def render_report_html(trace: Dict[str, Any]) -> str:
    meta = trace["meta"]
    css = """    body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}
    h1{margin:0 0 6px 0}
    .sub{font-size:12px;color:#444}
    .card{border:1px solid #bbb;border-radius:8px;padding:12px;margin:12px 0}
    .title{font-size:16px;font-weight:700;margin:0 0 8px 0}
    table{border-collapse:collapse;width:100%;font-size:12px}
    th,td{border:1px solid #ccc;padding:6px;vertical-align:top}
    th{background:#f3f3f3;text-align:left}
    .eq{font-family:Consolas,monospace;background:#fafafa;border:1px solid #ddd;padding:8px;border-radius:6px;white-space:pre-wrap}
    .mono{font-family:Consolas,monospace}
    .pf-pass{color:#0a6}
    .pf-fail{color:#c00}
    """

    def inputs_tbl():
        rows=[]
        for i in trace["inputs"]:
            rows.append(f"<tr><td class='mono'>{_e(i['id'])}</td><td>{_e(i['label'])}</td><td>{_e(i['value'])}</td><td>{_e(i['units'])}</td><td>{_e(i['source'])}</td></tr>")
        return "<table><tr><th>ID</th><th>Label</th><th>Value</th><th>Units</th><th>Source</th></tr>"+"".join(rows)+"</table>"

    def assump():
        a=trace.get("assumptions",[])
        if not a:
            return "<div class='sub'>None.</div>"
        return "<ul>"+"".join([f"<li>{_e(x['id'])}: {_e(x['text'])}</li>" for x in a])+"</ul>"

    def step(s):
        vars_rows="".join([f"<tr><td class='mono'>{_e(v['symbol'])}</td><td>{_e(v['description'])}</td><td>{_e(v['value'])}</td><td>{_e(v['units'])}</td><td>{_e(v['source'])}</td></tr>" for v in s['variables']])
        chk=""
        if s.get('checks'):
            chk_rows="".join([f"<tr><td>{_e(c['label'])}</td><td>{_e(c['demand'])}</td><td>{_e(c['capacity'])}</td><td>{_e(c['ratio'])}</td><td class='{ 'pf-pass' if c['pass_fail']=='PASS' else 'pf-fail' }'>{_e(c['pass_fail'])}</td></tr>" for c in s['checks']])
            chk=f"<div class='card'><div class='title'>Check</div><table><tr><th>Label</th><th>Demand</th><th>Capacity</th><th>Ratio</th><th>Status</th></tr>{chk_rows}</table></div>"
        rounded_val = s['result_rounded']['value']
        try:
            rounded_val = f"{float(rounded_val):.2f}"
        except (ValueError, TypeError):
            pass
        return f"""        <div class='card'>
          <div class='title'>{_e(s['id'])} — {_e(s['section'])}: {_e(s['title'])}</div>
          <div class='sub' style='margin-top:8px'><b>1) Equation</b></div>
          <div class='eq'>{_e(s['equation_latex'])}</div>
          <div class='sub' style='margin-top:8px'><b>2) Variables</b></div>
          <table><tr><th>Symbol</th><th>Description</th><th>Value</th><th>Units</th><th>Source</th></tr>{vars_rows}</table>
          <div class='sub' style='margin-top:8px'><b>3) Result</b></div>
          <div class='eq'><b>{_e(rounded_val)} {_e(s['result_rounded']['units'])}</b></div>
        </div>
        {chk}
        """

    steps_html="".join([step(s) for s in trace['steps']])
    summary_html=f"<pre class='eq'>{_e(json.dumps(trace.get('summary',{}), indent=2))}</pre>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Calc Package — {meta['tool_id']}</title><style>{css}</style></head>
<body>
<h1>Calc Package</h1>
<div class='sub'>{_e(meta['tool_id'])} v{_e(meta['tool_version'])} — Report v{_e(meta['report_version'])}</div>
<div class='sub'>Timestamp: {_e(meta['timestamp'])} — Input hash: <span class='mono'>{_e(meta['input_hash'])}</span></div>
<div class='sub'>Units: {_e(meta['units_system'])} — Code basis: {_e(meta.get('code_basis') or '')}</div>

<div class='card'><div class='title'>Inputs</div>{inputs_tbl()}</div>
<div class='card'><div class='title'>Assumptions</div>{assump()}</div>
<div class='card'><div class='title'>Summary</div>{summary_html}</div>
<div class='card'><div class='title'>Calculation Steps</div>{steps_html}</div>
</body></html>"""
