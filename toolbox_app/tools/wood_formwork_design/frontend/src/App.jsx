import React, { useEffect, useRef, useState } from "react";
import { ToolShell } from "@toolbox/ui-core";

const DIMLUMBER_DB = {
  "2x4": { b: 1.5, d: 3.5 },
  "2x6": { b: 1.5, d: 5.5 },
  "2x8": { b: 1.5, d: 7.25 },
  "2x10": { b: 1.5, d: 9.25 },
  "2x12": { b: 1.5, d: 11.25 },
  "4x4": { b: 3.5, d: 3.5 },
  "4x6": { b: 3.5, d: 5.5 },
  "4x8": { b: 3.5, d: 7.25 }
};

const formatNumber = (value, decimals = 2) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(decimals).replace(/\.?0+$/, "");
};

const formatNumberFixed = (value, decimals = 2) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(decimals);
};

const formatInt = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return Math.round(n).toString();
};

function NumberInput({ value, onChange, step = "any", disabled = false }) {
  const [draft, setDraft] = useState(value === null || value === undefined ? "" : String(value));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (focused) return;
    if (value === null || value === undefined || value === "") {
      setDraft("");
      return;
    }
    setDraft(String(value));
  }, [value, focused]);

  return (
    <input
      value={draft}
      type="number"
      step={step}
      onChange={(ev) => {
        const next = ev.target.value;
        setDraft(next);
        onChange(next === "" ? null : Number(next));
      }}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      disabled={disabled}
    />
  );
}

function Field({ label, units, value, onChange, type = "number", step = "any", disabled = false }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="row">
        {type === "number" ? (
          <NumberInput value={value} onChange={onChange} step={step} disabled={disabled} />
        ) : (
          <input
            type={type}
            value={value ?? ""}
            onChange={(ev) => onChange(ev.target.value)}
            disabled={disabled}
          />
        )}
        <div className="units">{units || ""}</div>
      </div>
    </div>
  );
}

function SelectField({ label, value, onChange, options, disabled = false }) {
  const safeOptions = options && options.length ? options : [{ label: "-", value: "" }];
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value} onChange={(ev) => onChange(ev.target.value)} disabled={disabled}>
        {safeOptions.map((opt) => (
          <option key={opt.value ?? opt} value={opt.value ?? opt}>
            {opt.label ?? opt}
          </option>
        ))}
      </select>
    </div>
  );
}

function CheckboxField({ label, checked, onChange }) {
  return (
    <label className="chip">
      <input type="checkbox" checked={checked} onChange={(ev) => onChange(ev.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function memberCenters(length, spacing, thickness) {
  const L = Math.max(length, thickness);
  const s = Math.max(spacing, 1);
  const first = thickness / 2;
  const last = L - thickness / 2;
  if (last <= first) return [L / 2];
  const count = Math.floor((last - first) / s) + 1;
  const centers = Array.from({ length: count }, (_, i) => first + i * s);
  if (last - centers[centers.length - 1] > 1e-6) {
    centers.push(last);
  }
  return centers;
}

function tiePositions(length, spacing) {
  const L = Math.max(length, 1);
  const s = Math.max(spacing, 1);
  if (L <= s) return [L / 2];
  const positions = [];
  let pos = s;
  while (pos <= L + 1e-6) {
    positions.push(pos);
    pos += s;
  }
  return positions;
}

function FormworkDiagram({
  studSize,
  walerSize,
  studSpacingIn,
  walerSpacingIn,
  tieSpacingIn,
  studOrientation,
  formHeightFt
}) {
  const stud = DIMLUMBER_DB[studSize] || { b: 1.5, d: 3.5 };
  const waler = DIMLUMBER_DB[walerSize] || { b: 1.5, d: 5.5 };

  const studSpacing = Math.max(Number(studSpacingIn) || 16, 4);
  const walerSpacing = Math.max(Number(walerSpacingIn) || 24, 4);
  const tieSpacing = Math.max(Number(tieSpacingIn) || 24, 4);
  const heightIn = Math.max((Number(formHeightFt) || 12) * 12, 12);

  const verticalStuds = studOrientation !== "horizontal";
  const studThk = stud.b;
  const walerThk = waler.b;

  const pad = 24;
  const labelPad = 40;
  const bayCount = 6;

  let width = 0;
  let studPositions = [];
  let walerPositions = [];
  let tieXPositions = [];
  let tieYPositions = [];

  if (verticalStuds) {
    width = bayCount * studSpacing + studThk;
    studPositions = memberCenters(width, studSpacing, studThk);
    walerPositions = memberCenters(heightIn, walerSpacing, walerThk);
    tieXPositions = tiePositions(width - studThk / 2, tieSpacing);
    tieYPositions = walerPositions;
  } else {
    width = bayCount * walerSpacing + walerThk;
    walerPositions = memberCenters(width, walerSpacing, walerThk);
    studPositions = memberCenters(heightIn, studSpacing, studThk);
    tieXPositions = walerPositions;
    tieYPositions = tiePositions(heightIn - studThk / 2, tieSpacing);
  }

  const viewBox = [
    -pad,
    -pad - 8,
    width + pad * 2 + 60,
    heightIn + labelPad + pad * 2
  ].join(" ");

  const tieR = Math.max(2, Math.min(studThk, walerThk) * 0.25);
  const dimColor = "#374151";
  const textColor = "#111827";

  return (
    <svg className="diagram" viewBox={viewBox} role="img">
      <defs>
        <marker id="arrowDim" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill={dimColor} />
        </marker>
      </defs>
      <rect x="0" y="0" width={width} height={heightIn} fill="none" stroke="#334155" strokeWidth="2" />
      {verticalStuds
        ? studPositions.map((x, idx) => (
            <rect
              key={`stud-${idx}`}
              x={x - studThk / 2}
              y={0}
              width={studThk}
              height={heightIn}
              fill="rgba(51,65,85,0.18)"
              stroke="#334155"
              strokeWidth="1"
            />
          ))
        : studPositions.map((y, idx) => (
            <rect
              key={`stud-${idx}`}
              x={0}
              y={y - studThk / 2}
              width={width}
              height={studThk}
              fill="rgba(51,65,85,0.18)"
              stroke="#334155"
              strokeWidth="1"
            />
          ))}
      {verticalStuds
        ? walerPositions.map((y, idx) => (
            <rect
              key={`waler-${idx}`}
              x={0}
              y={y - walerThk / 2}
              width={width}
              height={walerThk}
              fill="rgba(15,118,110,0.18)"
              stroke="#0f766e"
              strokeWidth="1"
            />
          ))
        : walerPositions.map((x, idx) => (
            <rect
              key={`waler-${idx}`}
              x={x - walerThk / 2}
              y={0}
              width={walerThk}
              height={heightIn}
              fill="rgba(15,118,110,0.18)"
              stroke="#0f766e"
              strokeWidth="1"
            />
          ))}
      {tieXPositions.flatMap((x, xi) =>
        tieYPositions.map((y, yi) => (
          <circle
            key={`tie-${xi}-${yi}`}
            cx={x}
            cy={y}
            r={tieR}
            fill="rgba(249,115,22,0.8)"
            stroke="#c2410c"
          />
        ))
      )}

      {verticalStuds && studPositions.length > 1 ? (
        <>
          <line
            x1={studPositions[0]}
            y1={-12}
            x2={studPositions[1]}
            y2={-12}
            stroke={dimColor}
            strokeWidth="1.2"
            markerStart="url(#arrowDim)"
            markerEnd="url(#arrowDim)"
          />
          <text x={(studPositions[0] + studPositions[1]) / 2} y={-18} textAnchor="middle" fill={textColor}>
            {`Stud spacing ${formatInt(studSpacing)} in`}
          </text>
        </>
      ) : null}

      {verticalStuds && walerPositions.length > 1 ? (
        <>
          <line
            x1={width + 20}
            y1={walerPositions[0]}
            x2={width + 20}
            y2={walerPositions[1]}
            stroke={dimColor}
            strokeWidth="1.2"
            markerStart="url(#arrowDim)"
            markerEnd="url(#arrowDim)"
          />
          <text x={width + 24} y={(walerPositions[0] + walerPositions[1]) / 2} fill={textColor}>
            {`Waler spacing ${formatInt(walerSpacing)} in`}
          </text>
        </>
      ) : null}

      {!verticalStuds && walerPositions.length > 1 ? (
        <>
          <line
            x1={walerPositions[0]}
            y1={-12}
            x2={walerPositions[1]}
            y2={-12}
            stroke={dimColor}
            strokeWidth="1.2"
            markerStart="url(#arrowDim)"
            markerEnd="url(#arrowDim)"
          />
          <text x={(walerPositions[0] + walerPositions[1]) / 2} y={-18} textAnchor="middle" fill={textColor}>
            {`Waler spacing ${formatInt(walerSpacing)} in`}
          </text>
        </>
      ) : null}

      {!verticalStuds && studPositions.length > 1 ? (
        <>
          <line
            x1={width + 20}
            y1={studPositions[0]}
            x2={width + 20}
            y2={studPositions[1]}
            stroke={dimColor}
            strokeWidth="1.2"
            markerStart="url(#arrowDim)"
            markerEnd="url(#arrowDim)"
          />
          <text x={width + 24} y={(studPositions[0] + studPositions[1]) / 2} fill={textColor}>
            {`Stud spacing ${formatInt(studSpacing)} in`}
          </text>
        </>
      ) : null}

      <line
        x1={-18}
        y1={0}
        x2={-18}
        y2={heightIn}
        stroke={dimColor}
        strokeWidth="1.2"
        markerStart="url(#arrowDim)"
        markerEnd="url(#arrowDim)"
      />
      <text x={-24} y={heightIn / 2} textAnchor="end" fill={textColor}>
        {`Form height ${formatNumber(heightIn / 12, 1)} ft`}
      </text>

      <text x={0} y={heightIn + labelPad} fill={textColor}>
        {`Stud size ${studSize} | Waler size ${walerSize}`}
      </text>
    </svg>
  );
}

function PressureDiagram({ depths, pressures }) {
  const svgW = 520;
  const svgH = 340;
  const pad = 48;
  const xs = Array.isArray(pressures) ? pressures.map((v) => Number(v)) : [];
  const ys = Array.isArray(depths) ? depths.map((v) => Number(v)) : [];
  const pairs = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!pairs.length) {
    return <div className="muted">No pressure profile data available.</div>;
  }
  const maxX = Math.max(...xs, 1);
  const maxY = Math.max(...ys, 1);
  const toX = (x) => pad + (x / maxX) * (svgW - 2 * pad);
  const toY = (y) => pad + (y / maxY) * (svgH - 2 * pad);

  const line = pairs.map(([x, y]) => `${toX(x)},${toY(y)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="diagram" role="img">
      <line x1={pad} y1={pad} x2={pad} y2={svgH - pad} stroke="#94a3b8" strokeWidth="1.2" />
      <line x1={pad} y1={pad} x2={svgW - pad} y2={pad} stroke="#94a3b8" strokeWidth="1.2" />
      <polyline points={line} fill="none" stroke="#b45309" strokeWidth="2.2" />
      <text x={pad} y={svgH - 18} fill="#475569">
        Depth (ft)
      </text>
      <text x={svgW - pad} y={pad - 18} textAnchor="end" fill="#475569">
        Pressure (psf)
      </text>
    </svg>
  );
}

function NdsSummary({ summary, note }) {
  if (!summary || !summary.enabled) {
    return <div className="note">{note || "NDS lookup disabled."}</div>;
  }
  const factors = summary.factors || {};
  return (
    <div className="stack">
      <div className="card">
        <div className="card-title">NDS Adjustments</div>
        <div className="kv-grid">
          <div className="kv">
            <div className="k">CD source</div>
            <div className="v">{summary.cd_source}</div>
          </div>
          <div className="kv">
            <div className="k">CM / Ct</div>
            <div className="v">
              {formatNumber(factors.CM, 2)} / {formatNumber(factors.Ct, 2)}
            </div>
          </div>
          <div className="kv">
            <div className="k">Cf (stud)</div>
            <div className="v">{formatNumber(factors.Cf_stud, 2)}</div>
          </div>
          <div className="kv">
            <div className="k">Cf (waler)</div>
            <div className="v">{formatNumber(factors.Cf_waler, 2)}</div>
          </div>
        </div>
      </div>

      {[summary.stud, summary.waler].map((item) => (
        <div className="card" key={item.size}>
          <div className="card-title">{item === summary.stud ? "Stud values" : "Waler values"}</div>
          <div className="note">Size: {item.size}</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Property</th>
                  <th>Base</th>
                  <th>Adjusted</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Fb (psi)</td>
                  <td className="mono">{formatInt(item.base.Fb)}</td>
                  <td className="mono">{formatInt(item.adjusted.Fb)}</td>
                </tr>
                <tr>
                  <td>Fv (psi)</td>
                  <td className="mono">{formatInt(item.base.Fv)}</td>
                  <td className="mono">{formatInt(item.adjusted.Fv)}</td>
                </tr>
                <tr>
                  <td>E (psi)</td>
                  <td className="mono">{formatInt(item.base.E)}</td>
                  <td className="mono">{formatInt(item.adjusted.E)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [inputs, setInputs] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState("inputs");
  const solveId = useRef(0);

  useEffect(() => {
    let mounted = true;
    fetch("/api/config")
      .then((res) => res.json())
      .then((data) => {
        if (!mounted) return;
        setConfig(data);
        setInputs(data.defaults || {});
      })
      .catch((err) => {
        if (!mounted) return;
        setError(String(err));
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!config || !inputs) return;
    const tieSizes = config.options?.tie_sizes || {};
    if (inputs.tie_type === "taper_tie" || inputs.tie_type === "coil_rod") {
      const opts = tieSizes[inputs.tie_type] || [];
      const next = opts.length ? opts[0].value : null;
      if (inputs.tie_size !== next) {
        setInputs((prev) => ({ ...prev, tie_size: next }));
      }
    } else if (inputs.tie_size) {
      setInputs((prev) => ({ ...prev, tie_size: null }));
    }
  }, [inputs?.tie_type, config]);

  useEffect(() => {
    if (!config || !inputs) return;
    const grades = config.options?.nds?.grades_by_species?.[inputs.nds_species] || [];
    if (!grades.length) return;
    const next = grades.includes(inputs.nds_grade) ? inputs.nds_grade : grades[0];
    if (next !== inputs.nds_grade) {
      setInputs((prev) => ({ ...prev, nds_grade: next }));
    }
  }, [inputs?.nds_species, config]);

  useEffect(() => {
    if (!inputs) return;
    const id = ++solveId.current;
    const timer = setTimeout(() => {
      setBusy(true);
      setError(null);
      fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs)
      })
        .then((res) => res.json())
        .then((data) => {
          if (id !== solveId.current) return;
          if (!data.ok) {
            setError(data.error || "Solve failed");
            setBusy(false);
            return;
          }
          setResults(data);
          setBusy(false);
        })
        .catch((err) => {
          if (id !== solveId.current) return;
          setError(String(err));
          setBusy(false);
        });
    }, 300);
    return () => clearTimeout(timer);
  }, [inputs]);

  const options = config?.options || {};
  const displayedInputs = results?.inputs || inputs || {};
  const pressureSummary = results?.pressure_summary;
  const utilTable = results?.util_table || [];
  const segmentTable = results?.segment_table || [];
  const profile = results?.pressure_profile || {};
  const downloads = results?.download || {};
  const tieCapacity = results?.tie_capacity;

  const orientationLabel =
    inputs?.stud_orientation === "horizontal"
      ? { stud: "Stud spacing (vertical, in)", waler: "Waler spacing (horizontal, in)" }
      : { stud: "Stud spacing (horizontal, in)", waler: "Waler spacing (vertical, in)" };

  const tabs = [
    { id: "inputs", label: "Inputs & Summary" },
    { id: "pressure", label: "Pressure Profile" },
    { id: "segments", label: "Segment Checks" },
    { id: "report", label: "Report Export" }
  ];

  const allowablesDisabled = !!inputs?.use_nds && !!results?.nds_summary;

  return (
    <ToolShell
      title="Wood Formwork Design"
      eyebrow="Concrete | Formwork"
      description="ACI 347R-14 lateral pressure with NDS-adjusted member checks for plywood, studs, walers, and ties."
      tags={[{ label: "ACI 347R-14" }, { label: "NDS 2018" }, { label: "ASD checks" }]}
    >
      <div className="tabs">
        {tabs.map((t) => (
          <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {error ? <div className="note">Error: {error}</div> : null}
      <div className="muted">{busy ? "Running calculations..." : "Auto-run enabled"}</div>

      {tab === "inputs" ? (
        <div className="layout-grid">
          <div className="stack">
            <div className="card">
              <div className="card-title">1. Lateral Pressure (ACI 347R-14)</div>
              <div className="input-grid">
                <SelectField
                  label="Element type"
                  value={inputs?.element_type || "wall"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, element_type: v }))}
                  options={options.element_types || []}
                />
                <Field
                  label="Form height"
                  units="ft"
                  value={inputs?.form_height_ft}
                  onChange={(v) => setInputs((prev) => ({ ...prev, form_height_ft: v }))}
                />
                <Field
                  label="Segment size"
                  units="ft"
                  value={inputs?.segment_ft}
                  onChange={(v) => setInputs((prev) => ({ ...prev, segment_ft: v }))}
                  step="0.25"
                />
                <Field
                  label="Unit weight"
                  units="pcf"
                  value={inputs?.unit_weight_pcf}
                  onChange={(v) => setInputs((prev) => ({ ...prev, unit_weight_pcf: v }))}
                />
                <Field
                  label="Concrete temperature"
                  units="F"
                  value={inputs?.temp_F}
                  onChange={(v) => setInputs((prev) => ({ ...prev, temp_F: v }))}
                />
                <Field
                  label="Placement rate"
                  units="ft/hr"
                  value={inputs?.rate_ftph}
                  onChange={(v) => setInputs((prev) => ({ ...prev, rate_ftph: v }))}
                />
                <Field
                  label="Slump"
                  units="in"
                  value={inputs?.slump_in}
                  onChange={(v) => setInputs((prev) => ({ ...prev, slump_in: v }))}
                />
                <Field
                  label="Vibration depth"
                  units="ft"
                  value={inputs?.vib_depth_ft}
                  onChange={(v) => setInputs((prev) => ({ ...prev, vib_depth_ft: v }))}
                />
                <SelectField
                  label="Mix category"
                  value={inputs?.mix_category || "normal"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, mix_category: v }))}
                  options={options.mix_categories || []}
                />
              </div>
              <div className="section-title">Special conditions</div>
              <div className="chip-row">
                {(options.special_conditions || []).map((opt) => (
                  <CheckboxField
                    key={opt.value}
                    label={opt.label}
                    checked={(inputs?.special_conditions || []).includes(opt.value)}
                    onChange={(checked) => {
                      const set = new Set(inputs?.special_conditions || []);
                      if (checked) {
                        set.add(opt.value);
                      } else {
                        set.delete(opt.value);
                      }
                      setInputs((prev) => ({ ...prev, special_conditions: Array.from(set) }));
                    }}
                  />
                ))}
              </div>
              {pressureSummary ? (
                <div className="kv-grid" style={{ marginTop: 12 }}>
                  <div className="kv">
                    <div className="k">p_cap</div>
                    <div className="v">{formatInt(pressureSummary.p_cap_psf)} psf</div>
                  </div>
                  <div className="kv">
                    <div className="k">Hydrostatic</div>
                    <div className="v">{formatInt(pressureSummary.p_hydro_psf)} psf</div>
                  </div>
                  <div className="kv">
                    <div className="k">Cc / Cw</div>
                    <div className="v">
                      {formatNumberFixed(pressureSummary.Cc, 2)} / {formatNumberFixed(pressureSummary.Cw, 2)}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Controlling</div>
                    <div className="v">{pressureSummary.case}</div>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="card">
              <div className="card-title">2. NDS Presets (Studs & Walers)</div>
              <div className="input-grid">
                <div className="field">
                  <label>Use NDS presets</label>
                  <div className="row">
                    <input
                      type="checkbox"
                      checked={!!inputs?.use_nds}
                      onChange={(ev) => setInputs((prev) => ({ ...prev, use_nds: ev.target.checked }))}
                    />
                    <div className="units" />
                  </div>
                </div>
                <SelectField
                  label="Species"
                  value={inputs?.nds_species || ""}
                  onChange={(v) => setInputs((prev) => ({ ...prev, nds_species: v }))}
                  options={(options.nds?.species || []).map((v) => ({ label: v, value: v }))}
                  disabled={!inputs?.use_nds}
                />
                <SelectField
                  label="Grade"
                  value={inputs?.nds_grade || ""}
                  onChange={(v) => setInputs((prev) => ({ ...prev, nds_grade: v }))}
                  options={(options.nds?.grades_by_species?.[inputs?.nds_species] || []).map((v) => ({
                    label: v,
                    value: v
                  }))}
                  disabled={!inputs?.use_nds}
                />
                <SelectField
                  label="CD (duration factor)"
                  value={inputs?.nds_cd_preset || 1.25}
                  onChange={(v) => setInputs((prev) => ({ ...prev, nds_cd_preset: Number(v) }))}
                  options={(options.nds?.cd_presets || []).map((v) => ({ label: v, value: v }))}
                  disabled={!inputs?.use_nds}
                />
                <Field
                  label="Custom CD (override)"
                  units=""
                  value={inputs?.nds_cd_custom}
                  onChange={(v) => setInputs((prev) => ({ ...prev, nds_cd_custom: v }))}
                  disabled={!inputs?.use_nds}
                />
                <SelectField
                  label="Moisture > 19%"
                  value={inputs?.nds_moisture || "no"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, nds_moisture: v }))}
                  options={(options.nds?.yes_no || []).map((v) => ({ label: v === "yes" ? "Yes" : "No", value: v }))}
                  disabled={!inputs?.use_nds}
                />
                <SelectField
                  label="Temperature > 100F"
                  value={inputs?.nds_temp || "no"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, nds_temp: v }))}
                  options={(options.nds?.yes_no || []).map((v) => ({ label: v === "yes" ? "Yes" : "No", value: v }))}
                  disabled={!inputs?.use_nds}
                />
              </div>
              <div className="note">Custom CD overrides the preset if provided.</div>
            </div>

            <div className="card">
              <div className="card-title">3. Geometry & Allowables</div>
              <div className="input-grid">
                <SelectField
                  label="Plywood thickness"
                  value={inputs?.ply_thk || 0.75}
                  onChange={(v) => setInputs((prev) => ({ ...prev, ply_thk: Number(v) }))}
                  options={(options.plywood_thickness || []).map((v) => ({ label: v, value: v }))}
                />
                <SelectField
                  label="Stud size"
                  value={inputs?.stud_size || "2x4"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, stud_size: v }))}
                  options={(options.dimlumber_sizes || []).map((v) => ({ label: v, value: v }))}
                />
                <SelectField
                  label="Stud orientation"
                  value={inputs?.stud_orientation || "vertical"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, stud_orientation: v }))}
                  options={options.stud_orientations || []}
                />
                <Field
                  label={orientationLabel.stud}
                  units="in"
                  value={inputs?.stud_spacing_in}
                  onChange={(v) => setInputs((prev) => ({ ...prev, stud_spacing_in: v }))}
                />
                <SelectField
                  label="Double waler size"
                  value={inputs?.waler_size || "2x6"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, waler_size: v }))}
                  options={(options.dimlumber_sizes || []).map((v) => ({ label: v, value: v }))}
                />
                <Field
                  label={orientationLabel.waler}
                  units="in"
                  value={inputs?.waler_spacing_in}
                  onChange={(v) => setInputs((prev) => ({ ...prev, waler_spacing_in: v }))}
                />
                <Field
                  label="Tie spacing along waler"
                  units="in"
                  value={inputs?.tie_spacing_in}
                  onChange={(v) => setInputs((prev) => ({ ...prev, tie_spacing_in: v }))}
                />
                <SelectField
                  label="Tie type"
                  value={inputs?.tie_type || "snap_tie"}
                  onChange={(v) => setInputs((prev) => ({ ...prev, tie_type: v }))}
                  options={options.tie_types || []}
                />
                <SelectField
                  label="Tie size (if applicable)"
                  value={inputs?.tie_size || ""}
                  onChange={(v) => setInputs((prev) => ({ ...prev, tie_size: v }))}
                  options={(options.tie_sizes?.[inputs?.tie_type] || []).map((opt) => ({
                    label: opt.label,
                    value: opt.value
                  }))}
                  disabled={!(inputs?.tie_type === "taper_tie" || inputs?.tie_type === "coil_rod")}
                />
                <Field
                  label="Tie SWL override"
                  units="lb"
                  value={inputs?.tie_swl_override}
                  onChange={(v) => setInputs((prev) => ({ ...prev, tie_swl_override: v }))}
                />
                <Field
                  label="Deflection limit plywood"
                  units="L/."
                  value={inputs?.defl_ply}
                  onChange={(v) => setInputs((prev) => ({ ...prev, defl_ply: v }))}
                />
                <Field
                  label="Deflection limit studs"
                  units="L/."
                  value={inputs?.defl_stud}
                  onChange={(v) => setInputs((prev) => ({ ...prev, defl_stud: v }))}
                />
                <Field
                  label="Deflection limit walers"
                  units="L/."
                  value={inputs?.defl_waler}
                  onChange={(v) => setInputs((prev) => ({ ...prev, defl_waler: v }))}
                />
              </div>

              <div className="section-title">Plywood allowables</div>
              <div className="input-grid three">
                <Field
                  label="Plywood Fb"
                  units="psi"
                  value={inputs?.ply_Fb}
                  onChange={(v) => setInputs((prev) => ({ ...prev, ply_Fb: v }))}
                />
                <Field
                  label="Plywood Fv"
                  units="psi"
                  value={inputs?.ply_Fv}
                  onChange={(v) => setInputs((prev) => ({ ...prev, ply_Fv: v }))}
                />
                <Field
                  label="Plywood E"
                  units="psi"
                  value={inputs?.ply_E}
                  onChange={(v) => setInputs((prev) => ({ ...prev, ply_E: v }))}
                />
              </div>

              <div className="section-title">Stud allowables</div>
              <div className="input-grid three">
                <Field
                  label="Stud Fb"
                  units="psi"
                  value={allowablesDisabled ? displayedInputs?.stud_Fb : inputs?.stud_Fb}
                  onChange={(v) => setInputs((prev) => ({ ...prev, stud_Fb: v }))}
                  disabled={allowablesDisabled}
                />
                <Field
                  label="Stud Fv"
                  units="psi"
                  value={allowablesDisabled ? displayedInputs?.stud_Fv : inputs?.stud_Fv}
                  onChange={(v) => setInputs((prev) => ({ ...prev, stud_Fv: v }))}
                  disabled={allowablesDisabled}
                />
                <Field
                  label="Stud E"
                  units="psi"
                  value={allowablesDisabled ? displayedInputs?.stud_E : inputs?.stud_E}
                  onChange={(v) => setInputs((prev) => ({ ...prev, stud_E: v }))}
                  disabled={allowablesDisabled}
                />
              </div>

              <div className="section-title">Double waler allowables</div>
              <div className="input-grid three">
                <Field
                  label="Waler Fb"
                  units="psi"
                  value={allowablesDisabled ? displayedInputs?.waler_Fb : inputs?.waler_Fb}
                  onChange={(v) => setInputs((prev) => ({ ...prev, waler_Fb: v }))}
                  disabled={allowablesDisabled}
                />
                <Field
                  label="Waler Fv"
                  units="psi"
                  value={allowablesDisabled ? displayedInputs?.waler_Fv : inputs?.waler_Fv}
                  onChange={(v) => setInputs((prev) => ({ ...prev, waler_Fv: v }))}
                  disabled={allowablesDisabled}
                />
                <Field
                  label="Waler E"
                  units="psi"
                  value={allowablesDisabled ? displayedInputs?.waler_E : inputs?.waler_E}
                  onChange={(v) => setInputs((prev) => ({ ...prev, waler_E: v }))}
                  disabled={allowablesDisabled}
                />
              </div>

              {tieCapacity ? (
                <div className="kv" style={{ marginTop: 12 }}>
                  <div className="k">Tie capacity (SWL)</div>
                  <div className="v">
                    {formatInt(tieCapacity.swl)} lb ({tieCapacity.note})
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          <div className="stack">
            <div className="card">
              <div className="card-title">4. Formwork Diagram</div>
              <div className="note">Indicative geometry and spacing.</div>
              <FormworkDiagram
                studSize={inputs?.stud_size}
                walerSize={inputs?.waler_size}
                studSpacingIn={inputs?.stud_spacing_in}
                walerSpacingIn={inputs?.waler_spacing_in}
                tieSpacingIn={inputs?.tie_spacing_in}
                studOrientation={inputs?.stud_orientation}
                formHeightFt={inputs?.form_height_ft}
              />
            </div>

            <div className="card">
              <div className="card-title">5. NDS Summary</div>
              <NdsSummary summary={results?.nds_summary} note={results?.nds_note} />
            </div>

            <div className="card">
              <div className="card-title">6. Uniform Utilization (p = p_cap)</div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Member</th>
                      <th>Check</th>
                      <th>Demand</th>
                      <th>Capacity</th>
                      <th>Util</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {utilTable.map((row, idx) => {
                      const util = Number(row.utilization);
                      const status = Number.isFinite(util) && util <= 1 ? "OK" : "NG";
                      return (
                        <tr key={`${row.member}-${idx}`}>
                          <td>{row.member}</td>
                          <td>{row.check}</td>
                          <td>
                            {formatNumber(row.demand, 2)} {row.units}
                          </td>
                          <td>
                            {formatNumber(row.capacity, 2)} {row.units}
                          </td>
                          <td className="mono">{formatNumber(row.utilization, 2)}</td>
                          <td className={`status ${status === "NG" ? "bad" : ""}`}>{status}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {tab === "pressure" ? (
        <div className="stack">
          <div className="card">
            <div className="card-title">Pressure Profile</div>
            <PressureDiagram depths={profile.depths_ft} pressures={profile.pressures_psf} />
            <div className="note">
              Profile assumed: p(z) = min(w * z, p_cap). Depth is measured from the top of placement.
            </div>
          </div>
        </div>
      ) : null}

      {tab === "segments" ? (
        <div className="card">
          <div className="card-title">Segment Checks</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Seg</th>
                  <th>Depth Top</th>
                  <th>Depth Bot</th>
                  <th>p_mid</th>
                  <th>Ply</th>
                  <th>Stud</th>
                  <th>Waler</th>
                  <th>Tie int</th>
                  <th>Tie edge</th>
                  <th>Tie corner</th>
                  <th>Control</th>
                  <th>Util</th>
                </tr>
              </thead>
              <tbody>
                {segmentTable.map((row) => (
                  <tr key={`seg-${row.Segment}`}>
                    <td>{row.Segment}</td>
                    <td>{formatNumber(row.DepthTop_ft, 2)} ft</td>
                    <td>{formatNumber(row.DepthBot_ft, 2)} ft</td>
                    <td>{formatNumber(row.p_mid_psf, 0)} psf</td>
                    <td>{formatNumber(row.ply_util, 2)}</td>
                    <td>{formatNumber(row.stud_util, 2)}</td>
                    <td>{formatNumber(row.waler_util, 2)}</td>
                    <td>{formatNumber(row.tie_util_interior, 2)}</td>
                    <td>{formatNumber(row.tie_util_edge, 2)}</td>
                    <td>{formatNumber(row.tie_util_corner, 2)}</td>
                    <td>{row.controlling_member}</td>
                    <td className="mono">{formatNumber(row.controlling_util, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {tab === "report" ? (
        <div className="card">
          <div className="card-title">Report Export</div>
          <div className="actions">
            <button
              className="btn"
              onClick={() => window.open(downloads.segment_csv || "/api/download/segment_checks.csv", "_blank")}
            >
              Download CSV (segment checks)
            </button>
            <button
              className="btn secondary"
              onClick={() => window.open(downloads.report_pdf || "/api/download/report.pdf", "_blank")}
            >
              Download PDF summary
            </button>
          </div>
          <div className="note">
            CSV includes the full per-segment table. PDF includes inputs, pressure summary, and governing utilizations.
          </div>
        </div>
      ) : null}
    </ToolShell>
  );
}
