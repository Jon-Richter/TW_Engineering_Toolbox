import React, { useEffect, useMemo, useState } from "react";

function Field({ label, units, value, onChange, type = "number", step = "any", disabled = false }) {
  return (
    <div className="row">
      <div>{label}</div>
      <input
        value={value}
        type={type}
        step={step}
        onChange={(ev) => onChange(ev.target.value)}
        disabled={disabled}
      />
      <div className="unit">{units || ""}</div>
    </div>
  );
}

// lightweight spinner CSS inlined to avoid modifying separate stylesheet
const spinnerStyle = `
.spinner { display: inline-block; padding: 6px 10px; background:#1f6feb; color:white; border-radius:4px; }
.muted { color: #666; }
`;

function Select({ label, value, onChange, options }) {
  return (
    <div className="row">
      <div>{label}</div>
      <select value={value} onChange={(ev) => onChange(ev.target.value)}>
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      <div />
    </div>
  );
}

function getInitialMode() {
  if (typeof window === "undefined") return "padeye";
  const params = new URLSearchParams(window.location.search);
  const raw = (params.get("mode") || "").toLowerCase();
  return raw === "spreader" ? "spreader" : "padeye";
}

function formatWll(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value ?? "");
  const fixed = n.toFixed(3);
  return fixed.replace(/\.?0+$/, "");
}

export default function App() {
  useEffect(() => {
    // inject inline spinner styles for quick visual affordance
    const el = document.createElement("style");
    el.setAttribute("data-inline-spinner", "1");
    el.textContent = spinnerStyle;
    document.head.appendChild(el);
    return () => {
      document.head.removeChild(el);
    };
  }, []);
  const apiBase = import.meta.env.VITE_API_BASE || "";
  const [mode, setMode] = useState(getInitialMode());
  const [designCategory, setDesignCategory] = useState("C");
  const [Fy, setFy] = useState(50.0);
  const [Fu, setFu] = useState(65.0);
  const [impact, setImpact] = useState(1.0);
  const designFactorByCategory = { A: 2.0, B: 3.0, C: 6.0 };
  const designFactor = designFactorByCategory[designCategory] ?? 6.0;

  const [pad, setPad] = useState({
    P: 50.0,
    theta_deg: 0.0,
    beta_deg: 90.0,
    H: 40.0,
    h: 38.0,
    w: 5.0,
    Wb: 8.0,
    t: 1.0,
    Dh: 1.4,
    Dp: 1.5,
    R: 2.0,
    tcheek: 0.0,
    ex: 0.0,
    ey: 0.0
  });

  const [shackles, setShackles] = useState([]);
  const [shackleId, setShackleId] = useState("custom");

  const [spr, setSpr] = useState({
    shape: "W10X49",
    span_L_ft: 20.0,
    Lb_ft: 20.0,
    KL_ft: 20.0,
    Cb: 1.0,
    V_kip: 0.0,
    P_kip: 0.0,
    Mx_app_kipft: 0.0,
    My_app_kipft: 0.0,
    include_self_weight: true,
    Cmx: 1.0,
    Cmy: 1.0,
    braced_against_twist: true,
    weld_check: false,
    weld_size_in: 0.25,
    weld_length_in: 0.0,
    weld_exx_ksi: 70.0
  });

  const [busyByMode, setBusyByMode] = useState({ padeye: false, spreader: false });
  const [resultsByMode, setResultsByMode] = useState({ padeye: null, spreader: null });
  const [errorByMode, setErrorByMode] = useState({ padeye: null, spreader: null });
  const [noteByMode, setNoteByMode] = useState({ padeye: null, spreader: null });

  function withBase(url) {
    if (!apiBase) return url;
    return url.startsWith("/") ? `${apiBase}${url}` : url;
  }

  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/api/shackles`)
      .then((res) => res.json())
      .then((json) => {
        if (!active) return;
        if (json.ok && Array.isArray(json.items)) {
          setShackles(json.items);
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [apiBase]);

  const shackleMap = useMemo(() => {
    const out = new Map();
    shackles.forEach((item) => {
      if (item && item.id) out.set(item.id, item);
    });
    return out;
  }, [shackles]);

  const selectedShackle = shackleId !== "custom" ? shackleMap.get(shackleId) : null;

  useEffect(() => {
    if (!selectedShackle) return;
    const e = Number(selectedShackle.eccentricity_in || 0);
    const theta = Number(pad.theta_deg || 0);
    const ex = e * Math.cos((theta * Math.PI) / 180.0);
    const ey = e * Math.sin((theta * Math.PI) / 180.0);
    const pin = Number(selectedShackle.pin_diameter_in || pad.Dp);
    setPad((prev) => ({
      ...prev,
      ex: Number(ex.toFixed(6)),
      ey: Number(ey.toFixed(6)),
      Dp: pin
    }));
  }, [selectedShackle, pad.theta_deg]);

  async function runSolve(targetMode, providedSignal) {
    const activeMode = targetMode || mode;
    setBusyByMode((prev) => ({ ...prev, [activeMode]: true }));
    setErrorByMode((prev) => ({ ...prev, [activeMode]: null }));
    setNoteByMode((prev) => ({ ...prev, [activeMode]: null }));

    let controller;
    let timeoutId;
    let signal = providedSignal;

    if (!providedSignal) {
      controller = new AbortController();
      signal = controller.signal;
      timeoutId = setTimeout(() => controller.abort(), 30000);
    }

    try {
      const base = {
        mode: activeMode,
        units_system: "US",
        design_category: designCategory,
        Nd: Number(designFactor),
        Fy: Number(Fy),
        Fu: Number(Fu),
        impact_factor: Number(impact)
      };
      const payload = activeMode === "padeye" ? { ...base, ...pad } : { ...base, ...spr };
      const response = await fetch(`${apiBase}/api/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal
      });
      const json = await response.json();
      if (!json.ok) throw new Error(json.error || "Solve failed");
      setResultsByMode((prev) => ({ ...prev, [activeMode]: json.results }));
    } catch (err) {
      const message =
        err && err.name === "AbortError"
          ? "Solve timed out after 30s. Check backend logs for errors."
          : String((err && err.message) || err || "Unknown error");
      setErrorByMode((prev) => ({ ...prev, [activeMode]: message }));
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      setBusyByMode((prev) => ({ ...prev, [activeMode]: false }));
    }
  }

  // Auto-run when inputs change (debounced)
  useEffect(() => {
    const payloadKey = JSON.stringify({ mode, designCategory, Fy, Fu, impact, pad, spr, shackleId });
    const debounceMs = 400;

    function inputsAreValid(targetMode) {
      if (typeof Fy === "undefined" || typeof Fu === "undefined") return false;
      if (!Number.isFinite(Number(Fy)) || !Number.isFinite(Number(Fu))) return false;
      if (!designCategory) return false;
      if (targetMode === "padeye") {
        const required = [pad.P, pad.h, pad.t, pad.Dp, pad.Dh, pad.Wb, pad.H];
        for (const v of required) {
          if (!Number.isFinite(Number(v))) return false;
        }
      } else {
        if (!spr || !spr.shape) return false;
        const required = [spr.span_L_ft, spr.Lb_ft, spr.KL_ft];
        for (const v of required) {
          if (!Number.isFinite(Number(v))) return false;
        }
      }
      return true;
    }

    const valid = inputsAreValid(mode);
    if (!valid) {
      setNoteByMode((prev) => ({ ...prev, [mode]: "Waiting for valid inputs" }));
      return;
    }
    // clear any prior note
    setNoteByMode((prev) => ({ ...prev, [mode]: null }));

    const id = setTimeout(() => {
      // fire-and-forget; runSolve handles its own busy state
      runSolve(mode);
    }, debounceMs);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify({ mode, designCategory, Fy, Fu, impact, pad, spr, shackleId })]);

  async function optimizeTheta() {
    const activeMode = "padeye";
    setBusyByMode((prev) => ({ ...prev, [activeMode]: true }));
    setErrorByMode((prev) => ({ ...prev, [activeMode]: null }));
    setNoteByMode((prev) => ({ ...prev, [activeMode]: null }));
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);
    try {
      const payload = {
        mode: "padeye",
        units_system: "US",
        design_category: designCategory,
        Nd: Number(designFactor),
        Fy: Number(Fy),
        Fu: Number(Fu),
        impact_factor: Number(impact),
        ...pad
      };
      const response = await fetch(`${apiBase}/api/optimize_theta`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      const json = await response.json();
      if (!json.ok) throw new Error(json.error || "Theta sweep failed");
      if (json.best && typeof json.best.theta_deg === "number") {
        setPad((prev) => ({ ...prev, theta_deg: json.best.theta_deg }));
      }
      if (json.results) {
        setResultsByMode((prev) => ({ ...prev, padeye: json.results }));
      }
      if (json.best) {
        setNoteByMode((prev) => ({
          ...prev,
          padeye: `Worst-case theta = ${json.best.theta_deg} deg (utilization ${Number(
            json.best.governing_ratio
          ).toFixed(3)})`
        }));
      }
    } catch (err) {
      const message =
        err.name === "AbortError"
          ? "Theta sweep timed out after 60s."
          : String(err.message || err);
      setErrorByMode((prev) => ({ ...prev, [activeMode]: message }));
    } finally {
      clearTimeout(timeoutId);
      setBusyByMode((prev) => ({ ...prev, [activeMode]: false }));
    }
  }

  async function optimizeSection() {
    const activeMode = "spreader";
    setBusyByMode((prev) => ({ ...prev, [activeMode]: true }));
    setErrorByMode((prev) => ({ ...prev, [activeMode]: null }));
    setNoteByMode((prev) => ({ ...prev, [activeMode]: null }));
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);
    try {
      const payload = {
        mode: "spreader",
        units_system: "US",
        design_category: designCategory,
        Nd: Number(designFactor),
        Fy: Number(Fy),
        Fu: Number(Fu),
        impact_factor: Number(impact),
        ...spr
      };
      const response = await fetch(`${apiBase}/api/optimize_shape`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      const json = await response.json();
      if (!json.ok) throw new Error(json.error || "Section optimization failed");
      if (json.best && json.best.shape) {
        setSpr((prev) => ({ ...prev, shape: json.best.shape }));
      }
      if (json.results) {
        setResultsByMode((prev) => ({ ...prev, spreader: json.results }));
      }
      if (json.best) {
        setNoteByMode((prev) => ({
          ...prev,
          spreader: `Optimal shape = ${json.best.shape} (${Number(json.best.weight_lbft).toFixed(
            2
          )} lb/ft)`
        }));
      }
    } catch (err) {
      const message =
        err.name === "AbortError"
          ? "Section optimization timed out after 120s."
          : String(err.message || err);
      setErrorByMode((prev) => ({ ...prev, [activeMode]: message }));
    } finally {
      clearTimeout(timeoutId);
      setBusyByMode((prev) => ({ ...prev, [activeMode]: false }));
    }
  }

  const results = resultsByMode[mode];
  const error = errorByMode[mode];
  const note = noteByMode[mode];
  const busy = busyByMode[mode];
  const reportUrl = results ? `${apiBase}/api/report.html?mode=${mode}` : "";
  const artifacts = results && results.artifacts ? results.artifacts : {};
  const checks = results && Array.isArray(results.checks) ? results.checks : [];
  const hasChecks = checks.length > 0;
  const outputLabelMap = {
    Px: "Load Component Px",
    Py: "Load Component Py",
    Pz: "Load Component Pz",
    sigma_eq: "Equivalent Stress",
    F_allow: "Allowable Stress",
    U_base: "Base Utilization",
    governing_ratio: "Governing Utilization",
    governing_check: "Governing Limit State",
    M_sw: "Self-Weight Moment",
    Mx_total: "Total Strong-Axis Moment",
    fbx: "Strong-Axis Bending Stress",
    fby: "Weak-Axis Bending Stress",
    fa: "Axial Stress",
    "U_3-29": "Eq. (3-29) Interaction",
    "U_3-31": "Eq. (3-31) Interaction",
    U_combined: "Governing Interaction",
    U_shear: "Shear Utilization",
    governing_step: "Calc Step ID"
  };
  const outputOrder = {
    padeye: [
      "Px",
      "Py",
      "Pz",
      "sigma_eq",
      "F_allow",
      "U_base",
      "governing_ratio",
      "governing_check"
    ],
    spreader: [
      "M_sw",
      "Mx_total",
      "fbx",
      "fby",
      "fa",
      "U_3-29",
      "U_3-31",
      "U_combined",
      "U_shear",
      "governing_ratio",
      "governing_check"
    ]
  };
  const hiddenOutputs = new Set(["governing_step"]);

  const orderedOutputs = (() => {
    if (!results || !results.key_outputs) return [];
    const outputs = results.key_outputs || {};
    const keys = outputOrder[mode] || Object.keys(outputs);
    const seen = new Set();
    const list = [];
    keys.forEach((key) => {
      if (hiddenOutputs.has(key)) return;
      if (outputs[key] !== undefined) {
        list.push([key, outputs[key]]);
        seen.add(key);
      }
    });
    Object.keys(outputs).forEach((key) => {
      if (hiddenOutputs.has(key) || seen.has(key)) return;
      list.push([key, outputs[key]]);
    });
    return list;
  })();

  const shackleOptions = [
    { value: "custom", label: "Custom (manual)" },
    ...shackles.map((item) => ({
      value: item.id,
      label: `${item.type} - ${formatWll(item.wll_t)} t`
    }))
  ];
  const shackleDisabled = Boolean(selectedShackle);
  const shackleBow = selectedShackle ? selectedShackle.bow_diameter_in : "";
  const shacklePin = selectedShackle ? selectedShackle.pin_diameter_in : "";
  const shackleE = selectedShackle ? selectedShackle.eccentricity_in : "";

  return (
    <div className="container">
      <header className="masthead">
        <div>
          <div className="eyebrow">ASME BTH-1-2023</div>
          <h1>Spreader Bar + Padeye</h1>
          <div className="sub">
            Professional BTH-1 checks with auto-generated calc package exports (HTML/Excel/JSON).
          </div>
        </div>
        <div className="masthead-tags">
          <div className="tag">Mode: {mode === "padeye" ? "Padeye" : "Spreader bar"}</div>
          <div className="tag">Design category: {designCategory}</div>
          <div className="tag">Design factor Nd: {designFactor.toFixed(1)}</div>
        </div>
      </header>

      <div className="grid">
        <div className="card">
          <div className="card-title">Inputs</div>
          <div className="tabs">
            <button
              className={`tab${mode === "padeye" ? " active" : ""}`}
              onClick={() => setMode("padeye")}
              type="button"
            >
              Padeye
            </button>
            <button
              className={`tab${mode === "spreader" ? " active" : ""}`}
              onClick={() => setMode("spreader")}
              type="button"
            >
              Spreader bar (rolled)
            </button>
          </div>

          <Select
            label="Design category"
            value={designCategory}
            onChange={(value) => setDesignCategory(value)}
            options={[
              { value: "A", label: "A" },
              { value: "B", label: "B" },
              { value: "C", label: "C" }
            ]}
          />
          <Field
            label="Design factor Nd (auto)"
            units="-"
            value={designFactor}
            onChange={() => {}}
            disabled
          />
          <Field label="Fy" units="ksi" value={Fy} onChange={(v) => setFy(Number(v))} />
          <Field label="Fu" units="ksi" value={Fu} onChange={(v) => setFu(Number(v))} />
          <Field
            label="Impact factor"
            units="-"
            value={impact}
            onChange={(v) => setImpact(Number(v))}
          />

          <div className="hr" />

          {mode === "padeye" ? (
            <div>
              <div className="section-title">Padeye loads</div>
              <Field
                label="Applied resultant load P"
                units="kip"
                value={pad.P}
                onChange={(v) => setPad({ ...pad, P: Number(v) })}
                disabled={busy}
              />
              <Field
                label="In-plane angle theta"
                units="deg"
                value={pad.theta_deg}
                onChange={(v) => setPad({ ...pad, theta_deg: Number(v) })}
                disabled={busy}
              />
              <Field
                label="Out-of-plane angle beta"
                units="deg"
                value={pad.beta_deg}
                onChange={(v) => setPad({ ...pad, beta_deg: Number(v) })}
              />
              <div className="section-title">Geometry</div>
              <Field
                label="Padeye height H"
                units="in"
                value={pad.H}
                onChange={(v) => setPad({ ...pad, H: Number(v) })}
              />
              <Field
                label="Hole height h"
                units="in"
                value={pad.h}
                onChange={(v) => setPad({ ...pad, h: Number(v) })}
              />
              <Field
                label="Width at hole w"
                units="in"
                value={pad.w}
                onChange={(v) => setPad({ ...pad, w: Number(v) })}
              />
              <Field
                label="Width at base Wb"
                units="in"
                value={pad.Wb}
                onChange={(v) => setPad({ ...pad, Wb: Number(v) })}
              />
              <Field
                label="Plate thickness t"
                units="in"
                value={pad.t}
                onChange={(v) => setPad({ ...pad, t: Number(v) })}
              />
              <Field
                label="Hole diameter Dh"
                units="in"
                value={pad.Dh}
                onChange={(v) => setPad({ ...pad, Dh: Number(v) })}
              />
              <Field
                label="Pin diameter Dp"
                units="in"
                value={shackleDisabled ? shacklePin : pad.Dp}
                onChange={(v) => setPad({ ...pad, Dp: Number(v) })}
                disabled={shackleDisabled}
              />
              <Field
                label="Hole center to top edge R"
                units="in"
                value={pad.R}
                onChange={(v) => setPad({ ...pad, R: Number(v) })}
              />
              <div className="section-title">Shackle selection</div>
              <Select
                label="Shackle"
                value={shackleId}
                onChange={(value) => setShackleId(value)}
                options={shackleOptions}
              />
              <Field
                label="Shackle bow diameter"
                units="in"
                value={shackleBow}
                onChange={() => {}}
                disabled
              />
              <Field
                label="Shackle pin diameter"
                units="in"
                value={shacklePin}
                onChange={() => {}}
                disabled
              />
              <Field
                label="Shackle eccentricity e"
                units="in"
                value={shackleE}
                onChange={() => {}}
                disabled
              />
              <div className="section-title">Boss / cheek plates and eccentricities</div>
              <Field
                label="Boss/cheek thickness tcheek"
                units="in"
                value={pad.tcheek}
                onChange={(v) => setPad({ ...pad, tcheek: Number(v) })}
              />
              <Field
                label="Eccentricity ex (torsion arm)"
                units="in"
                value={pad.ex}
                onChange={(v) => setPad({ ...pad, ex: Number(v) })}
                disabled={shackleDisabled}
              />
              <Field
                label="Eccentricity ey (moment arm add)"
                units="in"
                value={pad.ey}
                onChange={(v) => setPad({ ...pad, ey: Number(v) })}
                disabled={shackleDisabled}
              />
              <button className="btn btn-secondary" disabled={busy} onClick={optimizeTheta} type="button">
                {busy ? "Running..." : "Find Worst-Case Theta"}
              </button>
            </div>
          ) : (
            <div>
              <div className="section-title">Spreader inputs</div>
              <Field
                label="Shape (AISC label)"
                units=""
                type="text"
                value={spr.shape}
                onChange={(v) => setSpr({ ...spr, shape: v })}
              />
              <Field
                label="Span L"
                units="ft"
                value={spr.span_L_ft}
                onChange={(v) => setSpr({ ...spr, span_L_ft: Number(v) })}
              />
              <Field
                label="Unbraced length Lb"
                units="ft"
                value={spr.Lb_ft}
                onChange={(v) => setSpr({ ...spr, Lb_ft: Number(v) })}
              />
              <Field
                label="Effective length KL"
                units="ft"
                value={spr.KL_ft}
                onChange={(v) => setSpr({ ...spr, KL_ft: Number(v) })}
              />
              <Field label="Cb" units="-" value={spr.Cb} onChange={(v) => setSpr({ ...spr, Cb: Number(v) })} />
              <Field
                label="Cmx (eq. 3-29)"
                units="-"
                value={spr.Cmx}
                onChange={(v) => setSpr({ ...spr, Cmx: Number(v) })}
              />
              <Field
                label="Cmy (eq. 3-29)"
                units="-"
                value={spr.Cmy}
                onChange={(v) => setSpr({ ...spr, Cmy: Number(v) })}
              />
              <Select
                label="Braced against twist at Lb ends"
                value={String(spr.braced_against_twist)}
                onChange={(v) => setSpr({ ...spr, braced_against_twist: v === "true" })}
                options={[
                  { value: "true", label: "Yes" },
                  { value: "false", label: "No" }
                ]}
              />
              <Select
                label="Include weld sizing check"
                value={String(spr.weld_check)}
                onChange={(v) => setSpr({ ...spr, weld_check: v === "true" })}
                options={[
                  { value: "false", label: "No" },
                  { value: "true", label: "Yes" }
                ]}
              />
              <Field
                label="Fillet weld size"
                units="in"
                value={spr.weld_size_in}
                onChange={(v) => setSpr({ ...spr, weld_size_in: Number(v) })}
              />
              <Field
                label="Total effective weld length"
                units="in"
                value={spr.weld_length_in}
                onChange={(v) => setSpr({ ...spr, weld_length_in: Number(v) })}
              />
              <Field
                label="Weld metal strength Exx"
                units="ksi"
                value={spr.weld_exx_ksi}
                onChange={(v) => setSpr({ ...spr, weld_exx_ksi: Number(v) })}
              />
              <Field
                label="Shear V"
                units="kip"
                value={spr.V_kip}
                onChange={(v) => setSpr({ ...spr, V_kip: Number(v) })}
              />
              <Field
                label="Axial compression P"
                units="kip"
                value={spr.P_kip}
                onChange={(v) => setSpr({ ...spr, P_kip: Number(v) })}
              />
              <Field
                label="Applied strong-axis moment Mx"
                units="kip-ft"
                value={spr.Mx_app_kipft}
                onChange={(v) => setSpr({ ...spr, Mx_app_kipft: Number(v) })}
              />
              <Field
                label="Applied weak-axis moment My"
                units="kip-ft"
                value={spr.My_app_kipft}
                onChange={(v) => setSpr({ ...spr, My_app_kipft: Number(v) })}
              />
              <Select
                label="Include self-weight moment"
                value={String(spr.include_self_weight)}
                onChange={(v) => setSpr({ ...spr, include_self_weight: v === "true" })}
                options={[
                  { value: "true", label: "Yes" },
                  { value: "false", label: "No" }
                ]}
              />
              <button className="btn btn-secondary" disabled={busy} onClick={optimizeSection} type="button">
                {busy ? "Running..." : "Optimize Section (Min Weight)"}
              </button>
            </div>
          )}

          {/* Run now handled by auto-run; show busy indicator instead */}
          <div style={{ marginTop: 12 }}>
            {busy ? <span className="spinner">Running…</span> : <span className="muted">Auto-run</span>}
          </div>
          {error ? <div className="bad">{error}</div> : null}
          {note ? <div className="good">{note}</div> : null}
        </div>

        <div>
          <div className="card" style={{ marginBottom: "16px" }}>
            <div className="card-title">Results</div>
            {!results ? (
              <div className="sub">Run analysis to generate results and a calc package.</div>
            ) : (
              <div>
                <div className="kvgrid">
                  {orderedOutputs.map(([key, value]) => (
                    <div className="kv" key={key}>
                      <div className="k">{outputLabelMap[key] || key.replace(/_/g, " ")}</div>
                      <div className="v">
                        {typeof value === "object" ? `${value.value} ${value.units || ""}` : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="links">
                  <a className="link" href={reportUrl} target="_blank" rel="noreferrer">
                    View Calc Package
                  </a>
                  <a className="link" href={reportUrl} target="_blank" rel="noreferrer">
                    Print / Save PDF
                  </a>
                  {Object.entries(artifacts).map(([name, url]) => (
                    <a className="link" href={withBase(url)} key={name}>
                      {name}
                    </a>
                  ))}
                </div>
                {hasChecks ? (
                  <div className="limit-state">
                    <div className="section-title">Limit State Utilizations</div>
                    <div className="checks">
                      {checks.map((check, idx) => (
                        <div className="check-row" key={`${check.step_id}-${check.label}-${idx}`}>
                          <div className="check-label">{check.label}</div>
                          <div className="check-meta">{check.step_id}</div>
                          <div className={`check-ratio ${check.pass_fail === "PASS" ? "ok" : "bad"}`}>
                            {Number(check.ratio).toFixed(3)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </div>
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div className="card-title card-title--tight">
              Calc Package Viewer
            </div>
            {results ? (
              <iframe title="report" src={reportUrl} />
            ) : (
              <div className="placeholder">
                Run analysis to generate the calc package for this mode.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
