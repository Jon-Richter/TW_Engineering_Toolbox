import React, { useMemo, useState } from "react";
import { ToolShell } from "@toolbox/ui-core";

function num(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : 0;
}

function isNonEmpty(v) {
  return v !== null && v !== undefined && v !== "";
}

function openNew(url) {
  window.open(url, "_blank", "noopener,noreferrer");
}

function FieldNumber({ value, onChange, label, units }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="row">
        <input
          type="number"
          step="any"
          value={value}
          onChange={(ev) => onChange(num(ev.target.value))}
        />
        <div className="units">{units || ""}</div>
      </div>
    </div>
  );
}

function FieldText({ value, onChange, label }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="row">
        <input type="text" value={value} onChange={(ev) => onChange(ev.target.value)} />
      </div>
    </div>
  );
}

function FieldSelect({ value, onChange, label, units, options }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="row">
        <select value={value} onChange={(ev) => onChange(ev.target.value)}>
          {options.map((opt) => (
            <option value={opt.value} key={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <div className="units">{units || ""}</div>
      </div>
    </div>
  );
}

function FieldBool({ value, onChange, label }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="row">
        <input type="checkbox" checked={!!value} onChange={(ev) => onChange(!!ev.target.checked)} />
        <div className="units" />
      </div>
    </div>
  );
}

function ReinforcementEditor({ groupState, setGroupState, groupName }) {
  const layers = groupState.reinf_layers || [];

  const defaultLayer = useMemo(() => {
    if (groupName === "beams") {
      return {
        face: "bottom",
        offset_in: num(groupState.cover_in) + num(groupState.stirrup_dia_in) + num(groupState.bar_dia_in) / 2,
        n_bars: 3,
        bar_dia_in: num(groupState.bar_dia_in),
        As_override_in2: null
      };
    }
    // slabs
    return {
      face: "bottom",
      offset_in: num(groupState.cover_in) + num(groupState.bar_dia_in) / 2,
      n_bars: 3,
      bar_dia_in: num(groupState.bar_dia_in),
      As_override_in2: null
    };
  }, [groupName, groupState.bar_dia_in, groupState.cover_in, groupState.stirrup_dia_in]);

  function updateLayer(i, key, value) {
    const next = { ...groupState };
    const nextLayers = (next.reinf_layers || []).slice();
    const layer = { ...(nextLayers[i] || {}) };
    layer[key] = value;
    nextLayers[i] = layer;
    next.reinf_layers = nextLayers;
    setGroupState(next);
  }

  function addLayer() {
    const next = { ...groupState };
    const nextLayers = (next.reinf_layers || []).slice();
    nextLayers.push(defaultLayer);
    next.reinf_layers = nextLayers;
    setGroupState(next);
  }

  function removeLayer(i) {
    const next = { ...groupState };
    const nextLayers = (next.reinf_layers || []).slice();
    nextLayers.splice(i, 1);
    next.reinf_layers = nextLayers;
    setGroupState(next);
  }

  return (
    <div className="panel">
      <div style={{ fontWeight: 600, marginBottom: "6px" }}>
        Reinforcement layers (tension + compression allowed)
      </div>
      <div className="sub" style={{ marginBottom: "8px" }}>
        Option 1 (default): enter n bars and bar diameter. Option 2: check Override As and enter As for
        the layer. Offset is to bar centroid from selected face.
      </div>

      <table>
        <thead>
          <tr>
            <th>Layer</th>
            <th>Face</th>
            <th>Offset to centroid (in)</th>
            <th>n bars</th>
            <th>Bar dia (in)</th>
            <th>Override As</th>
            <th>As override (in²)</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {layers.map((ly, i) => {
            const override = isNonEmpty(ly.As_override_in2);
            return (
              <tr key={`${groupName}_ly_${i}`}>
                <td>{String(i + 1)}</td>
                <td>
                  <select
                    value={ly.face || "bottom"}
                    onChange={(ev) => updateLayer(i, "face", ev.target.value)}
                  >
                    <option value="bottom">bottom</option>
                    <option value="top">top</option>
                  </select>
                </td>
                <td>
                  <input
                    type="number"
                    step="0.01"
                    value={ly.offset_in}
                    onChange={(ev) => updateLayer(i, "offset_in", num(ev.target.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="1"
                    value={ly.n_bars}
                    disabled={override}
                    onChange={(ev) => updateLayer(i, "n_bars", num(ev.target.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.001"
                    value={ly.bar_dia_in}
                    disabled={override}
                    onChange={(ev) => updateLayer(i, "bar_dia_in", num(ev.target.value))}
                  />
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={override}
                    onChange={(ev) => {
                      const checked = !!ev.target.checked;
                      updateLayer(i, "As_override_in2", checked ? (ly.As_override_in2 || 0.01) : null);
                    }}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.0001"
                    value={override ? ly.As_override_in2 : ""}
                    disabled={!override}
                    onChange={(ev) => updateLayer(i, "As_override_in2", num(ev.target.value))}
                  />
                </td>
                <td>
                  <button className="secondary" onClick={() => removeLayer(i)}>
                    Remove
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="actions">
        <button className="secondary" onClick={addLayer}>
          Add layer
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("beams");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [resp, setResp] = useState(null);

  // Defaults are intentionally identical to the existing built UI.
  const [beams, setBeams] = useState({
    b_in: 12,
    h_in: 24,
    cover_in: 1.5,
    stirrup_dia_in: 0.375,
    bar_dia_in: 1.0,
    Mu_kipft: 180,
    fc_psi: 4000,
    fy_psi: 60000,
    Es_psi: 29000000,
    compression_reinf: false,
    transverse_type: "other",
    reinf_layers: [{ face: "bottom", offset_in: 2.375, n_bars: 3, bar_dia_in: 1.0, As_override_in2: null }]
  });

  const [slabs, setSlabs] = useState({
    thickness_in: 8,
    cover_in: 0.75,
    bar_dia_in: 0.5,
    Mu_kipft_per_ft: 12,
    fc_psi: 4000,
    fy_psi: 60000,
    Es_psi: 29000000,
    reinf_layers: [{ face: "bottom", offset_in: 1.0, n_bars: 3, bar_dia_in: 0.5, As_override_in2: null }]
  });

  const [columns, setColumns] = useState({
    shape: "rectangular",
    b_in: 16,
    h_in: 16,
    D_in: 18,
    Ast_in2: 4.0,
    Pu_kip: 500,
    fc_psi: 5000,
    fy_psi: 60000,
    transverse_type: "ties"
  });

  const [dev, setDev] = useState({
    calc_type: "tension_development",
    bar_size: "#5",
    db_in: 0.625,
    fc_psi: 4000,
    fy_psi: 60000,
    lambda_factor: 1.0,
    is_top_bar: false,
    is_epoxy: false,
    epoxy_cover_lt_3db_or_spacing_lt_6db: false,
    fy_gt_60000: false,
    As_provided_over_As_required: 1.0,
    percent_bars_spliced: 100.0,
    cover_ge_db_and_spacing_ge_2db: true
  });

  const [walls, setWalls] = useState({
    lw_in: 144,
    t_in: 10,
    lu_in: 144,
    cover_in: 1.5,
    bar_dia_in: 0.625,
    rho_v: 0.0025,
    Pu_kip: 200,
    M_top_oop_kipft: 20,
    M_bot_oop_kipft: -20,
    M_top_ip_kipft: 10,
    M_bot_ip_kipft: -10,
    end_bottom: "fixed",
    end_top: "fixed",
    fc_psi: 4000,
    fy_psi: 60000,
    Es_psi: 29000000,
    transverse_type: "other",
    cracked_section: true,
    beta_dns: 0.6,
    member_nonsway: true,
    transverse_loads_between_ends: true
  });

  const [anchors, setAnchors] = useState({
    Nu_kip: 0,
    Vu_kip: 10,
    fc_psi: 4000,
    cracked: true,
    lightweight: false,
    lambda_factor: 1.0,
    anchor_count_x: 1,
    anchor_count_y: 1,
    sx_in: 0,
    sy_in: 0,
    edge_x_neg_in: 12,
    edge_x_pos_in: 12,
    edge_y_neg_in: 12,
    edge_y_pos_in: 12,
    member_thickness_in: 8,
    anchor_family: "hilti_kh_ez",
    diameter_in: 0.5,
    hef_in: 3.0,
    fy_psi: null,
    fu_psi: null,
    redundant: false,
    steel_ductile_tension: true,
    steel_ductile_shear: true
  });

  const [punching, setPunching] = useState({
    Vu_kip: 200,
    Mux_kipft: 0,
    Muy_kipft: 0,
    column_bx_in: 16,
    column_by_in: 16,
    slab_thickness_in: 8,
    d_in: null,
    location: "interior",
    fc_psi: 4000,
    lambda_factor: 1.0
  });

  const tabs = [
    { id: "beams", label: "Beams" },
    { id: "slabs", label: "Slabs" },
    { id: "walls", label: "Walls (Slender)" },
    { id: "columns", label: "Columns" },
    { id: "dev", label: "Development & Splices" },
    { id: "anchors", label: "Anchors" },
    { id: "punching", label: "Punching Shear" }
  ];

  function moduleAndInputs() {
    if (tab === "beams") return { module: "beam_flexure", inputs: beams };
    if (tab === "slabs") return { module: "slab_oneway_flexure", inputs: slabs };
    if (tab === "columns") return { module: "column_axial", inputs: columns };
    if (tab === "dev") return { module: "development_length_splices", inputs: dev };
    if (tab === "walls") return { module: "wall_slender", inputs: walls };
    if (tab === "anchors") return { module: "anchors_ch17", inputs: anchors };
    if (tab === "punching") return { module: "punching_shear", inputs: punching };
    return { module: "beam_flexure", inputs: beams };
  }

  async function solve() {
    const mi = moduleAndInputs();
    setBusy(true);
    setError(null);
    setResp(null);

    try {
      const r = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mi)
      });

      const data = await r.json();
      if (!r.ok || data.ok === false) {
        setBusy(false);
        setError(data.error || data.summary_text || "Solve failed");
        setResp(data);
        return;
      }

      setBusy(false);
      setResp(data);
      setError(null);
    } catch (err) {
      setBusy(false);
      setError(String(err));
      setResp(null);
    }
  }

  function renderResults() {
    if (!resp) return null;

    const downloads = resp.download || {};
    const outputs = resp.outputs || {};
    const outKeys = Object.keys(outputs);

    return (
      <div>
        <div className="panel">
          <div style={{ fontWeight: 600, marginBottom: "6px" }}>Governing Summary</div>
          <pre>{resp.summary_text || ""}</pre>

          {resp.run_dir ? (
            <div className="kv">
              <div className="k">Run directory</div>
              <div>{resp.run_dir}</div>
              <div className="k">Input hash</div>
              <div>{resp.input_hash || ""}</div>
              <div className="k">Tool version</div>
              <div>{resp.tool_version || ""}</div>
            </div>
          ) : null}
        </div>

        <div className="actions">
          <button
            className="secondary"
            onClick={() => openNew(downloads.report_html || "/api/report.html")}
          >
            View Calc Package
          </button>
          <button
            className="secondary"
            onClick={() => openNew(downloads.report_html || "/api/report.html")}
          >
            Print / Save PDF
          </button>
          <button
            className="secondary"
            onClick={() => openNew(downloads.excel || "/api/download/results.xlsx")}
          >
            Download Excel
          </button>
          <button
            className="secondary"
            onClick={() => openNew(downloads.calc_trace || "/api/download/calc_trace.json")}
          >
            Download calc_trace.json
          </button>
          <button
            className="secondary"
            onClick={() => openNew(downloads.mathcad_inputs || "/api/download/mathcad_inputs.csv")}
          >
            Download Mathcad handoff
          </button>
        </div>

        {outKeys.length ? (
          <div className="panel">
            <div style={{ fontWeight: 600, marginBottom: "6px" }}>Key Outputs</div>
            <div className="kv">
              {outKeys.flatMap((k) => [
                <div className="k" key={`k_${k}`}>
                  {k}
                </div>,
                <div key={`v_${k}`}>{String(outputs[k])}</div>
              ])}
            </div>
          </div>
        ) : null}

        <div className="iframeWrap">
          <iframe src={(downloads.report_html || "/api/report.html") + "?t=" + Date.now()} />
        </div>
      </div>
    );
  }

  function renderForm() {
    if (tab === "beams") {
      return (
        <div>
          <div className="grid">
            <FieldNumber value={beams.b_in} onChange={(v) => setBeams({ ...beams, b_in: v })} label="Width b" units="in" />
            <FieldNumber value={beams.h_in} onChange={(v) => setBeams({ ...beams, h_in: v })} label="Depth h" units="in" />
            <FieldNumber value={beams.cover_in} onChange={(v) => setBeams({ ...beams, cover_in: v })} label="Cover to stirrup outside" units="in" />
            <FieldNumber value={beams.stirrup_dia_in} onChange={(v) => setBeams({ ...beams, stirrup_dia_in: v })} label="Stirrup diameter" units="in" />
            <FieldNumber value={beams.bar_dia_in} onChange={(v) => setBeams({ ...beams, bar_dia_in: v })} label="Tension bar diameter" units="in" />
            <FieldNumber value={beams.Mu_kipft} onChange={(v) => setBeams({ ...beams, Mu_kipft: v })} label="Factored moment Mu" units="kip-ft" />
            <FieldNumber value={beams.fc_psi} onChange={(v) => setBeams({ ...beams, fc_psi: v })} label="Concrete strength f'c" units="psi" />
            <FieldNumber value={beams.fy_psi} onChange={(v) => setBeams({ ...beams, fy_psi: v })} label="Steel yield strength fy" units="psi" />
            <FieldNumber value={beams.Es_psi} onChange={(v) => setBeams({ ...beams, Es_psi: v })} label="Steel modulus Es" units="psi" />
            <FieldBool
              value={beams.compression_reinf}
              onChange={(v) => setBeams({ ...beams, compression_reinf: v })}
              label="Compression reinforcement (use top layers below)"
            />
            <FieldSelect
              value={beams.transverse_type}
              onChange={(v) => setBeams({ ...beams, transverse_type: v })}
              label="Transverse type (φ cc only)"
              units=""
              options={[
                { value: "other", label: "Other (ties)" },
                { value: "spiral", label: "Spiral" }
              ]}
            />
          </div>
          <ReinforcementEditor groupState={beams} setGroupState={setBeams} groupName="beams" />
        </div>
      );
    }

    if (tab === "slabs") {
      return (
        <div>
          <div className="grid">
            <FieldNumber value={slabs.thickness_in} onChange={(v) => setSlabs({ ...slabs, thickness_in: v })} label="Thickness h" units="in" />
            <FieldNumber value={slabs.cover_in} onChange={(v) => setSlabs({ ...slabs, cover_in: v })} label="Cover to bar" units="in" />
            <FieldNumber value={slabs.bar_dia_in} onChange={(v) => setSlabs({ ...slabs, bar_dia_in: v })} label="Bar diameter" units="in" />
            <FieldNumber value={slabs.Mu_kipft_per_ft} onChange={(v) => setSlabs({ ...slabs, Mu_kipft_per_ft: v })} label="Factored moment Mu per ft" units="kip-ft/ft" />
            <FieldNumber value={slabs.fc_psi} onChange={(v) => setSlabs({ ...slabs, fc_psi: v })} label="Concrete strength f'c" units="psi" />
            <FieldNumber value={slabs.fy_psi} onChange={(v) => setSlabs({ ...slabs, fy_psi: v })} label="Steel yield strength fy" units="psi" />
            <FieldNumber value={slabs.Es_psi} onChange={(v) => setSlabs({ ...slabs, Es_psi: v })} label="Steel modulus Es" units="psi" />
          </div>
          <ReinforcementEditor groupState={slabs} setGroupState={setSlabs} groupName="slabs" />
        </div>
      );
    }

    if (tab === "columns") {
      return (
        <div className="grid">
          <FieldSelect
            value={columns.shape}
            onChange={(v) => setColumns({ ...columns, shape: v })}
            label="Shape"
            units=""
            options={[
              { value: "rectangular", label: "Rectangular" },
              { value: "circular", label: "Circular" }
            ]}
          />
          <FieldNumber value={columns.b_in} onChange={(v) => setColumns({ ...columns, b_in: v })} label="Width b (rectangular)" units="in" />
          <FieldNumber value={columns.h_in} onChange={(v) => setColumns({ ...columns, h_in: v })} label="Depth h (rectangular)" units="in" />
          <FieldNumber value={columns.D_in} onChange={(v) => setColumns({ ...columns, D_in: v })} label="Diameter D (circular)" units="in" />
          <FieldNumber value={columns.Ast_in2} onChange={(v) => setColumns({ ...columns, Ast_in2: v })} label="Longitudinal steel Ast" units="in²" />
          <FieldNumber value={columns.Pu_kip} onChange={(v) => setColumns({ ...columns, Pu_kip: v })} label="Factored axial load Pu" units="kip" />
          <FieldNumber value={columns.fc_psi} onChange={(v) => setColumns({ ...columns, fc_psi: v })} label="Concrete strength f'c" units="psi" />
          <FieldNumber value={columns.fy_psi} onChange={(v) => setColumns({ ...columns, fy_psi: v })} label="Steel yield strength fy" units="psi" />
          <FieldSelect
            value={columns.transverse_type}
            onChange={(v) => setColumns({ ...columns, transverse_type: v })}
            label="Transverse reinforcement"
            units=""
            options={[
              { value: "ties", label: "Ties" },
              { value: "spiral", label: "Spiral" }
            ]}
          />
        </div>
      );
    }

    if (tab === "dev") {
      return (
        <div className="grid">
          <FieldSelect
            value={dev.bar_size}
            onChange={(v) => setDev({ ...dev, bar_size: v })}
            label="Bar size"
            units=""
            options={[
              { value: "#3", label: "#3" },
              { value: "#4", label: "#4" },
              { value: "#5", label: "#5" },
              { value: "#6", label: "#6" },
              { value: "#7", label: "#7" },
              { value: "#8", label: "#8" },
              { value: "#9", label: "#9" },
              { value: "#10", label: "#10" },
              { value: "#11", label: "#11" },
              { value: "#14", label: "#14" },
              { value: "#18", label: "#18" }
            ]}
          />
          <FieldNumber value={dev.db_in} onChange={(v) => setDev({ ...dev, db_in: v })} label="Bar diameter db" units="in" />
          <FieldNumber value={dev.fc_psi} onChange={(v) => setDev({ ...dev, fc_psi: v })} label="Concrete strength f'c" units="psi" />
          <FieldNumber value={dev.fy_psi} onChange={(v) => setDev({ ...dev, fy_psi: v })} label="Steel yield strength fy" units="psi" />
          <FieldNumber value={dev.lambda_factor} onChange={(v) => setDev({ ...dev, lambda_factor: v })} label="Lightweight factor λ" units="" />
          <FieldBool value={dev.is_top_bar} onChange={(v) => setDev({ ...dev, is_top_bar: v })} label="Top bar (ψt)" />
          <FieldBool value={dev.is_epoxy} onChange={(v) => setDev({ ...dev, is_epoxy: v })} label="Epoxy-coated (ψe)" />
          <FieldBool
            value={dev.epoxy_cover_lt_3db_or_spacing_lt_6db}
            onChange={(v) => setDev({ ...dev, epoxy_cover_lt_3db_or_spacing_lt_6db: v })}
            label="Epoxy with cover<3db or spacing<6db"
          />
          <FieldBool
            value={dev.cover_ge_db_and_spacing_ge_2db}
            onChange={(v) => setDev({ ...dev, cover_ge_db_and_spacing_ge_2db: v })}
            label="Cover≥db and spacing≥2db (Table row selection)"
          />
          <FieldNumber
            value={dev.As_provided_over_As_required}
            onChange={(v) => setDev({ ...dev, As_provided_over_As_required: v })}
            label="As,prov / As,req (splice)"
            units=""
          />
          <FieldNumber
            value={dev.percent_bars_spliced}
            onChange={(v) => setDev({ ...dev, percent_bars_spliced: v })}
            label="Max % spliced (label only)"
            units="%"
          />
        </div>
      );
    }

    if (tab === "walls") {
      return (
        <div className="grid">
          <FieldNumber value={walls.lw_in} onChange={(v) => setWalls({ ...walls, lw_in: v })} label="Wall length ℓw" units="in" />
          <FieldNumber value={walls.t_in} onChange={(v) => setWalls({ ...walls, t_in: v })} label="Thickness t" units="in" />
          <FieldNumber value={walls.lu_in} onChange={(v) => setWalls({ ...walls, lu_in: v })} label="Unsupported height ℓu" units="in" />
          <FieldNumber value={walls.cover_in} onChange={(v) => setWalls({ ...walls, cover_in: v })} label="Cover to vertical bars" units="in" />
          <FieldNumber value={walls.bar_dia_in} onChange={(v) => setWalls({ ...walls, bar_dia_in: v })} label="Vertical bar diameter" units="in" />
          <FieldNumber value={walls.rho_v} onChange={(v) => setWalls({ ...walls, rho_v: v })} label="Vertical steel ratio ρv (As/Ag)" units="" />
          <FieldNumber value={walls.Pu_kip} onChange={(v) => setWalls({ ...walls, Pu_kip: v })} label="Factored axial load Pu (compression +)" units="kip" />
          <FieldNumber value={walls.M_top_oop_kipft} onChange={(v) => setWalls({ ...walls, M_top_oop_kipft: v })} label="Out-of-plane Mtop" units="kip-ft" />
          <FieldNumber value={walls.M_bot_oop_kipft} onChange={(v) => setWalls({ ...walls, M_bot_oop_kipft: v })} label="Out-of-plane Mbot" units="kip-ft" />
          <FieldNumber value={walls.M_top_ip_kipft} onChange={(v) => setWalls({ ...walls, M_top_ip_kipft: v })} label="In-plane Mtop" units="kip-ft" />
          <FieldNumber value={walls.M_bot_ip_kipft} onChange={(v) => setWalls({ ...walls, M_bot_ip_kipft: v })} label="In-plane Mbot" units="kip-ft" />
          <FieldSelect
            value={walls.end_bottom}
            onChange={(v) => setWalls({ ...walls, end_bottom: v })}
            label="Bottom end condition"
            units=""
            options={[
              { value: "fixed", label: "Fixed (rotation restrained)" },
              { value: "pinned", label: "Pinned (rotation free)" },
              { value: "unbraced", label: "Unbraced (translation free)" }
            ]}
          />
          <FieldSelect
            value={walls.end_top}
            onChange={(v) => setWalls({ ...walls, end_top: v })}
            label="Top end condition"
            units=""
            options={[
              { value: "fixed", label: "Fixed (rotation restrained)" },
              { value: "pinned", label: "Pinned (rotation free)" },
              { value: "unbraced", label: "Unbraced (translation free)" }
            ]}
          />
          <FieldNumber value={walls.fc_psi} onChange={(v) => setWalls({ ...walls, fc_psi: v })} label="Concrete strength f'c" units="psi" />
          <FieldNumber value={walls.fy_psi} onChange={(v) => setWalls({ ...walls, fy_psi: v })} label="Steel yield strength fy" units="psi" />
          <FieldNumber value={walls.Es_psi} onChange={(v) => setWalls({ ...walls, Es_psi: v })} label="Steel modulus Es" units="psi" />
          <FieldSelect
            value={walls.transverse_type}
            onChange={(v) => setWalls({ ...walls, transverse_type: v })}
            label="Transverse type (φ cc only)"
            units=""
            options={[
              { value: "other", label: "Other (ties)" },
              { value: "spiral", label: "Spiral" }
            ]}
          />
          <FieldBool value={walls.cracked_section} onChange={(v) => setWalls({ ...walls, cracked_section: v })} label="Use cracked I for stiffness (0.35Ig)" />
          <FieldNumber value={walls.beta_dns} onChange={(v) => setWalls({ ...walls, beta_dns: v })} label="βdns sustained/total axial" units="" />
          <FieldBool value={walls.member_nonsway} onChange={(v) => setWalls({ ...walls, member_nonsway: v })} label="Nonsway member (δns method)" />
          <FieldBool value={walls.transverse_loads_between_ends} onChange={(v) => setWalls({ ...walls, transverse_loads_between_ends: v })} label="Transverse loads between ends (Cm=1.0)" />
        </div>
      );
    }

    if (tab === "anchors") {
      return (
        <div className="grid">
          <FieldNumber value={anchors.Nu_kip} onChange={(v) => setAnchors({ ...anchors, Nu_kip: v })} label="Factored tension Nu (tension +)" units="kip" />
          <FieldNumber value={anchors.Vu_kip} onChange={(v) => setAnchors({ ...anchors, Vu_kip: v })} label="Factored shear Vu" units="kip" />
          <FieldNumber value={anchors.fc_psi} onChange={(v) => setAnchors({ ...anchors, fc_psi: v })} label="Concrete strength f'c" units="psi" />
          <FieldBool value={anchors.cracked} onChange={(v) => setAnchors({ ...anchors, cracked: v })} label="Cracked concrete" />
          <FieldBool value={anchors.lightweight} onChange={(v) => setAnchors({ ...anchors, lightweight: v })} label="Lightweight concrete" />
          <FieldNumber value={anchors.lambda_factor} onChange={(v) => setAnchors({ ...anchors, lambda_factor: v })} label="Lambda (λ)" units="" />
          <FieldSelect
            value={anchors.anchor_family}
            onChange={(v) => setAnchors({ ...anchors, anchor_family: v })}
            label="Anchor family"
            units=""
            options={[
              { value: "cast_in_headed", label: "Cast-in headed rod" },
              { value: "hilti_kb_tz2", label: "Hilti KB-TZ2 (wedge)" },
              { value: "hilti_kh_ez", label: "Hilti KH-EZ (screw)" },
              { value: "hilti_hit_hy200_v3", label: "Hilti HIT-HY 200 V3 (adhesive)" }
            ]}
          />
          <FieldNumber value={anchors.diameter_in} onChange={(v) => setAnchors({ ...anchors, diameter_in: v })} label="Anchor diameter" units="in" />
          <FieldNumber value={anchors.hef_in} onChange={(v) => setAnchors({ ...anchors, hef_in: v })} label="Effective embedment hef" units="in" />
          <FieldNumber value={anchors.anchor_count_x} onChange={(v) => setAnchors({ ...anchors, anchor_count_x: v })} label="Count in X" units="" />
          <FieldNumber value={anchors.anchor_count_y} onChange={(v) => setAnchors({ ...anchors, anchor_count_y: v })} label="Count in Y" units="" />
          <FieldNumber value={anchors.sx_in} onChange={(v) => setAnchors({ ...anchors, sx_in: v })} label="Spacing sx (0 for single)" units="in" />
          <FieldNumber value={anchors.sy_in} onChange={(v) => setAnchors({ ...anchors, sy_in: v })} label="Spacing sy (0 for single)" units="in" />
          <FieldNumber value={anchors.edge_x_neg_in} onChange={(v) => setAnchors({ ...anchors, edge_x_neg_in: v })} label="Edge -X" units="in" />
          <FieldNumber value={anchors.edge_x_pos_in} onChange={(v) => setAnchors({ ...anchors, edge_x_pos_in: v })} label="Edge +X" units="in" />
          <FieldNumber value={anchors.edge_y_neg_in} onChange={(v) => setAnchors({ ...anchors, edge_y_neg_in: v })} label="Edge -Y" units="in" />
          <FieldNumber value={anchors.edge_y_pos_in} onChange={(v) => setAnchors({ ...anchors, edge_y_pos_in: v })} label="Edge +Y" units="in" />
          <FieldNumber value={anchors.member_thickness_in} onChange={(v) => setAnchors({ ...anchors, member_thickness_in: v })} label="Member thickness" units="in" />
          <FieldBool value={anchors.redundant} onChange={(v) => setAnchors({ ...anchors, redundant: v })} label="Redundant anchor group (ϕ for concrete tension)" />
          <FieldBool value={anchors.steel_ductile_tension} onChange={(v) => setAnchors({ ...anchors, steel_ductile_tension: v })} label="Steel ductile in tension" />
          <FieldBool value={anchors.steel_ductile_shear} onChange={(v) => setAnchors({ ...anchors, steel_ductile_shear: v })} label="Steel ductile in shear" />
        </div>
      );
    }

    if (tab === "punching") {
      return (
        <div>
          <div className="grid">
            <FieldNumber value={punching.Vu_kip} onChange={(v) => setPunching({ ...punching, Vu_kip: v })} label="Factored shear Vu" units="kip" />
            <FieldNumber value={punching.Mux_kipft} onChange={(v) => setPunching({ ...punching, Mux_kipft: v })} label="Unbalanced moment Mux" units="kip-ft" />
            <FieldNumber value={punching.Muy_kipft} onChange={(v) => setPunching({ ...punching, Muy_kipft: v })} label="Unbalanced moment Muy" units="kip-ft" />
            <FieldNumber value={punching.column_bx_in} onChange={(v) => setPunching({ ...punching, column_bx_in: v })} label="Column size bx" units="in" />
            <FieldNumber value={punching.column_by_in} onChange={(v) => setPunching({ ...punching, column_by_in: v })} label="Column size by" units="in" />
            <FieldNumber value={punching.slab_thickness_in} onChange={(v) => setPunching({ ...punching, slab_thickness_in: v })} label="Slab thickness h" units="in" />
            <FieldNumber value={punching.fc_psi} onChange={(v) => setPunching({ ...punching, fc_psi: v })} label="Concrete strength f'c" units="psi" />
            <FieldNumber value={punching.lambda_factor} onChange={(v) => setPunching({ ...punching, lambda_factor: v })} label="Lambda (λ)" units="" />
            <FieldSelect
              value={punching.location}
              onChange={(v) => setPunching({ ...punching, location: v })}
              label="Column location"
              units=""
              options={[
                { value: "interior", label: "Interior" },
                { value: "edge", label: "Edge" },
                { value: "corner", label: "Corner" }
              ]}
            />
          </div>

          <div className="sub">
            Note: current implementation is conservative and does not yet amplify for full moment transfer provisions.
          </div>
        </div>
      );
    }

    return null;
  }

  return (
    <ToolShell
      title="ACI 318-25 Concrete Design Tool"
      description="Offline React UI + Python backend. Generates audit-grade calc packages."
    >
      <div className="aci318-tool">
        <div className="tabs">
          {tabs.map((t) => (
            <div
              key={t.id}
              className={"tab" + (tab === t.id ? " active" : "")}
              onClick={() => {
                setTab(t.id);
                setResp(null);
                setError(null);
              }}
            >
              {t.label}
            </div>
          ))}
        </div>

        {renderForm()}

        <div className="actions">
          <button onClick={solve} disabled={busy}>
            {busy ? "Running..." : "Solve / Run"}
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {renderResults()}
      </div>
    </ToolShell>
  );
}
