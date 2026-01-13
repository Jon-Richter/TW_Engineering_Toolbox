import React, { useMemo, useState, useEffect } from "react";
import { getLoadRecord, loadDatabase, type LoadRecord, type LoadCaseName } from "./data/loads";
import { BAR_SIZES, CALC_CONFIG, DEFAULTS, computeAll, getBarProps, optimizeB, type CalcResult } from "./lib/calc";

type Cfg = { model: string; jibLength: number; towerSections: number };
const parseNumber = (value: string) => {
  const s = value.trim();
  if (!s) return 0;
  const parseFrac = (part: string) => {
    const m = part.match(/^([+-]?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/);
    if (!m) return Number.NaN;
    const num = Number(m[1]);
    const den = Number(m[2]);
    if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return Number.NaN;
    return num / den;
  };
  const parts = s.split(/\s+/);
  if (parts.length === 2 && parts[1].includes("/")) {
    const whole = Number(parts[0]);
    const frac = parseFrac(parts[1]);
    if (!Number.isFinite(whole) || !Number.isFinite(frac)) return 0;
    const sign = whole < 0 ? -1 : 1;
    return whole + sign * Math.abs(frac);
  }
  if (s.includes("/")) {
    const frac = parseFrac(s);
    return Number.isFinite(frac) ? frac : 0;
  }
  const num = Number(s);
  return Number.isFinite(num) ? num : 0;
};
type NumberInputProps = {
  value: number;
  onChange: (value: number) => void;
  className?: string;
  disabled?: boolean;
};

function NumberInput({ value, onChange, className, disabled = false }: NumberInputProps) {
  const [draft, setDraft] = useState(() => (Number.isFinite(value) ? String(value) : ""));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (focused) return;
    if (draft === "" && Number.isFinite(value) && value === 0) return;
    if (!Number.isFinite(value)) {
      setDraft("");
      return;
    }
    setDraft(String(value));
  }, [value, focused]);

  return (
    <input
      type="text"
      inputMode="decimal"
      className={className}
      value={draft}
      onChange={(e) => {
        const next = e.target.value;
        setDraft(next);
        if (!next.trim()) {
          onChange(0);
          return;
        }
        onChange(parseNumber(next));
      }}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      disabled={disabled}
    />
  );
}
const toInput = (value: number, d = 2) => (Number.isFinite(value) ? Number(value.toFixed(d)).toString() : "");
const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "-");
const fmt0 = (x: number) => (Number.isFinite(x) ? Math.round(x).toString() : "-");
const FT_TO_M = 0.3048;

const CASE_ORDER: LoadCaseName[] = ["operation", "stormFront", "stormRear", "erection"];
const CASE_LABEL: Record<LoadCaseName, string> = {
  operation: "In operation",
  stormFront: "Storm front",
  stormRear: "Storm rear",
  erection: "Erection"
};

const MAST_SECTIONS = ["24HC 630", "1000 HC"] as const;
type MastSection = (typeof MAST_SECTIONS)[number];
const MM_TO_FT = 3.280839895013123 / 1000;
const MAST_WIDTH_FT: Record<MastSection, number> = {
  "24HC 630": 2200 * MM_TO_FT,
  "1000 HC": 3100 * MM_TO_FT
};

function Panel(p: { title: string; subtitle?: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_20px_50px_rgba(0,0,0,0.45)]">
      <div className="px-4 py-3 border-b border-white/10 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-semibold text-slate-100 truncate">{p.title}</div>
          {p.subtitle ? <div className="text-xs text-slate-300 mt-0.5">{p.subtitle}</div> : null}
        </div>
        {p.right ? <div className="shrink-0">{p.right}</div> : null}
      </div>
      <div className="p-4">{p.children}</div>
    </div>
  );
}

function Pill(p: { label: string; kind: "pass" | "fail" | "warn" | "info" }) {
  const cls =
    p.kind === "pass"
      ? "bg-emerald-500/15 text-emerald-200 border-emerald-400/25"
      : p.kind === "fail"
      ? "bg-rose-500/15 text-rose-200 border-rose-400/25"
      : p.kind === "warn"
      ? "bg-amber-500/15 text-amber-200 border-amber-400/25"
      : "bg-sky-500/15 text-sky-200 border-sky-400/25";
  return <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs border ${cls} font-semibold`}>{p.label}</span>;
}

function Row(p: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1 text-sm">
      <div className="text-slate-300">{p.label}</div>
      <div className="font-mono text-slate-100 text-right">{p.value}</div>
    </div>
  );
}

function ResultBadge(p: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-md border border-amber-300/25 bg-amber-400/15 px-2 py-0.5 text-amber-100 font-semibold">
      {p.children}
    </span>
  );
}

function CalcRow(p: { label: string; children: React.ReactNode; kind?: "result" | "note" }) {
  return (
    <div className="grid grid-cols-[92px_1fr] gap-3 py-1">
      <div className="text-xs font-semibold tracking-wide text-slate-300">{p.label}</div>
      <div className={p.kind === "note" ? "text-xs text-slate-400" : "text-sm text-slate-100"}>{p.children}</div>
    </div>
  );
}

function CalcBlock(p: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="font-semibold text-slate-100">{p.title}</div>
        {p.right ? <div className="shrink-0">{p.right}</div> : null}
      </div>
      <div className="mt-2">{p.children}</div>
    </div>
  );
}

function CraneDiagram(p: { jibLength: number; towerSections: number; hookHeight_ft: number; maxJib: number; maxHook: number }) {
  const width = 360;
  const height = 220;
  const baseY = 190;
  const towerX = 60;
  const maxTowerH = 140;
  const maxJibL = 240;
  const dimOffset = 34;

  const towerH = Math.max(30, (p.hookHeight_ft / Math.max(p.maxHook, 1)) * maxTowerH);
  const jibL = Math.max(40, (p.jibLength / Math.max(p.maxJib, 1)) * maxJibL);
  const jibY = baseY - towerH;
  const jibDimY = Math.max(12, jibY - dimOffset);

  const towerW = 18;
  const towerL = towerX - towerW / 2;
  const towerR = towerX + towerW / 2;

  const jibTopY = jibY - 8;
  const jibBotY = jibY + 6;
  const jibEndX = towerX + jibL;
  const counterL = Math.min(70, Math.max(44, jibL * 0.28));
  const counterEndX = towerX - counterL;
  const hookDimX = width - 44;

  const apexX = towerX + 18;
  const apexY = jibTopY - 26;
  const tieX = towerX + jibL * 0.58;
  const trolleyX = towerX + jibL * 0.72;
  const hookEndY = Math.min(baseY - 12, jibBotY + 44);

  const towerPanels = Math.max(7, Math.round(towerH / 14));
  const jibPanels = Math.max(7, Math.round(jibL / 28));
  const counterPanels = Math.max(4, Math.round(counterL / 16));
  const towerSegY = towerH / towerPanels;
  const jibSegX = jibL / jibPanels;
  const counterSegX = counterL / counterPanels;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-72">
      <defs>
        <marker id="arrowDim" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" fill="#60a5fa" />
        </marker>
      </defs>
      {/* Base / ground */}
      <rect x="14" y={baseY} width="160" height="10" rx="2" className="fill-slate-700" />

      {/* Crane (stylized) */}
      <g stroke="currentColor" className="text-slate-200" fill="none">
        {/* Tower */}
        <rect x={towerL} y={jibTopY} width={towerW} height={baseY - jibTopY} rx="1" strokeWidth="2" />
        {Array.from({ length: towerPanels }, (_, i) => {
          const y0 = baseY - towerSegY * i;
          const y1 = baseY - towerSegY * (i + 1);
          const ym = (y0 + y1) / 2;
          return (
            <g key={`tower-${i}`}>
              <line x1={towerL} y1={y1} x2={towerR} y2={y1} strokeWidth="1.2" />
              <line x1={towerL} y1={y0} x2={towerR} y2={y1} strokeWidth="1" />
              <line x1={towerR} y1={y0} x2={towerL} y2={y1} strokeWidth="1" />
              <line x1={towerX} y1={ym} x2={towerX} y2={y1} strokeWidth="0.9" opacity="0.75" />
            </g>
          );
        })}

        {/* Counter-jib */}
        <line x1={counterEndX} y1={jibTopY} x2={towerX} y2={jibTopY} strokeWidth="2" />
        <line x1={counterEndX} y1={jibBotY} x2={towerX} y2={jibBotY} strokeWidth="2" />
        {Array.from({ length: counterPanels }, (_, i) => {
          const x0 = towerX - counterSegX * i;
          const x1 = towerX - counterSegX * (i + 1);
          const diagUp = i % 2 === 0;
          return (
            <g key={`counter-${i}`}>
              <line x1={x1} y1={jibTopY} x2={x1} y2={jibBotY} strokeWidth="1" />
              <line
                x1={diagUp ? x0 : x0}
                y1={diagUp ? jibBotY : jibTopY}
                x2={diagUp ? x1 : x1}
                y2={diagUp ? jibTopY : jibBotY}
                strokeWidth="1"
              />
            </g>
          );
        })}
        <rect x={towerL - 16} y={jibBotY - 8} width="20" height="10" rx="1" strokeWidth="1.5" />

        {/* Main jib */}
        <line x1={towerX} y1={jibTopY} x2={jibEndX} y2={jibTopY} strokeWidth="2" />
        <line x1={towerX} y1={jibBotY} x2={jibEndX} y2={jibBotY} strokeWidth="2" />
        {Array.from({ length: jibPanels }, (_, i) => {
          const x0 = towerX + jibSegX * i;
          const x1 = towerX + jibSegX * (i + 1);
          const diagUp = i % 2 === 0;
          return (
            <g key={`jib-${i}`}>
              <line x1={x1} y1={jibTopY} x2={x1} y2={jibBotY} strokeWidth="1" />
              <line
                x1={diagUp ? x0 : x0}
                y1={diagUp ? jibBotY : jibTopY}
                x2={diagUp ? x1 : x1}
                y2={diagUp ? jibTopY : jibBotY}
                strokeWidth="1"
              />
            </g>
          );
        })}
        <rect x={jibEndX - 8} y={jibTopY - 2} width="10" height={jibBotY - jibTopY + 4} rx="1" strokeWidth="1.5" />

        {/* A-frame + tie lines */}
        <line x1={towerL} y1={jibTopY} x2={apexX} y2={apexY} strokeWidth="1.8" />
        <line x1={towerR} y1={jibTopY} x2={apexX} y2={apexY} strokeWidth="1.8" />
        <line x1={apexX} y1={apexY} x2={tieX} y2={jibTopY} strokeWidth="1.2" opacity="0.9" />
        <line x1={apexX} y1={apexY} x2={counterEndX + 6} y2={jibTopY} strokeWidth="1.2" opacity="0.9" />

        {/* Trolley + hook */}
        <rect x={trolleyX - 6} y={jibBotY - 3} width="12" height="6" rx="1" strokeWidth="1.4" />
        <line x1={trolleyX} y1={jibBotY + 3} x2={trolleyX} y2={hookEndY} strokeWidth="1.2" opacity="0.75" />
        <path
          d={`M${trolleyX - 4},${hookEndY} Q${trolleyX},${hookEndY + 5} ${trolleyX + 4},${hookEndY}`}
          strokeWidth="1.4"
          opacity="0.85"
        />
      </g>

      {/* Dimensions */}
      <g stroke="#60a5fa" fill="#60a5fa">
        <line x1={towerX} y1={jibY} x2={towerX} y2={jibDimY} strokeWidth="1" />
        <line x1={jibEndX} y1={jibY} x2={jibEndX} y2={jibDimY} strokeWidth="1" />
        <line
          x1={towerX}
          y1={jibDimY}
          x2={jibEndX}
          y2={jibDimY}
          strokeWidth="1.5"
          markerStart="url(#arrowDim)"
          markerEnd="url(#arrowDim)"
        />
        <text
          x={towerX + jibL / 2}
          y={jibDimY - 8}
          textAnchor="middle"
          className="text-[14px] font-semibold"
          stroke="#0f172a"
          strokeWidth="3"
          paintOrder="stroke"
        >
          Jib Length {fmt(p.jibLength, 1)} ft
        </text>

        <line
          x1={hookDimX}
          y1={baseY}
          x2={hookDimX}
          y2={baseY - towerH}
          strokeWidth="1.5"
          markerStart="url(#arrowDim)"
          markerEnd="url(#arrowDim)"
        />
        <text
          x={hookDimX - 6}
          y={baseY - towerH / 2 - 6}
          textAnchor="end"
          className="text-[14px] font-semibold"
          stroke="#0f172a"
          strokeWidth="3"
          paintOrder="stroke"
        >
          Hook Height {fmt(p.hookHeight_ft, 1)} ft
        </text>
      </g>

    </svg>
  );
}

export default function App() {
  const first =
    (loadDatabase[0] as LoadRecord | undefined) ??
    ({
      model: "N/A",
      jibLength: 0,
      towerSections: 0,
      hookHeight_ft: 0,
      slewTorque_kipft: 0,
      cases: {
        operation: { Pu: 0, M_kipft: 0, V_k: 0 },
        stormFront: { Pu: 0, M_kipft: 0, V_k: 0 },
        stormRear: { Pu: 0, M_kipft: 0, V_k: 0 },
        erection: { Pu: 0, M_kipft: 0, V_k: 0 }
      }
    } satisfies LoadRecord);

  const [cfg, setCfg] = useState<Cfg>({ model: first.model, jibLength: first.jibLength, towerSections: first.towerSections });
  const [mastSection, setMastSection] = useState<MastSection>("1000 HC");

  // Geometry
  const [BxInput, setBxInput] = useState(toInput(DEFAULTS.Bx_ft));
  const [ByInput, setByInput] = useState(toInput(DEFAULTS.By_ft));
  const [tInput, setTInput] = useState(toInput(DEFAULTS.t_ft));
  const [qa, setQa] = useState(DEFAULTS.qa_psf);
  const [pedInput, setPedInput] = useState(toInput(MAST_WIDTH_FT[mastSection]));

  // Sliding parameters (service)
  const [mu, setMu] = useState(DEFAULTS.mu);
  const [FSslide, setFSslide] = useState(DEFAULTS.FS_slide_req);
  const [gridN, setGridN] = useState(DEFAULTS.gridN);

  // Materials / rebar (screening)
  const [fc, setFc] = useState(DEFAULTS.fc_psi);
  const [fy, setFy] = useState(DEFAULTS.fy_ksi);
  const [cover, setCover] = useState(DEFAULTS.cover_in);
  const [barX, setBarX] = useState<(typeof BAR_SIZES)[number]>(DEFAULTS.barX);
  const [barY, setBarY] = useState<(typeof BAR_SIZES)[number]>(DEFAULTS.barY);
  const [spacingX, setSpacingX] = useState(DEFAULTS.spacingX_in);
  const [spacingY, setSpacingY] = useState(DEFAULTS.spacingY_in);

  const [tab, setTab] = useState<"summary" | "equations">("summary");

  const models = useMemo(() => Array.from(new Set(loadDatabase.map((r) => r.model))).sort(), []);
  const maxJib = useMemo(() => Math.max(...loadDatabase.map((r) => r.jibLength), 1), []);
  const maxHook = useMemo(() => Math.max(...loadDatabase.map((r) => r.hookHeight_ft), 1), []);
  const spacingOptions = useMemo(() => {
    const vals: number[] = [];
    for (let s = 3; s <= 18.0001; s += 0.5) vals.push(Number(s.toFixed(1)));
    return vals;
  }, []);
  const jibs = useMemo(
    () => Array.from(new Set(loadDatabase.filter((r) => r.model === cfg.model).map((r) => r.jibLength))).sort((a, b) => a - b),
    [cfg.model]
  );
  const hookOptions = useMemo(
    () =>
      loadDatabase
        .filter((r) => r.model === cfg.model && r.jibLength === cfg.jibLength)
        .map((r) => ({ towerSections: r.towerSections, hookHeight_ft: r.hookHeight_ft }))
        .sort((a, b) => a.hookHeight_ft - b.hookHeight_ft),
    [cfg.model, cfg.jibLength]
  );

  const record = useMemo<LoadRecord>(() => {
    try {
      return getLoadRecord(cfg.model, cfg.jibLength, cfg.towerSections);
    } catch {
      return first;
    }
  }, [cfg, first]);

  useEffect(() => {
    setPedInput(toInput(MAST_WIDTH_FT[mastSection]));
  }, [mastSection]);

  const Bx = parseNumber(BxInput);
  const By = parseNumber(ByInput);
  const t = parseNumber(tInput);
  const ped = parseNumber(pedInput);
  const isRectangular = Math.abs(Bx - By) > 1e-6;

  const calc: CalcResult = useMemo(
    () =>
      computeAll({
        Bx_ft: Bx,
        By_ft: By,
        t_ft: t,
        pedestal_ft: ped,
        qa_psf: qa,
        mu,
        FS_slide_req: FSslide,
        gridN,
        record,
        fc_psi: fc,
        fy_ksi: fy,
        cover_in: cover,
        barX,
        barY,
        spacingX_in: spacingX,
        spacingY_in: spacingY
      }),
    [Bx, By, t, ped, qa, mu, FSslide, gridN, record, fc, fy, cover, barX, barY, spacingX, spacingY]
  );

  const onOptimizeSquare = () => {
    const res = optimizeB({
      pedestal_ft: ped,
      t_ft: t,
      qa_psf: qa,
      mu,
      FS_slide_req: FSslide,
      gridN,
      record,
      Bmin_ft: 4,
      Bmax_ft: 40
    });
    const B = res.B_ft;
    setBxInput(toInput(B));
    setByInput(toInput(B));
    const auto = computeAll({
      Bx_ft: B,
      By_ft: B,
      t_ft: t,
      pedestal_ft: ped,
      qa_psf: qa,
      mu,
      FS_slide_req: FSslide,
      gridN,
      record,
      fc_psi: fc,
      fy_ksi: fy,
      cover_in: cover,
      barX,
      barY
    });
    setSpacingX(auto.strength.flexureX.spacing_in);
    setSpacingY(auto.strength.flexureY.spacing_in);
  };

  const hdrRight = <Pill label={calc.service.passAll ? "SERVICE OK" : "SERVICE CHECK"} kind={calc.service.passAll ? "pass" : "warn"} />;
  const craneRight = (
    <div className="text-right leading-4">
      <div className="text-[11px] text-slate-300">
        Jib <span className="font-mono text-slate-100">{fmt(record.jibLength, 1)}</span>{" "}
        <span className="text-slate-400">ft</span>{" "}
        <span className="text-slate-400">
          ({fmt(record.jibLength * FT_TO_M, 1)} m)
        </span>
      </div>
      <div className="text-[11px] text-slate-300">
        HUH <span className="font-mono text-slate-100">{fmt(record.hookHeight_ft, 1)}</span>{" "}
        <span className="text-slate-400">ft</span>{" "}
        <span className="text-slate-400">
          ({fmt(record.hookHeight_ft * FT_TO_M, 1)} m)
        </span>
      </div>
    </div>
  );
  const strengthOk =
    calc.strength.flexureX.utilization <= 1 &&
    calc.strength.flexureY.utilization <= 1 &&
    calc.strength.shear.oneWay.utilization <= 1 &&
    calc.strength.shear.punching.utilization <= 1;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 to-slate-900 text-slate-100">
      <div className="max-w-screen-2xl mx-auto px-4 py-6">
        <div className="flex items-baseline justify-between gap-4 mb-4">
          <div>
            <h1 className="text-xl font-bold">Tower Crane Spread Footing Designer</h1>
            <p className="text-xs text-slate-300">
              Always checks: operation, storm front, storm rear, erection; slewing torque from DB applied to operation + erection
            </p>
          </div>
          <div className="text-xs text-slate-400">Static-site friendly (build to dist/)</div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* LEFT */}
          <div className="lg:col-span-4">
            <Panel title="Crane diagram" subtitle="Updates with selected jib length and hook height." right={craneRight}>
              <CraneDiagram
                jibLength={record.jibLength}
                towerSections={record.towerSections}
                hookHeight_ft={record.hookHeight_ft}
                maxJib={maxJib}
                maxHook={maxHook}
              />
            </Panel>

            <div className="mt-4">
            <Panel title="Inputs" subtitle="Service stability + sliding + strength screening." right={hdrRight}>
              <div className="space-y-5">
                <div>
                  <div className="text-xs text-slate-300 mb-1">Crane configuration</div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <label className="text-xs text-slate-300">
                      Model
                      <select
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={cfg.model}
                        onChange={(e) => {
                          const m = e.target.value;
                          const rec = loadDatabase.find((r) => r.model === m) ?? loadDatabase[0];
                          if (rec) setCfg({ model: rec.model, jibLength: rec.jibLength, towerSections: rec.towerSections });
                        }}
                      >
                        {models.map((m) => (
                          <option key={m} value={m} className="bg-slate-900">
                            {m}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="text-xs text-slate-300">
                      Jib length
                      <select
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={cfg.jibLength}
                        onChange={(e) => {
                          const j = parseNumber(e.target.value);
                          const rec =
                            loadDatabase.find((r) => r.model === cfg.model && r.jibLength === j) ??
                            loadDatabase.find((r) => r.model === cfg.model) ??
                            loadDatabase[0];
                          if (rec) setCfg({ model: rec.model, jibLength: rec.jibLength, towerSections: rec.towerSections });
                        }}
                      >
                        {jibs.map((j) => (
                          <option key={j} value={j} className="bg-slate-900">
                            {fmt(j, 1)} ft ({fmt(j * FT_TO_M, 1)} m)
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="text-xs text-slate-300">
                      Hook height (HUH)
                      <select
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={cfg.towerSections}
                        onChange={(e) => setCfg({ ...cfg, towerSections: parseNumber(e.target.value) })}
                      >
                        {hookOptions.map((o) => (
                          <option key={o.towerSections} value={o.towerSections} className="bg-slate-900">
                            {fmt(o.hookHeight_ft, 1)} ft ({fmt(o.hookHeight_ft * FT_TO_M, 1)} m)
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="text-xs text-slate-300">
                      Mast Section (bottom)
                      <select
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={mastSection}
                        onChange={(e) => setMastSection(e.target.value as MastSection)}
                      >
                        {MAST_SECTIONS.map((m) => (
                          <option key={m} value={m} className="bg-slate-900">
                            {m}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1">Foundation (service)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    <label className="text-xs text-slate-300">
                      Bx (ft)
                      <input
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={BxInput}
                        onChange={(e) => setBxInput(e.target.value)}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      By (ft)
                      <input
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={ByInput}
                        onChange={(e) => setByInput(e.target.value)}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      t (ft)
                      <input
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={tInput}
                        onChange={(e) => setTInput(e.target.value)}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      qa (psf)
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={qa}
                        onChange={setQa}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      Mast Width (ft)
                      <input
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={pedInput}
                        onChange={(e) => setPedInput(e.target.value)}
                      />
                    </label>
                  </div>

                  <button
                    onClick={onOptimizeSquare}
                    className="mt-3 w-full rounded-xl px-3 py-2 text-sm font-semibold bg-amber-500/20 border border-amber-300/30 text-amber-100 hover:bg-amber-500/30"
                  >
                    Optimize (square) footing size
                  </button>
                  <div className="mt-2 text-[11px] text-slate-400">
                    Optimization is square-only: searches integer B and sets Bx=By=B. All 4 cases must pass.
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1">Sliding parameters (service)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    <label className="text-xs text-slate-300">
                      mu
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={mu}
                        onChange={setMu}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      FSreq
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={FSslide}
                        onChange={setFSslide}
                      />
                    </label>
                    <label className="text-xs text-slate-300 col-span-2 sm:col-span-1">
                      Grid n (default 40)
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={gridN}
                        onChange={(value) => setGridN(Math.max(10, Math.floor(value)))}
                      />
                    </label>
                  </div>
                  <div className="mt-2 text-[11px] text-slate-400">Slewing torque comes from the database; applied to operation + erection only.</div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1">Materials / rebar (screening)</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    <label className="text-xs text-slate-300">
                      f&apos;c (psi)
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={fc}
                        onChange={setFc}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      fy (ksi)
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={fy}
                        onChange={setFy}
                      />
                    </label>
                    <label className="text-xs text-slate-300">
                      cover (in)
                      <NumberInput
                        className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                        value={cover}
                        onChange={setCover}
                      />
                    </label>
                    {!isRectangular ? (
                      <>
                        <label className="text-xs text-slate-300">
                          bar (both dirs)
                          <select
                            className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                            value={barX}
                            onChange={(e) => {
                              const b = e.target.value as any;
                              setBarX(b);
                              setBarY(b);
                            }}
                          >
                            {BAR_SIZES.map((b) => (
                              <option key={b} value={b} className="bg-slate-900">
                                {b}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="text-xs text-slate-300">
                          spacing (in) (both dirs)
                          <select
                            className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                            value={spacingX}
                            onChange={(e) => {
                              const s = parseNumber(e.target.value);
                              setSpacingX(s);
                              setSpacingY(s);
                            }}
                          >
                            {spacingOptions.map((s) => (
                              <option key={s} value={s} className="bg-slate-900">
                                {s}
                              </option>
                            ))}
                          </select>
                        </label>
                      </>
                    ) : (
                      <>
                        <label className="text-xs text-slate-300">
                          bar (X-dir)
                          <select
                            className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                            value={barX}
                            onChange={(e) => setBarX(e.target.value as any)}
                          >
                            {BAR_SIZES.map((b) => (
                              <option key={b} value={b} className="bg-slate-900">
                                {b}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="text-xs text-slate-300">
                          spacing (in) (X-dir)
                          <select
                            className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                            value={spacingX}
                            onChange={(e) => setSpacingX(parseNumber(e.target.value))}
                          >
                            {spacingOptions.map((s) => (
                              <option key={s} value={s} className="bg-slate-900">
                                {s}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="text-xs text-slate-300">
                          bar (Y-dir)
                          <select
                            className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                            value={barY}
                            onChange={(e) => setBarY(e.target.value as any)}
                          >
                            {BAR_SIZES.map((b) => (
                              <option key={b} value={b} className="bg-slate-900">
                                {b}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="text-xs text-slate-300">
                          spacing (in) (Y-dir)
                          <select
                            className="mt-1 w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm"
                            value={spacingY}
                            onChange={(e) => setSpacingY(parseNumber(e.target.value))}
                          >
                            {spacingOptions.map((s) => (
                              <option key={s} value={s} className="bg-slate-900">
                                {s}
                              </option>
                            ))}
                          </select>
                        </label>
                      </>
                    )}
                  </div>
                </div>

                <div className="pt-2 border-t border-white/10 text-[11px] text-slate-400 space-y-1">
                  <div className="font-semibold text-slate-300">Disclaimers</div>
                  <div>No uplift resistance assumed. Bearing plane is compression-only (p &gt;= 0).</div>
                  <div>Check load directionality and rotation envelope separately if your tables are one-directional.</div>
                </div>
              </div>
            </Panel>
            </div>
          </div>

          {/* CENTER */}
          <div className="lg:col-span-5">
            <Panel
              title="Results"
              subtitle="All 4 service cases are checked; governing case is reported."
              right={
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setTab("summary")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border ${
                      tab === "summary" ? "bg-white/10 border-white/20" : "border-white/10 hover:bg-white/5 text-slate-300"
                    }`}
                  >
                    Summary
                  </button>
                  <button
                    onClick={() => setTab("equations")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border ${
                      tab === "equations" ? "bg-white/10 border-white/20" : "border-white/10 hover:bg-white/5 text-slate-300"
                    }`}
                  >
                    Equations
                  </button>
                </div>
              }
            >
              <div className="space-y-4">
                {calc.errors.length ? (
                  <div className="rounded-xl border border-rose-400/25 bg-rose-500/10 p-3">
                    <div className="font-semibold text-rose-200">Issues</div>
                    <ul className="mt-1 list-disc list-inside text-rose-100/90 text-xs space-y-1">
                      {calc.errors.map((e, i) => (
                        <li key={i}>{e}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {tab === "summary" ? (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                        <div className="text-xs text-slate-300">Footing</div>
                        <div className="mt-1 text-lg font-bold">
                          {fmt(calc.geometry.Bx_ft, 1)} ft x {fmt(calc.geometry.By_ft, 1)} ft
                        </div>
                        <div className="text-xs text-slate-400 mt-1">
                          t = {fmt(calc.geometry.t_ft, 1)} ft; W = {fmt(calc.geometry.weight_kips, 1)} k
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                        <div className="text-xs text-slate-300">Service overall</div>
                        <div className="mt-2 flex items-center justify-between">
                          <div className="text-sm">All 4 cases</div>
                          <Pill label={calc.service.passAll ? "PASS" : "FAIL"} kind={calc.service.passAll ? "pass" : "fail"} />
                        </div>
                        <div className="mt-2 text-xs text-slate-300">
                          Governing bearing: <span className="font-semibold">{CASE_LABEL[calc.service.governing.bearingCase]}</span>
                        </div>
                        <div className="text-xs text-slate-300">
                          Governing overturning: <span className="font-semibold">{CASE_LABEL[calc.service.governing.overturnCase]}</span>
                        </div>
                        <div className="text-xs text-slate-300">
                          Governing sliding: <span className="font-semibold">{CASE_LABEL[calc.service.governing.slidingCase]}</span>
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                        <div className="text-xs text-slate-300">Strength screening</div>
                        <div className="mt-2 text-xs text-slate-300">
                          Flexure X governed by: <span className="font-semibold">{CASE_LABEL[calc.strength.flexureX.governingCase]}</span>
                        </div>
                        <div className="text-xs text-slate-300">
                          Flexure Y governed by: <span className="font-semibold">{CASE_LABEL[calc.strength.flexureY.governingCase]}</span>
                        </div>
                        <div className="mt-2 text-xs text-slate-400">Uses 1.5x service qmax per case (screening).</div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-xs text-slate-300">Service case matrix</div>
                          <div className="text-lg font-bold mt-0.5">Bearing / Overturning / Sliding</div>
                        </div>
                        <Pill label={calc.service.passAll ? "SERVICE OK" : "CHECK"} kind={calc.service.passAll ? "pass" : "warn"} />
                      </div>

                      <div className="mt-3 overflow-auto">
                        <table className="w-full text-sm">
                          <thead className="text-xs text-slate-300">
                            <tr className="border-b border-white/10">
                              <th className="text-left py-2 pr-3">Case</th>
                              <th className="text-right py-2 px-2">qmax (psf)</th>
                              <th className="text-right py-2 px-2">theta (deg)</th>
                              <th className="text-center py-2 px-2">Bearing</th>
                              <th className="text-right py-2 px-2">FSot</th>
                              <th className="text-center py-2 px-2">Overturn</th>
                              <th className="text-right py-2 px-2">FSslide</th>
                              <th className="text-center py-2 pl-2">Sliding</th>
                            </tr>
                          </thead>
                          <tbody className="text-slate-100">
                            {calc.service.cases.map((c) => (
                              <tr key={c.name} className="border-b border-white/5">
                                <td className="py-2 pr-3">{CASE_LABEL[c.name]}</td>
                                <td className="py-2 px-2 text-right font-mono">{fmt0(c.field.qmax_psf)}</td>
                                <td className="py-2 px-2 text-right font-mono">{fmt(c.moment.theta_deg, 0)}</td>
                                <td className="py-2 px-2 text-center">
                                  <Pill label={c.bearingPass ? "PASS" : "FAIL"} kind={c.bearingPass ? "pass" : "fail"} />
                                </td>
                                <td className="py-2 px-2 text-right font-mono">{fmt(c.overturn.FS, 2)}</td>
                                <td className="py-2 px-2 text-center">
                                  <Pill label={c.overturn.pass ? "PASS" : "FAIL"} kind={c.overturn.pass ? "pass" : "fail"} />
                                </td>
                                <td className="py-2 px-2 text-right font-mono">{fmt(c.sliding.FS_gov, 3)}</td>
                                <td className="py-2 pl-2 text-center">
                                  <Pill label={c.sliding.pass ? "PASS" : "FAIL"} kind={c.sliding.pass ? "pass" : "fail"} />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <div className="mt-3 text-[11px] text-slate-400">
                        Note: slewing torque T is included in "operation" and "erection" only. Storm cases exclude T.
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-xs text-slate-300">Strength summary</div>
                          <div className="text-lg font-bold mt-0.5">Flexure + Shear (screening)</div>
                        </div>
                        <Pill label={strengthOk ? "STRENGTH OK" : "CHECK"} kind={strengthOk ? "pass" : "warn"} />
                      </div>

                      <div className="mt-3 grid grid-cols-1 md:grid-cols-5 gap-3">
                        <div className="rounded-xl border border-white/10 bg-white/5 p-3 md:col-span-2">
                          <div className="text-xs text-slate-300">Bottom mat reinforcement</div>
                          <div className="mt-2 text-sm">
                            <div className="font-semibold">
                              X-dir: {calc.strength.flexureX.bar} @ {fmt(calc.strength.flexureX.spacing_in, 1)} in
                            </div>
                            <div className="text-xs text-slate-400">Governing: {CASE_LABEL[calc.strength.flexureX.governingCase]}</div>
                          </div>
                          <div className="mt-2 text-sm">
                            <div className="font-semibold">
                              Y-dir: {calc.strength.flexureY.bar} @ {fmt(calc.strength.flexureY.spacing_in, 1)} in
                            </div>
                            <div className="text-xs text-slate-400">Governing: {CASE_LABEL[calc.strength.flexureY.governingCase]}</div>
                          </div>
                        </div>

                        <div className="rounded-xl border border-white/10 bg-white/5 p-3 md:col-span-3">
                          <div className="text-xs text-slate-300">Demand / capacity</div>
                          <div className="mt-2 overflow-auto">
                            <table className="w-full text-sm">
                              <thead className="text-xs text-slate-300">
                                <tr className="border-b border-white/10">
                                  <th className="text-left py-1 pr-2">Check</th>
                                  <th className="text-right py-1 px-2">Demand</th>
                                  <th className="text-right py-1 px-2">Capacity</th>
                                  <th className="text-right py-1 pl-2">Util</th>
                                </tr>
                              </thead>
                              <tbody className="text-slate-100">
                                <tr className="border-b border-white/5">
                                  <td className="py-1 pr-2">Flexure X (kip-ft/ft)</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.flexureX.Mu_kipft_perft, 2)}</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.flexureX.phiMn_kipft_perft, 2)}</td>
                                  <td className="py-1 pl-2 text-right font-mono">{fmt(calc.strength.flexureX.utilization, 3)}</td>
                                </tr>
                                <tr className="border-b border-white/5">
                                  <td className="py-1 pr-2">Flexure Y (kip-ft/ft)</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.flexureY.Mu_kipft_perft, 2)}</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.flexureY.phiMn_kipft_perft, 2)}</td>
                                  <td className="py-1 pl-2 text-right font-mono">{fmt(calc.strength.flexureY.utilization, 3)}</td>
                                </tr>
                                <tr className="border-b border-white/5">
                                  <td className="py-1 pr-2">One-way shear (k/ft)</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.shear.oneWay.Vu_k_perft, 2)}</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.shear.oneWay.phiVc_k_perft, 2)}</td>
                                  <td className="py-1 pl-2 text-right font-mono">{fmt(calc.strength.shear.oneWay.utilization, 3)}</td>
                                </tr>
                                <tr>
                                  <td className="py-1 pr-2">Punching shear (k)</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.shear.punching.Vu_k, 1)}</td>
                                  <td className="py-1 px-2 text-right font-mono">{fmt(calc.strength.shear.punching.phiVc_k, 1)}</td>
                                  <td className="py-1 pl-2 text-right font-mono">{fmt(calc.strength.shear.punching.utilization, 3)}</td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                    <div className="text-xs text-slate-300 mb-2">Detailed calculations</div>

                    <div className="space-y-4">
                      <CalcBlock title="0. Inputs, assumptions, and constants">
                        <CalcRow
                          label="Units"
                          kind="note"
                          children={
                            <span className="font-mono">
                              Length: ft (geometry), in (rebar); Forces: k; Moments: kip-ft; Pressure: psf
                            </span>
                          }
                        />
                        <CalcRow
                          label="Inputs"
                          kind="note"
                          children={
                            <span className="font-mono">
                              qa={fmt0(calc.inputs.qa_psf)} psf; mu={fmt(calc.inputs.mu, 3)}; FSslide,req={fmt(calc.inputs.FS_slide_req, 2)}; grid n={calc.inputs.gridN}; f&apos;c={fmt0(fc)} psi; fy={fmt(fy, 1)} ksi; cover={fmt(cover, 2)} in
                            </span>
                          }
                        />
                        <CalcRow
                          label="Factors"
                          kind="note"
                          children={
                            <span className="font-mono">
                              gamma={CALC_CONFIG.gammaConc_pcf} pcf; strength factor={CALC_CONFIG.strengthFactor}; phiFlex={CALC_CONFIG.phi_flex}; phiShear={CALC_CONFIG.phi_shear}; FSot,req={CALC_CONFIG.FS_overturn_req}
                            </span>
                          }
                        />
                      </CalcBlock>

                      <CalcBlock title="A. Derived values">
                        <CalcRow label="Area" children={<span className="font-mono">A = Bx * By</span>} />
                        <CalcRow
                          label="Substitute"
                          children={
                            <span className="font-mono">
                              A = {fmt(calc.geometry.Bx_ft, 2)} * {fmt(calc.geometry.By_ft, 2)} ={" "}
                              <ResultBadge>{fmt(calc.geometry.area_ft2, 2)} ft^2</ResultBadge>
                            </span>
                          }
                        />

                        <div className="h-2" />

                        <CalcRow label="Weight" children={<span className="font-mono">W = gamma * Bx * By * t / 1000</span>} />
                        <CalcRow
                          label="Substitute"
                          children={
                            <span className="font-mono">
                              W = 150 * {fmt(calc.geometry.Bx_ft, 2)} * {fmt(calc.geometry.By_ft, 2)} * {fmt(calc.geometry.t_ft, 2)} / 1000 ={" "}
                              <ResultBadge>{fmt(calc.geometry.weight_kips, 2)} k</ResultBadge>
                            </span>
                          }
                        />
                        <CalcRow
                          label="Note"
                          kind="note"
                          children={<span className="font-mono">gamma = 150 pcf; result in kips</span>}
                        />
                      </CalcBlock>

                      <CalcBlock title="B. Service checks (all cases)">
                        <div className="space-y-4">
                          {calc.service.cases.map((c) => {
                            const a = record.cases[c.name];
                            const Tapp = c.name === "operation" || c.name === "erection" ? record.slewTorque_kipft : 0;
                            const leverArm = Math.min(calc.geometry.Bx_ft, calc.geometry.By_ft) / 2;

                            return (
                              <CalcBlock
                                key={c.name}
                                title={CASE_LABEL[c.name]}
                                right={<Pill label={c.passAll ? "PASS" : "CHECK"} kind={c.passAll ? "pass" : "warn"} />}
                              >
                                <CalcRow
                                  label="Inputs"
                                  children={
                                    <span className="font-mono">
                                      P={fmt(a.Pu, 2)} k, M={fmt(a.M_kipft, 2)} kip-ft, V={fmt(a.V_k, 2)} k, T={fmt(Tapp, 2)} kip-ft
                                    </span>
                                  }
                                />

                                <div className="h-2" />

                                <CalcRow label="Equation" children={<span className="font-mono">N_total = P + W</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      N_total = {fmt(a.Pu, 2)} + {fmt(calc.geometry.weight_kips, 2)} = <ResultBadge>{fmt(c.N_total_k, 2)} k</ResultBadge>
                                    </span>
                                  }
                                />

                                <div className="h-2" />

                                <CalcRow label="Equation" children={<span className="font-mono">M_base = M + V * t</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      M_base = {fmt(a.M_kipft, 2)} + {fmt(a.V_k, 2)} * {fmt(calc.geometry.t_ft, 2)} ={" "}
                                      <ResultBadge>{fmt(c.moment.M_base_kipft, 2)} kip-ft</ResultBadge>
                                    </span>
                                  }
                                />

                                <div className="h-2" />

                                <CalcRow label="Bearing plane" children={<span className="font-mono">p(x,y) = max(0, a + b*x + c*y)</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      a={fmt(c.field.a, 1)} psf, b={fmt(c.field.b, 1)} psf/ft, c={fmt(c.field.c, 1)} psf/ft
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="Result"
                                  children={
                                    <span className="font-mono">
                                      qmax=<ResultBadge>{fmt0(c.field.qmax_psf)} psf</ResultBadge>, qmin={fmt0(c.field.qmin_psf)} psf, Acontact={fmt(c.field.Acontact_ft2, 2)} ft^2
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="Check"
                                  children={
                                    <span className="font-mono">
                                      qmax &lt;= qa: {fmt0(c.field.qmax_psf)} &lt;= {fmt0(calc.inputs.qa_psf)} =&gt;{" "}
                                      <ResultBadge>{c.bearingPass ? "PASS" : "FAIL"}</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="Note"
                                  kind="note"
                                  children={
                                    <span className="font-mono">
                                      Grid n={c.field.n} (dx={fmt(c.field.dx_ft, 3)} ft, dy={fmt(c.field.dy_ft, 3)} ft); iterations={c.field.iterations}; worst bearing axis theta={fmt(c.moment.theta_deg, 0)} deg
                                    </span>
                                  }
                                />
                                <details className="mt-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                                  <summary className="cursor-pointer text-xs text-slate-300 select-none">How the bearing plane is solved</summary>
                                  <div className="mt-2 text-[12px] leading-5 text-slate-100">
                                    <div className="font-mono">
                                      Solve for plane coefficients over the compression-only contact cells:
                                      <br />
                                      p(x,y) = a + b*x + c*y, with p &gt;= 0 enforced by iterating contactMask
                                      <br />
                                      S0=ΣdA, Sx=Σx·dA, Sy=Σy·dA, Sxx=Σx²·dA, Syy=Σy²·dA, Sxy=Σx·y·dA
                                      <br />
                                      [S0 Sx Sy; Sx Sxx Sxy; Sy Sxy Syy]·[a b c]^T = [N_lb My_lbft Mx_lbft]^T
                                    </div>
                                    <div className="mt-1 text-xs text-slate-400 font-mono">
                                      x,y are measured from footing centroid; N_lb = N_total(k)*1000; Mx/My in kip-ft are converted to lb-ft.
                                    </div>
                                  </div>
                                </details>

                                <div className="h-2" />

                                <CalcRow label="Overturn" children={<span className="font-mono">FS = (N_total * lever) / M_base</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      FS = ({fmt(c.N_total_k, 2)} * {fmt(leverArm, 2)}) / {fmt(c.moment.M_base_kipft, 2)} ={" "}
                                      <ResultBadge>{fmt(c.overturn.FS, 2)}</ResultBadge>
                                    </span>
                                  }
                                />

                                <div className="h-2" />

                                <CalcRow label="Sliding (simp.)" children={<span className="font-mono">U = V/(mu*N) + |T|/(mu*N*r_eff)</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      U = {fmt(a.V_k, 2)}/({fmt(calc.inputs.mu, 3)}*{fmt(c.sliding.N_contact_k, 2)}) + |{fmt(Tapp, 2)}|/({fmt(calc.inputs.mu, 3)}*{fmt(c.sliding.N_contact_k, 2)}*{fmt(c.sliding.r_eff_ft, 2)}) ={" "}
                                      <ResultBadge>{fmt(c.sliding.interaction, 3)}</ResultBadge> ({c.sliding.simplifiedPass ? "PASS" : "FAIL"})
                                    </span>
                                  }
                                />

                                <div className="h-2" />

                                <CalcRow label="Sliding (det.)" children={<span className="font-mono">FS = 1 / Umax</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      FS = 1 / {fmt(c.sliding.Umax, 4)} = <ResultBadge>{fmt(c.sliding.FS_gov, 3)}</ResultBadge> (req {fmt(calc.inputs.FS_slide_req, 2)}) =&gt;{" "}
                                      <ResultBadge>{c.sliding.pass ? "PASS" : "FAIL"}</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="Note"
                                  kind="note"
                                  children={
                                    <span className="font-mono">
                                      Worst V direction theta={fmt(c.sliding.thetaV_deg, 0)} deg; N(contact)={fmt(c.sliding.N_contact_k, 2)} k; alphaV=V/N={fmt(c.sliding.alphaV, 4)}; kT={fmt(c.sliding.k_torsion, 6)}; gov cell (x,y)=({fmt(c.sliding.govCell?.x_ft ?? 0, 2)}, {fmt(c.sliding.govCell?.y_ft ?? 0, 2)}) ft; U={fmt(c.sliding.govCell?.U ?? 0, 4)}
                                    </span>
                                  }
                                />
                                <details className="mt-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                                  <summary className="cursor-pointer text-xs text-slate-300 select-none">How sliding is checked (detailed)</summary>
                                  <div className="mt-2 text-[12px] leading-5 text-slate-100">
                                    <div className="font-mono">
                                      Per contact cell (i,j):
                                      <br />
                                      capacity = mu * p_ij
                                      <br />
                                      traction from V: tV = alphaV * p_ij * v_hat
                                      <br />
                                      traction from T: tT = kT * p_ij * (-y, x)
                                      <br />
                                      U_ij = |tV + tT| / capacity; Umax governs; FS = 1/Umax
                                    </div>
                                    <div className="mt-1 text-xs text-slate-400 font-mono">
                                      alphaV = V/N(contact); kT = (T*1000) / Σ(p_ij * r_ij^2 * dA); v_hat is rotated to the reported worst-case direction.
                                    </div>
                                  </div>
                                </details>
                              </CalcBlock>
                            );
                          })}
                        </div>
                      </CalcBlock>

                      <CalcBlock title="C. Strength screening (flexure + shear)">
                        {(() => {
                          const fx = calc.strength.flexureX;
                          const fy2 = calc.strength.flexureY;
                          const oneWay = calc.strength.shear.oneWay;
                          const punching = calc.strength.shear.punching;

                          const xBar = getBarProps(fx.bar);
                          const yBar = getBarProps(fy2.bar);
                          const AbX = xBar.area_in2;
                          const AbY = yBar.area_in2;

                          const Bx = calc.geometry.Bx_ft;
                          const By = calc.geometry.By_ft;
                          const t_ft = calc.geometry.t_ft;
                          const t_in = t_ft * 12;
                          const ped = calc.geometry.pedestal_ft;
                          const Lx = (Bx - ped) / 2;
                          const Ly = (By - ped) / 2;

                          const qmaxX = calc.service.cases.find((c) => c.name === fx.governingCase)?.field.qmax_psf ?? 0;
                          const qmaxY = calc.service.cases.find((c) => c.name === fy2.governingCase)?.field.qmax_psf ?? 0;
                          const qmaxV = calc.service.cases.find((c) => c.name === oneWay.governingCase)?.field.qmax_psf ?? 0;
                          const Np = calc.service.cases.find((c) => c.name === punching.governingCase)?.N_total_k ?? 0;
                          const quX = CALC_CONFIG.strengthFactor * qmaxX;
                          const quY = CALC_CONFIG.strengthFactor * qmaxY;
                          const quV = CALC_CONFIG.strengthFactor * qmaxV;

                          const dShear_in = t_in - cover - Math.max(xBar.dia_in, yBar.dia_in) / 2;
                          const dShear_ft = dShear_in / 12;
                          const Lv = Math.max(0, Math.max(Lx - dShear_ft, Ly - dShear_ft));
                          const c_in = ped * 12;
                          return (
                            <div className="space-y-4">
                              <CalcBlock title="Flexure X (strip)">
                                <CalcRow label="d" children={<span className="font-mono">d = 12*t - cover - db/2</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      d = 12*{fmt(t_ft, 2)} - {fmt(cover, 2)} - {fmt(xBar.dia_in, 3)}/2 = <ResultBadge>{fmt(fx.d_in, 3)} in</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="qu (gov)"
                                  children={
                                    <span className="font-mono">
                                      Governing case: {CASE_LABEL[fx.governingCase]} ⇒ qu = {CALC_CONFIG.strengthFactor}*{fmt0(qmaxX)} ={" "}
                                      <ResultBadge>{fmt0(quX)} psf</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Mu" children={<span className="font-mono">Mu = qu * Ly^2 / (2*1000)</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      Mu = {fmt0(quX)} * {fmt(Ly, 2)}^2 / (2*1000) = <ResultBadge>{fmt(fx.Mu_kipft_perft, 2)} kip-ft/ft</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="As_req" children={<span className="font-mono">As_req = <ResultBadge>{fmt(fx.As_req_in2_perft, 3)} in^2/ft</ResultBadge></span>} />
                                <CalcRow label="Equation" children={<span className="font-mono">As_prov = Ab * 12 / s</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      As_prov = {fmt(AbX, 3)} * 12 / {fmt(fx.spacing_in, 1)} ={" "}
                                      <ResultBadge>{fmt(fx.As_prov_in2_perft, 3)} in^2/ft</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Result" children={<span className="font-mono">phiMn = <ResultBadge>{fmt(fx.phiMn_kipft_perft, 2)} kip-ft/ft</ResultBadge>; Util = {fmt(fx.utilization, 3)}</span>} />
                              </CalcBlock>

                              <CalcBlock title="Flexure Y (strip)">
                                <CalcRow label="d" children={<span className="font-mono">d = 12*t - cover - db/2</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      d = 12*{fmt(t_ft, 2)} - {fmt(cover, 2)} - {fmt(yBar.dia_in, 3)}/2 = <ResultBadge>{fmt(fy2.d_in, 3)} in</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="qu (gov)"
                                  children={
                                    <span className="font-mono">
                                      Governing case: {CASE_LABEL[fy2.governingCase]} ⇒ qu = {CALC_CONFIG.strengthFactor}*{fmt0(qmaxY)} ={" "}
                                      <ResultBadge>{fmt0(quY)} psf</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Mu" children={<span className="font-mono">Mu = qu * Lx^2 / (2*1000)</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      Mu = {fmt0(quY)} * {fmt(Lx, 2)}^2 / (2*1000) = <ResultBadge>{fmt(fy2.Mu_kipft_perft, 2)} kip-ft/ft</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="As_req" children={<span className="font-mono">As_req = <ResultBadge>{fmt(fy2.As_req_in2_perft, 3)} in^2/ft</ResultBadge></span>} />
                                <CalcRow label="Equation" children={<span className="font-mono">As_prov = Ab * 12 / s</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      As_prov = {fmt(AbY, 3)} * 12 / {fmt(fy2.spacing_in, 1)} ={" "}
                                      <ResultBadge>{fmt(fy2.As_prov_in2_perft, 3)} in^2/ft</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Result" children={<span className="font-mono">phiMn = <ResultBadge>{fmt(fy2.phiMn_kipft_perft, 2)} kip-ft/ft</ResultBadge>; Util = {fmt(fy2.utilization, 3)}</span>} />
                              </CalcBlock>

                              <details className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                                <summary className="cursor-pointer text-xs text-slate-300 select-none">How flexure As_req / phiMn are computed</summary>
                                <div className="mt-2 text-[12px] leading-5 text-slate-100">
                                  <div className="font-mono">
                                    For a 12-in strip (b=12 in), singly-reinforced:
                                    <br />
                                    a = As*fy / (0.85*fc*b)
                                    <br />
                                    Mn = As*fy*(d - a/2)
                                    <br />
                                    phiMn = phiFlex*Mn
                                    <br />
                                    Solve phiMn(As) = Mu for As_req; then compute phiMn using As_prov.
                                  </div>
                                </div>
                              </details>

                              <CalcBlock title="One-way shear (screening)">
                                <CalcRow label="d" children={<span className="font-mono">d = 12*t - cover - max(db)/2</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      d = 12*{fmt(t_ft, 2)} - {fmt(cover, 2)} - max({fmt(xBar.dia_in, 3)}, {fmt(yBar.dia_in, 3)})/2 ={" "}
                                      <ResultBadge>{fmt(dShear_in, 3)} in</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow
                                  label="qu (gov)"
                                  children={
                                    <span className="font-mono">
                                      Governing case: {CASE_LABEL[oneWay.governingCase]} ⇒ qu = {CALC_CONFIG.strengthFactor}*{fmt0(qmaxV)} ={" "}
                                      <ResultBadge>{fmt0(quV)} psf</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Equation" children={<span className="font-mono">Vu = qu * Lv / 1000, Lv = max(Lx-d, Ly-d)</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      Lv = max({fmt(Lx, 2)} - {fmt(dShear_ft, 3)}, {fmt(Ly, 2)} - {fmt(dShear_ft, 3)}) = <ResultBadge>{fmt(Lv, 3)} ft</ResultBadge>; Vu ={" "}
                                      {fmt0(quV)} * {fmt(Lv, 3)} / 1000 = <ResultBadge>{fmt(oneWay.Vu_k_perft, 2)} k/ft</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Equation" children={<span className="font-mono">phiVc ≈ phiShear * 2*sqrt(fc) * 12 * d / 1000</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      phiVc = {CALC_CONFIG.phi_shear} * 2*sqrt({fmt0(fc)}) * 12 * {fmt(dShear_in, 3)} / 1000 ={" "}
                                      <ResultBadge>{fmt(oneWay.phiVc_k_perft, 2)} k/ft</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Result" children={<span className="font-mono">Util = <ResultBadge>{fmt(oneWay.utilization, 3)}</ResultBadge></span>} />
                              </CalcBlock>

                              <CalcBlock title="Punching shear (screening)">
                                <CalcRow label="Equation" children={<span className="font-mono">bo = 4*(c + d), phiVc ≈ phiShear * 4*sqrt(fc) * bo * d / 1000</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      c = {fmt(c_in, 1)} in, d = {fmt(dShear_in, 3)} in ⇒ bo = <ResultBadge>{fmt(punching.bo_in, 1)} in</ResultBadge>; phiVc ={" "}
                                      <ResultBadge>{fmt(punching.phiVc_k, 1)} k</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Equation" children={<span className="font-mono">Vu ≈ strengthFactor * N_total (governing case)</span>} />
                                <CalcRow
                                  label="Substitute"
                                  children={
                                    <span className="font-mono">
                                      Governing case: {CASE_LABEL[punching.governingCase]} ⇒ Vu = {CALC_CONFIG.strengthFactor}*{fmt(Np, 2)} ={" "}
                                      <ResultBadge>{fmt(punching.Vu_k, 1)} k</ResultBadge>
                                    </span>
                                  }
                                />
                                <CalcRow label="Result" children={<span className="font-mono">Util = <ResultBadge>{fmt(punching.utilization, 3)}</ResultBadge></span>} />
                              </CalcBlock>
                            </div>
                          );
                        })()}
                        <div className="mt-2 text-[11px] text-slate-400">
                          Note: strength uses 1.5× service qmax (screening) and does not replace full code design.
                        </div>
                      </CalcBlock>

                      <details className="rounded-xl border border-white/10 bg-white/5 p-3">
                        <summary className="cursor-pointer text-xs text-slate-300 select-none">Raw calc text (copy/paste)</summary>
                        <pre className="mt-2 whitespace-pre-wrap text-[12px] leading-5 text-slate-100">{calc.eq.join("\n")}</pre>
                      </details>
                    </div>
                  </div>
                )}

                {calc.warnings.length ? (
                  <div className="rounded-xl border border-amber-400/20 bg-amber-500/10 p-3">
                    <div className="font-semibold text-amber-200 text-sm">Warnings / assumptions</div>
                    <ul className="mt-1 list-disc list-inside text-amber-100/90 text-xs space-y-1">
                      {calc.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </Panel>
          </div>

          {/* RIGHT */}
          <div className="lg:col-span-3">
            <Panel title="Load details" subtitle="Database loads per case + slewing torque." right={<Pill label="DB" kind="info" />}>
              <div className="space-y-4">
                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <div className="text-xs text-slate-300">Selected config</div>
                  <div className="mt-1 text-sm font-semibold">{record.model}</div>
                  <div className="text-xs text-slate-400">
                    Jib {fmt(record.jibLength, 1)} ft ({fmt(record.jibLength * FT_TO_M, 1)} m); HUH {fmt(record.hookHeight_ft, 1)} ft ({fmt(record.hookHeight_ft * FT_TO_M, 1)} m)
                  </div>
                  <div className="text-xs text-slate-400">Tower {record.towerSections} sections</div>
                  <div className="mt-2 text-xs text-slate-300">
                    Slewing torque (service): <span className="font-mono">{fmt(record.slewTorque_kipft, 2)}</span> kip-ft
                  </div>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <div className="text-xs text-slate-300">Service case loads</div>
                  <div className="mt-2 overflow-auto">
                    <table className="w-full text-sm">
                      <thead className="text-xs text-slate-300">
                        <tr className="border-b border-white/10">
                          <th className="text-left py-2 pr-3">Load</th>
                          {CASE_ORDER.map((k) => (
                            <th key={k} className="text-right py-2 px-2">
                              {CASE_LABEL[k]}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="text-slate-100">
                        <tr className="border-b border-white/5">
                          <td className="py-2 pr-3 text-slate-300">P (k)</td>
                          {CASE_ORDER.map((k) => (
                            <td key={k} className="py-2 px-2 text-right font-mono">
                              {fmt0(record.cases[k].Pu)}
                            </td>
                          ))}
                        </tr>
                        <tr className="border-b border-white/5">
                          <td className="py-2 pr-3 text-slate-300">M (kip-ft)</td>
                          {CASE_ORDER.map((k) => (
                            <td key={k} className="py-2 px-2 text-right font-mono">
                              {fmt0(record.cases[k].M_kipft)}
                            </td>
                          ))}
                        </tr>
                        <tr className="border-b border-white/5">
                          <td className="py-2 pr-3 text-slate-300">V (k)</td>
                          {CASE_ORDER.map((k) => (
                            <td key={k} className="py-2 px-2 text-right font-mono">
                              {fmt0(record.cases[k].V_k)}
                            </td>
                          ))}
                        </tr>
                        <tr>
                          <td className="py-2 pr-3 text-slate-300">T (kip-ft)</td>
                          {CASE_ORDER.map((k) => {
                            const Tapp = k === "operation" || k === "erection" ? record.slewTorque_kipft : 0;
                            return (
                              <td key={k} className="py-2 px-2 text-right font-mono">
                                {fmt0(Tapp)}
                              </td>
                            );
                          })}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-2 text-[11px] text-slate-400">T is applied to operation + erection only.</div>
                </div>
              </div>
            </Panel>

          </div>
        </div>
      </div>
    </div>
  );
}
