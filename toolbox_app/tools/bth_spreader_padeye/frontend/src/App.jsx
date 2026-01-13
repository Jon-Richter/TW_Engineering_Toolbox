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

function Select({ label, value, onChange, options, disabled = false }) {
  return (
    <div className="row">
      <div>{label}</div>
      <select value={value} onChange={(ev) => onChange(ev.target.value)} disabled={disabled}>
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
  if (raw === "spreader_two_way") return "spreader_two_way";
  return raw === "spreader" ? "spreader" : "padeye";
}

function formatWll(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value ?? "");
  const fixed = n.toFixed(3);
  return fixed.replace(/\.?0+$/, "");
}

function formatDim(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  const fixed = n.toFixed(2);
  return fixed.replace(/\.?0+$/, "");
}

function getOutputNumber(value, fallback = NaN) {
  if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, "value")) {
    const n = Number(value.value);
    return Number.isFinite(n) ? n : fallback;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeBeamDiagrams(beamSolver) {
  if (!beamSolver) return null;
  const diag = beamSolver.diagrams || {};
  if (Array.isArray(diag.x_ft) && (Array.isArray(diag.moment_kipft) || Array.isArray(diag.shear_kip))) {
    return {
      x_ft: diag.x_ft,
      moment_kipft: Array.isArray(diag.moment_kipft) ? diag.moment_kipft : [],
      shear_kip: Array.isArray(diag.shear_kip) ? diag.shear_kip : []
    };
  }
  if (Array.isArray(diag.x_user) && (Array.isArray(diag.moment_user) || Array.isArray(diag.shear_user))) {
    return {
      x_ft: diag.x_user,
      moment_kipft: Array.isArray(diag.moment_user) ? diag.moment_user.map((v) => v / 1000.0) : [],
      shear_kip: Array.isArray(diag.shear_user) ? diag.shear_user.map((v) => v / 1000.0) : []
    };
  }
  if (Array.isArray(beamSolver.x_ft) && (Array.isArray(beamSolver.moment_kipft) || Array.isArray(beamSolver.shear_kip))) {
    return {
      x_ft: beamSolver.x_ft,
      moment_kipft: Array.isArray(beamSolver.moment_kipft) ? beamSolver.moment_kipft : [],
      shear_kip: Array.isArray(beamSolver.shear_kip) ? beamSolver.shear_kip : []
    };
  }
  if (Array.isArray(beamSolver.x_user) && (Array.isArray(beamSolver.moment_user) || Array.isArray(beamSolver.shear_user))) {
    return {
      x_ft: beamSolver.x_user,
      moment_kipft: Array.isArray(beamSolver.moment_user) ? beamSolver.moment_user.map((v) => v / 1000.0) : [],
      shear_kip: Array.isArray(beamSolver.shear_user) ? beamSolver.shear_user.map((v) => v / 1000.0) : []
    };
  }
  return null;
}

function PadeyeDiagram({ H, h, a1, Wb, Wb1, t, Dh, R }) {
  const svgW = 640;
  const svgH = 420;
  const pad = 60;

  const safe = (value, fallback) => {
    const n = Number(value);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };

  const WbVal = safe(Wb, 1);
  const HVal = safe(H, 1);
  const hVal = Math.min(safe(h, HVal * 0.7), HVal);
  const a1Raw = Number(a1);
  const a1Val = Number.isFinite(a1Raw) ? Math.min(Math.max(a1Raw, 0), HVal) : 0;
  const Wb1Val = safe(Wb1, WbVal / 2);
  const RVal = Math.max(Number(R) || 0, 0);
  const yCenter = HVal - RVal;

  const eZ = WbVal / 2 - Wb1Val;
  const baseLeftX = -WbVal / 2;
  const baseRightX = WbVal / 2;

  const tangentPoint = (xEdge) => {
    if (RVal <= 0) return null;
    const dx = xEdge - eZ;
    const dy = a1Val - yCenter;
    const d2 = dx * dx + dy * dy;
    const r2 = RVal * RVal;
    if (d2 <= r2) return null;
    const sqrtTerm = Math.sqrt(Math.max(d2 - r2, 0));
    const coeff1 = r2 / d2;
    const coeff2 = (RVal * sqrtTerm) / d2;
    const px = -dy;
    const py = dx;
    const t1 = { x: eZ + coeff1 * dx + coeff2 * px, y: yCenter + coeff1 * dy + coeff2 * py };
    const t2 = { x: eZ + coeff1 * dx - coeff2 * px, y: yCenter + coeff1 * dy - coeff2 * py };
    const candidates = [t1, t2];
    const valid = candidates.filter((c) => c.y >= a1Val - 1e-6);
    return (valid.length ? valid : candidates).reduce((best, cur) => (cur.y > best.y ? cur : best));
  };

  const edgeXAtHole = (xEdge, tangent, sideSign) => {
    if (hVal <= a1Val + 1e-6 || RVal <= 0) return xEdge;
    if (tangent) {
      if (hVal <= tangent.y + 1e-6 && Math.abs(tangent.y - a1Val) > 1e-6) {
        const t = (hVal - a1Val) / (tangent.y - a1Val);
        return xEdge + t * (tangent.x - xEdge);
      }
    }
    const dy = hVal - yCenter;
    if (Math.abs(dy) <= RVal) {
      const dx = Math.sqrt(Math.max(RVal * RVal - dy * dy, 0));
      return eZ + sideSign * dx;
    }
    return xEdge;
  };

  const leftT = tangentPoint(baseLeftX);
  const rightT = tangentPoint(baseRightX);
  const leftX = edgeXAtHole(baseLeftX, leftT, -1);
  const rightX = edgeXAtHole(baseRightX, rightT, 1);
  const wVal = rightX > leftX ? rightX - leftX : WbVal;
  const DhVal = Math.min(safe(Dh, Math.max(wVal * 0.35, 0.1)), wVal * 0.9);

  const maxY = HVal;
  const minX = Math.min(baseLeftX, leftX, eZ - RVal);
  const maxX = Math.max(baseRightX, rightX, eZ + RVal);
  const spanX = Math.max(maxX - minX, WbVal);

  const scale = Math.min((svgW - 2 * pad) / spanX, (svgH - 2 * pad) / maxY);
  const baseX = pad - minX * scale;
  const baseY = svgH - pad;
  const toX = (x) => baseX + x * scale;
  const toY = (y) => baseY - y * scale;

  const leftBase = { x: toX(baseLeftX), y: toY(0) };
  const rightBase = { x: toX(baseRightX), y: toY(0) };
  const leftA1 = { x: toX(baseLeftX), y: toY(a1Val) };
  const rightA1 = { x: toX(baseRightX), y: toY(a1Val) };
  const centerlineX = toX(0);
  const holeX = toX(eZ);
  const holeY = toY(hVal);
  const holeR = (DhVal * scale) / 2;
  const topY = toY(HVal);

  const leftHole = { x: toX(leftX), y: holeY };
  const rightHole = { x: toX(rightX), y: holeY };

  const outline = [];
  outline.push(`M ${leftBase.x} ${leftBase.y}`);
  outline.push(`L ${rightBase.x} ${rightBase.y}`);
  outline.push(`L ${rightA1.x} ${rightA1.y}`);

  if (RVal > 0 && leftT && rightT) {
    const rightTsvg = { x: toX(rightT.x), y: toY(rightT.y) };
    const leftTsvg = { x: toX(leftT.x), y: toY(leftT.y) };
    outline.push(`L ${rightTsvg.x} ${rightTsvg.y}`);

    const TAU = Math.PI * 2;
    const norm = (ang) => {
      let a = ang % TAU;
      if (a < 0) a += TAU;
      return a;
    };
    const start = norm(Math.atan2(rightT.y - yCenter, rightT.x - eZ));
    const end = norm(Math.atan2(leftT.y - yCenter, leftT.x - eZ));
    const target = norm(Math.PI / 2);
    const ccwContains =
      start <= end ? target >= start && target <= end : target >= start || target <= end;
    const steps = 20;
    if (ccwContains) {
      const sweep = start <= end ? end - start : TAU - start + end;
      for (let i = 1; i <= steps; i += 1) {
        const ang = start + (sweep * i) / steps;
        const x = eZ + RVal * Math.cos(ang);
        const y = yCenter + RVal * Math.sin(ang);
        outline.push(`L ${toX(x)} ${toY(y)}`);
      }
    } else {
      const sweep = start >= end ? start - end : start + (TAU - end);
      for (let i = 1; i <= steps; i += 1) {
        const ang = start - (sweep * i) / steps;
        const x = eZ + RVal * Math.cos(ang);
        const y = yCenter + RVal * Math.sin(ang);
        outline.push(`L ${toX(x)} ${toY(y)}`);
      }
    }

    outline.push(`L ${leftTsvg.x} ${leftTsvg.y}`);
  } else {
    const topRight = { x: toX(baseRightX), y: toY(HVal) };
    const topLeft = { x: toX(baseLeftX), y: toY(HVal) };
    outline.push(`L ${topRight.x} ${topRight.y}`);
    outline.push(`L ${topLeft.x} ${topLeft.y}`);
  }

  outline.push(`L ${leftA1.x} ${leftA1.y}`);
  outline.push(`L ${leftBase.x} ${leftBase.y}`);
  outline.push("Z");

  const leftDimX = leftBase.x - 26;
  const leftDimX2 = leftBase.x - 12;
  const a1DimX = leftBase.x - 40;
  const wDimY = holeY - 26;
  const WbDimY = baseY + 28;
  const DhDimY = holeY + holeR + 22;
  const rDimX = Math.max(rightHole.x, rightBase.x) + 26;
  const rTopY = topY;
  const rCenterY = toY(yCenter);
  const tLabelX = rightBase.x + 18;
  const tLabelY = baseY - (HVal * scale) * 0.45;

  return (
    <svg className="padeye-diagram" viewBox={`0 0 ${svgW} ${svgH}`} role="img">
      <defs>
        <marker
          id="arrow"
          markerWidth="8"
          markerHeight="8"
          refX="4"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L8,4 L0,8 Z" fill="#e2e8f0" />
        </marker>
      </defs>
      <rect x="0" y="0" width={svgW} height={svgH} fill="none" />
      <line
        x1={centerlineX}
        y1={topY - 18}
        x2={centerlineX}
        y2={baseY + 18}
        stroke="#94a3b8"
        strokeDasharray="6 6"
        strokeWidth="1"
      />
      <line
        x1={leftBase.x - 10}
        y1={holeY}
        x2={rightBase.x + 10}
        y2={holeY}
        stroke="#94a3b8"
        strokeDasharray="6 6"
        strokeWidth="1"
      />
      <path d={outline.join(" ")} fill="none" stroke="#e2e8f0" strokeWidth="2" />
      <circle cx={holeX} cy={holeY} r={holeR} fill="none" stroke="#e2e8f0" strokeWidth="2" />

      <line x1={leftDimX} y1={baseY} x2={leftDimX} y2={topY} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
      <text x={leftDimX - 6} y={(baseY + topY) / 2} textAnchor="end">
        {`H ${formatDim(HVal)} in`}
      </text>

      <line x1={leftDimX2} y1={baseY} x2={leftDimX2} y2={holeY} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
      <text x={leftDimX2 - 6} y={(baseY + holeY) / 2} textAnchor="end">
        {`h ${formatDim(hVal)} in`}
      </text>

      {a1Val > 0 ? (
        <g>
          <line x1={a1DimX} y1={baseY} x2={a1DimX} y2={leftA1.y} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
          <text x={a1DimX - 6} y={(baseY + leftA1.y) / 2} textAnchor="end">
            {`a1 ${formatDim(a1Val)} in`}
          </text>
        </g>
      ) : null}

      <line x1={leftBase.x} y1={WbDimY} x2={rightBase.x} y2={WbDimY} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
      <text x={(leftBase.x + rightBase.x) / 2} y={WbDimY + 16} textAnchor="middle">
        {`Wb ${formatDim(WbVal)} in`}
      </text>

      <line x1={leftHole.x} y1={wDimY} x2={rightHole.x} y2={wDimY} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
      <text x={(leftHole.x + rightHole.x) / 2} y={wDimY - 8} textAnchor="middle">
        {`w ${formatDim(wVal)} in`}
      </text>

      <line x1={holeX - holeR} y1={DhDimY} x2={holeX + holeR} y2={DhDimY} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
      <text x={holeX} y={DhDimY + 16} textAnchor="middle">
        {`Dh ${formatDim(DhVal)} in`}
      </text>

      {RVal > 0 ? (
        <g>
          <line x1={rDimX} y1={rTopY} x2={rDimX} y2={rCenterY} stroke="#e2e8f0" strokeWidth="1.2" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
          <text x={rDimX + 6} y={(rTopY + rCenterY) / 2} textAnchor="start">
            {`R ${formatDim(RVal)} in`}
          </text>
        </g>
      ) : null}

      <line x1={rightBase.x + 4} y1={tLabelY} x2={rightBase.x + 22} y2={tLabelY - 8} stroke="#e2e8f0" strokeWidth="1.2" markerEnd="url(#arrow)" />
      <text x={tLabelX + 6} y={tLabelY - 12} textAnchor="start">
        {`t ${formatDim(t)} in`}
      </text>
    </svg>
  );
}

function SpreaderTwoWayDiagram({
  lengthFt,
  padeyeEdgeFt,
  padeyeHeightIn,
  pointLoads,
  shapeLabel,
  shapeDepthIn,
  selfWeightKipFt,
  slingAngleMinDeg,
  outputs
}) {
  const svgW = 860;
  const svgH = 520;
  const pad = 60;

  const L = Number(lengthFt) || 0;
  const edge = Number(padeyeEdgeFt) || 0;
  const padeyeHeightFt = (Number(padeyeHeightIn) || 0) / 12.0;
  const xLeft = edge;
  const xRight = Math.max(L - edge, edge);
  const cgX = getOutputNumber(outputs && outputs.cg_x);
  const totalLoadOut = getOutputNumber(outputs && outputs.total_load);
  const angleLeftOut = getOutputNumber(outputs && outputs.sling_angle_left);
  const angleRightOut = getOutputNumber(outputs && outputs.sling_angle_right);
  const lenLeftOut = getOutputNumber(outputs && outputs.sling_length_left);
  const lenRightOut = getOutputNumber(outputs && outputs.sling_length_right);

  const loads = Array.isArray(pointLoads) ? pointLoads : [];
  const loadSum = loads.reduce((sum, load) => sum + (Number(load.P_kip) || 0), 0);
  const loadMoment = loads.reduce((sum, load) => sum + (Number(load.P_kip) || 0) * (Number(load.x_ft) || 0), 0);
  const selfWeight = Number(selfWeightKipFt) || 0;
  const totalLoadCalc = loadSum + selfWeight * L;
  const cgCalc = totalLoadCalc > 0 ? (loadMoment + selfWeight * L * (L / 2.0)) / totalLoadCalc : NaN;

  const cgVal = Number.isFinite(cgX) ? cgX : Number.isFinite(cgCalc) ? cgCalc : L / 2.0;
  const dLeft = Math.abs(cgVal - xLeft);
  const dRight = Math.abs(xRight - cgVal);
  const dLong = Math.max(dLeft, dRight);
  const slingAngleDegNum = Number(slingAngleMinDeg);
  const slingAngleRad = Number.isFinite(slingAngleDegNum) ? (slingAngleDegNum * Math.PI) / 180 : 0;
  const hookHeight = dLong > 0 ? dLong * Math.tan(slingAngleRad) : 0.0;
  const angleLeftCalc = hookHeight > 0 || dLeft > 0 ? (Math.atan2(hookHeight, dLeft || 1e-9) * 180) / Math.PI : 90.0;
  const angleRightCalc = hookHeight > 0 || dRight > 0 ? (Math.atan2(hookHeight, dRight || 1e-9) * 180) / Math.PI : 90.0;
  const lenLeftCalc = Math.hypot(hookHeight, dLeft);
  const lenRightCalc = Math.hypot(hookHeight, dRight);

  const angleLeft = Number.isFinite(angleLeftOut) ? angleLeftOut : angleLeftCalc;
  const angleRight = Number.isFinite(angleRightOut) ? angleRightOut : angleRightCalc;
  const lenLeft = Number.isFinite(lenLeftOut) ? lenLeftOut : lenLeftCalc;
  const lenRight = Number.isFinite(lenRightOut) ? lenRightOut : lenRightCalc;
  const hookY = padeyeHeightFt + (Number.isFinite(hookHeight) ? hookHeight : 0);

  const minY = -1.4;
  const maxY = Math.max(hookY + 1.2, padeyeHeightFt + 2.2);
  const spanX = Math.max(L, 1);
  const spanY = Math.max(maxY - minY, 1);
  const scale = Math.min((svgW - 2 * pad) / spanX, (svgH - 2 * pad) / spanY);

  const toX = (x) => pad + x * scale;
  const toY = (y) => svgH - pad - (y - minY) * scale;

  const beamY = 0;
  const loadStartY = Math.max(padeyeHeightFt + 1.0, padeyeHeightFt + hookHeight * 0.35);
  const cgArrowStartY = loadStartY + 0.6;

  const totalLoadLabel = Number.isFinite(totalLoadOut)
    ? formatDim(totalLoadOut)
    : Number.isFinite(totalLoadCalc)
    ? formatDim(totalLoadCalc)
    : "-";
  const markerId = "tw-arrow";
  const dashedId = "tw-dash";

  return (
    <svg className="padeye-diagram" viewBox={`0 0 ${svgW} ${svgH}`} role="img">
      <defs>
        <marker
          id={markerId}
          markerWidth="8"
          markerHeight="8"
          refX="4"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L8,4 L0,8 Z" fill="#e2e8f0" />
        </marker>
        <pattern id={dashedId} width="6" height="6" patternUnits="userSpaceOnUse">
          <path d="M0 0 L6 0" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 6" />
        </pattern>
      </defs>

      {/* beam */}
      <line x1={toX(0)} y1={toY(beamY)} x2={toX(L)} y2={toY(beamY)} stroke="#e2e8f0" strokeWidth="4" />
      {/* padeye holes */}
      <circle cx={toX(xLeft)} cy={toY(padeyeHeightFt)} r="5" fill="#e2e8f0" />
      <circle cx={toX(xRight)} cy={toY(padeyeHeightFt)} r="5" fill="#e2e8f0" />

      {/* length dimension */}
      <line
        x1={toX(0)}
        y1={toY(minY + 0.3)}
        x2={toX(L)}
        y2={toY(minY + 0.3)}
        stroke="#e2e8f0"
        strokeWidth="1.2"
        markerStart={`url(#${markerId})`}
        markerEnd={`url(#${markerId})`}
      />
      <text x={toX(L / 2)} y={toY(minY + 0.1)} textAnchor="middle">
        {`L ${formatDim(L)} ft`}
      </text>

      {/* member size */}
      <text x={toX(0)} y={toY(beamY + 0.35)} textAnchor="start">
        {shapeDepthIn
          ? `${shapeLabel} (d ${formatDim(shapeDepthIn)} in)`
          : `${shapeLabel}`}
      </text>

      {/* sling geometry */}
      <circle cx={toX(cgVal)} cy={toY(hookY)} r="5" fill="#f59e0b" />
      <line x1={toX(xLeft)} y1={toY(padeyeHeightFt)} x2={toX(cgVal)} y2={toY(hookY)} stroke="#f59e0b" strokeWidth="2" />
      <line x1={toX(xRight)} y1={toY(padeyeHeightFt)} x2={toX(cgVal)} y2={toY(hookY)} stroke="#f59e0b" strokeWidth="2" />

      <text x={toX((xLeft + cgVal) / 2)} y={toY((padeyeHeightFt + hookY) / 2 + 0.2)} textAnchor="start">
        {Number.isFinite(angleLeft) ? `${formatDim(angleLeft)}°` : ""}
      </text>
      <text x={toX((xLeft + cgVal) / 2)} y={toY((padeyeHeightFt + hookY) / 2 - 0.05)} textAnchor="start">
        {Number.isFinite(lenLeft) ? `${formatDim(lenLeft)} ft` : ""}
      </text>
      <text x={toX((xRight + cgVal) / 2)} y={toY((padeyeHeightFt + hookY) / 2 + 0.2)} textAnchor="end">
        {Number.isFinite(angleRight) ? `${formatDim(angleRight)}°` : ""}
      </text>
      <text x={toX((xRight + cgVal) / 2)} y={toY((padeyeHeightFt + hookY) / 2 - 0.05)} textAnchor="end">
        {Number.isFinite(lenRight) ? `${formatDim(lenRight)} ft` : ""}
      </text>

      {/* point loads */}
      {loads.map((load, idx) => {
        const x = Number(load.x_ft) || 0;
        const P = Number(load.P_kip) || 0;
        const labelY = loadStartY + 0.4 + (idx % 2) * 0.5;
        return (
          <g key={`load-${idx}`}>
            <line
              x1={toX(x)}
              y1={toY(loadStartY)}
              x2={toX(x)}
              y2={toY(beamY)}
              stroke="#e2e8f0"
              strokeWidth="1.6"
              markerEnd={`url(#${markerId})`}
            />
            <text x={toX(x)} y={toY(labelY)} textAnchor="middle">
              {`P ${formatDim(P)} kip @ ${formatDim(x)} ft`}
            </text>
          </g>
        );
      })}

      {/* centroid arrow */}
      <line
        x1={toX(cgVal)}
        y1={toY(cgArrowStartY)}
        x2={toX(cgVal)}
        y2={toY(beamY)}
        stroke="#94a3b8"
        strokeWidth="1.6"
        strokeDasharray="6 6"
        markerEnd={`url(#${markerId})`}
      />
      <text x={toX(cgVal)} y={toY(cgArrowStartY + 0.35)} textAnchor="middle">
        {`CG ${formatDim(cgVal)} ft, W ${totalLoadLabel} kip`}
      </text>
    </svg>
  );
}

function LineDiagram({ title, xVals, yVals, units, color = "#38bdf8" }) {
  const svgW = 860;
  const svgH = 140;
  const pad = 50;
  const xsRaw = Array.isArray(xVals) ? xVals : [];
  const ysRaw = Array.isArray(yVals) ? yVals : [];
  const pairs = xsRaw
    .map((x, i) => [Number(x), Number(ysRaw[i])])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!pairs.length) {
    return (
      <div style={{ padding: "10px 16px" }}>
        <div className="section-title">{title}</div>
        <div className="sub">No diagram data available.</div>
      </div>
    );
  }

  const xs = pairs.map((pair) => pair[0]);
  const ys = pairs.map((pair) => pair[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const toX = (x) => pad + ((x - minX) / spanX) * (svgW - 2 * pad);
  const toY = (y) => svgH - pad - ((y - minY) / spanY) * (svgH - 2 * pad);

  const points = xs.map((x, i) => `${toX(x)},${toY(ys[i])}`).join(" ");
  const zeroY = minY <= 0 && maxY >= 0 ? toY(0) : null;

  return (
    <div style={{ marginTop: 8 }}>
      <div className="section-title">{title}</div>
      <svg viewBox={`0 0 ${svgW} ${svgH}`} role="img">
        {zeroY !== null ? (
          <line x1={pad} y1={zeroY} x2={svgW - pad} y2={zeroY} stroke="#94a3b8" strokeDasharray="5 6" />
        ) : null}
        <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
        <text x={svgW - pad} y={pad - 8} textAnchor="end">{units}</text>
      </svg>
    </div>
  );
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
    a1: 0.0,
    Wb: 8.0,
    Wb1: 4.0,
    t: 1.0,
    Dh: 1.4,
    Dp: 1.5,
    R: 2.0,
    tcheek: 0.0,
    ex: 0.0,
    ey: 0.0,
    weld_type: "Fillet",
    weld_group: "Long Sides Only",
    weld_size_16: 4,
    weld_exx_ksi: 70.0
  });

  const [shackles, setShackles] = useState([]);
  const [shackleId, setShackleId] = useState("custom");
  const [spreaderShapes, setSpreaderShapes] = useState([]);
  const [spreaderShapeLoading, setSpreaderShapeLoading] = useState(false);

  const [spr, setSpr] = useState({
    shape: "W10X49",
    span_L_ft: 20.0,
    Lb_ft: 20.0,
    KL_ft: 20.0,
    ey: 0.0,
    mx_includes_total: false,
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

  const [twoWay, setTwoWay] = useState({
    shape: "W10X49",
    length_ft: 20.0,
    padeye_edge_ft: 2.0,
    padeye_height_in: 6.0,
    sling_angle_deg: 45.0,
    point_loads: [{ x_ft: 10.0, P_kip: 10.0 }]
  });

  const [busyByMode, setBusyByMode] = useState({ padeye: false, spreader: false, spreader_two_way: false });
  const [resultsByMode, setResultsByMode] = useState({ padeye: null, spreader: null, spreader_two_way: null });
  const [errorByMode, setErrorByMode] = useState({ padeye: null, spreader: null, spreader_two_way: null });
  const [noteByMode, setNoteByMode] = useState({ padeye: null, spreader: null, spreader_two_way: null });
  const [reportVisibleByMode, setReportVisibleByMode] = useState({ padeye: false, spreader: false, spreader_two_way: false });
  const [reportTokenByMode, setReportTokenByMode] = useState({ padeye: 0, spreader: 0, spreader_two_way: 0 });

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

  useEffect(() => {
    let active = true;
    setSpreaderShapeLoading(true);
    fetch(`${apiBase}/api/spreader_shapes`)
      .then((res) => res.json())
      .then((json) => {
        if (!active) return;
        if (json.ok && Array.isArray(json.items)) {
          setSpreaderShapes(json.items);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!active) return;
        setSpreaderShapeLoading(false);
      });
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

  const spreaderShapeOptions = useMemo(() => {
    if (spreaderShapes.length) {
      return spreaderShapes.map((item) => ({ value: item.label, label: item.label }));
    }
    const fallback = spr.shape || "Loading...";
    return [{ value: fallback, label: fallback }];
  }, [spreaderShapes, spr.shape]);

  useEffect(() => {
    if (!spreaderShapes.length) return;
    const labels = new Set(spreaderShapes.map((item) => item.label));
    if (!labels.has(spr.shape)) {
      setSpr((prev) => ({ ...prev, shape: spreaderShapes[0].label }));
    }
  }, [spreaderShapes, spr.shape]);

  useEffect(() => {
    if (!spreaderShapes.length) return;
    const labels = new Set(spreaderShapes.map((item) => item.label));
    if (!labels.has(twoWay.shape)) {
      setTwoWay((prev) => ({ ...prev, shape: spreaderShapes[0].label }));
    }
  }, [spreaderShapes, twoWay.shape]);

  const selectedShackle = shackleId !== "custom" ? shackleMap.get(shackleId) : null;

  useEffect(() => {
    if (!selectedShackle) return;
    const e = Number(selectedShackle.eccentricity_in || 0);
    const theta = Number(pad.theta_deg || 0);
    const thetaRad = (theta * Math.PI) / 180.0;
    const ex = e * Math.cos(thetaRad);
    const ey = e * Math.sin(thetaRad);
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
      const payload =
        activeMode === "padeye"
          ? { ...base, ...pad }
          : activeMode === "spreader"
          ? { ...base, ...spr }
          : { ...base, ...twoWay };
      const response = await fetch(`${apiBase}/api/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal
      });
      const json = await response.json();
      if (!json.ok) throw new Error(json.error || "Solve failed");
      setResultsByMode((prev) => ({ ...prev, [activeMode]: json.results }));
      return json.results;
    } catch (err) {
      const message =
        err && err.name === "AbortError"
          ? "Solve timed out after 30s. Check backend logs for errors."
          : String((err && err.message) || err || "Unknown error");
      setErrorByMode((prev) => ({ ...prev, [activeMode]: message }));
      return null;
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      setBusyByMode((prev) => ({ ...prev, [activeMode]: false }));
    }
  }

  // Auto-run when inputs change (debounced)
  useEffect(() => {
    const payloadKey = JSON.stringify({ mode, designCategory, Fy, Fu, impact, pad, spr, twoWay, shackleId });
    const debounceMs = 400;

    function inputsAreValid(targetMode) {
      if (typeof Fy === "undefined" || typeof Fu === "undefined") return false;
      if (!Number.isFinite(Number(Fy)) || !Number.isFinite(Number(Fu))) return false;
      if (!designCategory) return false;
      if (targetMode === "padeye") {
        const required = [
          pad.P,
          pad.h,
          pad.t,
          pad.Dp,
          pad.Dh,
          pad.Wb,
          pad.Wb1,
          pad.H,
          pad.R,
          pad.a1,
          pad.weld_size_16,
          pad.weld_exx_ksi
        ];
        for (const v of required) {
          if (!Number.isFinite(Number(v))) return false;
        }
        if (!pad.weld_type || !pad.weld_group) return false;
      } else if (targetMode === "spreader") {
        if (!spr || !spr.shape) return false;
        const required = [spr.span_L_ft, spr.Lb_ft, spr.KL_ft, spr.ey];
        for (const v of required) {
          if (!Number.isFinite(Number(v))) return false;
        }
      } else if (targetMode === "spreader_two_way") {
        if (!twoWay || !twoWay.shape) return false;
        const required = [
          twoWay.length_ft,
          twoWay.padeye_edge_ft,
          twoWay.padeye_height_in,
          twoWay.sling_angle_deg
        ];
        for (const v of required) {
          if (!Number.isFinite(Number(v))) return false;
        }
        if (Number(twoWay.padeye_edge_ft) * 2 >= Number(twoWay.length_ft)) return false;
        const loads = Array.isArray(twoWay.point_loads) ? twoWay.point_loads : [];
        for (const load of loads) {
          if (!Number.isFinite(Number(load.x_ft)) || !Number.isFinite(Number(load.P_kip))) return false;
          if (Number(load.x_ft) > Number(twoWay.length_ft) + 1e-9) return false;
        }
      }
      return true;
    }

    const valid = inputsAreValid(mode);
    setReportVisibleByMode((prev) =>
      prev[mode] ? { ...prev, [mode]: false } : prev
    );
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
  }, [JSON.stringify({ mode, designCategory, Fy, Fu, impact, pad, spr, twoWay, shackleId })]);

  function buildReportUrl(targetMode, token) {
    const stamp = token ? `&t=${token}` : "";
    return `${apiBase}/api/report.html?mode=${targetMode}${stamp}`;
  }

  function handleViewReport(ev) {
    const activeMode = mode;
    if (busyByMode[activeMode]) {
      if (ev) ev.preventDefault();
      setNoteByMode((prev) => ({ ...prev, [activeMode]: "Analysis running. Try again in a moment." }));
      return;
    }
    const token = Date.now();
    const url = buildReportUrl(activeMode, token);
    setReportTokenByMode((prev) => ({ ...prev, [activeMode]: token }));
    setReportVisibleByMode((prev) => ({ ...prev, [activeMode]: true }));
    if (ev && ev.currentTarget) {
      ev.currentTarget.href = url;
    }
  }

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

  function updateTwoWayLoad(index, field, value) {
    setTwoWay((prev) => {
      const next = Array.isArray(prev.point_loads) ? [...prev.point_loads] : [];
      const current = next[index] || { x_ft: 0, P_kip: 0 };
      next[index] = { ...current, [field]: Number(value) };
      return { ...prev, point_loads: next };
    });
  }

  function addTwoWayLoad() {
    setTwoWay((prev) => ({
      ...prev,
      point_loads: [...(prev.point_loads || []), { x_ft: 0, P_kip: 0 }]
    }));
  }

  function removeTwoWayLoad(index) {
    setTwoWay((prev) => {
      const next = Array.isArray(prev.point_loads) ? [...prev.point_loads] : [];
      next.splice(index, 1);
      return { ...prev, point_loads: next.length ? next : [{ x_ft: 0, P_kip: 0 }] };
    });
  }

  function pushTwoWayToSpreader() {
    const results = resultsByMode.spreader_two_way;
    const outputs = results && results.key_outputs ? results.key_outputs : null;
    if (!outputs) {
      setNoteByMode((prev) => ({ ...prev, spreader_two_way: "Run the two-way analysis first." }));
      return;
    }
    const getVal = (key) => {
      const entry = outputs[key];
      if (entry && typeof entry === "object" && entry.value !== undefined) return Number(entry.value);
      const v = Number(entry);
      return Number.isFinite(v) ? v : 0;
    };
    const spanL = Number(twoWay.length_ft) || 0;
    const spacing = spanL - 2.0 * (Number(twoWay.padeye_edge_ft) || 0);
    const Mtotal = getVal("max_moment_total") || getVal("max_moment");
    const Vmax = getVal("max_shear");
    const Paxial = getVal("axial_compression");

    setSpr((prev) => ({
      ...prev,
      shape: twoWay.shape || prev.shape,
      span_L_ft: spanL,
      Lb_ft: spanL,
      KL_ft: spacing > 0 ? spacing : prev.KL_ft,
      V_kip: Vmax,
      P_kip: Paxial,
      Mx_app_kipft: Mtotal,
      ey: Number(twoWay.padeye_height_in) || prev.ey,
      mx_includes_total: true,
      include_self_weight: false
    }));
    setMode("spreader");
    setNoteByMode((prev) => ({ ...prev, spreader: "Two-way results pushed into spreader tab." }));
  }

  const results = resultsByMode[mode];
  const error = errorByMode[mode];
  const note = noteByMode[mode];
  const busy = busyByMode[mode];
  const reportToken = reportTokenByMode[mode];
  const reportUrl = results ? buildReportUrl(mode, reportToken) : "";
  const reportVisible = reportVisibleByMode[mode];
  const artifacts = results && results.artifacts ? results.artifacts : {};
  const checks = results && Array.isArray(results.checks) ? results.checks : [];
  const hasChecks = checks.length > 0;
  const twoWayOutputs = resultsByMode.spreader_two_way && resultsByMode.spreader_two_way.key_outputs ? resultsByMode.spreader_two_way.key_outputs : null;
  const twoWayTables = resultsByMode.spreader_two_way && resultsByMode.spreader_two_way.tables ? resultsByMode.spreader_two_way.tables : null;
  const twoWayBeamSolver = twoWayTables && twoWayTables.two_way ? twoWayTables.two_way.beam_solver : null;
  const twoWayDiagrams = normalizeBeamDiagrams(twoWayBeamSolver);
  const padeyeLimitStateOrder = [
    "Allowable Tensile Strength Through Pin Hole, Pt",
    "Allowable Single Plane Fracture Strength, Pb",
    "Allowable Double Plane Shear Strength, Pv",
    "Pin Bearing Stress",
    "Shear at Base of Padeye",
    "In-Plane Bending at Base of Padeye",
    "Out-of-plane Bending at Base of Padeye",
    "Tension at Base of Padeye",
    "Combined Stress at Base of Padeye",
    "Weld Group Combined Stress"
  ];
  const orderedChecks = (() => {
    if (!hasChecks) return [];
    if (mode !== "padeye") return checks;
    const orderMap = new Map(padeyeLimitStateOrder.map((label, idx) => [label, idx]));
    return [...checks].sort((a, b) => {
      const ai = orderMap.has(a.label) ? orderMap.get(a.label) : padeyeLimitStateOrder.length;
      const bi = orderMap.has(b.label) ? orderMap.get(b.label) : padeyeLimitStateOrder.length;
      return ai - bi;
    });
  })();
  const outputLabelMap = {
    Px: "Load Component Px",
    Py: "Load Component Py",
    Pz: "Load Component Pz",
    governing_ratio: "Governing Utilization",
    governing_check: "Governing Limit State",
    support_spacing: "Padeye Spacing",
    total_load: "Total Vertical Load",
    cg_x: "Load CG (from left)",
    R_left: "Left Padeye Reaction",
    R_right: "Right Padeye Reaction",
    sling_angle_left: "Left Sling Angle",
    sling_angle_right: "Right Sling Angle",
    sling_length_left: "Left Sling Length",
    sling_length_right: "Right Sling Length",
    sling_tension_left: "Left Sling Tension",
    sling_tension_right: "Right Sling Tension",
    axial_compression: "Axial Compression (from slings)",
    max_shear: "Max Shear (strong axis)",
    max_moment: "Max Moment (vertical)",
    max_moment_total: "Max Moment (incl. eccentric)",
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
    ],
    spreader_two_way: [
      "support_spacing",
      "total_load",
      "cg_x",
      "R_left",
      "R_right",
      "sling_angle_left",
      "sling_angle_right",
      "sling_length_left",
      "sling_length_right",
      "sling_tension_left",
      "sling_tension_right",
      "axial_compression",
      "max_shear",
      "max_moment",
      "max_moment_total"
    ]
  };
  const hiddenOutputs = new Set(["governing_step"]);

  function formatOutputValue(key, value) {
    if (key === "governing_ratio") {
      const numeric = typeof value === "number" ? value : Number(value);
      if (Number.isFinite(numeric)) return numeric.toFixed(3);
      if (value && typeof value === "object" && Number.isFinite(Number(value.value))) {
        const units = value.units ? ` ${value.units}` : "";
        return `${Number(value.value).toFixed(3)}${units}`;
      }
    }
    if (value && typeof value === "object") {
      if (Object.prototype.hasOwnProperty.call(value, "value")) {
        return `${value.value} ${value.units || ""}`.trim();
      }
    }
    return String(value);
  }

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
  const modeLabel =
    mode === "padeye"
      ? "Padeye"
      : mode === "spreader_two_way"
      ? "Spreader (Two-way)"
      : "Spreader bar";
  const weldSize16 = Number(pad.weld_size_16);
  const weldSizeIn = Number.isFinite(weldSize16) ? weldSize16 / 16.0 : 0;
  const weldSizeInDisplay = Number.isFinite(weldSizeIn) ? formatDim(weldSizeIn) : "-";
  const effectiveThroatValue = (() => {
    if (pad.weld_type === "CJP") return null;
    if (pad.weld_type === "Fillet") return 0.707 * weldSizeIn;
    if (pad.weld_type === "PJP 60° Bevel") return Number(pad.t) || 0;
    if (pad.weld_type === "PJP 45° Bevel") return Math.max((Number(pad.t) || 0) - 0.125, 0);
    return null;
  })();
  const effectiveThroatDisplay =
    effectiveThroatValue === null ? "N/A" : formatDim(effectiveThroatValue);
  const weldTypeOptions = [
    { value: "Fillet", label: "Fillet" },
    { value: "PJP 60° Bevel", label: "PJP 60° Bevel" },
    { value: "PJP 45° Bevel", label: "PJP 45° Bevel" },
    { value: "CJP", label: "CJP" }
  ];
  const weldGroupOptions =
    pad.weld_type === "Fillet"
      ? [
          { value: "Long Sides Only", label: "Long Sides Only" },
          { value: "All Around", label: "All Around" }
        ]
      : [{ value: "Long Sides Only", label: "Long Sides Only" }];

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
          <div className="tag">Mode: {modeLabel}</div>
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
            <button
              className={`tab${mode === "spreader_two_way" ? " active" : ""}`}
              onClick={() => setMode("spreader_two_way")}
              type="button"
            >
              Spreader (Two-way)
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
                label="Straight corner height a1"
                units="in"
                value={pad.a1}
                onChange={(v) => setPad({ ...pad, a1: Number(v) })}
              />
              <Field
                label="Width at base Wb"
                units="in"
                value={pad.Wb}
                onChange={(v) => setPad({ ...pad, Wb: Number(v) })}
              />
              <Field
                label="Hole center to edge Wb1"
                units="in"
                value={pad.Wb1}
                onChange={(v) => setPad({ ...pad, Wb1: Number(v) })}
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
                label="Top radius R"
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
              <div className="section-title">Base weld</div>
              <Select
                label="Weld type"
                value={pad.weld_type}
                onChange={(value) =>
                  setPad((prev) => ({
                    ...prev,
                    weld_type: value,
                    weld_group: value === "Fillet" ? prev.weld_group : "Long Sides Only"
                  }))
                }
                options={weldTypeOptions}
              />
              <Select
                label="Weld group"
                value={pad.weld_group}
                onChange={(value) => setPad({ ...pad, weld_group: value })}
                options={weldGroupOptions}
              />
              <Field
                label="Weld size (sixteenths)"
                units="1/16 in"
                value={pad.weld_size_16}
                step="1"
                onChange={(v) => {
                  const raw = Number(v);
                  const next = Number.isFinite(raw) ? Math.max(0, Math.round(raw)) : 0;
                  setPad({ ...pad, weld_size_16: next });
                }}
              />
              <Field
                label="Weld size (calc)"
                units="in"
                value={weldSizeInDisplay}
                type="text"
                onChange={() => {}}
                disabled
              />
              <Field
                label="Effective throat te"
                units="in"
                value={effectiveThroatDisplay}
                type="text"
                onChange={() => {}}
                disabled
              />
              <Field
                label="Weld metal strength Exx"
                units="ksi"
                value={pad.weld_exx_ksi}
                onChange={(v) => setPad({ ...pad, weld_exx_ksi: Number(v) })}
              />
              <button className="btn btn-secondary" disabled={busy} onClick={optimizeTheta} type="button">
                {busy ? "Running..." : "Find Worst-Case Theta"}
              </button>
            </div>
          ) : mode === "spreader" ? (
            <div>
              <div className="section-title">Spreader inputs</div>
              <Select
                label="Shape (AISC label)"
                value={spr.shape}
                onChange={(v) => setSpr({ ...spr, shape: v })}
                options={spreaderShapeOptions}
                disabled={spreaderShapeLoading || !spreaderShapes.length}
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
              <Field
                label="Top padeye height ey"
                units="in"
                value={spr.ey}
                onChange={(v) => setSpr({ ...spr, ey: Number(v) })}
                disabled={spr.mx_includes_total}
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
              <Select
                label="Mx includes self-weight + axial eccentricity"
                value={String(spr.mx_includes_total)}
                onChange={(v) => setSpr({ ...spr, mx_includes_total: v === "true" })}
                options={[
                  { value: "false", label: "No (compute automatically)" },
                  { value: "true", label: "Yes (Mx is total)" }
                ]}
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
                disabled={spr.mx_includes_total}
              />
              <button className="btn btn-secondary" disabled={busy} onClick={optimizeSection} type="button">
                {busy ? "Running..." : "Optimize Section (Min Weight)"}
              </button>
            </div>
          ) : (
            <div>
              <div className="section-title">Spreader two-way inputs</div>
              <Select
                label="Shape (AISC label)"
                value={twoWay.shape}
                onChange={(v) => setTwoWay({ ...twoWay, shape: v })}
                options={spreaderShapeOptions}
                disabled={spreaderShapeLoading || !spreaderShapes.length}
              />
              <Field
                label="Total beam length"
                units="ft"
                value={twoWay.length_ft}
                onChange={(v) => setTwoWay({ ...twoWay, length_ft: Number(v) })}
              />
              <Field
                label="Edge to top padeye hole"
                units="ft"
                value={twoWay.padeye_edge_ft}
                onChange={(v) => setTwoWay({ ...twoWay, padeye_edge_ft: Number(v) })}
              />
              <Field
                label="Top of beam to padeye hole"
                units="in"
                value={twoWay.padeye_height_in}
                onChange={(v) => setTwoWay({ ...twoWay, padeye_height_in: Number(v) })}
              />
              <Field
                label="Minimum sling angle (from horizontal)"
                units="deg"
                value={twoWay.sling_angle_deg}
                onChange={(v) => setTwoWay({ ...twoWay, sling_angle_deg: Number(v) })}
              />
              <div className="section-title">Point loads (vertical)</div>
              <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 8 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", paddingBottom: 6 }}>x (ft)</th>
                    <th style={{ textAlign: "left", paddingBottom: 6 }}>P (kip)</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(twoWay.point_loads || []).map((load, idx) => (
                    <tr key={`tw-load-${idx}`}>
                      <td style={{ paddingBottom: 6 }}>
                        <input
                          type="number"
                          step="any"
                          value={load.x_ft}
                          onChange={(ev) => updateTwoWayLoad(idx, "x_ft", ev.target.value)}
                        />
                      </td>
                      <td style={{ paddingBottom: 6 }}>
                        <input
                          type="number"
                          step="any"
                          value={load.P_kip}
                          onChange={(ev) => updateTwoWayLoad(idx, "P_kip", ev.target.value)}
                        />
                      </td>
                      <td style={{ paddingBottom: 6 }}>
                        <button type="button" className="btn btn-secondary" onClick={() => removeTwoWayLoad(idx)}>
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button type="button" className="btn btn-secondary" onClick={addTwoWayLoad}>
                Add Load
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ marginLeft: 8 }}
                onClick={pushTwoWayToSpreader}
                disabled={busy || !resultsByMode.spreader_two_way}
              >
                Push to Spreader Tab
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
          {mode === "padeye" ? (
            <div className="card" style={{ marginBottom: "16px" }}>
              <div className="card-title">Padeye Diagram</div>
              <PadeyeDiagram
                H={pad.H}
                h={pad.h}
                a1={pad.a1}
                Wb={pad.Wb}
                Wb1={pad.Wb1}
                t={pad.t}
                Dh={pad.Dh}
                R={pad.R}
              />
            </div>
          ) : null}
          {mode === "spreader_two_way" ? (
            <div className="card" style={{ marginBottom: "16px" }}>
              <div className="card-title">Spreader Geometry + Rigging</div>
              <SpreaderTwoWayDiagram
                lengthFt={twoWay.length_ft}
                padeyeEdgeFt={twoWay.padeye_edge_ft}
                padeyeHeightIn={twoWay.padeye_height_in}
                pointLoads={twoWay.point_loads}
                shapeLabel={twoWay.shape}
                shapeDepthIn={twoWayTables && twoWayTables.two_way ? twoWayTables.two_way.shape_depth_in : null}
                selfWeightKipFt={twoWayTables && twoWayTables.two_way ? twoWayTables.two_way.self_weight_kipft : null}
                slingAngleMinDeg={twoWay.sling_angle_deg}
                outputs={twoWayOutputs}
              />
              <div style={{ padding: "0 10px 12px 10px" }}>
                <LineDiagram
                  title="Strong-Axis Moment Diagram"
                  xVals={twoWayDiagrams ? twoWayDiagrams.x_ft : []}
                  yVals={twoWayDiagrams ? twoWayDiagrams.moment_kipft : []}
                  units="kip-ft"
                  color="#f59e0b"
                />
                <LineDiagram
                  title="Shear Diagram"
                  xVals={twoWayDiagrams ? twoWayDiagrams.x_ft : []}
                  yVals={twoWayDiagrams ? twoWayDiagrams.shear_kip : []}
                  units="kip"
                  color="#38bdf8"
                />
                <LineDiagram
                  title="Axial Diagram"
                  xVals={
                    twoWayOutputs
                      ? [0, twoWay.length_ft]
                      : []
                  }
                  yVals={
                    twoWayOutputs
                      ? [Number(twoWayOutputs.axial_compression && twoWayOutputs.axial_compression.value !== undefined ? twoWayOutputs.axial_compression.value : twoWayOutputs.axial_compression) || 0,
                         Number(twoWayOutputs.axial_compression && twoWayOutputs.axial_compression.value !== undefined ? twoWayOutputs.axial_compression.value : twoWayOutputs.axial_compression) || 0]
                      : []
                  }
                  units="kip"
                  color="#22c55e"
                />
              </div>
            </div>
          ) : null}
          <div className="card" style={{ marginBottom: "16px" }}>
            <div className="card-title">Results</div>
            {!results ? (
              <div className="sub">Run analysis to generate results and a calc package.</div>
            ) : (
              <div>
                <div className="kvgrid">
                  {orderedOutputs.map(([key, value]) => {
                    let governingUtilStyle = {};
                    if (key === "governing_ratio") {
                      const numeric = typeof value === "number" ? value : Number(value);
                      const numVal = Number.isFinite(numeric) ? numeric : (value && typeof value === "object" && Number.isFinite(Number(value.value)) ? Number(value.value) : null);
                      if (numVal !== null) {
                        governingUtilStyle = numVal > 1
                          ? { backgroundColor: "#7f1d1d", borderColor: "#5f1414" }
                          : { backgroundColor: "#166534", borderColor: "#14532d" };
                      }
                    }
                    return (
                      <div className="kv" key={key} style={governingUtilStyle}>
                        <div className="k">{outputLabelMap[key] || key.replace(/_/g, " ")}</div>
                        <div className="v">
                          {formatOutputValue(key, value)}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="links">
                  <a className="link" href={reportUrl} target="_blank" rel="noreferrer" onClick={handleViewReport}>
                    View Calc Package
                  </a>
                  <a className="link" href={reportUrl} target="_blank" rel="noreferrer" onClick={handleViewReport}>
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
                      {orderedChecks.map((check, idx) => (
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
            {reportVisible && results ? (
              <iframe title="report" src={reportUrl} />
            ) : (
              <div className="placeholder">
                Click View Calc Package to load the latest report for this mode.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
