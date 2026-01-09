// src/lib/calc.ts
//
// Updates:
//  - Evaluates 4 mandatory SERVICE cases per config.
//  - Slewing torque T from DB, applied to operation + erection.
//  - Service checks per case: bearing plane (no tension), overturning FS, detailed coupled sliding (V+T).
//  - Reports governing case(s) and full calc-sheet output.
//
// Notes:
//  - Bearing plane is iterated p(x,y)=a+bx+cy with p>=0 enforced by dropping negative cells and re-solving.
//  - Sliding uses local friction capacity mu p per cell, coupled translation + torsion without double-counting.

import type { CaseActions, LoadCaseName, LoadRecord } from "../data/loads";

export const BAR_SIZES = ["#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#14"] as const;
const BAR_DB: Record<(typeof BAR_SIZES)[number], { dia_in: number; area_in2: number }> = {
  "#4": { dia_in: 0.5, area_in2: 0.20 },
  "#5": { dia_in: 0.625, area_in2: 0.31 },
  "#6": { dia_in: 0.75, area_in2: 0.44 },
  "#7": { dia_in: 0.875, area_in2: 0.60 },
  "#8": { dia_in: 1.0, area_in2: 0.79 },
  "#9": { dia_in: 1.128, area_in2: 1.00 },
  "#10": { dia_in: 1.27, area_in2: 1.27 },
  "#11": { dia_in: 1.41, area_in2: 1.56 },
  "#14": { dia_in: 1.693, area_in2: 2.25 }
};

export function getBarProps(bar: (typeof BAR_SIZES)[number]) {
  return BAR_DB[bar];
}

export const DEFAULTS = {
  qa_psf: 4000,
  pedestal_ft: 10.17,
  Bx_ft: 16,
  By_ft: 16,
  t_ft: 6,
  fc_psi: 4000,
  fy_ksi: 60,
  cover_in: 3,
  barX: "#6" as (typeof BAR_SIZES)[number],
  barY: "#6" as (typeof BAR_SIZES)[number],
  spacingX_in: 12,
  spacingY_in: 12,
  mu: 0.30,
  FS_slide_req: 1.5,
  gridN: 40
};

const C = {
  gammaConc_pcf: 150,
  strengthFactor: 1.5,
  FS_overturn_req: 1.5,
  phi_flex: 0.9,
  phi_shear: 0.75
};

export const CALC_CONFIG = {
  gammaConc_pcf: C.gammaConc_pcf,
  strengthFactor: C.strengthFactor,
  FS_overturn_req: C.FS_overturn_req,
  phi_flex: C.phi_flex,
  phi_shear: C.phi_shear
} as const;

const CASES: LoadCaseName[] = ["operation", "stormFront", "stormRear", "erection"];
const caseTitle: Record<LoadCaseName, string> = {
  operation: "In operation",
  stormFront: "Storm from front",
  stormRear: "Storm from rear",
  erection: "Crane during erection"
};
const caseHasSlew: Record<LoadCaseName, boolean> = {
  operation: true,
  stormFront: false,
  stormRear: false,
  erection: true
};

export type PressureField = {
  n: number;
  dx_ft: number;
  dy_ft: number;
  dA_ft2: number;
  xCenters_ft: number[];
  yCenters_ft: number[];
  p_psf: Float64Array;
  contactMask: Uint8Array;
  converged: boolean;
  iterations: number;
  a: number;
  b: number;
  c: number;
  qmax_psf: number;
  qmin_psf: number;
  Acontact_ft2: number;
};

export type SlidingOut = {
  Umax: number;
  FS_gov: number;
  pass: boolean;
  govCell: { x_ft: number; y_ft: number; p_psf: number; U: number } | null;
  thetaV_deg: number;
  N_contact_k: number;
  alphaV: number;

  r_eff_ft: number;
  interaction: number;
  simplifiedPass: boolean;
  simplifiedPassDetailedFail: boolean;

  denomPr2: number;
  k_torsion: number;
};

export type MomentResult = {
  M_top_kipft: number;
  V_k: number;
  M_base_kipft: number;
  theta_deg: number;
  Mx_kipft: number;
  My_kipft: number;
};

export type ServiceCaseResult = {
  name: LoadCaseName;
  title: string;
  actions: CaseActions;
  T_kipft: number;
  moment: MomentResult;

  N_total_k: number; // Pu + footing weight
  field: PressureField;
  bearingPass: boolean;
  overturn: { FS: number; pass: boolean; leverArm_ft: number };
  sliding: SlidingOut;

  passAll: boolean;
};

export type FlexureOut = {
  Mu_kipft_perft: number;
  d_in: number;
  As_req_in2_perft: number;
  bar: (typeof BAR_SIZES)[number];
  spacing_in: number;
  As_prov_in2_perft: number;
  phiMn_kipft_perft: number;
  utilization: number;
  governingCase: LoadCaseName;
};

export type ShearOut = {
  oneWay: { Vu_k_perft: number; phiVc_k_perft: number; pass: boolean; utilization: number; governingCase: LoadCaseName };
  punching: { Vu_k: number; phiVc_k: number; pass: boolean; utilization: number; governingCase: LoadCaseName; bo_in: number };
};

export type CalcResult = {
  geometry: {
    Bx_ft: number;
    By_ft: number;
    t_ft: number;
    pedestal_ft: number;
    area_ft2: number;
    weight_kips: number;
  };
  inputs: {
    qa_psf: number;
    mu: number;
    FS_slide_req: number;
    gridN: number;
  };
  service: {
    cases: ServiceCaseResult[];
    passAll: boolean;
    governing: {
      bearingCase: LoadCaseName;
      overturnCase: LoadCaseName;
      slidingCase: LoadCaseName;
    };
  };
  strength: { flexureX: FlexureOut; flexureY: FlexureOut; shear: ShearOut };
  warnings: string[];
  errors: string[];
  eq: string[];
};

function idx(i: number, j: number, n: number) {
  return i + n * j;
}
function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}
function roundDownToHalfIn(x: number) {
  return Math.floor(x * 2) / 2;
}
function roundUpToHalfIn(x: number) {
  return Math.ceil(x * 2) / 2;
}
const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "-");
const fmt0 = (v: number) => (Number.isFinite(v) ? Math.round(v).toString() : "-");

function footingWeight_kips(Bx_ft: number, By_ft: number, t_ft: number) {
  const vol_ft3 = Bx_ft * By_ft * t_ft;
  return (C.gammaConc_pcf * vol_ft3) / 1000;
}

function solve3x3(A: number[][], b: number[]) {
  const M = A.map((row) => row.slice());
  const x = b.slice();
  for (let k = 0; k < 3; k++) {
    let piv = k;
    let best = Math.abs(M[k][k]);
    for (let i = k + 1; i < 3; i++) {
      const v = Math.abs(M[i][k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (best < 1e-18) return null;
    if (piv !== k) {
      [M[k], M[piv]] = [M[piv], M[k]];
      [x[k], x[piv]] = [x[piv], x[k]];
    }
    for (let i = k + 1; i < 3; i++) {
      const f = M[i][k] / M[k][k];
      for (let j = k; j < 3; j++) M[i][j] -= f * M[k][j];
      x[i] -= f * x[k];
    }
  }
  const sol = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < 3; j++) s -= M[i][j] * sol[j];
    sol[i] = s / M[i][i];
  }
  return sol as [number, number, number];
}

export function computePressureFieldNoTension(args: {
  Bx_ft: number;
  By_ft: number;
  N_k: number;
  Mx_kipft: number;
  My_kipft: number;
  n: number;
  iterMax?: number;
}): PressureField {
  const { Bx_ft, By_ft, N_k, Mx_kipft, My_kipft } = args;
  const n = Math.max(10, Math.floor(args.n));
  const iterMax = args.iterMax ?? 60;

  const dx_ft = Bx_ft / n;
  const dy_ft = By_ft / n;
  const dA = dx_ft * dy_ft;

  const xCenters = Array.from({ length: n }, (_, i) => -Bx_ft / 2 + (i + 0.5) * dx_ft);
  const yCenters = Array.from({ length: n }, (_, j) => -By_ft / 2 + (j + 0.5) * dy_ft);

  const p = new Float64Array(n * n);
  const mask = new Uint8Array(n * n);
  mask.fill(1);

  let a = 0,
    b = 0,
    c = 0;
  let converged = false;

  const N_lb = N_k * 1000;
  const Mx_lbft = Mx_kipft * 1000;
  const My_lbft = My_kipft * 1000;

  if (!(N_k > 0) || !(Bx_ft > 0) || !(By_ft > 0)) {
    return {
      n,
      dx_ft,
      dy_ft,
      dA_ft2: dA,
      xCenters_ft: xCenters,
      yCenters_ft: yCenters,
      p_psf: p,
      contactMask: mask,
      converged: false,
      iterations: 0,
      a: 0,
      b: 0,
      c: 0,
      qmax_psf: 0,
      qmin_psf: 0,
      Acontact_ft2: 0
    };
  }

  let iterations = 0;
  for (let it = 0; it < iterMax; it++) {
    iterations = it + 1;
    let S0 = 0,
      Sx = 0,
      Sy = 0,
      Sxx = 0,
      Syy = 0,
      Sxy = 0,
      count = 0;

    for (let j = 0; j < n; j++) {
      const y = yCenters[j];
      for (let i = 0; i < n; i++) {
        const id = idx(i, j, n);
        if (!mask[id]) continue;
        const x = xCenters[i];
        count++;
        S0 += dA;
        Sx += x * dA;
        Sy += y * dA;
        Sxx += x * x * dA;
        Syy += y * y * dA;
        Sxy += x * y * dA;
      }
    }
    if (count === 0 || S0 <= 0) break;

    const sol = solve3x3(
      [
        [S0, Sx, Sy],
        [Sx, Sxx, Sxy],
        [Sy, Sxy, Syy]
      ],
      [N_lb, My_lbft, Mx_lbft]
    );
    if (!sol) break;
    [a, b, c] = sol;

    let changed = 0;
    for (let j = 0; j < n; j++) {
      const y = yCenters[j];
      for (let i = 0; i < n; i++) {
        const id = idx(i, j, n);
        const x = xCenters[i];
        const pij = a + b * x + c * y;
        const pos = pij > 0 ? pij : 0;
        p[id] = pos;
        const newMask = pos > 0 ? 1 : 0;
        if (newMask !== mask[id]) {
          mask[id] = newMask;
          changed++;
        }
      }
    }
    if (changed === 0) {
      converged = true;
      break;
    }
  }

  let qmax = -Infinity;
  let qmin = Infinity;
  let Acontact = 0;
  for (let k = 0; k < n * n; k++) {
    const pij = p[k];
    if (pij > 0) {
      Acontact += dA;
      if (pij > qmax) qmax = pij;
      if (pij < qmin) qmin = pij;
    }
  }
  if (!Number.isFinite(qmax)) qmax = 0;
  if (!Number.isFinite(qmin)) qmin = 0;

  return {
    n,
    dx_ft,
    dy_ft,
    dA_ft2: dA,
    xCenters_ft: xCenters,
    yCenters_ft: yCenters,
    p_psf: p,
    contactMask: mask,
    converged,
    iterations,
    a,
    b,
    c,
    qmax_psf: qmax,
    qmin_psf: Acontact > 0 ? qmin : 0,
    Acontact_ft2: Acontact
  };
}

export function computeSlidingDetailed(args: {
  field: PressureField;
  mu: number;
  FSreq: number;
  V_k: number;
  T_kipft: number;
}): SlidingOut {
  const { field, mu, FSreq, V_k, T_kipft } = args;
  const n = field.n;
  const p = field.p_psf;
  const dA = field.dA_ft2;
  const xC = field.xCenters_ft;
  const yC = field.yCenters_ft;

  const V_lb = V_k * 1000;
  const T_lbft = T_kipft * 1000;

  let N_lb = 0;
  let sumPr = 0;
  let sumPr2 = 0;

  for (let j = 0; j < n; j++) {
    const y = yC[j];
    for (let i = 0; i < n; i++) {
      const id = idx(i, j, n);
      const pij = p[id];
      if (pij <= 0) continue;
      const x = xC[i];
      const r = Math.hypot(x, y);
      N_lb += pij * dA;
      sumPr += pij * r * dA;
      sumPr2 += pij * r * r * dA;
    }
  }

  const N_k = N_lb / 1000;
  const r_eff = N_lb > 0 ? sumPr / N_lb : 0;
  const alphaV = N_k > 0 ? V_k / N_k : 0;

  const simplified =
    mu > 0 && N_k > 0 && r_eff > 1e-9 ? V_k / (mu * N_k) + Math.abs(T_kipft) / (mu * N_k * r_eff) : Infinity;
  const simplifiedPass = simplified <= 1;

  if (!(mu > 0) || !(N_lb > 0)) {
    const demandExists = V_k > 0 || Math.abs(T_kipft) > 0;
    const Umax = demandExists ? Infinity : 0;
    const FS_gov = Umax > 0 ? 1 / Umax : Infinity;
    return {
      Umax,
      FS_gov,
      pass: !demandExists,
      govCell: null,
      thetaV_deg: 0,
      N_contact_k: N_k,
      alphaV,
      r_eff_ft: r_eff,
      interaction: simplified,
      simplifiedPass,
      simplifiedPassDetailedFail: simplifiedPass && demandExists,
      denomPr2: sumPr2,
      k_torsion: 0
    };
  }

  const kT = sumPr2 > 1e-18 ? T_lbft / sumPr2 : 0;

  function evalDirection(vxh: number, vyh: number): { Umax: number; gov: SlidingOut["govCell"] } {
    let Umax = 0;
    let gov: SlidingOut["govCell"] = null;

    for (let j = 0; j < n; j++) {
      const y = yC[j];
      for (let i = 0; i < n; i++) {
        const id = idx(i, j, n);
        const pij = p[id];
        if (pij <= 0) continue;
        const x = xC[i];

        const tVx = alphaV * pij * vxh;
        const tVy = alphaV * pij * vyh;

        const tTx = kT * pij * (-y);
        const tTy = kT * pij * x;

        const tx = tVx + tTx;
        const ty = tVy + tTy;

        const cap = mu * pij;
        const demand = Math.hypot(tx, ty);
        const U = cap > 0 ? demand / cap : Infinity;

        if (U > Umax) {
          Umax = U;
          gov = { x_ft: x, y_ft: y, p_psf: pij, U };
        }
      }
    }

    return { Umax, gov };
  }

  let worstUmax = 0;
  let worstGov: SlidingOut["govCell"] = null;
  let thetaV = 0;

  if (!(V_k > 1e-12)) {
    const res = evalDirection(0, 0);
    worstUmax = res.Umax;
    worstGov = res.gov;
    thetaV = 0;
  } else {
    const stepDeg = 5;
    for (let deg = 0; deg <= 180; deg += stepDeg) {
      const th = (deg * Math.PI) / 180;
      const vxh = Math.cos(th);
      const vyh = Math.sin(th);
      const res = evalDirection(vxh, vyh);
      if (res.Umax > worstUmax + 1e-12) {
        worstUmax = res.Umax;
        worstGov = res.gov;
        thetaV = deg;
      }
    }
  }

  const FS_gov = worstUmax > 1e-18 ? 1 / worstUmax : Infinity;
  const pass = worstUmax <= 1 / FSreq;

  return {
    Umax: worstUmax,
    FS_gov,
    pass,
    govCell: worstGov,
    thetaV_deg: thetaV,
    N_contact_k: N_k,
    alphaV,
    r_eff_ft: r_eff,
    interaction: simplified,
    simplifiedPass,
    simplifiedPassDetailedFail: simplifiedPass && !pass,
    denomPr2: sumPr2,
    k_torsion: kT
  };
}

function resultantMomentAtBase(actions: CaseActions, t_ft: number) {
  const M_top = Math.abs(actions.M_kipft);
  const V = Math.abs(actions.V_k);
  const M_base = M_top + V * t_ft;
  return { M_top_kipft: M_top, V_k: V, M_base_kipft: M_base };
}

function findWorstCasePressureField(args: {
  Bx_ft: number;
  By_ft: number;
  N_k: number;
  M_kipft: number;
  n: number;
}) {
  const { Bx_ft, By_ft, N_k, M_kipft, n } = args;
  if (!(M_kipft > 0)) {
    const field = computePressureFieldNoTension({ Bx_ft, By_ft, N_k, Mx_kipft: 0, My_kipft: 0, n });
    return { field, theta_deg: 0, Mx_kipft: 0, My_kipft: 0 };
  }

  const stepDeg = 1;
  let bestScore = -Infinity;
  let bestField: PressureField | null = null;
  let bestTheta = 0;
  let bestMx = M_kipft;
  let bestMy = 0;
  const isSquare = Math.abs(Bx_ft - By_ft) <= 1e-6;

  for (let deg = 0; deg <= 90; deg += stepDeg) {
    const theta = (deg * Math.PI) / 180;
    const Mx = M_kipft * Math.cos(theta);
    const My = M_kipft * Math.sin(theta);
    const field = computePressureFieldNoTension({ Bx_ft, By_ft, N_k, Mx_kipft: Mx, My_kipft: My, n });
    const unstable = !(field.Acontact_ft2 > 0) || !field.converged;
    const score = unstable ? Number.POSITIVE_INFINITY : field.qmax_psf;
    const improve = score > bestScore + 1e-9;
    const tie =
      !improve && Math.abs(score - bestScore) <= 1e-9 && isSquare && Math.abs(deg - 45) < Math.abs(bestTheta - 45);
    if (improve || tie) {
      bestScore = score;
      bestField = field;
      bestTheta = deg;
      bestMx = Mx;
      bestMy = My;
    }
  }

  if (!bestField) {
    const field = computePressureFieldNoTension({ Bx_ft, By_ft, N_k, Mx_kipft: M_kipft, My_kipft: 0, n });
    return { field, theta_deg: 0, Mx_kipft: M_kipft, My_kipft: 0 };
  }

  const unstable = !(bestField.Acontact_ft2 > 0) || !bestField.converged;
  if (unstable) bestField.qmax_psf = Number.POSITIVE_INFINITY;

  return { field: bestField, theta_deg: bestTheta, Mx_kipft: bestMx, My_kipft: bestMy };
}

function solveAsReq(Mu_kipft_perft: number, fc_psi: number, fy_ksi: number, d_in: number): number {
  const Mu_lb_in = Mu_kipft_perft * 12 * 1000;
  if (Mu_lb_in <= 1e-6) return 0;

  const b = 12;
  const fy = fy_ksi * 1000;
  const fc = fc_psi;
  const phi = C.phi_flex;

  const A = (-phi * fy * fy) / (2 * 0.85 * fc * b);
  const B = phi * fy * d_in;
  const Cc = -Mu_lb_in;

  const disc = B * B - 4 * A * Cc;
  if (disc <= 0 || Math.abs(A) < 1e-18) return Math.max(0, Mu_lb_in / (phi * fy * d_in));
  const r1 = (-B + Math.sqrt(disc)) / (2 * A);
  const r2 = (-B - Math.sqrt(disc)) / (2 * A);
  const roots = [r1, r2].filter((r) => Number.isFinite(r) && r > 0);
  if (!roots.length) return Math.max(0, Mu_lb_in / (phi * fy * d_in));
  return Math.min(...roots);
}

function barSpacing(As_req: number, bar: (typeof BAR_SIZES)[number]) {
  const Ab = BAR_DB[bar].area_in2;
  const s_raw = As_req <= 0 ? 18 : (Ab * 12) / As_req;
  let s = clamp(roundUpToHalfIn(s_raw), 3, 18);
  while (s > 3 && (Ab * 12) / s + 1e-12 < As_req) s -= 0.5;
  return { s_in: s, As_prov: (Ab * 12) / s };
}

function normalizeSpacing(spacing_in?: number) {
  if (!spacing_in || !(spacing_in > 0)) return null;
  const s = clamp(spacing_in, 3, 18);
  return roundDownToHalfIn(s);
}

function flexureCapacityPhiMn(As_prov_in2_perft: number, fc_psi: number, fy_ksi: number, d_in: number): number {
  if (!(As_prov_in2_perft > 0) || !(fc_psi > 0) || !(fy_ksi > 0) || !(d_in > 0)) return 0;
  const b = 12;
  const fy = fy_ksi * 1000;
  const a = (As_prov_in2_perft * fy) / (0.85 * fc_psi * b);
  const Mn_lb_in = As_prov_in2_perft * fy * (d_in - a / 2);
  const phiMn_lb_in = C.phi_flex * Mn_lb_in;
  return phiMn_lb_in / (12 * 1000);
}

function computeServiceCase(args: {
  name: LoadCaseName;
  actions: CaseActions;
  T_kipft: number;
  Bx: number;
  By: number;
  t_ft: number;
  W: number;
  qa: number;
  mu: number;
  FS_slide_req: number;
  gridN: number;
}): ServiceCaseResult {
  const { name, actions, T_kipft, Bx, By, t_ft, W, qa, mu, FS_slide_req, gridN } = args;

  const N_total = actions.Pu + W;
  const baseMoment = resultantMomentAtBase(actions, t_ft);
  const worstField = findWorstCasePressureField({
    Bx_ft: Bx,
    By_ft: By,
    N_k: N_total,
    M_kipft: baseMoment.M_base_kipft,
    n: gridN
  });
  const field = worstField.field;

  const unstable = !(field.Acontact_ft2 > 0) || !field.converged;
  const bearingPass = !unstable && field.qmax_psf <= qa;

  const leverArm = Math.min(Bx, By) / 2;
  const FS =
    !(baseMoment.M_base_kipft > 1e-12) || N_total <= 0 ? Infinity : (N_total * leverArm) / baseMoment.M_base_kipft;
  const overturnPass = FS >= C.FS_overturn_req;

  const sliding = computeSlidingDetailed({
    field,
    mu,
    FSreq: FS_slide_req,
    V_k: actions.V_k,
    T_kipft
  });

  const passAll = bearingPass && overturnPass && sliding.pass && !unstable;

  return {
    name,
    title: caseTitle[name],
    actions,
    T_kipft,
    moment: {
      M_top_kipft: baseMoment.M_top_kipft,
      V_k: baseMoment.V_k,
      M_base_kipft: baseMoment.M_base_kipft,
      theta_deg: worstField.theta_deg,
      Mx_kipft: worstField.Mx_kipft,
      My_kipft: worstField.My_kipft
    },
    N_total_k: N_total,
    field,
    bearingPass,
    overturn: { FS, pass: overturnPass, leverArm_ft: leverArm },
    sliding,
    passAll
  };
}

export function computeAll(input: {
  Bx_ft: number;
  By_ft: number;
  t_ft: number;
  pedestal_ft: number;
  qa_psf: number;

  mu: number;
  FS_slide_req: number;
  gridN: number;

  record: LoadRecord;

  fc_psi: number;
  fy_ksi: number;
  cover_in: number;
  barX: (typeof BAR_SIZES)[number];
  barY: (typeof BAR_SIZES)[number];
  spacingX_in?: number;
  spacingY_in?: number;
}): CalcResult {
  const errors: string[] = [];
  const warnings: string[] = [
    "Preliminary prototype only. Verify per code and project criteria.",
    "Four service load cases are always checked (operation, storm front, storm rear, erection).",
    "Slewing torque T is taken from the database and applied to operation + erection only.",
    "Overturning uses base moment M_base = M + V*t and is rotated to the worst-case bearing axis.",
    "Bearing plane is iterated with no tension (p>=0).",
    "Sliding uses detailed local friction coupling (V+T) without double-counting friction; simplified interaction is also reported."
  ];

  const Bx = input.Bx_ft;
  const By = input.By_ft;
  const t = input.t_ft;
  const ped = input.pedestal_ft;

  if (!(Bx > 0 && By > 0)) errors.push("Footing plan dimensions Bx and By must be > 0.");
  if (!(t > 0)) errors.push("Footing thickness t must be > 0.");
  if (!(ped > 0)) errors.push("Mast width must be > 0.");
  if (Bx <= ped || By <= ped) errors.push("Footing dimensions must exceed mast width.");
  if (!(input.qa_psf > 0)) errors.push("Allowable bearing qa must be > 0 psf.");
  if (!(input.mu >= 0)) errors.push("Friction coefficient mu must be >= 0.");
  if (!(input.FS_slide_req > 0)) errors.push("Required sliding FS must be > 0.");
  if (!(input.gridN >= 10)) errors.push("Grid n must be >= 10.");
  if (!(input.fy_ksi > 0)) errors.push("fy must be > 0.");

  const W = footingWeight_kips(Bx, By, t);

  const cases: ServiceCaseResult[] = CASES.map((name) =>
    computeServiceCase({
      name,
      actions: input.record.cases[name],
      T_kipft: caseHasSlew[name] ? input.record.slewTorque_kipft : 0,
      Bx,
      By,
      t_ft: t,
      W,
      qa: input.qa_psf,
      mu: input.mu,
      FS_slide_req: input.FS_slide_req,
      gridN: input.gridN
    })
  );

  const bearingGov = cases.reduce((a, b) => (b.field.qmax_psf > a.field.qmax_psf ? b : a)).name;
  const otGov = cases.reduce((a, b) => (b.overturn.FS < a.overturn.FS ? b : a)).name;
  const slideGov = cases.reduce((a, b) => (b.sliding.Umax > a.sliding.Umax ? b : a)).name;

  const passAll = cases.every((c) => c.passAll);

  // Strength screening
  const t_in = t * 12;
  const dbX = BAR_DB[input.barX].dia_in;
  const dbY = BAR_DB[input.barY].dia_in;
  const dX_in = t_in - input.cover_in - dbX / 2;
  const dY_in = t_in - input.cover_in - dbY / 2;
  const dShear_in = t_in - input.cover_in - Math.max(dbX, dbY) / 2;
  const dShear_ft = dShear_in / 12;

  const Lx = (Bx - ped) / 2;
  const Ly = (By - ped) / 2;
  if (Lx <= 0 || Ly <= 0) errors.push("Cantilever lengths (Bx-mast)/2 and (By-mast)/2 must be > 0.");

  let govMuX = -Infinity,
    govMuY = -Infinity;
  let govCaseMuX: LoadCaseName = "operation";
  let govCaseMuY: LoadCaseName = "operation";

  for (const c of cases) {
    const qumax_u = C.strengthFactor * c.field.qmax_psf;
    const MuX = (qumax_u * (Ly ** 2)) / (2 * 1000);
    const MuY = (qumax_u * (Lx ** 2)) / (2 * 1000);
    if (MuX > govMuX) {
      govMuX = MuX;
      govCaseMuX = c.name;
    }
    if (MuY > govMuY) {
      govMuY = MuY;
      govCaseMuY = c.name;
    }
  }

  const AsX = solveAsReq(govMuX, input.fc_psi, input.fy_ksi, dX_in);
  const AsY = solveAsReq(govMuY, input.fc_psi, input.fy_ksi, dY_in);
  const spX = barSpacing(AsX, input.barX);
  const spY = barSpacing(AsY, input.barY);
  const spacingProvidedX = normalizeSpacing(input.spacingX_in);
  const spacingProvidedY = normalizeSpacing(input.spacingY_in);
  const spacingX = spacingProvidedX ?? spX.s_in;
  const spacingY = spacingProvidedY ?? spY.s_in;
  const AbX = BAR_DB[input.barX].area_in2;
  const AbY = BAR_DB[input.barY].area_in2;
  const AsProvX = (AbX * 12) / spacingX;
  const AsProvY = (AbY * 12) / spacingY;
  const phiMnX = flexureCapacityPhiMn(AsProvX, input.fc_psi, input.fy_ksi, dX_in);
  const phiMnY = flexureCapacityPhiMn(AsProvY, input.fc_psi, input.fy_ksi, dY_in);
  const utilX = phiMnX > 0 ? govMuX / phiMnX : Infinity;
  const utilY = phiMnY > 0 ? govMuY / phiMnY : Infinity;

  const flexureX: FlexureOut = {
    Mu_kipft_perft: govMuX,
    d_in: dX_in,
    As_req_in2_perft: AsX,
    bar: input.barX,
    spacing_in: spacingX,
    As_prov_in2_perft: AsProvX,
    phiMn_kipft_perft: phiMnX,
    utilization: utilX,
    governingCase: govCaseMuX
  };
  const flexureY: FlexureOut = {
    Mu_kipft_perft: govMuY,
    d_in: dY_in,
    As_req_in2_perft: AsY,
    bar: input.barY,
    spacing_in: spacingY,
    As_prov_in2_perft: AsProvY,
    phiMn_kipft_perft: phiMnY,
    utilization: utilY,
    governingCase: govCaseMuY
  };

  const Vc1_lb = 2 * Math.sqrt(Math.max(input.fc_psi, 0)) * 12 * dShear_in;
  const phiVc1_k = (C.phi_shear * Vc1_lb) / 1000;

  let govVu = -Infinity;
  let govCaseVu: LoadCaseName = "operation";
  for (const c of cases) {
    const qumax_u = C.strengthFactor * c.field.qmax_psf;
    const Lv = Math.max(0, Math.max(Lx - dShear_ft, Ly - dShear_ft));
    const Vu = (qumax_u * Lv) / 1000;
    if (Vu > govVu) {
      govVu = Vu;
      govCaseVu = c.name;
    }
  }
  const oneWayUtil = phiVc1_k > 0 ? govVu / phiVc1_k : Infinity;
  const oneWay = {
    Vu_k_perft: govVu,
    phiVc_k_perft: phiVc1_k,
    pass: oneWayUtil <= 1,
    utilization: oneWayUtil,
    governingCase: govCaseVu
  };

  const c_in = ped * 12;
  const bo_in = 4 * (c_in + dShear_in);
  const Vcp_lb = 4 * Math.sqrt(Math.max(input.fc_psi, 0)) * bo_in * dShear_in;
  const phiVcp_k = (C.phi_shear * Vcp_lb) / 1000;

  let govPunch = -Infinity;
  let govCasePunch: LoadCaseName = "operation";
  for (const c of cases) {
    const Pu_total_u = C.strengthFactor * c.N_total_k;
    if (Pu_total_u > govPunch) {
      govPunch = Pu_total_u;
      govCasePunch = c.name;
    }
  }
  const punchingUtil = phiVcp_k > 0 ? govPunch / phiVcp_k : Infinity;
  const punching = {
    Vu_k: govPunch,
    phiVc_k: phiVcp_k,
    pass: punchingUtil <= 1,
    utilization: punchingUtil,
    governingCase: govCasePunch,
    bo_in
  };

  const eq: string[] = [];
  eq.push("=== GLOBAL INPUTS ===");
  eq.push(`Bx=${fmt(Bx, 2)} ft, By=${fmt(By, 2)} ft, t=${fmt(t, 2)} ft, mast width=${fmt(ped, 2)} ft`);
  eq.push(`qa=${fmt0(input.qa_psf)} psf, mu=${fmt(input.mu, 3)}, FS_slide_req=${fmt(input.FS_slide_req, 2)}, grid n=${input.gridN}`);
  eq.push(`Slewing torque from DB: T_slew=${fmt(input.record.slewTorque_kipft, 2)} kip-ft (applied to operation + erection)`);
  eq.push(`Footing weight W=${fmt(W, 2)} k`);
  eq.push("");

  for (const c of cases) {
    eq.push(`=== SERVICE CASE: ${caseTitle[c.name].toUpperCase()} ===`);
    eq.push(`Pu=${fmt(c.actions.Pu, 2)} k, M=${fmt(c.actions.M_kipft, 2)} kip-ft, V=${fmt(c.actions.V_k, 2)} k`);
    eq.push(
      `Resultant moment: M_top=${fmt(c.moment.M_top_kipft, 2)} kip-ft, V=${fmt(c.moment.V_k, 2)} k, M_base=${fmt(c.moment.M_base_kipft, 2)} kip-ft`
    );
    eq.push(
      `Worst-case bearing axis: theta=${fmt(c.moment.theta_deg, 0)} deg => resolved M about x=${fmt(c.moment.Mx_kipft, 2)} kip-ft, about y=${fmt(c.moment.My_kipft, 2)} kip-ft`
    );
    eq.push(`T applied = ${fmt(c.T_kipft, 2)} kip-ft`);
    eq.push(`N_total = Pu + W = ${fmt(c.N_total_k, 2)} k`);
    eq.push(`Bearing plane (no tension): converged=${c.field.converged ? "YES" : "NO"}, Acontact=${fmt(c.field.Acontact_ft2, 2)} ft^2`);
    eq.push(`qmax=${fmt0(c.field.qmax_psf)} psf, qmin=${fmt0(c.field.qmin_psf)} psf  => qmax<=qa? ${c.bearingPass ? "PASS" : "FAIL"}`);
    eq.push(
      `Overturning FS: FS=${fmt(c.overturn.FS, 2)} (req ${fmt(C.FS_overturn_req, 2)}) => ${c.overturn.pass ? "PASS" : "FAIL"} (lever arm ${fmt(c.overturn.leverArm_ft, 2)} ft)`
    );
    eq.push(`Sliding detailed: Umax=${fmt(c.sliding.Umax, 4)} => FSgov=${fmt(c.sliding.FS_gov, 3)} (req ${fmt(input.FS_slide_req, 2)}) => ${c.sliding.pass ? "PASS" : "FAIL"}`);
    if (c.actions.V_k > 1e-12) eq.push(`  Sliding worst-case V direction: theta=${fmt(c.sliding.thetaV_deg, 0)} deg`);
    if (c.sliding.govCell) {
      eq.push(`  Governing cell: (x,y)=(${fmt(c.sliding.govCell.x_ft, 3)}, ${fmt(c.sliding.govCell.y_ft, 3)}) ft, p=${fmt0(c.sliding.govCell.p_psf)} psf, U=${fmt(c.sliding.govCell.U, 4)}`);
    }
    eq.push(`Sliding simplified: r_eff=${fmt(c.sliding.r_eff_ft, 3)} ft, interaction=${fmt(c.sliding.interaction, 4)} <=1? ${c.sliding.simplifiedPass ? "PASS" : "FAIL"}`);
    if (c.sliding.simplifiedPassDetailedFail) eq.push("  FLAG: simplified passes but detailed fails.");
    eq.push(`CASE RESULT: ${c.passAll ? "PASS" : "FAIL"}`);
    eq.push("");
  }

  eq.push("=== GOVERNING (SERVICE) ===");
  eq.push(`Bearing governed by: ${caseTitle[bearingGov]}`);
  eq.push(`Overturning governed by: ${caseTitle[otGov]}`);
  eq.push(`Sliding governed by: ${caseTitle[slideGov]}`);
  eq.push(`OVERALL SERVICE: ${passAll ? "PASS" : "FAIL"}`);
  eq.push("");

  eq.push("=== STRENGTH SCREENING (effective 1.5x service) ===");
  eq.push("Use qumax,u = 1.5*qmax(service) per case; take governing strip moments and shear.");
  eq.push(`Flexure X (uses Ly): Mu/ft=${fmt(flexureX.Mu_kipft_perft, 2)} governed by ${caseTitle[flexureX.governingCase]}`);
  eq.push(
    `  As_req=${fmt(flexureX.As_req_in2_perft, 3)} in^2/ft; provide ${flexureX.bar} @ ${fmt(flexureX.spacing_in, 1)} in => As_prov=${fmt(flexureX.As_prov_in2_perft, 3)} in^2/ft`
  );
  eq.push(`  phiMn=${fmt(flexureX.phiMn_kipft_perft, 2)} kip-ft/ft; utilization=${fmt(flexureX.utilization, 3)}`);
  eq.push(`Flexure Y (uses Lx): Mu/ft=${fmt(flexureY.Mu_kipft_perft, 2)} governed by ${caseTitle[flexureY.governingCase]}`);
  eq.push(
    `  As_req=${fmt(flexureY.As_req_in2_perft, 3)} in^2/ft; provide ${flexureY.bar} @ ${fmt(flexureY.spacing_in, 1)} in => As_prov=${fmt(flexureY.As_prov_in2_perft, 3)} in^2/ft`
  );
  eq.push(`  phiMn=${fmt(flexureY.phiMn_kipft_perft, 2)} kip-ft/ft; utilization=${fmt(flexureY.utilization, 3)}`);
  eq.push(
    `One-way shear: Vu=${fmt(oneWay.Vu_k_perft, 2)} k/ft (gov ${caseTitle[oneWay.governingCase]}), phiVc~${fmt(oneWay.phiVc_k_perft, 2)} k/ft; utilization=${fmt(oneWay.utilization, 3)}`
  );
  eq.push(
    `Punching: Vu~${fmt(punching.Vu_k, 1)} k (gov ${caseTitle[punching.governingCase]}), phiVc~${fmt(punching.phiVc_k, 1)} k; utilization=${fmt(punching.utilization, 3)}`
  );

  return {
    geometry: { Bx_ft: Bx, By_ft: By, t_ft: t, pedestal_ft: ped, area_ft2: Bx * By, weight_kips: W },
    inputs: { qa_psf: input.qa_psf, mu: input.mu, FS_slide_req: input.FS_slide_req, gridN: input.gridN },
    service: {
      cases,
      passAll,
      governing: { bearingCase: bearingGov, overturnCase: otGov, slidingCase: slideGov }
    },
    strength: { flexureX, flexureY, shear: { oneWay, punching } },
    warnings,
    errors,
    eq
  };
}

export function optimizeB(input: {
  pedestal_ft: number;
  t_ft: number;
  qa_psf: number;
  mu: number;
  FS_slide_req: number;
  gridN: number;
  record: LoadRecord;
  Bmin_ft: number;
  Bmax_ft: number;
}) {
  const start = Math.ceil(Math.max(input.Bmin_ft, input.pedestal_ft + 1));
  const end = Math.floor(Math.max(start, input.Bmax_ft));

  for (let B = start; B <= end; B += 1) {
    const W = footingWeight_kips(B, B, input.t_ft);
    let ok = true;

    for (const name of CASES) {
      const actions = input.record.cases[name];
      const T = caseHasSlew[name] ? input.record.slewTorque_kipft : 0;
      const N = actions.Pu + W;
      if (!(N > 0)) {
        ok = false;
        break;
      }

      const baseMoment = resultantMomentAtBase(actions, input.t_ft);
      const worstField = findWorstCasePressureField({
        Bx_ft: B,
        By_ft: B,
        N_k: N,
        M_kipft: baseMoment.M_base_kipft,
        n: input.gridN
      });
      const field = worstField.field;
      if (!field.converged || !(field.Acontact_ft2 > 0) || field.qmax_psf > input.qa_psf) {
        ok = false;
        break;
      }

      const FS = baseMoment.M_base_kipft > 1e-12 ? (N * (B / 2)) / baseMoment.M_base_kipft : Infinity;
      if (!(FS >= C.FS_overturn_req)) {
        ok = false;
        break;
      }

      const sliding = computeSlidingDetailed({
        field,
        mu: input.mu,
        FSreq: input.FS_slide_req,
        V_k: actions.V_k,
        T_kipft: T
      });
      if (!sliding.pass) {
        ok = false;
        break;
      }
    }

    if (ok) return { B_ft: B };
  }

  return { B_ft: end };
}
