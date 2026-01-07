from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ----------------------------
# Units & sign conventions
# ----------------------------
# Internal solver uses:
#   length: inches
#   force:  pounds-force (lbf)
#   moment: inch-lbf
#   E:      psi
#   I:      in^4
#
# User-facing:
#   - See README for unit systems.
#
# Internally, we use positive upward forces and positive CCW moments to match standard FEM dof signs:
#   v (deflection) positive upward
#   theta positive counterclockwise
#
# Therefore for user inputs (default):
#   w (distributed) positive downward  -> q_internal = -w
#   P (point load)  positive downward  -> P_internal = -P
#   M (moment)      positive clockwise -> M_internal = -M (converted to internal units)


UnitSystem = Literal["IMPERIAL_FT_LB", "SI_M_KN"]


class Span(BaseModel):
    """Beam property segment."""
    length: float = Field(..., gt=0, description="Span length (ft for Imperial, m for SI)")
    E: float = Field(..., gt=0, description="Elastic modulus (psi for Imperial, Pa for SI)")
    I: float = Field(..., gt=0, description="Second moment of area (in^4 for Imperial, m^4 for SI)")


class Joint(BaseModel):
    """Joint/support definition at a beam station."""
    restraint_v: bool = Field(True, description="Vertical displacement fixed?")
    restraint_theta: bool = Field(False, description="Rotation fixed?")
    internal_hinge: bool = Field(False, description="Internal hinge (moment release) at joint? (interior joints only)")


class DistributedLoad(BaseModel):
    """Linearly varying distributed load between x_start and x_end along global axis."""
    x_start: float = Field(..., ge=0)
    x_end: float = Field(..., gt=0)
    w_start: float = Field(..., description="Load intensity at x_start (positive downward).")
    w_end: float = Field(..., description="Load intensity at x_end (positive downward).")

    @model_validator(mode="after")
    def _check_order(self):
        if self.x_end <= self.x_start:
            raise ValueError("DistributedLoad: x_end must be greater than x_start.")
        return self


class PointLoad(BaseModel):
    """Concentrated load at global x. P positive downward."""
    x: float = Field(..., ge=0)
    P: float = Field(..., description="Point load magnitude (positive downward).")


class PointMoment(BaseModel):
    """Concentrated moment at global x. M positive clockwise."""
    x: float = Field(..., ge=0)
    M: float = Field(..., description="Point moment magnitude (positive clockwise).")


class BeamModel(BaseModel):
    unit_system: UnitSystem = Field("IMPERIAL_FT_LB")
    spans: List[Span] = Field(..., min_length=1)
    joints: List[Joint] = Field(..., min_length=2)
    distributed_loads: List[DistributedLoad] = Field(default_factory=list)
    point_loads: List[PointLoad] = Field(default_factory=list)
    point_moments: List[PointMoment] = Field(default_factory=list)

    mesh_max_element_length: float = Field(
        2.0,
        gt=0,
        description="Mesh target max element length (ft for Imperial, m for SI). Smaller -> more accuracy for diagrams/deflections.",
    )

    @model_validator(mode="after")
    def _check_sizes(self):
        if len(self.joints) != len(self.spans) + 1:
            raise ValueError("BeamModel: joints must have length = len(spans) + 1.")
        total_L = sum(s.length for s in self.spans)
        for dl in self.distributed_loads:
            if dl.x_end > total_L + 1e-9:
                raise ValueError(f"DistributedLoad ends beyond beam length: x_end={dl.x_end} > {total_L}")
        for pl in self.point_loads:
            if pl.x > total_L + 1e-9:
                raise ValueError(f"PointLoad beyond beam length: x={pl.x} > {total_L}")
        for pm in self.point_moments:
            if pm.x > total_L + 1e-9:
                raise ValueError(f"PointMoment beyond beam length: x={pm.x} > {total_L}")

        # Hinges allowed only at interior joints
        if len(self.joints) >= 3:
            for i, j in enumerate(self.joints):
                if j.internal_hinge and (i == 0 or i == len(self.joints) - 1):
                    raise ValueError("Internal hinge not allowed at end joints; use support type instead.")
        return self


@dataclass(frozen=True)
class Node:
    x_in: float
    is_physical_joint: bool
    joint_index: Optional[int] = None


@dataclass(frozen=True)
class Element:
    n1: int
    n2: int
    L_in: float
    E_psi: float
    I_in4: float
    q0_lb_in: float  # positive upward
    q1_lb_in: float  # positive upward


@dataclass(frozen=True)
class DofMap:
    v_dof: List[int]
    theta_dof_left: List[int]
    theta_dof_right: List[int]
    hinge_nodes: List[bool]
    ndof: int


# --------- unit conversions ---------

def _to_internal_units(model: BeamModel) -> Tuple[List[float], List[float], List[float], float]:
    if model.unit_system == "IMPERIAL_FT_LB":
        lengths_in = [s.length * 12.0 for s in model.spans]
        E_psi = [s.E for s in model.spans]
        I_in4 = [s.I for s in model.spans]
        max_elem_len_in = model.mesh_max_element_length * 12.0
    else:
        m_to_in = 39.37007874015748
        Pa_to_psi = 0.00014503773773020923
        m4_to_in4 = m_to_in ** 4

        lengths_in = [s.length * m_to_in for s in model.spans]
        E_psi = [s.E * Pa_to_psi for s in model.spans]
        I_in4 = [s.I * m4_to_in4 for s in model.spans]
        max_elem_len_in = model.mesh_max_element_length * m_to_in
    return lengths_in, E_psi, I_in4, max_elem_len_in


def _distributed_load_user_to_internal(model: BeamModel, w_user: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return -(w_user / 12.0)  # lb/ft -> lb/in, downward -> negative
    else:
        # kN/m -> lbf/in
        kN_to_N = 1000.0
        N_to_lbf = 0.22480894387096
        m_to_in = 39.37007874015748
        w_lbf_in = (w_user * kN_to_N) * N_to_lbf / m_to_in
        return -w_lbf_in


def _point_load_user_to_internal(model: BeamModel, P_user: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return -P_user
    else:
        kN_to_N = 1000.0
        N_to_lbf = 0.22480894387096
        return -(P_user * kN_to_N * N_to_lbf)


def _moment_user_to_internal(model: BeamModel, M_user: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return -(M_user * 12.0)  # ft-lb -> in-lb, clockwise -> negative
    else:
        kN_to_N = 1000.0
        N_to_lbf = 0.22480894387096
        m_to_in = 39.37007874015748
        return -(M_user * kN_to_N * N_to_lbf * m_to_in)


def _x_user_to_internal_in(model: BeamModel, x_user: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return x_user * 12.0
    else:
        m_to_in = 39.37007874015748
        return x_user * m_to_in


def _x_internal_to_user(model: BeamModel, x_in: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return x_in / 12.0
    else:
        m_to_in = 39.37007874015748
        return x_in / m_to_in


def _shear_internal_to_user(model: BeamModel, V_lbf: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return V_lbf
    else:
        lbf_to_N = 4.4482216152605
        return V_lbf * lbf_to_N / 1000.0  # kN


def _moment_internal_to_user(model: BeamModel, M_inlb: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return M_inlb / 12.0  # ft-lb
    else:
        lbf_to_N = 4.4482216152605
        in_to_m = 0.0254
        return M_inlb * lbf_to_N * in_to_m / 1000.0  # kN-m


def _deflection_internal_to_user(model: BeamModel, v_in: float) -> float:
    if model.unit_system == "IMPERIAL_FT_LB":
        return v_in
    else:
        return v_in * 0.0254  # m


# --------- FEM helpers ---------

def _shape_functions(xi: float, L: float) -> np.ndarray:
    N1 = 1 - 3 * xi**2 + 2 * xi**3
    N2 = L * (xi - 2 * xi**2 + xi**3)
    N3 = 3 * xi**2 - 2 * xi**3
    N4 = L * (-xi**2 + xi**3)
    return np.array([N1, N2, N3, N4], dtype=float)


def _consistent_load_vector_linear(q0: float, q1: float, L: float, n_gauss: int = 4) -> np.ndarray:
    pts, wts = np.polynomial.legendre.leggauss(n_gauss)  # [-1,1]
    xis = 0.5 * (pts + 1.0)   # [0,1]
    ws = 0.5 * wts

    fe = np.zeros(4, dtype=float)
    for xi, w in zip(xis, ws):
        q = q0 + (q1 - q0) * xi
        N = _shape_functions(xi, L)
        fe += N * q * L * w
    return fe


def _beam_element_stiffness(E: float, I: float, L: float) -> np.ndarray:
    k = (E * I / (L**3)) * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2],
        ],
        dtype=float,
    )
    return k


# --------- Mesh build ---------

def build_mesh(model: BeamModel) -> Tuple[List[Node], List[Element], DofMap, Dict[str, float]]:
    lengths_in, E_psi, I_in4, max_len_in = _to_internal_units(model)

    span_stations = [0.0]
    acc = 0.0
    for L in lengths_in:
        acc += L
        span_stations.append(acc)
    total_L_in = span_stations[-1]

    stations = set(span_stations)
    for dl in model.distributed_loads:
        stations.add(_x_user_to_internal_in(model, dl.x_start))
        stations.add(_x_user_to_internal_in(model, dl.x_end))
    for pl in model.point_loads:
        stations.add(_x_user_to_internal_in(model, pl.x))
    for pm in model.point_moments:
        stations.add(_x_user_to_internal_in(model, pm.x))

    stations_list = sorted(stations)
    refined = [stations_list[0]]
    for a, b in zip(stations_list[:-1], stations_list[1:]):
        seg = b - a
        n = int(np.ceil(seg / max_len_in)) if seg > max_len_in else 1
        for i in range(1, n + 1):
            refined.append(a + seg * (i / n))

    # unique with tolerance
    refined.sort()
    uniq: List[float] = []
    tol = 1e-6
    for x in refined:
        if not uniq or abs(x - uniq[-1]) > tol:
            uniq.append(x)

    # identify physical joints
    joint_key = {round(x, 6): j for j, x in enumerate(span_stations)}
    nodes: List[Node] = []
    for x in uniq:
        key = round(x, 6)
        if key in joint_key:
            nodes.append(Node(x_in=x, is_physical_joint=True, joint_index=joint_key[key]))
        else:
            nodes.append(Node(x_in=x, is_physical_joint=False, joint_index=None))

    def span_index_for(x_in: float) -> int:
        for i in range(len(span_stations) - 1):
            if span_stations[i] - tol <= x_in <= span_stations[i + 1] + tol:
                return min(i, len(lengths_in) - 1)
        return len(lengths_in) - 1

    def q_internal_at(x_in: float) -> float:
        x_user = _x_internal_to_user(model, x_in)
        q = 0.0
        for dl in model.distributed_loads:
            if dl.x_start - 1e-12 <= x_user <= dl.x_end + 1e-12:
                t = 0.0 if abs(dl.x_end - dl.x_start) < 1e-12 else (x_user - dl.x_start) / (dl.x_end - dl.x_start)
                w_user = dl.w_start + (dl.w_end - dl.w_start) * t
                q += _distributed_load_user_to_internal(model, w_user)
        return q

    elements: List[Element] = []
    for i in range(len(nodes) - 1):
        x1 = nodes[i].x_in
        x2 = nodes[i + 1].x_in
        L = x2 - x1
        if L <= tol:
            continue
        mid = 0.5 * (x1 + x2)
        si = span_index_for(mid)
        q0 = q_internal_at(x1)
        q1 = q_internal_at(x2)
        elements.append(Element(i, i + 1, L, E_psi[si], I_in4[si], q0, q1))

    hinge_nodes = [False] * len(nodes)
    for i, node in enumerate(nodes):
        if node.is_physical_joint and node.joint_index is not None:
            if model.joints[node.joint_index].internal_hinge:
                hinge_nodes[i] = True

    v_dof = [-1] * len(nodes)
    t_left = [-1] * len(nodes)
    t_right = [-1] * len(nodes)

    dof = 0
    for i in range(len(nodes)):
        v_dof[i] = dof
        dof += 1
        if hinge_nodes[i] and 0 < i < len(nodes) - 1:
            t_left[i] = dof; dof += 1
            t_right[i] = dof; dof += 1
        else:
            t_left[i] = dof
            t_right[i] = dof
            dof += 1

    dof_map = DofMap(v_dof, t_left, t_right, hinge_nodes, dof)
    return nodes, elements, dof_map, {"total_length_in": total_L_in}


def _element_dofs(e: Element, dof_map: DofMap) -> List[int]:
    i, j = e.n1, e.n2
    # For rotation: element to right of node uses theta_right; to left uses theta_left
    return [dof_map.v_dof[i], dof_map.theta_dof_right[i], dof_map.v_dof[j], dof_map.theta_dof_left[j]]


# --------- Solve ---------

def solve_beam(model: BeamModel) -> Dict[str, object]:
    nodes, elements, dof_map, info = build_mesh(model)
    ndof = dof_map.ndof

    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    for e in elements:
        ke = _beam_element_stiffness(e.E_psi, e.I_in4, e.L_in)
        fe = _consistent_load_vector_linear(e.q0_lb_in, e.q1_lb_in, e.L_in)
        dofs = _element_dofs(e, dof_map)

        for a in range(4):
            A = dofs[a]
            F[A] += fe[a]
            for b in range(4):
                B = dofs[b]
                K[A, B] += ke[a, b]

    # Map x(in) to node index
    x_to_node = {round(n.x_in, 6): i for i, n in enumerate(nodes)}
    tol = 1e-4

    def find_node(x_in: float) -> int:
        key = round(x_in, 6)
        if key in x_to_node:
            return x_to_node[key]
        dists = [abs(n.x_in - x_in) for n in nodes]
        i = int(np.argmin(dists))
        if dists[i] > tol:
            raise ValueError(f"No node found at x={x_in} in (nearest {dists[i]} in).")
        return i

    for pl in model.point_loads:
        xi = _x_user_to_internal_in(model, pl.x)
        ni = find_node(xi)
        F[dof_map.v_dof[ni]] += _point_load_user_to_internal(model, pl.P)

    for pm in model.point_moments:
        xi = _x_user_to_internal_in(model, pm.x)
        ni = find_node(xi)
        M = _moment_user_to_internal(model, pm.M)
        if dof_map.hinge_nodes[ni] and 0 < ni < len(nodes) - 1:
            F[dof_map.theta_dof_left[ni]] += M
            F[dof_map.theta_dof_right[ni]] += M
        else:
            F[dof_map.theta_dof_left[ni]] += M

    # Boundary conditions at physical joints
    fixed: List[int] = []
    for i, node in enumerate(nodes):
        if not node.is_physical_joint or node.joint_index is None:
            continue
        j = model.joints[node.joint_index]
        if j.restraint_v:
            fixed.append(dof_map.v_dof[i])
        if j.restraint_theta:
            if dof_map.hinge_nodes[i] and 0 < i < len(nodes) - 1:
                fixed.append(dof_map.theta_dof_left[i])
                fixed.append(dof_map.theta_dof_right[i])
            else:
                fixed.append(dof_map.theta_dof_left[i])

    fixed_set = set(fixed)
    free = [i for i in range(ndof) if i not in fixed_set]

    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    d = np.zeros(ndof, dtype=float)

    try:
        d_free = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        d_free, *_ = np.linalg.lstsq(K_ff, F_f, rcond=None)

    for idx, dof_idx in enumerate(free):
        d[dof_idx] = d_free[idx]

    # reactions
    R = K @ d - F

    # Diagrams
    xs_in: List[float] = []
    Vs: List[float] = []
    Ms: List[float] = []
    vs_in: List[float] = []

    def v_in_element(e: Element, xloc: float) -> float:
        xi = xloc / e.L_in
        N = _shape_functions(xi, e.L_in)
        dofs = _element_dofs(e, dof_map)
        return float(N @ d[dofs])

    def end_forces(e: Element) -> np.ndarray:
        ke = _beam_element_stiffness(e.E_psi, e.I_in4, e.L_in)
        fe = _consistent_load_vector_linear(e.q0_lb_in, e.q1_lb_in, e.L_in)
        dofs = _element_dofs(e, dof_map)
        return ke @ d[dofs] - fe

    # choose points per element
    lengths_in, _, _, max_len_in = _to_internal_units(model)
    for e in elements:
        npts = max(10, int(np.ceil(e.L_in / max(1.0, max_len_in))) * 20)
        x0 = nodes[e.n1].x_in
        f = end_forces(e)
        V1 = float(f[0])
        M1 = float(f[1])
        L = e.L_in
        q0, q1 = e.q0_lb_in, e.q1_lb_in
        for k in range(npts + 1):
            xi = k / npts
            xloc = xi * L
            xg = x0 + xloc

            Vx = V1 - (q0 * xloc + (q1 - q0) * xloc**2 / (2.0 * L))
            Mx = M1 + V1 * xloc - q0 * xloc**2 / 2.0 - (q1 - q0) * xloc**3 / (6.0 * L)
            xs_in.append(xg)
            Vs.append(Vx)
            Ms.append(Mx)
            vs_in.append(v_in_element(e, xloc))

    x_user = np.array([_x_internal_to_user(model, x) for x in xs_in], dtype=float)
    V_user = np.array([_shear_internal_to_user(model, v) for v in Vs], dtype=float)
    M_user = np.array([_moment_internal_to_user(model, m) for m in Ms], dtype=float)
    v_user = np.array([_deflection_internal_to_user(model, vv) for vv in vs_in], dtype=float)

    mesh_x_user = np.array([_x_internal_to_user(model, n.x_in) for n in nodes], dtype=float)
    mesh_v_user = np.array([_deflection_internal_to_user(model, d[dof_map.v_dof[i]]) for i in range(len(nodes))], dtype=float)
    mesh_theta = np.array([d[dof_map.theta_dof_right[i]] for i in range(len(nodes))], dtype=float)

    # reactions at supports
    # Build physical joint x list from spans
    joint_x_user = [0.0]
    acc_u = 0.0
    for s in model.spans:
        acc_u += s.length
        joint_x_user.append(acc_u)

    joint_node_idx = [find_node(_x_user_to_internal_in(model, x)) for x in joint_x_user]

    reactions_user = []
    for j_idx, ni in enumerate(joint_node_idx):
        j = model.joints[j_idx]
        if j.restraint_v:
            reactions_user.append({
                "joint": j_idx,
                "x_user": joint_x_user[j_idx],
                "dof": "V",
                "reaction_user": _shear_internal_to_user(model, float(R[dof_map.v_dof[ni]])),
            })
        if j.restraint_theta:
            if dof_map.hinge_nodes[ni] and 0 < ni < len(nodes) - 1:
                reactions_user.append({
                    "joint": j_idx,
                    "x_user": joint_x_user[j_idx],
                    "dof": "M_left",
                    "reaction_user": _moment_internal_to_user(model, float(R[dof_map.theta_dof_left[ni]])),
                })
                reactions_user.append({
                    "joint": j_idx,
                    "x_user": joint_x_user[j_idx],
                    "dof": "M_right",
                    "reaction_user": _moment_internal_to_user(model, float(R[dof_map.theta_dof_right[ni]])),
                })
            else:
                reactions_user.append({
                    "joint": j_idx,
                    "x_user": joint_x_user[j_idx],
                    "dof": "M",
                    "reaction_user": _moment_internal_to_user(model, float(R[dof_map.theta_dof_left[ni]])),
                })

    labels = {
        "x": "ft" if model.unit_system == "IMPERIAL_FT_LB" else "m",
        "V": "lb" if model.unit_system == "IMPERIAL_FT_LB" else "kN",
        "M": "ft-lb" if model.unit_system == "IMPERIAL_FT_LB" else "kN-m",
        "v": "in" if model.unit_system == "IMPERIAL_FT_LB" else "m",
    }

    return {
        "unit_system": model.unit_system,
        "sign_convention": {
            "v_positive": "upward",
            "theta_positive": "CCW",
            "user_loads": {"w": "positive downward", "P": "positive downward", "M": "positive clockwise"},
        },
        "inputs": model.model_dump(),
        "mesh": {
            "node_x_user": mesh_x_user.tolist(),
            "node_v_user": mesh_v_user.tolist(),
            "node_theta_rad": mesh_theta.tolist(),
            "hinge_nodes": dof_map.hinge_nodes,
            "n_nodes": len(nodes),
            "n_elements": len(elements),
        },
        "reactions": reactions_user,
        "diagrams": {
            "x_user": x_user.tolist(),
            "shear_user": V_user.tolist(),
            "moment_user": M_user.tolist(),
            "deflection_user": v_user.tolist(),
            "labels": labels,
        },
        "stability": {
            "note": "If the structure is unstable/singular, a least-squares solution is used; verify supports/hinges.",
        },
    }
