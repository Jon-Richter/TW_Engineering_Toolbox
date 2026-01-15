from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets


_beam_solver = importlib.import_module("toolbox_app.blocks.1D_beam_solver")
BeamModel = _beam_solver.BeamModel
Span = _beam_solver.Span
Joint = _beam_solver.Joint
DistributedLoad = _beam_solver.DistributedLoad
PointLoad = _beam_solver.PointLoad
PointMoment = _beam_solver.PointMoment
solve_beam = _beam_solver.solve_beam

from .exports import export_results_excel, export_mathcad_assignments, export_results_json


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)


class _SolveWorker(QtCore.QRunnable):
    def __init__(self, model: BeamModel):
        super().__init__()
        self.model = model
        self.signals = _WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            res = solve_beam(self.model)
            self.signals.finished.emit(res)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


@dataclass
class UiDefaults:
    unit_system: str = "IMPERIAL_FT_LB"
    span_E_default: float = 29_000_000.0
    span_I_default: float = 100.0
    span_L_default: float = 10.0
    mesh_max_imperial_ft: float = 0.1
    mesh_max_si_m: float = 0.03048


class BeamAnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, output_dir: Path, defaults: Optional[UiDefaults] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Beam Analysis (Continuous Beam FEM)")
        self.setMinimumSize(1100, 700)
        self.resize(int(1100 * 1.2), 700)

        self._output_dir = output_dir
        self._defaults = defaults or UiDefaults()
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._last_results: Optional[Dict[str, Any]] = None
        self._suppress_live_update = False

        self._build_ui()
        self._init_plot_axes()
        self._apply_defaults()
        self._rebuild_spans_and_joints()

    def _build_ui(self) -> None:
        # Lazy matplotlib import to minimize any UI blocking during tool start.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: inputs
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        gbox = QtWidgets.QGroupBox("General")
        gl = QtWidgets.QGridLayout(gbox)

        gl.addWidget(QtWidgets.QLabel("Unit system:"), 0, 0)
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItem("Imperial (ft, lb, psi, in^4)", "IMPERIAL_FT_LB")
        self.unit_combo.addItem("SI (m, kN, Pa, m^4)", "SI_M_KN")
        gl.addWidget(self.unit_combo, 0, 1)

        gl.addWidget(QtWidgets.QLabel("Spans:"), 1, 0)
        self.span_count = QtWidgets.QSpinBox()
        self.span_count.setRange(1, 20)
        gl.addWidget(self.span_count, 1, 1)

        left_layout.addWidget(gbox)

        sbox = QtWidgets.QGroupBox("Spans (properties)")
        sl = QtWidgets.QVBoxLayout(sbox)
        self.spans_table = QtWidgets.QTableWidget()
        self.spans_table.setColumnCount(3)
        self.spans_table.setHorizontalHeaderLabels(["Length (ft)", "E (psi)", "I (in^4)"])
        self.spans_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.spans_table.setMinimumHeight(180)
        sl.addWidget(self.spans_table)
        left_layout.addWidget(sbox, 3)

        jbox = QtWidgets.QGroupBox("Joints / Supports / Releases")
        jl = QtWidgets.QVBoxLayout(jbox)
        self.joints_table = QtWidgets.QTableWidget()
        self.joints_table.setColumnCount(5)
        self.joints_table.setHorizontalHeaderLabels(["Joint", "X", "V restraint", "θ restraint", "Internal hinge"])
        self.joints_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.joints_table.setMinimumHeight(200)
        jl.addWidget(self.joints_table)
        left_layout.addWidget(jbox, 3)

        tbox = QtWidgets.QGroupBox("Loads (global X)")
        min_input_table_h = max(self.spans_table.minimumHeight(), self.joints_table.minimumHeight())
        # This panel has tabs + buttons + sign note, so it needs extra height to keep the tables usable.
        tbox.setMinimumHeight(min_input_table_h + 120)
        tl = QtWidgets.QVBoxLayout(tbox)
        self.load_tabs = QtWidgets.QTabWidget()
        self.load_tabs.setMinimumHeight(min_input_table_h + 60)

        # Distributed loads
        self.dl_table = QtWidgets.QTableWidget()
        self.dl_table.setColumnCount(4)
        self.dl_table.setHorizontalHeaderLabels(["x_start", "x_end", "w_start", "w_end"])
        self.dl_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.dl_table.setMinimumHeight(min_input_table_h)
        dl_page = QtWidgets.QWidget()
        dl_l = QtWidgets.QVBoxLayout(dl_page)
        dl_l.addWidget(self.dl_table)
        dl_btns = QtWidgets.QHBoxLayout()
        self.dl_add = QtWidgets.QPushButton("Add")
        self.dl_del = QtWidgets.QPushButton("Remove selected")
        dl_btns.addWidget(self.dl_add)
        dl_btns.addWidget(self.dl_del)
        dl_btns.addStretch(1)
        dl_l.addLayout(dl_btns)
        self.load_tabs.addTab(dl_page, "Distributed")

        # Point loads
        self.pl_table = QtWidgets.QTableWidget()
        self.pl_table.setColumnCount(2)
        self.pl_table.setHorizontalHeaderLabels(["x", "P"])
        self.pl_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.pl_table.setMinimumHeight(min_input_table_h)
        pl_page = QtWidgets.QWidget()
        pl_l = QtWidgets.QVBoxLayout(pl_page)
        pl_l.addWidget(self.pl_table)
        pl_btns = QtWidgets.QHBoxLayout()
        self.pl_add = QtWidgets.QPushButton("Add")
        self.pl_del = QtWidgets.QPushButton("Remove selected")
        pl_btns.addWidget(self.pl_add)
        pl_btns.addWidget(self.pl_del)
        pl_btns.addStretch(1)
        pl_l.addLayout(pl_btns)
        self.load_tabs.addTab(pl_page, "Point Loads")

        # Point moments
        self.pm_table = QtWidgets.QTableWidget()
        self.pm_table.setColumnCount(2)
        self.pm_table.setHorizontalHeaderLabels(["x", "M"])
        self.pm_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.pm_table.setMinimumHeight(min_input_table_h)
        pm_page = QtWidgets.QWidget()
        pm_l = QtWidgets.QVBoxLayout(pm_page)
        pm_l.addWidget(self.pm_table)
        pm_btns = QtWidgets.QHBoxLayout()
        self.pm_add = QtWidgets.QPushButton("Add")
        self.pm_del = QtWidgets.QPushButton("Remove selected")
        pm_btns.addWidget(self.pm_add)
        pm_btns.addWidget(self.pm_del)
        pm_btns.addStretch(1)
        pm_l.addLayout(pm_btns)
        self.load_tabs.addTab(pm_page, "Point Moments")

        tl.addWidget(self.load_tabs)

        self.sign_label = QtWidgets.QLabel(
            "Sign convention: v up, θ CCW. Enter loads as: w>0 downward, P>0 downward, M>0 clockwise."
        )
        self.sign_label.setWordWrap(True)
        tl.addWidget(self.sign_label)

        left_layout.addWidget(tbox, 3)

        abox = QtWidgets.QGroupBox("Actions")
        al = QtWidgets.QGridLayout(abox)
        self.calc_btn = QtWidgets.QPushButton("Calculate")
        self.export_xlsx_btn = QtWidgets.QPushButton("Export Excel (.xlsx)")
        self.export_mathcad_btn = QtWidgets.QPushButton("Export Mathcad handoff (.txt)")
        self.export_json_btn = QtWidgets.QPushButton("Export JSON (.json)")
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)

        al.addWidget(self.calc_btn, 0, 0, 1, 2)
        al.addWidget(self.export_xlsx_btn, 1, 0)
        al.addWidget(self.export_mathcad_btn, 1, 1)
        al.addWidget(self.export_json_btn, 2, 0, 1, 2)
        al.addWidget(self.status_lbl, 3, 0, 1, 2)
        left_layout.addWidget(abox)

        left_layout.addStretch(1)

        # Right panel: plots + reactions
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 5)

        rbox = QtWidgets.QGroupBox("Reactions")
        rl = QtWidgets.QVBoxLayout(rbox)
        self.reactions_table = QtWidgets.QTableWidget()
        self.reactions_table.setColumnCount(4)
        self.reactions_table.setHorizontalHeaderLabels(["Joint", "X", "DOF", "Reaction"])
        self.reactions_table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.reactions_table)
        right_layout.addWidget(rbox, 2)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # connections
        self.span_count.valueChanged.connect(self._rebuild_spans_and_joints)
        self.unit_combo.currentIndexChanged.connect(self._unit_changed)
        self.spans_table.itemChanged.connect(self._on_span_table_changed)
        self.joints_table.itemChanged.connect(self._on_inputs_changed)
        self.dl_table.itemChanged.connect(self._on_inputs_changed)
        self.pl_table.itemChanged.connect(self._on_inputs_changed)
        self.pm_table.itemChanged.connect(self._on_inputs_changed)

        self.dl_add.clicked.connect(lambda: self._add_row(self.dl_table, [0.0, 0.0, 0.0, 0.0]))
        self.dl_del.clicked.connect(lambda: self._remove_selected(self.dl_table))
        self.pl_add.clicked.connect(lambda: self._add_row(self.pl_table, [0.0, 0.0]))
        self.pl_del.clicked.connect(lambda: self._remove_selected(self.pl_table))
        self.pm_add.clicked.connect(lambda: self._add_row(self.pm_table, [0.0, 0.0]))
        self.pm_del.clicked.connect(lambda: self._remove_selected(self.pm_table))

        self.calc_btn.clicked.connect(self._calculate)
        self.export_xlsx_btn.clicked.connect(self._export_xlsx)
        self.export_mathcad_btn.clicked.connect(self._export_mathcad)
        self.export_json_btn.clicked.connect(self._export_json)

    def _init_plot_axes(self) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(4, 1, height_ratios=[0.7, 1.0, 1.0, 1.0], hspace=0.35)
        self.ax_load = self.fig.add_subplot(gs[0, 0])
        self.ax_shear = self.fig.add_subplot(gs[1, 0], sharex=self.ax_load)
        self.ax_moment = self.fig.add_subplot(gs[2, 0], sharex=self.ax_load)
        self.ax_defl = self.fig.add_subplot(gs[3, 0], sharex=self.ax_load)

        self.line_shear, = self.ax_shear.plot([], [])
        self.line_moment, = self.ax_moment.plot([], [])
        self.line_defl, = self.ax_defl.plot([], [])

        self.ax_shear.grid(True)
        self.ax_moment.grid(True)
        self.ax_defl.grid(True)

        self.ax_shear.tick_params(axis="y", pad=6)
        self.ax_moment.tick_params(axis="y", pad=6)
        self.ax_defl.tick_params(axis="y", pad=6)

        self.ax_load.tick_params(labelbottom=False)
        self.ax_shear.tick_params(labelbottom=False)
        self.ax_moment.tick_params(labelbottom=False)

        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.35)

    def _apply_defaults(self) -> None:
        self._suppress_live_update = True
        try:
            self.unit_combo.setCurrentIndex(0 if self._defaults.unit_system == "IMPERIAL_FT_LB" else 1)
            self.span_count.setValue(3)
            self._unit_changed()

            # sample rows
            self._add_row(self.dl_table, [0.0, 6.0, 10.0, 10.0])
            self._add_row(self.pl_table, [6.0, 100.0])
        finally:
            self._suppress_live_update = False
        self._render_plots()

    def _unit_changed(self) -> None:
        us = self.unit_combo.currentData()
        if us == "IMPERIAL_FT_LB":
            self.spans_table.setHorizontalHeaderLabels(["Length (ft)", "E (psi)", "I (in^4)"])
            self.dl_table.setHorizontalHeaderLabels(["x_start (ft)", "x_end (ft)", "w_start (lb/ft)", "w_end (lb/ft)"])
            self.pl_table.setHorizontalHeaderLabels(["x (ft)", "P (lb)"])
            self.pm_table.setHorizontalHeaderLabels(["x (ft)", "M (ft-lb)"])
        else:
            self.spans_table.setHorizontalHeaderLabels(["Length (m)", "E (Pa)", "I (m^4)"])
            self.dl_table.setHorizontalHeaderLabels(["x_start (m)", "x_end (m)", "w_start (kN/m)", "w_end (kN/m)"])
            self.pl_table.setHorizontalHeaderLabels(["x (m)", "P (kN)"])
            self.pm_table.setHorizontalHeaderLabels(["x (m)", "M (kN-m)"])
        self._on_inputs_changed()

    def _on_span_table_changed(self, *_args) -> None:
        if self._suppress_live_update:
            return
        self._suppress_live_update = True
        try:
            self._update_joint_positions()
        finally:
            self._suppress_live_update = False
        self._on_inputs_changed()

    def _on_inputs_changed(self, *_args) -> None:
        if self._suppress_live_update:
            return
        if self._last_results is not None:
            self._last_results = None
            self.status_lbl.setText("Inputs changed. Run Calculate to update results.")
        self._render_plots()

    def _collect_load_plot_data(self) -> Dict[str, Any]:
        unit_system = str(self.unit_combo.currentData())
        spans: List[float] = []
        for r in range(self.spans_table.rowCount()):
            L = self._cell_float(self.spans_table, r, 0, default=0.0)
            spans.append(max(0.0, L))

        joint_x = [0.0]
        acc = 0.0
        for L in spans:
            acc += L
            joint_x.append(acc)

        total_length = acc
        plot_length = total_length if total_length > 0 else 1.0

        supports = []
        for r in range(self.joints_table.rowCount()):
            x = joint_x[r] if r < len(joint_x) else plot_length
            v_combo: QtWidgets.QComboBox = self.joints_table.cellWidget(r, 2)  # type: ignore
            t_combo: QtWidgets.QComboBox = self.joints_table.cellWidget(r, 3)  # type: ignore
            hinge_cb: QtWidgets.QCheckBox = self.joints_table.cellWidget(r, 4)  # type: ignore
            supports.append({
                "x": x,
                "v_fixed": bool(v_combo.currentData()) if v_combo else True,
                "t_fixed": bool(t_combo.currentData()) if t_combo else False,
                "hinge": bool(hinge_cb.isChecked()) if hinge_cb else False,
            })

        distributed = []
        for r in range(self.dl_table.rowCount()):
            xs = self._cell_float(self.dl_table, r, 0, 0.0)
            xe = self._cell_float(self.dl_table, r, 1, 0.0)
            w1 = self._cell_float(self.dl_table, r, 2, 0.0)
            w2 = self._cell_float(self.dl_table, r, 3, 0.0)
            if xe <= xs:
                continue
            distributed.append({"x_start": xs, "x_end": xe, "w_start": w1, "w_end": w2})

        point_loads = []
        for r in range(self.pl_table.rowCount()):
            x = self._cell_float(self.pl_table, r, 0, 0.0)
            P = self._cell_float(self.pl_table, r, 1, 0.0)
            if P == 0:
                continue
            point_loads.append({"x": x, "P": P})

        point_moments = []
        for r in range(self.pm_table.rowCount()):
            x = self._cell_float(self.pm_table, r, 0, 0.0)
            M = self._cell_float(self.pm_table, r, 1, 0.0)
            if M == 0:
                continue
            point_moments.append({"x": x, "M": M})

        unit_label = "ft" if unit_system == "IMPERIAL_FT_LB" else "m"
        return {
            "unit_system": unit_system,
            "unit_label": unit_label,
            "plot_length": plot_length,
            "total_length": total_length,
            "supports": supports,
            "distributed_loads": distributed,
            "point_loads": point_loads,
            "point_moments": point_moments,
        }

    def _plot_load_diagram(self, ax, reactions: Optional[List[Dict[str, Any]]] = None) -> None:
        ax.clear()
        data = self._collect_load_plot_data()
        L = data["plot_length"]

        mags: List[float] = []
        for dl in data["distributed_loads"]:
            mags.extend([abs(dl["w_start"]), abs(dl["w_end"])])
        for pl in data["point_loads"]:
            mags.append(abs(pl["P"]))
        if reactions:
            for r in reactions:
                if str(r.get("dof")) == "V":
                    mags.append(abs(float(r.get("reaction_user", 0.0))))

        max_mag = max(mags) if mags else 1.0
        if max_mag == 0:
            max_mag = 1.0

        scale = 0.9 / max_mag
        y_limit = 1.6

        ax.axhline(0, color="#333333", linewidth=1.0)

        # supports / hinges
        support_offset = -y_limit * 0.12
        edge_tol = L * 0.05

        def draw_fixed_support(x: float) -> None:
            wall_top = 0.0
            wall_bot = support_offset
            ax.plot([x, x], [wall_bot, wall_top], color="#333333", linewidth=2.0)

            hatch_len = max(L * 0.02, 0.06)
            hatch_drop = max(abs(wall_top - wall_bot) * 0.10, 0.04)

            if x <= edge_tol:
                dirs = [-1.0]
            elif x >= (L - edge_tol):
                dirs = [1.0]
            else:
                dirs = [-1.0, 1.0]

            ys = np.linspace(wall_top - 0.02, wall_bot + 0.02, 4)
            for d in dirs:
                for yy in ys:
                    ax.plot([x, x + d * hatch_len], [yy, yy - hatch_drop], color="#333333", linewidth=1.2)

        for s in data["supports"]:
            x = float(s["x"])
            if s["hinge"]:
                ax.plot(x, 0, marker="o", markersize=5, markerfacecolor="white", color="#333333")

            v_fixed = bool(s.get("v_fixed"))
            t_fixed = bool(s.get("t_fixed"))

            if v_fixed and t_fixed:
                draw_fixed_support(x)
            elif v_fixed:
                ax.plot(x, support_offset, marker="^", markersize=9, markerfacecolor="white", color="#333333")
            else:
                ax.plot(x, support_offset, marker="o", markersize=4, markerfacecolor="white", color="#333333")

            if t_fixed and not v_fixed:
                ax.plot(x, support_offset, marker="s", markersize=6, color="#333333")

        # distributed loads (plot above the beam)
        for dl in data["distributed_loads"]:
            xs = max(0.0, float(dl["x_start"]))
            xe = min(L, float(dl["x_end"]))
            if xe <= xs:
                continue
            y1 = abs(float(dl["w_start"])) * scale
            y2 = abs(float(dl["w_end"])) * scale
            ax.fill([xs, xs, xe, xe], [0, y1, y2, 0], color="#4c78a8", alpha=0.25, linewidth=0)
            ax.plot([xs, xe], [y1, y2], color="#4c78a8", linewidth=1.3)
            ax.plot([xs, xs], [0, y1], color="#4c78a8", linewidth=1.0)
            ax.plot([xe, xe], [0, y2], color="#4c78a8", linewidth=1.0)

        # point loads (plot above the beam)
        for pl in data["point_loads"]:
            x = min(L, max(0.0, float(pl["x"])))
            P = float(pl["P"])
            y = min(y_limit * 0.95, max(abs(P) * scale * 1.35, y_limit * 0.30))
            ax.annotate(
                "",
                xy=(x, 0 if P >= 0 else y),
                xytext=(x, y if P >= 0 else 0),
                arrowprops=dict(arrowstyle="-|>", color="#e45756", lw=1.8),
            )

        # point moments
        for pm in data["point_moments"]:
            x = min(L, max(0.0, float(pm["x"])))
            M = float(pm["M"])
            rad = -0.5 if M > 0 else 0.5
            dx = max(L * 0.02, 0.05)
            y = y_limit * 0.55
            ax.annotate(
                "",
                xy=(x + dx, y),
                xytext=(x - dx, y),
                arrowprops=dict(arrowstyle="->", color="#f28e2b", lw=1.6, connectionstyle=f"arc3,rad={rad}"),
            )

        # reactions
        if reactions:
            for r in reactions:
                if str(r.get("dof")) != "V":
                    continue
                x = min(L, max(0.0, float(r.get("x_user", 0.0))))
                R = float(r.get("reaction_user", 0.0))
                y = abs(R) * scale
                ax.annotate(
                    "",
                    xy=(x, y if R >= 0 else 0),
                    xytext=(x, 0 if R >= 0 else y),
                    arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=1.8),
                )

        ax.set_xlim(0, L)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title("Loading / Supports", fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelbottom=False)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    def _adjust_left_margin(self) -> None:
        try:
            self.canvas.draw()
            renderer = self.canvas.get_renderer()
        except Exception:
            return

        fig = self.fig
        min_x = None
        for ax in fig.axes:
            for label in ax.get_yticklabels():
                if not label.get_visible():
                    continue
                bbox = label.get_window_extent(renderer)
                min_x = bbox.x0 if min_x is None else min(min_x, bbox.x0)
            ylabel = ax.yaxis.label
            if ylabel.get_text():
                bbox = ylabel.get_window_extent(renderer)
                min_x = bbox.x0 if min_x is None else min(min_x, bbox.x0)

        if min_x is None:
            return

        pad_px = 6
        fig_width_px = fig.get_figwidth() * fig.dpi
        left_current = fig.subplotpars.left
        left_new = left_current + (pad_px - min_x) / fig_width_px
        left_new = max(0.05, min(0.45, left_new))
        if abs(left_new - left_current) > 1e-3:
            fig.subplots_adjust(left=left_new)

    def _resample_diagram(self, x: List[float], y: List[float], target_points: int = 1200) -> tuple[List[float], List[float]]:
        if not x or not y or len(x) != len(y) or target_points < 2:
            return x, y

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size < 3:
            return x, y

        if np.any(np.diff(x_arr) < 0):
            idx = np.argsort(x_arr, kind="mergesort")
            x_arr = x_arr[idx]
            y_arr = y_arr[idx]

        x_span = float(x_arr[-1] - x_arr[0])
        x_tol = max(1e-12, abs(x_span) * 1e-12)

        y_max = float(np.nanmax(y_arr))
        y_min = float(np.nanmin(y_arr))
        y_range = max(0.0, y_max - y_min)
        close_tol = max(1e-12, y_range * 1e-4)
        jump_tol = max(close_tol * 10.0, y_range * 1e-2)

        # Collapse repeated x points that arise from per-element sampling. Preserve true jumps (e.g. point loads).
        cx: List[float] = []
        cy: List[float] = []
        n = int(x_arr.size)
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(float(x_arr[j] - x_arr[i])) <= x_tol:
                j += 1
            if j == i + 1:
                cx.append(float(x_arr[i]))
                cy.append(float(y_arr[i]))
            else:
                ys = y_arr[i:j]
                if float(np.nanmax(ys) - np.nanmin(ys)) <= close_tol:
                    cx.append(float(x_arr[i]))
                    cy.append(float(np.nanmean(ys)))
                else:
                    cx.append(float(x_arr[i]))
                    cy.append(float(ys[0]))
                    cx.append(float(x_arr[i]))
                    cy.append(float(ys[-1]))
            i = j

        x2 = np.asarray(cx, dtype=float)
        y2 = np.asarray(cy, dtype=float)
        if x2.size < 3:
            return cx, cy

        x_span2 = float(x2[-1] - x2[0])
        if x_span2 <= x_tol:
            return cx, cy

        dx = x_span2 / float(target_points - 1)

        out_x: List[float] = []
        out_y: List[float] = []

        def append_segment(xs: np.ndarray, ys: np.ndarray) -> None:
            if xs.size == 0:
                return
            if xs.size == 1 or float(xs[-1] - xs[0]) <= x_tol:
                out_x.append(float(xs[0]))
                out_y.append(float(ys[0]))
                return

            npts = int(max(2, round(float(xs[-1] - xs[0]) / dx) + 1))
            xi = np.linspace(float(xs[0]), float(xs[-1]), npts)
            yi = np.interp(xi, xs, ys)

            if out_x and abs(out_x[-1] - float(xi[0])) <= x_tol:
                xi = xi[1:]
                yi = yi[1:]

            out_x.extend([float(v) for v in xi])
            out_y.extend([float(v) for v in yi])

        start = 0
        k = 0
        while k < x2.size - 1:
            is_jump = abs(float(x2[k + 1] - x2[k])) <= x_tol and abs(float(y2[k + 1] - y2[k])) >= jump_tol
            if is_jump:
                append_segment(x2[start : k + 1], y2[start : k + 1])
                out_x.append(float(x2[k + 1]))
                out_y.append(float(y2[k + 1]))
                start = k + 1
                k += 1
            k += 1
        append_segment(x2[start:], y2[start:])

        return out_x, out_y

    def _render_plots(self, results: Optional[Dict[str, Any]] = None) -> None:
        if results:
            self._plot_load_diagram(self.ax_load, reactions=results.get("reactions", []))

            d = results["diagrams"]
            x = d["x_user"]
            V = d["shear_user"]
            M = d["moment_user"]
            v = d["deflection_user"]
            labels = d.get("labels", {})

            xV, VV = self._resample_diagram(x, V)
            self.line_shear.set_data(xV, VV)
            self.ax_shear.set_ylabel(f"Shear ({labels.get('V','')})")
            self.ax_shear.relim()
            self.ax_shear.autoscale_view(scalex=False, scaley=True)

            xM, MM = self._resample_diagram(x, M)
            self.line_moment.set_data(xM, MM)
            self.ax_moment.set_ylabel(f"Moment ({labels.get('M','')})")
            self.ax_moment.relim()
            self.ax_moment.autoscale_view(scalex=False, scaley=True)

            xv, vv = self._resample_diagram(x, v)
            self.line_defl.set_data(xv, vv)
            self.ax_defl.set_xlabel(f"x ({labels.get('x','')})")
            self.ax_defl.set_ylabel(f"Defl. ({labels.get('v','')})")
            self.ax_defl.relim()
            self.ax_defl.autoscale_view(scalex=False, scaley=True)
        else:
            self._plot_load_diagram(self.ax_load, reactions=None)
            self.line_shear.set_data([], [])
            self.line_moment.set_data([], [])
            self.line_defl.set_data([], [])
            self.ax_shear.relim()
            self.ax_shear.autoscale_view(scalex=False, scaley=True)
            self.ax_moment.relim()
            self.ax_moment.autoscale_view(scalex=False, scaley=True)
            self.ax_defl.relim()
            self.ax_defl.autoscale_view(scalex=False, scaley=True)
            self.ax_defl.set_xlabel(f"x ({self._collect_load_plot_data()['unit_label']})")

        self.canvas.draw_idle()

    # ----- table helpers -----

    def _ensure_rows(self, table: QtWidgets.QTableWidget, rows: int) -> None:
        table.setRowCount(rows)
        for r in range(rows):
            for c in range(table.columnCount()):
                if table.item(r, c) is None and table.cellWidget(r, c) is None:
                    table.setItem(r, c, QtWidgets.QTableWidgetItem(""))

    def _set_if_empty(self, table: QtWidgets.QTableWidget, r: int, c: int, value: float) -> None:
        it = table.item(r, c)
        if it is None:
            table.setItem(r, c, QtWidgets.QTableWidgetItem(str(value)))
            return
        if str(it.text()).strip() == "":
            it.setText(str(value))

    def _cell_float(self, table: QtWidgets.QTableWidget, r: int, c: int, default: float = 0.0) -> float:
        it = table.item(r, c)
        if it is None:
            return default
        try:
            return float(it.text())
        except Exception:
            return default

    def _add_row(self, table: QtWidgets.QTableWidget, values: List[float]) -> None:
        r = table.rowCount()
        table.setRowCount(r + 1)
        for c, v in enumerate(values):
            table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self._on_inputs_changed()

    def _remove_selected(self, table: QtWidgets.QTableWidget) -> None:
        rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
        for r in rows:
            table.removeRow(r)
        self._on_inputs_changed()

    # ----- rebuild spans / joints -----

    def _rebuild_spans_and_joints(self) -> None:
        self._suppress_live_update = True
        try:
            n = int(self.span_count.value())
            self._ensure_rows(self.spans_table, n)
            self._ensure_rows(self.joints_table, n + 1)

            for r in range(n):
                self._set_if_empty(self.spans_table, r, 0, self._defaults.span_L_default)
                self._set_if_empty(self.spans_table, r, 1, self._defaults.span_E_default)
                self._set_if_empty(self.spans_table, r, 2, self._defaults.span_I_default)

            for r in range(n + 1):
                self.joints_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))
                self.joints_table.setItem(r, 1, QtWidgets.QTableWidgetItem(""))

                v_combo = QtWidgets.QComboBox()
                v_combo.addItem("Fixed", True)
                v_combo.addItem("Free", False)
                v_combo.setCurrentIndex(0)
                v_combo.currentIndexChanged.connect(self._on_inputs_changed)
                self.joints_table.setCellWidget(r, 2, v_combo)

                t_combo = QtWidgets.QComboBox()
                t_combo.addItem("Free (pinned/roller)", False)
                t_combo.addItem("Fixed (built-in)", True)
                t_combo.setCurrentIndex(0)
                t_combo.currentIndexChanged.connect(self._on_inputs_changed)
                self.joints_table.setCellWidget(r, 3, t_combo)

                cb = QtWidgets.QCheckBox()
                cb.setChecked(False)
                cb.setEnabled(0 < r < n)
                cb.stateChanged.connect(self._on_inputs_changed)
                self.joints_table.setCellWidget(r, 4, cb)

            self._update_joint_positions()
        finally:
            self._suppress_live_update = False
        self._on_inputs_changed()

    def _update_joint_positions(self) -> None:
        n = int(self.span_count.value())
        xs = [0.0]
        acc = 0.0
        for r in range(n):
            L = self._cell_float(self.spans_table, r, 0, default=0.0)
            acc += L
            xs.append(acc)

        for r, x in enumerate(xs):
            item = QtWidgets.QTableWidgetItem(f"{x:.6g}")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.joints_table.setItem(r, 1, item)

    # ----- build model / compute -----

    def _build_model(self) -> BeamModel:
        self._suppress_live_update = True
        try:
            self._update_joint_positions()
        finally:
            self._suppress_live_update = False
        unit_system = str(self.unit_combo.currentData())
        mesh_max = self._defaults.mesh_max_imperial_ft if unit_system == "IMPERIAL_FT_LB" else self._defaults.mesh_max_si_m

        spans: List[Span] = []
        for r in range(self.spans_table.rowCount()):
            spans.append(Span(
                length=self._cell_float(self.spans_table, r, 0, 0.0),
                E=self._cell_float(self.spans_table, r, 1, 0.0),
                I=self._cell_float(self.spans_table, r, 2, 0.0),
            ))

        joints: List[Joint] = []
        for r in range(self.joints_table.rowCount()):
            v_combo: QtWidgets.QComboBox = self.joints_table.cellWidget(r, 2)  # type: ignore
            t_combo: QtWidgets.QComboBox = self.joints_table.cellWidget(r, 3)  # type: ignore
            hinge_cb: QtWidgets.QCheckBox = self.joints_table.cellWidget(r, 4)  # type: ignore
            joints.append(Joint(
                restraint_v=bool(v_combo.currentData()) if v_combo else True,
                restraint_theta=bool(t_combo.currentData()) if t_combo else False,
                internal_hinge=bool(hinge_cb.isChecked()) if hinge_cb else False,
            ))

        dls: List[DistributedLoad] = []
        for r in range(self.dl_table.rowCount()):
            xs = self._cell_float(self.dl_table, r, 0, 0.0)
            xe = self._cell_float(self.dl_table, r, 1, 0.0)
            w1 = self._cell_float(self.dl_table, r, 2, 0.0)
            w2 = self._cell_float(self.dl_table, r, 3, 0.0)
            if xe == xs and w1 == 0 and w2 == 0:
                continue
            dls.append(DistributedLoad(x_start=xs, x_end=xe, w_start=w1, w_end=w2))

        pls: List[PointLoad] = []
        for r in range(self.pl_table.rowCount()):
            x = self._cell_float(self.pl_table, r, 0, 0.0)
            P = self._cell_float(self.pl_table, r, 1, 0.0)
            if P == 0:
                continue
            pls.append(PointLoad(x=x, P=P))

        pms: List[PointMoment] = []
        for r in range(self.pm_table.rowCount()):
            x = self._cell_float(self.pm_table, r, 0, 0.0)
            M = self._cell_float(self.pm_table, r, 1, 0.0)
            if M == 0:
                continue
            pms.append(PointMoment(x=x, M=M))

        return BeamModel(
            unit_system=unit_system,
            spans=spans,
            joints=joints,
            distributed_loads=dls,
            point_loads=pls,
            point_moments=pms,
            mesh_max_element_length=mesh_max,
        )

    def _calculate(self) -> None:
        self.status_lbl.setText("Solving...")
        self.calc_btn.setEnabled(False)

        try:
            model = self._build_model()
        except Exception as e:
            self.calc_btn.setEnabled(True)
            self.status_lbl.setText(f"Input validation failed: {e}")
            return

        worker = _SolveWorker(model)
        worker.signals.finished.connect(self._on_solved)
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    def _on_error(self, tb: str) -> None:
        self.calc_btn.setEnabled(True)
        self.status_lbl.setText("Solve failed. See error dialog.")
        QtWidgets.QMessageBox.critical(self, "Beam Analysis - Error", tb)

    def _on_solved(self, results: dict) -> None:
        self._last_results = results
        self.calc_btn.setEnabled(True)
        self.status_lbl.setText("Solved. You can export results.")
        self._render_plots(results)
        self._populate_reactions(results)

    def _populate_reactions(self, results: Dict[str, Any]) -> None:
        rs = results.get("reactions", [])
        self.reactions_table.setRowCount(len(rs))
        for r, row in enumerate(rs):
            self.reactions_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(row.get("joint"))))
            self.reactions_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{row.get('x_user'):.6g}"))
            self.reactions_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(row.get("dof"))))
            self.reactions_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{row.get('reaction_user'):.6g}"))

    # ----- exports -----

    def _export_xlsx(self) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.information(self, "Export", "Run Calculate first.")
            return
        p = self._output_dir / "beam_results.xlsx"
        export_results_excel(self._last_results, p)
        QtWidgets.QMessageBox.information(self, "Export", f"Excel saved to:\n{p}")

    def _export_mathcad(self) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.information(self, "Export", "Run Calculate first.")
            return
        p = self._output_dir / "beam_results_mathcad.txt"
        export_mathcad_assignments(self._last_results, p)
        QtWidgets.QMessageBox.information(self, "Export", f"Mathcad handoff saved to:\n{p}")

    def _export_json(self) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.information(self, "Export", "Run Calculate first.")
            return
        p = self._output_dir / "beam_results.json"
        export_results_json(self._last_results, p)
        QtWidgets.QMessageBox.information(self, "Export", f"JSON saved to:\n{p}")
