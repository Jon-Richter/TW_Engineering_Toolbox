from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .constants import EDGE_LOAD_OUTBOARD_IN, TOP_DIAG_END_OFFSET_IN

PYSIDE6_AVAILABLE = False

try:
    from PySide6.QtCore import QObject, QPointF, QRunnable, Qt, QThreadPool, Signal
    from PySide6.QtGui import QBrush, QPainter, QPen, QPolygonF
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE6_AVAILABLE = True
except Exception:
    # Allow importing the tool package in non-Qt environments (e.g., CI tests).
    PYSIDE6_AVAILABLE = False


def launch_ui(tool: Any, initial_inputs: Optional[Dict[str, Any]] = None, existing_window: Any = None) -> Any:
    """Launch the interactive Qt UI.

    The host Engineering Toolbox app provides a QApplication.
    This function reuses it if present.
    """

    if not PYSIDE6_AVAILABLE:
        raise RuntimeError(
            "PySide6 is not available in this Python environment. "
            "The C49 tool UI requires the Engineering Toolbox host app (Qt)."
        )

    app = QApplication.instance()
    if app is None:
        # Fallback for standalone debugging.
        app = QApplication([])

    if existing_window is not None:
        try:
            existing_window.show()
            existing_window.raise_()
            existing_window.activateWindow()
        except Exception:
            pass
        return existing_window

    win = C49MainWindow(tool=tool, initial_inputs=initial_inputs or {})
    win.show()
    return win


if PYSIDE6_AVAILABLE:

    class _TaskSignals(QObject):
        finished = Signal(object)
        failed = Signal(str)


    class _FunctionTask(QRunnable):
        """Run a function in the Qt global threadpool."""

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self.signals = _TaskSignals()

        def run(self) -> None:
            try:
                res = self.fn(*self.args, **self.kwargs)
                self.signals.finished.emit(res)
            except Exception as e:
                self.signals.failed.emit(str(e))


    class DiagramView(QGraphicsView):
        """Scaled section diagram that updates from inputs."""

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._scene = QGraphicsScene(self)
            self.setScene(self._scene)
            self.setRenderHint(QPainter.Antialiasing, True)
            self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def draw_state(self, *, girder_geom, placement, inputs_model) -> None:
            """Redraw the diagram.

            Parameters:
              - girder_geom: GirderGeometry
              - placement: PlacementOutcome
              - inputs_model: C49Inputs
            """

            self._scene.clear()

            # Pens/brushes (cosmetic pens so line weight stays readable when zooming)
            pen_outline = QPen(Qt.black)
            pen_outline.setCosmetic(True)
            pen_outline.setWidth(2)
            pen_thin = QPen(Qt.black)
            pen_thin.setCosmetic(True)
            pen_thin.setStyle(Qt.DashLine)
            pen_thin.setWidth(1)
            pen_member = QPen(Qt.black)
            pen_member.setCosmetic(True)
            pen_member.setWidth(2)
            pen_bearing = QPen(Qt.darkRed)
            pen_bearing.setCosmetic(True)
            pen_bearing.setWidth(3)
            pen_dim = QPen(Qt.darkBlue)
            pen_dim.setCosmetic(True)
            pen_dim.setWidth(1)

            brush_girder = QBrush(Qt.white)
            brush_slab = QBrush(Qt.white)
            brush_form = QBrush(Qt.lightGray)
            brush_bracket = QBrush(Qt.NoBrush)

            def draw_h_dim(x1: float, x2: float, y: float, label: str) -> None:
                if not (math.isfinite(x1) and math.isfinite(x2) and math.isfinite(y)):
                    return
                if abs(x2 - x1) < 1e-6:
                    return
                left = min(x1, x2)
                right = max(x1, x2)
                arrow = max(0.05, min(0.15, 0.06 * (right - left)))
                self._scene.addLine(left, y, right, y, pen_dim).setZValue(6)
                self._scene.addLine(left, y, left + arrow, y + arrow * 0.35, pen_dim).setZValue(6)
                self._scene.addLine(left, y, left + arrow, y - arrow * 0.35, pen_dim).setZValue(6)
                self._scene.addLine(right, y, right - arrow, y + arrow * 0.35, pen_dim).setZValue(6)
                self._scene.addLine(right, y, right - arrow, y - arrow * 0.35, pen_dim).setZValue(6)
                txt = self._scene.addText(label)
                txt.setDefaultTextColor(Qt.darkBlue)
                txt.setPos((left + right) / 2.0 - 0.2, y - arrow * 1.8)
                txt.setZValue(7)

            # --- Girder outline
            poly = QPolygonF()
            for x_ft, y_ft in girder_geom.outline_poly_ft:
                poly.append(QPointF(x_ft, y_ft))
            girder_item = self._scene.addPolygon(poly, pen_outline, brush_girder)
            girder_item.setZValue(0)

            if placement is not None and getattr(placement, "feasible", False):
                for face in getattr(girder_geom, "bearing_faces", []):
                    if face.tag == getattr(placement, "bearing_face", ""):
                        self._scene.addLine(face.p1[0], face.p1[1], face.p2[0], face.p2[1], pen_bearing).setZValue(2)
                        break

            # Determine top-flange outboard edge (x at y=0)
            x_edge_top_ft = max((x for x, y in girder_geom.outline_poly_ft if abs(y) < 1e-9), default=0.0)
            y_girder_bot_ft = max((y for _x, y in girder_geom.outline_poly_ft), default=0.0)

            # --- Deck + form stack
            y_soffit_ft = float(inputs_model.deck_soffit_offset_in) / 12.0
            t_slab_ft = float(inputs_model.slab_thickness_in) / 12.0

            # Diagram convention: slab extends upward (negative y)
            y_slab_bot = y_soffit_ft
            y_slab_top = y_slab_bot - t_slab_ft

            x_deck_edge_ft = x_edge_top_ft + float(inputs_model.overhang_length_ft)
            edge_out_ft = EDGE_LOAD_OUTBOARD_IN / 12.0
            top_member_end_ft = x_deck_edge_ft + edge_out_ft

            # Slab
            slab_rect = self._scene.addRect(
                0.0,
                y_slab_top,
                max(0.01, x_deck_edge_ft),
                t_slab_ft,
                pen_outline,
                brush_slab,
            )
            slab_rect.setZValue(1)

            # Form stack below soffit (plywood + 4x4 + 2x6)
            t_ply_ft = float(inputs_model.plywood_thickness_in) / 12.0
            t_4x4_ft = float(inputs_model.fourbyfour_thickness_in) / 12.0
            t_2x6_ft = float(inputs_model.twobysix_thickness_in) / 12.0

            y_ply_bot = y_slab_bot + t_ply_ft
            y_4x4_bot = y_ply_bot + t_4x4_ft
            y_2x6_bot = y_4x4_bot + t_2x6_ft

            # Plywood
            self._scene.addRect(0.0, y_slab_bot, max(0.01, x_deck_edge_ft), t_ply_ft, pen_outline, brush_form).setZValue(1)
            # 4x4
            self._scene.addRect(0.0, y_ply_bot, max(0.01, x_deck_edge_ft), t_4x4_ft, pen_outline, brush_form).setZValue(1)
            # 2x6
            self._scene.addRect(0.0, y_4x4_bot, max(0.01, x_deck_edge_ft), t_2x6_ft, pen_outline, brush_form).setZValue(1)

            # Edge load marker (deck edge + 3 in)
            self._scene.addLine(
                top_member_end_ft,
                y_slab_top - 0.05,
                top_member_end_ft,
                y_slab_top + 0.05,
                pen_thin,
            ).setZValue(2)

            # Bracket top line (bottom of stack)
            y_top_ft = y_2x6_bot
            self._scene.addLine(0.0, y_top_ft, max(0.01, top_member_end_ft), y_top_ft, pen_thin).setZValue(2)

            # --- Bracket (if feasible)
            if placement is not None and getattr(placement, "feasible", False):
                x_top_ft = float(placement.x_top_ft)
                y_top_ft = float(placement.y_top_ft)
                x_bot_ft = float(placement.x_bot_ft)
                y_bot_ft = float(placement.y_bot_ft)
                fallback_diag = top_member_end_ft - (TOP_DIAG_END_OFFSET_IN / 12.0)
                x_diag_top_ft = float(getattr(placement, "x_diag_top_ft", fallback_diag))

                top_h_ft = float(inputs_model.top_member_height_in) / 12.0
                b_vert_ft = float(inputs_model.vertical_member_width_in) / 12.0
                extra_up_ft = float(inputs_model.top_hanger_edge_clear_in) / 12.0

                # Top member envelope (matches placement check)
                self._scene.addRect(
                    x_edge_top_ft,
                    y_top_ft - top_h_ft,
                    max(0.01, top_member_end_ft - x_edge_top_ft),
                    top_h_ft,
                    pen_member,
                    brush_bracket,
                ).setZValue(3)

                # Vertical member envelope (matches placement check)
                self._scene.addRect(
                    x_top_ft - b_vert_ft / 2.0,
                    -extra_up_ft,
                    b_vert_ft,
                    (y_top_ft - (-extra_up_ft)),
                    pen_member,
                    brush_bracket,
                ).setZValue(3)

                # Diagonal (scaled thickness)
                diag_thk_ft = float(inputs_model.diagonal_envelope_thickness_in) / 12.0
                dx = x_bot_ft - x_diag_top_ft
                dy = y_bot_ft - y_top_ft
                length = math.hypot(dx, dy)
                if length > 1e-6 and diag_thk_ft > 0:
                    nx = -dy / length
                    ny = dx / length
                    hx = nx * diag_thk_ft / 2.0
                    hy = ny * diag_thk_ft / 2.0
                    poly = QPolygonF()
                    poly.append(QPointF(x_diag_top_ft + hx, y_top_ft + hy))
                    poly.append(QPointF(x_bot_ft + hx, y_bot_ft + hy))
                    poly.append(QPointF(x_bot_ft - hx, y_bot_ft - hy))
                    poly.append(QPointF(x_diag_top_ft - hx, y_top_ft - hy))
                    self._scene.addPolygon(poly, pen_member, brush_bracket).setZValue(3)
                else:
                    self._scene.addLine(x_diag_top_ft, y_top_ft, x_bot_ft, y_bot_ft, pen_member).setZValue(3)

                hole_r = 0.03
                self._scene.addEllipse(
                    x_diag_top_ft - hole_r,
                    y_top_ft - hole_r,
                    hole_r * 2.0,
                    hole_r * 2.0,
                    pen_member,
                ).setZValue(4)
                self._scene.addEllipse(
                    x_bot_ft - hole_r,
                    y_bot_ft - hole_r,
                    hole_r * 2.0,
                    hole_r * 2.0,
                    pen_member,
                ).setZValue(4)

                # Bottom pad indication (purely visual)
                pad_h_ft = float(inputs_model.bottom_pad_height_in) / 12.0
                pad_thk_ft = 0.25 / 12.0
                self._scene.addRect(
                    x_bot_ft,
                    y_bot_ft - pad_h_ft / 2.0,
                    pad_thk_ft,
                    pad_h_ft,
                    pen_member,
                    brush_bracket,
                ).setZValue(4)

                top_offset_ft = float(getattr(placement, "top_offset_ft", x_diag_top_ft - x_top_ft))
                bottom_offset_ft = float(getattr(placement, "bottom_offset_ft", abs(x_top_ft - x_bot_ft)))
                if top_offset_ft > 1e-6:
                    y_dim_top = y_slab_top - 0.25
                    draw_h_dim(
                        x_top_ft,
                        x_diag_top_ft,
                        y_dim_top,
                        f"Top offset: {top_offset_ft * 12.0:.2f} in",
                    )
                    end_offset_ft = top_member_end_ft - x_diag_top_ft
                    if end_offset_ft > 1e-6:
                        draw_h_dim(
                            x_diag_top_ft,
                            top_member_end_ft,
                            y_dim_top - 0.2,
                            f"End offset: {end_offset_ft * 12.0:.2f} in",
                        )
                if bottom_offset_ft > 1e-6:
                    y_dim_bot = max(y_bot_ft + pad_h_ft / 2.0 + 0.25, y_girder_bot_ft + 0.25)
                    draw_h_dim(
                        x_bot_ft,
                        x_top_ft,
                        y_dim_bot,
                        f"Bottom offset: {bottom_offset_ft * 12.0:.2f} in",
                    )
            else:
                # Indicate infeasible
                txt = self._scene.addText("Placement infeasible")
                txt.setDefaultTextColor(Qt.black)
                txt.setPos(x_edge_top_ft + 0.1, 0.1)
                txt.setZValue(10)

            # Fit view to content
            rect = self._scene.itemsBoundingRect()
            if not rect.isNull():
                margin = max(rect.width(), rect.height()) * 0.03 + 0.05
                rect = rect.adjusted(-margin, -margin, margin, margin)
                self.fitInView(rect, Qt.KeepAspectRatio)


    class C49MainWindow(QMainWindow):
        def __init__(self, tool: Any, initial_inputs: Dict[str, Any]):
            super().__init__()
            self._tool = tool
            self._threadpool = QThreadPool.globalInstance()

            self.setWindowTitle("C49 Overhang Bracket (TxDOT) – Interactive")
            self.resize(1200, 760)

            # Central layout
            splitter = QSplitter(Qt.Horizontal)
            self.setCentralWidget(splitter)

            # Left: Inputs
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(8, 8, 8, 8)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            left_layout.addWidget(scroll)

            form_host = QWidget()
            self._form_layout = QFormLayout(form_host)
            self._form_layout.setLabelAlignment(Qt.AlignLeft)
            self._form_layout.setFormAlignment(Qt.AlignTop)
            scroll.setWidget(form_host)

            # Status/output
            self._status = QLabel("")
            self._status.setWordWrap(True)
            left_layout.addWidget(self._status)

            # Buttons
            btn_row = QHBoxLayout()
            self._btn_run = QPushButton("Generate Calc Package")
            self._btn_run.clicked.connect(self._on_generate)
            btn_row.addWidget(self._btn_run)

            self._btn_refresh = QPushButton("Refresh Preview")
            self._btn_refresh.clicked.connect(self._update_preview)
            btn_row.addWidget(self._btn_refresh)

            btn_row.addStretch(1)
            left_layout.addLayout(btn_row)

            # Right: Diagram
            self._diagram = DiagramView()

            splitter.addWidget(left)
            splitter.addWidget(self._diagram)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([380, 820])

            # Controls
            self._controls: Dict[str, Any] = {}
            self._build_controls(initial_inputs)

            # Initial render
            self._update_preview()

        def _build_controls(self, initial_inputs: Dict[str, Any]) -> None:
            """Create widgets for a practical subset of inputs."""

            # NOTE: We intentionally expose the geometry-driving inputs prominently.
            # Additional load/capacity inputs remain available through the batch calc package.

            defaults = self._tool.default_inputs()
            merged = dict(defaults)
            merged.update(initial_inputs or {})
            self._girder_dim_keys = [
                "girder_depth_in",
                "girder_top_flange_width_in",
                "girder_bottom_flange_width_in",
                "girder_web_thickness_in",
                "girder_top_flange_thickness_in",
                "girder_bottom_flange_thickness_in",
            ]
            self._girder_dim_initial_overrides = {
                key for key in self._girder_dim_keys if merged.get(key) is not None
            }

            def add_combo(key: str, label: str, options: list[str]):
                cb = QComboBox()
                cb.addItems(options)
                if key in merged:
                    i = cb.findText(str(merged[key]).upper())
                    if i >= 0:
                        cb.setCurrentIndex(i)
                if key == "girder_type":
                    cb.currentTextChanged.connect(self._on_girder_type_changed)
                else:
                    cb.currentTextChanged.connect(lambda _=None: self._update_preview())
                self._form_layout.addRow(label, cb)
                self._controls[key] = cb

            def add_spin(
                key: str,
                label: str,
                *,
                step: float,
                decimals: int,
                min_v: float,
                max_v: float,
                layout: Optional[QFormLayout] = None,
            ):
                sb = QDoubleSpinBox()
                sb.setDecimals(decimals)
                sb.setSingleStep(step)
                sb.setRange(min_v, max_v)
                if key in merged and merged[key] is not None:
                    sb.setValue(float(merged[key]))
                sb.valueChanged.connect(lambda _=None: self._update_preview())
                (layout or self._form_layout).addRow(label, sb)
                self._controls[key] = sb

            # --- Geometry drivers
            add_combo("girder_type", "Girder type", ["TX28", "TX34", "TX40", "TX46", "TX54", "TX62", "TX70"])

            girder_group = QGroupBox("Girder dimensions (in)")
            girder_layout = QFormLayout(girder_group)
            girder_layout.setLabelAlignment(Qt.AlignLeft)
            self._form_layout.addRow(girder_group)
            add_spin(
                "girder_depth_in",
                "Overall depth",
                step=0.25,
                decimals=2,
                min_v=1.0,
                max_v=200.0,
                layout=girder_layout,
            )
            add_spin(
                "girder_top_flange_width_in",
                "Top flange width",
                step=0.25,
                decimals=2,
                min_v=1.0,
                max_v=100.0,
                layout=girder_layout,
            )
            add_spin(
                "girder_bottom_flange_width_in",
                "Bottom flange width",
                step=0.25,
                decimals=2,
                min_v=1.0,
                max_v=100.0,
                layout=girder_layout,
            )
            add_spin(
                "girder_web_thickness_in",
                "Web thickness",
                step=0.25,
                decimals=2,
                min_v=0.1,
                max_v=20.0,
                layout=girder_layout,
            )
            add_spin(
                "girder_top_flange_thickness_in",
                "Top flange thickness",
                step=0.25,
                decimals=2,
                min_v=0.1,
                max_v=20.0,
                layout=girder_layout,
            )
            add_spin(
                "girder_bottom_flange_thickness_in",
                "Bottom flange thickness",
                step=0.25,
                decimals=2,
                min_v=0.1,
                max_v=30.0,
                layout=girder_layout,
            )

            self._apply_girder_defaults(merged.get("girder_type", "TX54"), force=False)
            add_spin("overhang_length_ft", "Overhang length (ft)", step=0.25, decimals=2, min_v=0.5, max_v=20.0)
            add_spin("slab_thickness_in", "Slab thickness (in)", step=0.25, decimals=2, min_v=2.0, max_v=24.0)

            add_spin("deck_soffit_offset_in", "Deck soffit offset from girder top (in)", step=0.25, decimals=2, min_v=-6.0, max_v=24.0)
            add_spin("plywood_thickness_in", "Plywood thickness (in)", step=0.0625, decimals=4, min_v=0.0, max_v=3.0)
            add_spin("fourbyfour_thickness_in", "4x4 thickness (in)", step=0.0625, decimals=4, min_v=0.0, max_v=6.0)
            add_spin("twobysix_thickness_in", "2x6 thickness (in)", step=0.0625, decimals=4, min_v=0.0, max_v=4.0)

            add_spin("max_bracket_depth_in", "Max bracket depth (in)", step=1.0, decimals=1, min_v=6.0, max_v=80.0)
            add_spin("min_bracket_depth_in", "Min bracket depth (in)", step=1.0, decimals=1, min_v=0.0, max_v=80.0)
            add_spin("clearance_in", "Clearance to girder outline (in)", step=0.125, decimals=3, min_v=0.0, max_v=6.0)

            add_spin("top_member_height_in", "Top member envelope height (in)", step=0.25, decimals=2, min_v=0.5, max_v=12.0)
            add_spin("vertical_member_width_in", "Vertical member envelope width (in)", step=0.25, decimals=2, min_v=0.5, max_v=12.0)
            add_spin("bottom_pad_height_in", "Bottom pad envelope height (in)", step=0.25, decimals=2, min_v=0.5, max_v=12.0)
            add_spin("top_hanger_edge_clear_in", "Vertical member ext. above girder top (in)", step=0.25, decimals=2, min_v=0.0, max_v=24.0)

            # --- Load driver (kept per user request)
            add_spin("screed_wheel_load_kip", "Screed wheel load (kip)", step=0.1, decimals=3, min_v=0.0, max_v=20.0)

        def _gather_inputs(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, w in self._controls.items():
                if isinstance(w, QComboBox):
                    out[k] = w.currentText()
                elif isinstance(w, QDoubleSpinBox):
                    out[k] = float(w.value())
                else:
                    # fallback
                    try:
                        out[k] = w.value()
                    except Exception:
                        pass
            return out

        def _on_girder_type_changed(self, girder_type: str) -> None:
            self._apply_girder_defaults(girder_type, force=True)
            self._update_preview()

        def _apply_girder_defaults(self, girder_type: str, *, force: bool) -> None:
            from .db.txdot_girders import get_txdot_profile

            profile = get_txdot_profile(girder_type)
            defaults = {
                "girder_depth_in": profile.depth_in,
                "girder_top_flange_width_in": profile.top_flange_width_in,
                "girder_bottom_flange_width_in": profile.bottom_flange_width_in,
                "girder_web_thickness_in": profile.web_thickness_in,
                "girder_top_flange_thickness_in": 3.5,
                "girder_bottom_flange_thickness_in": profile.F_in,
            }
            for key, value in defaults.items():
                control = self._controls.get(key)
                if not isinstance(control, QDoubleSpinBox):
                    continue
                if not force and key in self._girder_dim_initial_overrides:
                    continue
                control.blockSignals(True)
                control.setValue(float(value))
                control.blockSignals(False)

        def _update_preview(self) -> None:
            from .models import C49Inputs
            from .section_library import get_txdot_girder
            from .analysis.placement import solve_best_placement_fast

            try:
                inputs = self._gather_inputs()
                model = C49Inputs.model_validate({**self._tool.default_inputs(), **inputs})

                girder = get_txdot_girder(model.girder_type, overrides=model.girder_override_dict())
                geom = girder.geometry()

                placement = solve_best_placement_fast(model.model_dump(), geom)

                self._diagram.draw_state(girder_geom=geom, placement=placement, inputs_model=model)

                if placement.feasible:
                    self._status.setText(
                        f"Preview: FEASIBLE\n"
                        f"Bracket depth: {placement.bracket_depth_in:.1f} in\n"
                        f"Diagonal angle: {placement.theta_deg:.1f}°\n"
                        f"sin(theta): {placement.sin_theta:.3f}"
                    )
                else:
                    self._status.setText(f"Preview: INFEASIBLE\n{placement.message}")

            except Exception as e:
                # Keep the UI responsive even if validation fails.
                self._status.setText(f"Preview error: {e}")

        def _on_generate(self) -> None:
            """Run the full calc package export in a background thread."""

            inputs = self._gather_inputs()
            all_inputs = {**self._tool.default_inputs(), **inputs}

            self._btn_run.setEnabled(False)
            self._status.setText("Running... (calc package will be written to %LOCALAPPDATA%\\EngineeringToolbox\\...)")

            task = _FunctionTask(self._tool.run_batch, all_inputs)
            task.signals.finished.connect(self._on_generate_done)
            task.signals.failed.connect(self._on_generate_failed)
            self._threadpool.start(task)

        def _on_generate_done(self, res: Any) -> None:
            self._btn_run.setEnabled(True)
            if isinstance(res, dict) and res.get("ok"):
                def _fmt(value: Any, decimals: int) -> str:
                    try:
                        return f"{float(value):.{decimals}f}"
                    except Exception:
                        return "n/a"

                run_dir = res.get("run_dir", "")
                spacing_in = _fmt(res.get("optimal_spacing_in_rounded_up"), 0)
                spacing_ft = _fmt(res.get("optimal_spacing_ft_rounded_up"), 3)
                hanger_demand = _fmt(res.get("hanger_demand_kip"), 3)
                diag_demand = _fmt(res.get("diagonal_demand_kip"), 3)
                hanger_util = _fmt(res.get("util_hanger"), 3)
                diag_util = _fmt(res.get("util_diagonal"), 3)
                self._status.setText(
                    "Run complete.\n"
                    f"Run dir: {run_dir}\n"
                    f"Optimal spacing (rounded up): {spacing_in} in ({spacing_ft} ft)\n"
                    f"Hanger demand: {hanger_demand} kip (util: {hanger_util})\n"
                    f"Diagonal demand: {diag_demand} kip (util: {diag_util})\n"
                    f"Placement feasible: {res.get('placement_feasible', 'n/a')}"
                )
            else:
                self._status.setText(f"Run failed: {res}")

        def _on_generate_failed(self, msg: str) -> None:
            self._btn_run.setEnabled(True)
            self._status.setText(f"Run failed: {msg}")
            QMessageBox.critical(self, "C49 Tool", msg)
