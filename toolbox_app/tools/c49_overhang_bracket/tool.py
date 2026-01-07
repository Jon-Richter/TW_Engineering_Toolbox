from __future__ import annotations

import traceback
import math
from pathlib import Path
from typing import Any, Dict

from .compat import ToolMeta
from .models import C49Inputs
from .paths import create_run_dir, compute_input_hash
from .logging_utils import get_run_logger, remove_run_logger_sink
from .calc_trace import Assumption, CalcTrace
from .section_library import get_txdot_girder
from .analysis.placement import solve_best_placement
from .solver import solve_optimal_spacing
from .exports import export_all

# Plugin contract hint: because this tool launches a Qt UI, the host must run it on the UI thread.
RUNS_ON_UI_THREAD = True


class C49OverhangBracketTool:
    """C49 overhang bracket tool.

    - UI mode (default): run() launches the interactive window and returns immediately.
    - Batch mode: run_batch() performs deterministic computation + full calc package exports.
    """

    meta = ToolMeta(
        id="c49_overhang_bracket",
        name="C49 Overhang Bracket",
        category="Temporary Works",
        version="0.5.0",
        description="C49 overhang bracket spacing tool with TxDOT girder geometry and audit-grade calc package exports.",
    )

    InputModel = C49Inputs

    # Also expose on the instance/class for host apps that introspect the TOOL object.
    RUNS_ON_UI_THREAD = True

    def __init__(self) -> None:
        self._window = None

    def default_inputs(self) -> dict:
        return self.InputModel().model_dump()

    # ------------------------------
    # Batch calculation API (headless)
    # ------------------------------
    def run_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full calculation + exports and return results.

        Safe to execute in a background thread.
        """

        model = self.InputModel.model_validate(inputs)
        inputs_norm = model.model_dump()
        input_hash = compute_input_hash(inputs_norm)
        run_dir = create_run_dir(self.meta.id, input_hash)
        log, _log_sink = get_run_logger(run_dir, self.meta.id, input_hash)

        try:
            log.info("Starting C49 batch run")
            log.info(f"Inputs (validated): {inputs_norm}")

            # Calc trace
            trace = CalcTrace.new(
                tool_id=self.meta.id,
                tool_version=self.meta.version,
                units_system="US",
                code_basis="Temporary works (method per internal C49 workflow)",
                inputs=inputs_norm,
                input_hash=input_hash,
            )

            # Assumptions
            trace.assumptions.extend(
                [
                    Assumption(
                        id="A1",
                        text=(
                            "Girder outline uses TxDOT Std. Sheet A-21 piecewise-linear geometry; "
                            "fillets are approximated as straight segments."
                        ),
                    ),
                    Assumption(
                        id="A2",
                        text=(
                            "Bottom bearing is compression-only horizontal reaction; therefore it must bear on a "
                            "vertical girder surface (web or bottom flange side)."
                        ),
                    ),
                    Assumption(
                        id="A3",
                        text=(
                            "Top hanger provides the only positive connection to the girder; remaining members "
                            "are treated as bearing/strut action."
                        ),
                    ),
                    Assumption(
                        id="A4",
                        text=(
                            "Edge line load of 75 plf and screed wheel load act at 3 in outboard of the deck edge."
                        ),
                    ),
                ]
            )

            # Girder
            girder = get_txdot_girder(model.girder_type, overrides=model.girder_override_dict())
            geom = girder.geometry()
            trace.tables["girder"] = girder.metadata()

            # Placement
            placement = solve_best_placement(trace, inputs_norm, geom)
            trace.tables["placement"] = placement.__dict__

            # Solver (includes traced evaluation for governing spacing)
            solve = solve_optimal_spacing(trace, inputs_norm, placement)

            governing_ratio = float("nan")
            controlling_case = "none"
            util_hanger = float("nan")
            util_diagonal = float("nan")
            hanger_demand_kip = float("nan")
            diagonal_demand_kip = float("nan")
            optimal_spacing_in_rounded_up = None
            optimal_spacing_ft_rounded_up = None
            if solve.feasible and solve.final is not None:
                util_hanger = float(solve.final.util_hanger)
                util_diagonal = float(solve.final.util_diagonal)
                hanger_demand_kip = float(solve.final.hanger_demand_kip)
                diagonal_demand_kip = float(solve.final.diagonal_demand_kip)
                governing_ratio = max(util_hanger, util_diagonal)
                controlling_case = str(solve.final.governs)
                spacing_in = float(solve.optimal_spacing_ft) * 12.0
                optimal_spacing_in_rounded_up = int(math.ceil(spacing_in - 1e-9))
                optimal_spacing_ft_rounded_up = optimal_spacing_in_rounded_up / 12.0

            results: Dict[str, Any] = {
                "ok": True,
                "run_dir": str(run_dir),
                "input_hash": trace.meta.input_hash,
                "placement_feasible": placement.feasible,
                "placement_bearing_face": placement.bearing_face,
                "placement_bracket_depth_in": placement.bracket_depth_in,
                "optimal_spacing_ft": solve.optimal_spacing_ft,
                "optimal_spacing_in_rounded_up": optimal_spacing_in_rounded_up,
                "optimal_spacing_ft_rounded_up": optimal_spacing_ft_rounded_up,
                "solver_feasible": solve.feasible,
                "solver_message": solve.message,
                "hanger_demand_kip": hanger_demand_kip,
                "diagonal_demand_kip": diagonal_demand_kip,
                "util_hanger": util_hanger,
                "util_diagonal": util_diagonal,
                "governing_ratio": governing_ratio,
                "controlling_case": controlling_case,
            }

            trace.summary = {
                "placement_feasible": placement.feasible,
                "bearing_face": placement.bearing_face,
                "bracket_depth_in": placement.bracket_depth_in,
                "optimal_spacing_ft": solve.optimal_spacing_ft,
                "optimal_spacing_in_rounded_up": optimal_spacing_in_rounded_up,
                "optimal_spacing_ft_rounded_up": optimal_spacing_ft_rounded_up,
                "solver_feasible": solve.feasible,
                "hanger_demand_kip": hanger_demand_kip,
                "diagonal_demand_kip": diagonal_demand_kip,
                "util_hanger": util_hanger,
                "util_diagonal": util_diagonal,
                "governing_ratio": governing_ratio,
                "controlling_case": controlling_case,
            }

            # Export calc package
            out_paths = export_all(trace, run_dir, results)
            results["outputs"] = {k: str(v) for k, v in out_paths.items()}

            log.info("Batch run complete")
            return results

        except Exception as e:
            log.exception("Batch run failed")
            return {
                "ok": False,
                "run_dir": str(run_dir),
                "input_hash": input_hash,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        finally:
            remove_run_logger_sink(_log_sink)

    # ------------------------------
    # UI entry point (host app calls this)
    # ------------------------------
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Launch the interactive UI and return immediately.

        In environments without PySide6 (e.g., CI/headless tests), this function
        falls back to run_batch(...).
        """

        # Explicit opt-in headless batch mode
        if bool(inputs.get("__batch__") or inputs.get("headless")):
            return self.run_batch(inputs)

        try:
            from .ui_app import launch_ui

            self._window = launch_ui(tool=self, initial_inputs=inputs, existing_window=self._window)
            return {"ok": True, "status": "launched"}

        except ImportError:
            # No PySide6 in this runtime; do batch instead.
            return self.run_batch(inputs)

        except Exception as e:
            return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


TOOL = C49OverhangBracketTool()
