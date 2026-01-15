from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from pydantic import ValidationError

from .models import InputModel
from .paths import make_run_dir, write_text_log
from toolbox_app.blocks.aisc_shapes_db import ShapeDatabase, ShapeNotFoundError
from .design_axial import design_tension, design_compression
from .design_shear import design_shear
from .design_flexure import design_flexure
from .design_interaction import design_interaction
from .exports import build_report_html, export_excel, export_mathcad_handoff, export_report_html


# Try to import your framework's ToolMeta. Provide a safe fallback for local testing.
try:
    from toolbox_app.core.tooling import ToolMeta  # type: ignore
except Exception:  # pragma: no cover
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ToolMeta:  # minimal fallback
        id: str
        name: str
        category: str
        version: str
        description: str


class AISC360MemberDesignTool:
    """
    AISC 360-16 member design checks (foundational implementation).
    Pure computation tool suitable for background execution.
    """

    meta = ToolMeta(
        id="aisc360_16_member_design",
        name="AISC 360-16 Steel Member Design",
        category="Steel",
        version="0.1.0",
        description=(
            "Evaluates flexure, shear, axial tension/compression, and combined axial+flexure "
            "per AISC 360-16 with LRFD/ASD toggle and AISC Shapes Database (v16) section selection."
        ),
    )

    InputModel = InputModel

    def default_inputs(self) -> dict:
        return InputModel().model_dump()

    def run(self, inputs: dict) -> dict:
        run_dir = make_run_dir(self.meta.id)

        trace: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        # Validate inputs via Pydantic
        try:
            model = InputModel.model_validate(inputs)
        except ValidationError as e:
            # Return validation info in a deterministic way
            msg = e.json(indent=2)
            write_text_log(run_dir, "validation_error.json", msg)
            return {
                "ok": False,
                "run_dir": str(run_dir),
                "error": "Input validation failed (see validation_error.json).",
                "validation_error_path": str(run_dir / "validation_error.json"),
            }

        # Load shape
        db = ShapeDatabase()
        try:
            designation = model.selected_designation()
        except ValueError as e:
            errors.append(str(e))
            write_text_log(run_dir, "error.txt", "\n".join(errors))
            return {
                "ok": False,
                "run_dir": str(run_dir),
                "error": str(e),
                "error_path": str(run_dir / "error.txt"),
            }
        try:
            shape = db.get_shape(designation)
        except ShapeNotFoundError as e:
            errors.append(str(e))
            write_text_log(run_dir, "error.txt", "\n".join(errors))
            return {"ok": False, "run_dir": str(run_dir), "error": str(e), "error_path": str(run_dir / "error.txt")}

        # Resolve material properties (explicit + deterministic)
        mat = model.resolved_material()
        trace.append(f"Material: Fy={mat.Fy_ksi:.3f} ksi, Fu={mat.Fu_ksi:.3f} ksi, E={mat.E_ksi:.3f} ksi")

        # Run checks
        flex = design_flexure(model=model, shape=shape, mat=mat, trace=trace, warnings=warnings)
        shr = design_shear(model=model, shape=shape, mat=mat, trace=trace, warnings=warnings)
        ten = design_tension(model=model, shape=shape, mat=mat, trace=trace, warnings=warnings)
        comp = design_compression(model=model, shape=shape, mat=mat, trace=trace, warnings=warnings)
        inter = design_interaction(
            model=model, shape=shape, flex=flex, ten=ten, comp=comp, trace=trace, warnings=warnings
        )

        results = {
            "ok": True,
            "run_dir": str(run_dir),
            "inputs": model.model_dump(),
            "shape": shape.to_public_dict(),
            "material": asdict(mat),
            "checks": {
                "flexure": flex,
                "shear": shr,
                "tension": ten,
                "compression": comp,
                "interaction": inter,
            },
            "warnings": warnings,
        }

        # Exports
        report = build_report_html(results=results, trace=trace)
        report_path = export_report_html(run_dir=run_dir, html=report["html"])
        excel_path = export_excel(run_dir=run_dir, results=results, trace=trace)
        mathcad = export_mathcad_handoff(run_dir=run_dir, results=results)

        results["exports"] = {
            "excel_xlsx": str(excel_path),
            "mathcad_json": str(mathcad["json_path"]),
            "mathcad_csv": str(mathcad["csv_path"]),
            "mathcad_assignments_txt": str(mathcad["assignments_path"]),
            "report_html": str(report_path),
        }
        results["summary"] = report["summary"]
        results["limit_states"] = report["limit_states"]
        results["report_html"] = report["html"]

        # Logs
        write_text_log(run_dir, "trace.txt", "\n".join(trace))
        if warnings:
            write_text_log(run_dir, "warnings.txt", "\n".join(warnings))

        return results
