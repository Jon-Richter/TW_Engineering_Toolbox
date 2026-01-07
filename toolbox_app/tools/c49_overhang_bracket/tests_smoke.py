from __future__ import annotations

from pathlib import Path

from .tool import TOOL


def _assert_exists(p: Path) -> None:
    assert p.exists(), f"Missing: {p}"


def _check_outputs(run_dir: Path) -> None:
    _assert_exists(run_dir / "report.html")
    _assert_exists(run_dir / "calculation_report.html")
    _assert_exists(run_dir / "report.pdf")
    _assert_exists(run_dir / "calc_trace.json")
    _assert_exists(run_dir / "results.json")
    _assert_exists(run_dir / "results.xlsx")
    _assert_exists(run_dir / "mathcad_inputs.csv")
    _assert_exists(run_dir / "mathcad_steps.json")
    _assert_exists(run_dir / "run.log")


def test_smoke_case_1():
    inputs = TOOL.default_inputs()
    res = TOOL.run_batch(inputs)
    assert res["ok"] is True
    run_dir = Path(res["run_dir"])
    _check_outputs(run_dir)


def test_smoke_case_2():
    inputs = TOOL.default_inputs()
    inputs.update(
        {
            "girder_type": "TX34",
            "overhang_length_ft": 4.0,
            "screed_wheel_load_kip": 3.5,
            # force a thinner web to test override plumbing
            "girder_web_thickness_in": 6.5,
        }
    )
    res = TOOL.run_batch(inputs)
    assert res["ok"] is True
    run_dir = Path(res["run_dir"])
    _check_outputs(run_dir)
