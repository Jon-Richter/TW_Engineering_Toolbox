\
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

# Ensure tool root on path for direct invocation
TOOL_ROOT = Path(__file__).resolve().parent
import sys
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from backend.solver import solve
from models import BeamFlexureInputs, DevLengthTensionInputs


def _assert_artifacts(run_dir: Path) -> None:
    required = ["report.html", "calc_trace.json", "results.xlsx", "mathcad_inputs.csv"]
    missing = [f for f in required if not (run_dir / f).exists()]
    assert not missing, f"Missing artifacts in {run_dir}: {missing}"


def test_smoke_two_cases():
    tmp = Path(tempfile.mkdtemp(prefix="engtoolbox_localappdata_"))
    try:
        os.environ["LOCALAPPDATA"] = str(tmp)  # force outputs to temp

        # Case 1: Beam flexure
        beam_inputs = BeamFlexureInputs(
            Mu_kipft=180.0,
            reinf_layers=[
                {"face":"bottom", "offset_in": 2.375, "n_bars": 3, "bar_dia_in": 1.0, "As_override_in2": None},
                {"face":"top", "offset_in": 2.0, "As_override_in2": 0.62}
            ]
        ).model_dump()
        r1 = solve("aci318_25_concrete_design", "beam_flexure", beam_inputs)
        assert r1["ok"] is True
        rd1 = Path(r1["run_dir"])
        _assert_artifacts(rd1)

        # Case 2: Development length in tension
        dev_inputs = DevLengthTensionInputs().model_dump()
        r2 = solve("aci318_25_concrete_design", "development_length_tension", dev_inputs)
        assert r2["ok"] is True
        rd2 = Path(r2["run_dir"])
        _assert_artifacts(rd2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_smoke_two_cases()
    print("tests_smoke.py: PASS")
