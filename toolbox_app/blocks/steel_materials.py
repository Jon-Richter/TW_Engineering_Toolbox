from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Material:
    Fy_ksi: float
    Fu_ksi: float
    E_ksi: float


_MATERIAL_PRESETS: Dict[str, Tuple[float, float]] = {
    "A992": (50.0, 65.0),
    "A36": (36.0, 58.0),
    "A572 Gr 50": (50.0, 65.0),
    "A500 Gr B": (46.0, 58.0),
    "A500 Gr C": (50.0, 62.0),
    "Custom": (50.0, 65.0),
}

_MATERIAL_PRESET_VALUES: Dict[str, Dict[str, float]] = {
    name: {"Fy_ksi": vals[0], "Fu_ksi": vals[1]} for name, vals in _MATERIAL_PRESETS.items()
}