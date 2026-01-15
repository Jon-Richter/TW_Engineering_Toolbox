from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

DesignMethod = Literal["LRFD", "ASD"]


class ShapeFamily(str, Enum):
    W_HP = "W / HP"
    S_M = "S / M"
    C_MC = "C / MC"
    TEE = "WT / MT / ST"
    ANGLE = "L"
    DOUBLE_ANGLE = "2L"
    HSS_RECT = "HSS Rect / Square"
    HSS_ROUND = "HSS Round"
    PIPE = "PIPE"


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


class InputModel(BaseModel):
    # ---- Design setup ----
    design_method: DesignMethod = Field(default="ASD", description="Design methodology.")

    material_grade: Literal["A992", "A36", "A572 Gr 50", "A500 Gr B", "A500 Gr C", "Custom"] = Field(
        default="A992",
        description="Material grade preset. Select 'Custom' to control Fy/Fu directly.",
        json_schema_extra={
            "material_presets": _MATERIAL_PRESET_VALUES,
            "custom_value": "Custom",
            "preset_targets": ["Fy_ksi", "Fu_ksi"],
            "override_field": "override_material_properties",
        },
    )
    override_material_properties: bool = Field(
        default=False,
        description="If true, use Fy/Fu values below even when a preset is selected.",
    )
    Fy_ksi: float = Field(default=50.0, gt=0.0, title="Fy (ksi)", description="Steel yield stress Fy (ksi).")
    Fu_ksi: float = Field(default=65.0, gt=0.0, title="Fu (ksi)", description="Steel ultimate stress Fu (ksi).")
    E_ksi: float = Field(default=29000.0, gt=0.0, title="E (ksi)", description="Modulus of elasticity E (ksi).")

    # ---- Section selection (family + dropdown per family) ----
    shape_family: ShapeFamily = Field(default=ShapeFamily.W_HP, description="Section family filter / UI grouping.")

    w_hp_section: Optional[str] = Field(
        default=None,
        description="W/HP designation from AISC database.",
        json_schema_extra={
            "ui_tab": "W / HP",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["W", "HP"]},
        },
    )
    s_m_section: Optional[str] = Field(
        default=None,
        description="S/M designation from AISC database.",
        json_schema_extra={
            "ui_tab": "S / M",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["S", "M"]},
        },
    )
    c_mc_section: Optional[str] = Field(
        default=None,
        description="C/MC designation from AISC database.",
        json_schema_extra={
            "ui_tab": "C / MC",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["C", "MC"]},
        },
    )
    tee_section: Optional[str] = Field(
        default=None,
        description="WT/MT/ST designation from AISC database.",
        json_schema_extra={
            "ui_tab": "WT / MT / ST",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["WT", "MT", "ST"]},
        },
    )
    angle_section: Optional[str] = Field(
        default=None,
        description="Single angle (L) designation from AISC database.",
        json_schema_extra={
            "ui_tab": "L",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["L"]},
        },
    )
    double_angle_section: Optional[str] = Field(
        default=None,
        description="Double angle (2L) designation from AISC database.",
        json_schema_extra={
            "ui_tab": "2L",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["2L"]},
        },
    )
    hss_rect_section: Optional[str] = Field(
        default=None,
        description="Rectangular/Square HSS designation from AISC database.",
        json_schema_extra={
            "ui_tab": "HSS Rect / Square",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["HSS"], "hss": "rect"},
        },
    )
    hss_round_section: Optional[str] = Field(
        default=None,
        description="Round HSS designation from AISC database.",
        json_schema_extra={
            "ui_tab": "HSS Round",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["HSS"], "hss": "round"},
        },
    )
    pipe_section: Optional[str] = Field(
        default=None,
        description="Pipe designation from AISC database.",
        json_schema_extra={
            "ui_tab": "PIPE",
            "choices_source": {"provider": "aisc_shapes_v16", "type_codes": ["PIPE"]},
        },
    )

    # ---- Geometry / bracing ----
    # Effective lengths (ft)
    Lx_ft: float = Field(default=0.0, ge=0.0, title="Lx (ft)", description="Member length for buckling about x (ft).")
    Ly_ft: float = Field(default=0.0, ge=0.0, title="Ly (ft)", description="Member length for buckling about y (ft).")
    Lz_ft: float = Field(default=0.0, ge=0.0, title="Lz (ft)", description="Member length for torsional buckling (ft). (Used as note only in this version.)")

    Kx: float = Field(default=1.0, gt=0.0, title="Kx", description="Effective length factor Kx.")
    Ky: float = Field(default=1.0, gt=0.0, title="Ky", description="Effective length factor Ky.")
    Kz: float = Field(default=1.0, gt=0.0, title="Kz", description="Effective length factor Kz. (Note only in this version.)")

    # Flexural bracing
    Lb_ft: float = Field(default=0.0, ge=0.0, title="Lb (ft)", description="Unbraced length for major-axis flexure Lb (ft). 0 = fully braced.")
    Cb: float = Field(default=1.0, gt=0.0, title="Cb", description="Moment gradient factor Cb (AISC lateral-torsional buckling).")

    # ---- Applied demands ----
    # Convention: axial compression positive, tension negative.
    Pu_k: float = Field(default=0.0, title="Pu (kips)", description="Required axial strength Pu (kips). Compression positive; tension negative.")
    Mux_kft: float = Field(default=0.0, title="Mux (kips)", description="Required major-axis moment Mux (kip-ft).")
    Muy_kft: float = Field(default=0.0, title="Muy (kips)", description="Required minor-axis moment Muy (kip-ft).")
    Vux_k: float = Field(default=0.0, title="Vux (kips)", description="Required shear Vux (kips) (major-axis shear / web shear).")
    Vuy_k: float = Field(default=0.0, title="Vuy (kips)", description="Required shear Vuy (kips) (minor-axis shear).")
    @model_validator(mode="after")
    def _validate_section_choice(self) -> "InputModel":
        # Enforce that at most one section field is populated.
        # (UI needs to render defaults with no section selected.)
        fields = [
            self.w_hp_section,
            self.s_m_section,
            self.c_mc_section,
            self.tee_section,
            self.angle_section,
            self.double_angle_section,
            self.hss_rect_section,
            self.hss_round_section,
            self.pipe_section,
        ]
        chosen = [f for f in fields if f not in (None, "")]
        if len(chosen) > 1:
            raise ValueError(
                "Select only one section designation (one dropdown). "
                "All other section fields must be blank."
            )
        return self

    def selected_designation(self) -> str:
        # Return the selected label as a string (Enum value or raw string).
        for v in [
            self.w_hp_section,
            self.s_m_section,
            self.c_mc_section,
            self.tee_section,
            self.angle_section,
            self.double_angle_section,
            self.hss_rect_section,
            self.hss_round_section,
            self.pipe_section,
        ]:
            if v not in (None, ""):
                # If Enum, UI should store member value; but handle both.
                return str(getattr(v, "value", v))
        raise ValueError("No section selected. Choose one section designation before running.")

    def resolved_material(self) -> Material:
        Fy, Fu = _MATERIAL_PRESETS[self.material_grade]
        if self.material_grade != "Custom" and not self.override_material_properties:
            return Material(Fy_ksi=Fy, Fu_ksi=Fu, E_ksi=self.E_ksi)
        return Material(Fy_ksi=float(self.Fy_ksi), Fu_ksi=float(self.Fu_ksi), E_ksi=float(self.E_ksi))
