from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

DesignCategory = Literal["A","B","C"]
UnitsSystem = Literal["US"]
WeldType = Literal["Fillet", "PJP 60° Bevel", "PJP 45° Bevel", "CJP"]
WeldGroup = Literal["Long Sides Only", "All Around"]

class Common(BaseModel):
    mode: Literal["padeye","spreader","spreader_two_way"]
    units_system: UnitsSystem = "US"
    design_category: DesignCategory = "C"
    Nd: float = Field(6.0, gt=0, description="BTH design factor Nd")
    Fy: float = Field(50.0, gt=0, description="Yield strength [ksi]")
    Fu: float = Field(65.0, gt=0, description="Ultimate strength [ksi]")
    impact_factor: float = Field(1.0, ge=1.0)

class PadeyeInputs(Common):
    mode: Literal["padeye"] = "padeye"
    P: float = Field(..., gt=0, description="Applied resultant load [kip]")
    theta_deg: float = Field(0.0, description="In-plane angle θ [deg]")
    beta_deg: float = Field(90.0, description="Out-of-plane angle β [deg]")
    H: float = Field(40.0, gt=0, description="Padeye height [in]")
    h: float = Field(38.0, gt=0, description="Height to hole center [in]")
    a1: float = Field(0.0, ge=0.0, description="Straight vertical corner height a1 [in]")
    Wb: float = Field(8.0, gt=0, description="Plate width at base [in]")
    Wb1: float = Field(4.0, gt=0, description="Hole center to edge distance Wb1 [in]")
    t: float = Field(1.0, gt=0, description="Main plate thickness [in]")
    Dh: float = Field(1.4, gt=0, description="Hole diameter [in]")
    Dp: float = Field(1.5, gt=0, description="Pin diameter [in]")
    R: float = Field(2.0, ge=0.0, description="Top radius [in]")
    tcheek: float = Field(0.0, ge=0.0, description="Total cheek/boss plate thickness at hole [in]")
    ex: float = Field(0.0, ge=0.0, description="Out-of-plane eccentricity e_x for torsion [in]")
    ey: float = Field(0.0, ge=0.0, description="Vertical eccentricity e_y for out-of-plane moment [in]")
    weld_type: WeldType = Field("Fillet", description="Weld type at base")
    weld_group: WeldGroup = Field("Long Sides Only", description="Weld group configuration at base")
    weld_size_16: int = Field(4, ge=0, description="Weld size in sixteenths of an inch")
    weld_exx_ksi: float = Field(70.0, gt=0, description="Weld metal tensile strength Exx [ksi]")

    @model_validator(mode="after")
    def _validate_weld_inputs(self) -> "PadeyeInputs":
        if self.weld_type != "Fillet" and self.weld_group == "All Around":
            raise ValueError("weld_group must be 'Long Sides Only' unless weld_type is Fillet.")
        return self

class SpreaderInputs(Common):
    mode: Literal["spreader"] = "spreader"
    shape: str = Field(..., description="AISC shape label")
    span_L_ft: float = Field(..., gt=0, description="Spreader span between nodes [ft]")
    Lb_ft: float = Field(..., gt=0, description="Flexure unbraced length Lb [ft]")
    KL_ft: float = Field(..., gt=0, description="Compression effective length KL [ft]")
    ey: float = Field(0.0, ge=0.0, description="Top padeye height (top of spreader to padeye hole) [in]")
    mx_includes_total: bool = Field(False, description="Mx already includes self-weight and eccentric axial load")
    Cb: float = Field(1.0, gt=0, description="Bending coefficient Cb")
    V_kip: float = Field(0.0, ge=0.0, description="Shear V [kip]")
    P_kip: float = Field(0.0, ge=0.0, description="Axial compression P [kip]")
    Mx_app_kipft: float = Field(0.0, description="Applied strong-axis end moment [kip-ft]")
    My_app_kipft: float = Field(0.0, description="Applied weak-axis end moment [kip-ft]")
    Cmx: float = Field(1.0, gt=0, description="Moment gradient factor Cmx for eq. (3-29)")
    Cmy: float = Field(1.0, gt=0, description="Moment gradient factor Cmy for eq. (3-29)")
    braced_against_twist: bool = Field(True, description="Whether compression flange is braced against twist/lateral displacement at ends of unbraced length (for CLTB)")
    weld_check: bool = Field(False, description="Include end connection fillet weld sizing check (direct shear + axial only)")
    weld_size_in: float = Field(0.0, ge=0.0, description="Fillet weld leg size [in]")
    weld_length_in: float = Field(0.0, ge=0.0, description="Total effective weld length [in] (sum of all weld segments)")
    weld_exx_ksi: float = Field(70.0, gt=0, description="Weld metal tensile strength Exx [ksi]")
    include_self_weight: bool = True

    @model_validator(mode="after")
    def _validate_weld_inputs(self) -> "SpreaderInputs":
        if self.weld_check:
            if self.weld_size_in <= 0:
                raise ValueError("weld_size_in must be > 0 when weld_check is enabled.")
            if self.weld_length_in <= 0:
                raise ValueError("weld_length_in must be > 0 when weld_check is enabled.")
        return self


class TwoWayPointLoad(BaseModel):
    x_ft: float = Field(..., ge=0, description="Point load location from left end [ft]")
    P_kip: float = Field(..., ge=0, description="Point load magnitude (downward) [kip]")


class SpreaderTwoWayInputs(Common):
    mode: Literal["spreader_two_way"] = "spreader_two_way"
    shape: str = Field(..., description="AISC shape label")
    length_ft: float = Field(..., gt=0, description="Total spreader beam length [ft]")
    padeye_edge_ft: float = Field(..., ge=0, description="Edge distance to top padeye hole [ft]")
    padeye_height_in: float = Field(..., ge=0, description="Top of beam to top padeye hole [in]")
    sling_angle_deg: float = Field(..., gt=0, lt=90, description="Minimum sling angle from horizontal [deg]")
    point_loads: list[TwoWayPointLoad] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_geometry(self) -> "SpreaderTwoWayInputs":
        if self.padeye_edge_ft * 2.0 >= self.length_ft:
            raise ValueError("padeye_edge_ft must be less than half the total length.")
        for pl in self.point_loads:
            if pl.x_ft > self.length_ft + 1e-9:
                raise ValueError(f"Point load beyond beam length: x={pl.x_ft} > {self.length_ft}")
        return self
