from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

TxGirderType = Literal["TX28", "TX34", "TX40", "TX46", "TX54", "TX62", "TX70"]

class C49Inputs(BaseModel):
    """
    Inputs for construction-stage overhang bracket spacing + placement solver.

    Workflow:
      1) Select a girder type (TxDOT Tx-girder library).
      2) Provide overhang and loading inputs (including screed wheel load).
      3) Tool performs:
         - placement feasibility screening (girder interference + valid bearing face)
         - spacing optimization subject to member SWL and simplified demand model

    Geometry:
      - The shipped TxDOT library uses A-21 for depth and bottom-flange width.
      - Web / flange thicknesses default to conservative placeholders for constructability screening
        and can be overridden in the inputs below.
    """

    # --- Primary project inputs
    girder_type: TxGirderType = Field("TX54", description="TxDOT Tx I-girder type.")
    overhang_length_ft: float = Field(6.0, gt=0.1, description="Deck overhang length from girder CL to deck edge (ft).")
    slab_thickness_in: float = Field(8.5, gt=1.0, description="Deck thickness (in).")
    screed_wheel_load_kip: float = Field(2.0, ge=0.0, description="Screed wheel load (kip).")

    # Distributed load components (construction)
    concrete_unit_weight_pcf: float = Field(150.0, gt=0.0, description="Unit weight of concrete (pcf).")
    construction_live_load_psf: float = Field(50.0, ge=0.0, description="Construction live load (psf).")
    formwork_misc_psf: float = Field(10.0, ge=0.0, description="Formwork + misc dead load (psf).")

    # Allowables / capacities
    hanger_swl_kip: float = Field(6.0, gt=0.0, description="Safe working load of top hanger connection (kip).")
    diagonal_swl_kip: float = Field(3.75, gt=0.0, description="Safe working load of diagonal (kip).")

    # Spacing search bounds
    min_spacing_ft: float = Field(2.0, gt=0.5, description="Minimum bracket spacing to consider (ft).")
    max_spacing_ft: float = Field(12.0, gt=0.5, description="Maximum bracket spacing to consider (ft).")
    spacing_step_ft: float = Field(0.25, gt=0.0, description="Spacing increment for scan (ft).")

    # --- Placement inputs (decking stack and constructability)
    deck_soffit_offset_in: float = Field(
        0.0,
        description="Deck vertical offset from girder top (positive downward). "
                    ,
    )
    plywood_thickness_in: float = Field(0.75, gt=0.0, description="Plywood thickness (in).")
    fourbyfour_thickness_in: float = Field(3.50, gt=0.0, description="Joist thickness (in).")
    twobysix_thickness_in: float = Field(1.50, gt=0.0, description="Nailer thickness flat (in).")

    max_bracket_depth_in: float = Field(50.0, gt=0.0, description="Maximum allowable bracket depth (in).")
    min_bracket_depth_in: float = Field(32.0, ge=0.0, description="Minimum practical bracket depth to consider (in).")
    clearance_in: float = Field(0.25, ge=0.0, description="Minimum geometric clearance from girder outline (in).")

    # Conservative member envelopes for interference screening
    top_member_height_in: float = Field(3.0, gt=0.0, description="Envelope height for top member in section (in).")
    vertical_member_width_in: float = Field(2.0, gt=0.0, description="Envelope width for vertical/hanger member in section (in).")
    diagonal_envelope_thickness_in: float = Field(2.0, gt=0.0, description="Envelope thickness for diagonal member in section (in).")
    bottom_pad_height_in: float = Field(2.0, gt=0.0, description="Bearing pad/seat envelope height along vertical face (in).")
    top_hanger_edge_clear_in: float = Field(
        0.0, ge=0.0, description="Vertical extension of vertical member above girder top for screening (in)."
    )

    # --- Girder thickness overrides (advanced; inches)
    girder_web_thickness_in: Optional[float] = Field(None, gt=0.0, description="Girder web thickness (in).")
    girder_top_flange_thickness_in: Optional[float] = Field(None, gt=0.0, description="Girder top flange thickness (in).")
    girder_bottom_flange_thickness_in: Optional[float] = Field(None, gt=0.0, description="Girder bottom flange thickness (in).")
    girder_top_flange_width_in: Optional[float] = Field(None, gt=0.0, description="Girder top flange width (in).")
    girder_bottom_flange_width_in: Optional[float] = Field(None, gt=0.0, description="Girder: bottom flange width (in).")
    girder_depth_in: Optional[float] = Field(None, gt=0.0, description="Girder depth (in).")

    # --- Demand model controls
    use_geometry_based_diagonal_angle: bool = Field(
        True, description="If True, diagonal demand uses sin(theta) from placement geometry."
    )
    bracket_lever_arm_ft: float = Field(
        3.0, gt=0.0, description="Fallback lever arm if placement is infeasible or disabled (ft)."
    )

    @field_validator("girder_type")
    @classmethod
    def _normalize_girder_type(cls, v: str) -> str:
        return v.upper().replace(" ", "")

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.max_spacing_ft <= self.min_spacing_ft:
            raise ValueError("max_spacing_ft must be greater than min_spacing_ft.")
        if self.spacing_step_ft <= 0.0:
            raise ValueError("spacing_step_ft must be > 0.")
        if self.max_bracket_depth_in <= 0.0:
            raise ValueError("max_bracket_depth_in must be > 0.")
        if self.min_bracket_depth_in < 0.0:
            raise ValueError("min_bracket_depth_in must be >= 0.")
        if self.min_bracket_depth_in >= self.max_bracket_depth_in:
            raise ValueError("min_bracket_depth_in must be less than max_bracket_depth_in.")
        if self.bracket_lever_arm_ft <= 0.0:
            raise ValueError("bracket_lever_arm_ft must be > 0.")
        return self

    def girder_override_dict(self) -> dict:
        d = {}
        if self.girder_web_thickness_in is not None:
            d["web_thickness_in"] = self.girder_web_thickness_in
        if self.girder_top_flange_thickness_in is not None:
            d["top_flange_thickness_in"] = self.girder_top_flange_thickness_in
        if self.girder_bottom_flange_thickness_in is not None:
            d["bottom_flange_thickness_in"] = self.girder_bottom_flange_thickness_in
        if self.girder_top_flange_width_in is not None:
            d["top_flange_width_in"] = self.girder_top_flange_width_in
        if self.girder_bottom_flange_width_in is not None:
            d["bottom_flange_width_in"] = self.girder_bottom_flange_width_in
        if self.girder_depth_in is not None:
            d["depth_in"] = self.girder_depth_in
        return d
