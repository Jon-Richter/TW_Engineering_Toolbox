\
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, conlist, confloat, model_validator


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module: str = Field(..., description="Design module identifier")
    inputs: dict = Field(..., description="Module-specific inputs")


class ReinfLayer(BaseModel):
    """Reinforcement layer definition for rectangular sections.

    Default entry is by bar count + bar diameter (Option 1).
    User may override by directly entering area As for the layer (Option 2).

    Coordinates:
    - face = 'top' or 'bottom'
    - offset_in = distance from selected face to bar centroid
      (e.g., bottom layer: cover + stirrup_dia + db/2).
    """

    model_config = ConfigDict(extra="forbid")

    face: Literal["top", "bottom"] = Field("bottom", description="Reference face for offset")
    offset_in: confloat(gt=0) = Field(3.0, description="Distance from selected face to bar centroid", json_schema_extra={"units": "in"})

    # Option 1 (default): bar count and bar diameter
    n_bars: Optional[confloat(gt=0)] = Field(3, description="Number of bars in the layer (Option 1)")
    bar_dia_in: Optional[confloat(gt=0)] = Field(1.0, description="Bar diameter for the layer (Option 1)", json_schema_extra={"units": "in"})

    # Option 2 (override): directly enter area
    As_override_in2: Optional[confloat(gt=0)] = Field(None, description="Override steel area As for layer (Option 2)", json_schema_extra={"units": "in^2"})

    @model_validator(mode="after")
    def _validate_layer(self):
        if self.As_override_in2 is not None:
            return self
        if self.n_bars is None or self.bar_dia_in is None:
            raise ValueError("Provide either As_override_in2 OR (n_bars and bar_dia_in) for each reinforcement layer")
        return self



class BeamFlexureInputs(BaseModel):
    """
    Rectangular singly reinforced beam / strip design for flexure (strength design).
    Units are US customary (in, psi, kip-ft).
    """
    model_config = ConfigDict(extra="forbid")

    # Geometry
    b_in: confloat(gt=0) = Field(12.0, description="Beam width b", json_schema_extra={"units": "in"})
    h_in: confloat(gt=0) = Field(24.0, description="Total depth h", json_schema_extra={"units": "in"})
    cover_in: confloat(ge=0) = Field(1.5, description="Concrete cover to stirrup outside", json_schema_extra={"units": "in"})
    stirrup_dia_in: confloat(ge=0) = Field(0.375, description="Stirrup bar diameter", json_schema_extra={"units": "in"})
    bar_dia_in: confloat(gt=0) = Field(1.0, description="Legacy main tension bar diameter (used only if reinf_layers is empty)", json_schema_extra={"units": "in"})

    reinf_layers: list[ReinfLayer] = Field(default_factory=list, description="Reinforcement layers. If provided, tool evaluates capacity using multi-layer strain compatibility. Option 1 (default): n_bars + bar_dia_in; Option 2: As_override_in2 override.")

    # Loads
    Mu_kipft: confloat(gt=0) = Field(180.0, description="Factored design moment Mu", json_schema_extra={"units": "kip-ft"})

    # Materials
    fc_psi: confloat(gt=0) = Field(4000.0, description="Specified concrete compressive strength f'c", json_schema_extra={"units": "psi"})
    fy_psi: confloat(gt=0) = Field(60000.0, description="Steel yield strength fy", json_schema_extra={"units": "psi"})
    Es_psi: confloat(gt=0) = Field(29000000.0, description="Steel modulus Es", json_schema_extra={"units": "psi"})

    # Options
    compression_reinf: bool = Field(False, description="Legacy flag (deprecated). Use reinf_layers with top layers for compression reinforcement.")
    transverse_type: Literal["other", "spiral"] = Field("other", description="Transverse reinforcement type for phi per Table 21.2.2 (spiral only affects compression-controlled)")


class SlabOnewayFlexureInputs(BaseModel):
    """
    One-way slab strip design for flexure, per 12-in wide strip.
    """
    model_config = ConfigDict(extra="forbid")

    thickness_in: confloat(gt=0) = Field(8.0, description="Slab thickness h", json_schema_extra={"units": "in"})
    cover_in: confloat(ge=0) = Field(0.75, description="Clear cover to main bar", json_schema_extra={"units": "in"})
    bar_dia_in: confloat(gt=0) = Field(0.5, description="Legacy main bar diameter (used only if reinf_layers is empty)", json_schema_extra={"units": "in"})

    reinf_layers: list[ReinfLayer] = Field(default_factory=list, description="Reinforcement layers for 12-in strip. If provided, tool evaluates capacity using multi-layer strain compatibility.")
    Mu_kipft_per_ft: confloat(gt=0) = Field(12.0, description="Factored moment per foot width", json_schema_extra={"units": "kip-ft/ft"})

    fc_psi: confloat(gt=0) = Field(4000.0, description="Specified concrete compressive strength f'c", json_schema_extra={"units": "psi"})
    fy_psi: confloat(gt=0) = Field(60000.0, description="Steel yield strength fy", json_schema_extra={"units": "psi"})
    Es_psi: confloat(gt=0) = Field(29000000.0, description="Steel modulus Es", json_schema_extra={"units": "psi"})


class ColumnAxialInputs(BaseModel):
    """
    Axial compressive strength of nonprestressed reinforced column (short, concentric).
    """
    model_config = ConfigDict(extra="forbid")

    shape: Literal["rectangular", "circular"] = Field("rectangular")
    b_in: confloat(gt=0) = Field(16.0, description="Column width (rectangular)", json_schema_extra={"units": "in"})
    h_in: confloat(gt=0) = Field(16.0, description="Column depth (rectangular)", json_schema_extra={"units": "in"})
    D_in: confloat(gt=0) = Field(18.0, description="Column diameter (circular)", json_schema_extra={"units": "in"})
    Ast_in2: confloat(gt=0) = Field(4.0, description="Total longitudinal reinforcement area Ast", json_schema_extra={"units": "in^2"})

    Pu_kip: confloat(gt=0) = Field(500.0, description="Factored axial compressive load Pu", json_schema_extra={"units": "kip"})

    fc_psi: confloat(gt=0) = Field(5000.0, description="Specified concrete compressive strength f'c", json_schema_extra={"units": "psi"})
    fy_psi: confloat(gt=0) = Field(60000.0, description="Steel yield strength fy", json_schema_extra={"units": "psi"})

    transverse_type: Literal["ties", "spiral"] = Field("ties", description="Transverse reinforcement type (affects Pn,max and phi)")


class DevLengthTensionInputs(BaseModel):
    """
    Straight deformed bar development length in tension using Table 25.4.2.3 and Table 25.4.2.5.
    """
    model_config = ConfigDict(extra="forbid")

    bar_size: Literal["#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#14", "#18"] = Field("#5")
    db_in: confloat(gt=0) = Field(0.625, description="Bar diameter db", json_schema_extra={"units": "in"})

    fc_psi: confloat(gt=0) = Field(4000.0, description="Specified concrete compressive strength f'c", json_schema_extra={"units": "psi"})
    fy_psi: confloat(gt=0) = Field(60000.0, description="Steel yield strength fy", json_schema_extra={"units": "psi"})

    lambda_factor: confloat(gt=0, le=1.0) = Field(1.0, description="Lightweight concrete modification factor λ (1.0 normalweight, 0.75 lightweight)", json_schema_extra={"units": ""})

    is_top_bar: bool = Field(False, description="Top reinforcement (more than 12 in. fresh concrete below)")
    is_epoxy: bool = Field(False, description="Epoxy-coated reinforcement")
    epoxy_cover_lt_3db_or_spacing_lt_6db: bool = Field(False, description="Epoxy with cover < 3db OR clear spacing < 6db (affects ψe)")

    fy_gt_60000: bool = Field(False, description="Use grade factor ψg for fy > 60 ksi per Table 25.4.2.5")

    # Condition selection per Table 25.4.2.3
    # Optional: lap splice in tension per Table 25.5.2.1
    As_provided_over_As_required: confloat(gt=0) = Field(1.0, description="As,provided / As,required at splice location (for lap splice length per Table 25.5.2.1)", json_schema_extra={"units": ""})
    max_percent_spliced: confloat(gt=0, le=100) = Field(100.0, description="Maximum percent of As spliced within required lap length (Table 25.5.2.1). Tool uses this only to label; detailed splice distribution checks not implemented.", json_schema_extra={"units": "%"} )

    # Condition selection per Table 25.4.2.3
    cover_ge_db_and_spacing_ge_2db: bool = Field(True, description="Clear cover >= db AND clear spacing >= 2db (Table 25.4.2.3, first row)")


class WallSlenderInputs(BaseModel):
    """Slender reinforced concrete wall check for combined axial + in-plane + out-of-plane bending.

    This module uses the ACI 318-25 first-order moment magnification method (6.6.4.5) with
    critical buckling load per 6.6.4.4.2 and effective stiffness per 6.6.4.4.4(c), using
    wall section properties per Table 6.6.3.1.1(a)/(b). Section strength is evaluated using
    strain compatibility per Chapter 22 and strength reduction factors per 21.2.2.

    Units are US customary (in, psi, kip, kip-ft).
    """

    model_config = ConfigDict(extra="forbid")

    # Geometry
    lw_in: confloat(gt=0) = Field(144.0, description="Wall length \u2113w", json_schema_extra={"units": "in"})
    t_in: confloat(gt=0) = Field(10.0, description="Wall thickness t", json_schema_extra={"units": "in"})
    lu_in: confloat(gt=0) = Field(144.0, description="Unsupported height/length \u2113u (out-of-plane axis)", json_schema_extra={"units": "in"})
    cover_in: confloat(ge=0) = Field(1.5, description="Clear cover to vertical reinforcement", json_schema_extra={"units": "in"})
    bar_dia_in: confloat(gt=0) = Field(0.625, description="Vertical bar diameter (assumed 2 face layers)", json_schema_extra={"units": "in"})

    rho_v: confloat(gt=0, lt=0.10) = Field(0.0025, description="Total vertical reinforcement ratio \u03c1v = As/(Ag)", json_schema_extra={"units": ""})

    # Loads
    Pu_kip: confloat(ge=0) = Field(200.0, description="Factored axial load Pu (compression positive)", json_schema_extra={"units": "kip"})

    # Out-of-plane end moments (sign matters for Cm)
    M_top_oop_kipft: float = Field(20.0, description="Out-of-plane end moment at top (kip-ft; sign per convention)", json_schema_extra={"units": "kip-ft"})
    M_bot_oop_kipft: float = Field(-20.0, description="Out-of-plane end moment at bottom (kip-ft; sign per convention)", json_schema_extra={"units": "kip-ft"})

    # In-plane end moments
    M_top_ip_kipft: float = Field(10.0, description="In-plane end moment at top (kip-ft; sign per convention)", json_schema_extra={"units": "kip-ft"})
    M_bot_ip_kipft: float = Field(-10.0, description="In-plane end moment at bottom (kip-ft; sign per convention)", json_schema_extra={"units": "kip-ft"})

    # End conditions (requested two dropdowns)
    end_bottom: Literal["fixed", "pinned", "unbraced"] = Field("fixed", description="Bottom end condition")
    end_top: Literal["fixed", "pinned", "unbraced"] = Field("fixed", description="Top end condition")

    # Materials
    fc_psi: confloat(gt=0) = Field(4000.0, description="Specified concrete compressive strength f'c", json_schema_extra={"units": "psi"})
    fy_psi: confloat(gt=0) = Field(60000.0, description="Steel yield strength fy", json_schema_extra={"units": "psi"})
    Es_psi: confloat(gt=0) = Field(29000000.0, description="Steel modulus Es", json_schema_extra={"units": "psi"})

    transverse_type: Literal["other", "spiral"] = Field("other", description="Transverse reinforcement type for \u03d5 per Table 21.2.2")

    # Slenderness / stiffness options
    cracked_section: bool = Field(True, description="Use cracked section I for stiffness per Table 6.6.3.1.1(a)")
    beta_dns: confloat(ge=0, le=1.0) = Field(0.6, description="\u03b2dns = sustained axial / total axial for same combo (6.6.4.4.4); default 0.6", json_schema_extra={"units": ""})
    member_nonsway: bool = Field(True, description="Treat member as nonsway for \u03b4ns (6.6.4.5). If false, tool reports as not implemented.")
    transverse_loads_between_ends: bool = Field(True, description="If true, Cm = 1.0 per 6.6.4.5.3b")



class AnchorsCh17Inputs(BaseModel):
    """ACI 318-25 Chapter 17 anchor design (nonseismic). Units: US."""
    model_config = ConfigDict(extra="forbid")

    # Loads (factored)
    Nu_kip: float = Field(0.0, description="Factored tensile load (tension +)", json_schema_extra={"units": "kip"})
    Vu_kip: float = Field(0.0, description="Factored shear load (magnitude)", json_schema_extra={"units": "kip"})

    # Base material
    fc_psi: confloat(gt=0) = Field(4000.0, description="Concrete compressive strength f'c", json_schema_extra={"units":"psi"})
    cracked: bool = Field(True, description="Cracked concrete for Ch. 17 modifiers")
    lightweight: bool = Field(False, description="Lightweight concrete")
    lambda_factor: confloat(gt=0) = Field(1.0, description="ACI lambda modifier (1.0 normalweight)", json_schema_extra={"units":""})

    # Geometry (single anchor or group in rectangular pattern)
    anchor_count_x: int = Field(1, ge=1, description="Number of anchors in X")
    anchor_count_y: int = Field(1, ge=1, description="Number of anchors in Y")
    sx_in: confloat(gt=0) = Field(0.0, description="Spacing in X (0 for single)", json_schema_extra={"units":"in"})
    sy_in: confloat(gt=0) = Field(0.0, description="Spacing in Y (0 for single)", json_schema_extra={"units":"in"})
    edge_x_neg_in: confloat(ge=0) = Field(0.0, description="Edge distance to -X edge", json_schema_extra={"units":"in"})
    edge_x_pos_in: confloat(ge=0) = Field(0.0, description="Edge distance to +X edge", json_schema_extra={"units":"in"})
    edge_y_neg_in: confloat(ge=0) = Field(0.0, description="Edge distance to -Y edge", json_schema_extra={"units":"in"})
    edge_y_pos_in: confloat(ge=0) = Field(0.0, description="Edge distance to +Y edge", json_schema_extra={"units":"in"})
    member_thickness_in: confloat(gt=0) = Field(8.0, description="Member thickness (for breakout/pryout limits)", json_schema_extra={"units":"in"})

    # Anchor selection
    anchor_family: Literal["cast_in_headed", "hilti_kb_tz2", "hilti_kh_ez", "hilti_hit_hy200_v3"] = Field(
        "hilti_kh_ez", description="Anchor family / dataset"
    )
    diameter_in: confloat(gt=0) = Field(0.5, description="Anchor diameter", json_schema_extra={"units":"in"})
    hef_in: confloat(gt=0) = Field(3.0, description="Effective embedment depth", json_schema_extra={"units":"in"})
    # Steel properties (used for cast-in; post-installed defaults from dataset if not provided)
    fy_psi: Optional[confloat(gt=0)] = Field(None, description="Anchor steel yield strength (if applicable)", json_schema_extra={"units":"psi"})
    fu_psi: Optional[confloat(gt=0)] = Field(None, description="Anchor steel ultimate strength (if applicable)", json_schema_extra={"units":"psi"})

    # Design flags for phi per Table 21.2.1
    redundant: bool = Field(False, description="Anchor system redundant for concrete failure phi")
    steel_ductile_tension: bool = Field(True, description="Steel ductile in tension")
    steel_ductile_shear: bool = Field(True, description="Steel ductile in shear")


class PunchingShearInputs(BaseModel):
    """Two-way (punching) shear check for nonprestressed slabs without shear reinforcement."""
    model_config = ConfigDict(extra="forbid")

    # Loads
    Vu_kip: confloat(ge=0) = Field(200.0, description="Factored shear at column", json_schema_extra={"units":"kip"})
    Mux_kipft: float = Field(0.0, description="Factored unbalanced moment about slab x-axis", json_schema_extra={"units":"kip-ft"})
    Muy_kipft: float = Field(0.0, description="Factored unbalanced moment about slab y-axis", json_schema_extra={"units":"kip-ft"})

    # Geometry
    column_bx_in: confloat(gt=0) = Field(16.0, description="Column dimension in x", json_schema_extra={"units":"in"})
    column_by_in: confloat(gt=0) = Field(16.0, description="Column dimension in y", json_schema_extra={"units":"in"})
    slab_thickness_in: confloat(gt=0) = Field(8.0, description="Slab thickness", json_schema_extra={"units":"in"})
    d_in: Optional[confloat(gt=0)] = Field(None, description="Effective depth (if omitted, 0.8*h)", json_schema_extra={"units":"in"})
    location: Literal["interior", "edge", "corner"] = Field("interior", description="Column location for perimeter reduction")

    # Materials
    fc_psi: confloat(gt=0) = Field(4000.0, description="Concrete compressive strength f'c", json_schema_extra={"units":"psi"})
    lambda_factor: confloat(gt=0) = Field(1.0, description="ACI lambda modifier", json_schema_extra={"units":""})


class DevelopmentLengthSpliceInputs(BaseModel):
    """ACI 318-25 Chapter 25 development length and splice provisions (nonseismic)."""
    model_config = ConfigDict(extra="forbid")

    calc_type: Literal["tension_development", "compression_development", "tension_lap_splice", "compression_lap_splice"] = Field(
        "tension_development", description="Calculation type"
    )

    # Bar
    bar_size: str = Field("#5", description="Bar size key (e.g., #5)")
    db_in: Optional[confloat(gt=0)] = Field(None, description="Bar diameter (if omitted, from rebar db)", json_schema_extra={"units":"in"})

    # Materials
    fc_psi: confloat(gt=0) = Field(4000.0, description="Concrete strength f'c", json_schema_extra={"units":"psi"})
    fy_psi: confloat(gt=0) = Field(60000.0, description="Steel yield fy", json_schema_extra={"units":"psi"})
    lambda_factor: confloat(gt=0) = Field(1.0, description="Lightweight modifier λ", json_schema_extra={"units":""})

    # Modifiers
    is_top_bar: bool = Field(False, description="Top-cast bar factor ψt")
    is_epoxy: bool = Field(False, description="Epoxy-coated bar factor ψe")
    epoxy_cover_lt_3db_or_spacing_lt_6db: bool = Field(False, description="Epoxy confinement condition for ψe")
    fy_gt_60000: bool = Field(False, description="Use fy > 60 ksi adjustment per ACI 25.4")
    cover_ge_db_and_spacing_ge_2db: bool = Field(True, description="Confinement condition used for conservative checks")

    # Splice-specific
    As_provided_over_As_required: confloat(gt=0) = Field(1.0, description="As_provided / As_required (for Class A splice eligibility)")
    percent_bars_spliced: confloat(ge=0, le=100) = Field(100.0, description="Percent of bars spliced at location")
