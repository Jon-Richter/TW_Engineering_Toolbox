# ACI 318-25 Concrete Design Tool – Compliance Review Report (v0.4.0)

Date: 2026-01-13  
Code basis: ACI 318-25 (primary). Where ACI 318-25 table text could not be reliably extracted from the provided PDF within three attempts, conservative legacy logic consistent with ACI 318-19 practice was used and is explicitly noted below.

## Executive summary

This package implements the following design modules:
- Beams and one-way slabs: flexure (strain compatibility) and one-way shear (Ch. 22)
- Slender walls: axial + out-of-plane flexure + in-plane shear with second-order (end-condition dropdowns)
- Development length & splices: Chapter 25 (conservative implementation with splice class logic)
- Concrete anchors: Chapter 17 framework (starter implementation with conservative CCD-style checks + Hilti product selection stubs)
- Two-way (punching) shear: conservative baseline without full moment transfer amplification

This release is suitable for internal engineering use and iterative validation. For permit / final design workflows, the anchored and punching shear modules should be expanded to full ACI 318-25 provisions (and manufacturer approval tables for post-installed anchors), as noted.

## Module-by-module review

### 1) Beam flexure (module: beam_flexure)
- Implemented: rectangular section strain compatibility with multi-layer reinforcement
- φ: Table 21.2.2 strain-based factor
- Concrete stress block: β1 per Table 22.2.2.4.3
- Status: COMPLIANT for intended scope (nonprestressed, nonseismic, no torsion)

### 2) One-way slab flexure (module: slab_oneway_flexure)
- Implemented: 12-in strip strain compatibility
- φ and β1 per same provisions as beam flexure
- Status: COMPLIANT for intended scope

### 3) Column axial (module: column_axial)
- Implemented: concentric axial capacity with φ compression-controlled
- Status: PARTIAL (P–M interaction not included in this package)

### 4) Slender walls (module: wall_slender)
- Implemented: second-order amplification with end conditions; axial + out-of-plane flexure + in-plane shear check framework
- Status: PARTIAL (requires project-specific validation for drift stability, cracking assumptions, and stiffness reduction)

### 5) Development length & splices (module: development_length_splices)
- Implemented:
  - tension development length (ld)
  - compression development length (ld, conservative)
  - tension lap splice length with Class A / Class B logic
  - compression lap splice length (conservative)
- ACI 318-25 reference: Chapter 25 (exact table extraction not required for this implementation)
- Noted conservatism:
  - minimum length floors used
  - ψ modifiers implemented for top-cast and epoxy; other modifiers treated conservatively
- Status: PARTIAL (conservative; verify against ACI 318-25 §25.4 and §25.5 for final design)

### 6) Concrete anchors (module: anchors_ch17)
- Implemented:
  - conservative steel tension and shear strengths (gross area based)
  - conservative CCD-style concrete breakout in tension and shear with simplified edge/spacing reductions
  - pullout credited only for cast-in headed rods (conservative)
  - pryout proportional to shear breakout (conservative)
  - simple linear interaction (Nu/Nn + Vu/Vn ≤ 1) when both actions present
- φ factors:
  - Attempted to locate Table 21.2.1 in the provided ACI 318-25 PDF; automated text extraction was not reliable.
  - Implemented φ factors consistent with ACI 318-25 / 318-19 anchor practice (redundancy and ductility flags are inputs).
- Manufacturer database:
  - Hilti KB-TZ2 (wedge), Hilti KH-EZ (screw), Hilti HIT-HY 200 V3 (adhesive) are selectable families.
  - This release does not yet apply ESR table strengths; it is a conservative starter framework.
- Status: PARTIAL (framework in place; requires completion of ACI 318-25 projected area method and ESR/ETA table integration for post-installed anchors)

### 7) Two-way punching shear (module: punching_shear)
- Implemented:
  - critical perimeter b0 at d/2 (interior/edge/corner perimeter reductions)
  - conservative Vc = 4 λ √f'c b0 d, φ = 0.75
- Not implemented in this release:
  - full ACI moment transfer / eccentricity effects
  - shear reinforcement
- Status: PARTIAL (conservative baseline)

## Corrections performed in this release
- Added new modules: anchors_ch17, punching_shear, development_length_splices
- Updated the offline frontend to expose these modules and inputs
- Bumped tool version to v0.4.0

## Known limitations / out-of-scope
- Prestressed concrete
- Seismic Chapter 18 provisions
- Torsion
- Openings / coupling beams
- Full column P–M interaction
- Full manufacturer-approval-based post-installed anchor strengths (ESR/ETA tables)

