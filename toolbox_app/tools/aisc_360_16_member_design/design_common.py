from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


DesignMethod = Literal["LRFD", "ASD"]


@dataclass(frozen=True)
class StrengthFactors:
    phi: float
    omega: float


def factors_flexure(method: DesignMethod) -> StrengthFactors:
    # AISC Table 1-1 (typical): flexure φ=0.90, Ω=1.67
    return StrengthFactors(phi=0.90, omega=1.67) if method == "LRFD" else StrengthFactors(phi=0.90, omega=1.67)


def factors_shear(method: DesignMethod) -> StrengthFactors:
    # AISC Table 1-1: shear φ=1.00, Ω=1.50
    return StrengthFactors(phi=1.00, omega=1.50)


def factors_tension_yield(method: DesignMethod) -> StrengthFactors:
    # AISC Table 1-1: tension yielding φ=0.90, Ω=1.67
    return StrengthFactors(phi=0.90, omega=1.67)


def factors_tension_rupture(method: DesignMethod) -> StrengthFactors:
    # AISC Table 1-1: tension rupture φ=0.75, Ω=2.00
    return StrengthFactors(phi=0.75, omega=2.00)


def factors_compression(method: DesignMethod) -> StrengthFactors:
    # AISC Table 1-1: compression φ=0.90, Ω=1.67
    return StrengthFactors(phi=0.90, omega=1.67)


def design_strength(method: DesignMethod, Rn: float, phi: float, omega: float) -> float:
    if method == "LRFD":
        return phi * Rn
    return Rn / omega


def ft_to_in(ft: float) -> float:
    return 12.0 * ft


def kipft_to_kipin(kft: float) -> float:
    return 12.0 * kft


def kipin_to_kipft(kin: float) -> float:
    return kin / 12.0


def nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0


def safe_div(n: float, d: float) -> float:
    if abs(d) < 1e-12:
        return math.inf
    return n / d
