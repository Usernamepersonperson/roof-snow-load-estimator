"""Design-load helpers from publicly documented formulas.

Ground snow load (Switzerland), as published on the free German-language encyclopedia page
"Schneelast in der Schweiz" (https://de.wikipedia.org/wiki/Schneelast_in_der_Schweiz) and mirrored by
several free calculators:

    s_k = [1 + (h0 / 350)^2] * 0.4 kN/m2  >=  0.9 kN/m2,   h0 = h + Z,  valid up to h = 2000 m

h is the site altitude (m a.s.l.); Z in {-200, 0, +200, +400, +500} m is a location offset read from the
standard's zone map. The map is part of the copyrighted SIA 261 and is NOT reproduced here: Z is a user
parameter (default 0). The tool is not a substitute for the standard or a structural engineer.

Roof shape coefficient mu1 and exposure coefficient Ce follow the structure of EN 1991-1-3 (public
Eurocode concepts, sections 5.3.2 and 5.2): mu1 = 0.8 for roof pitch <= 30 deg, falling linearly to 0 at
60 deg. Ce = 0.8 / 1.0 / 1.2 for windswept / normal / sheltered topography. Ct = 1.0 (unheated or
well-insulated roof assumed). Drifting at parapets and roof steps is not modelled.
"""

from __future__ import annotations

import math

G = 9.81  # m/s2; 1 mm snow water equivalent (= 1 kg/m2) weighs 0.00981 kN/m2
VALID_Z = (-200.0, 0.0, 200.0, 400.0, 500.0)
EXPOSURE_CE = {"windswept": 0.8, "normal": 1.0, "sheltered": 1.2}
FORMULA_MAX_ALT_M = 2000.0
SK_MIN = 0.9


def swe_mm_to_kn_m2(swe_mm: float) -> float:
    """Snow water equivalent (mm = kg/m2) to load in kN/m2."""
    return swe_mm * G / 1000.0


def kn_m2_to_swe_mm(load: float) -> float:
    return load * 1000.0 / G


def ground_snow_load_sk(altitude_m: float, z_offset_m: float = 0.0) -> float:
    """Characteristic ground snow load s_k in kN/m2 from the public altitude formula.

    Raises ValueError for non-finite input or a Z offset outside the documented set.
    """
    if not math.isfinite(altitude_m) or not math.isfinite(z_offset_m):
        raise ValueError("altitude and Z must be finite")
    if z_offset_m not in VALID_Z:
        raise ValueError(f"Z must be one of {VALID_Z}")
    h0 = altitude_m + z_offset_m
    return max((1.0 + (h0 / 350.0) ** 2) * 0.4, SK_MIN)


def altitude_in_validity(altitude_m: float) -> bool:
    """The public formula is documented only up to 2000 m."""
    return altitude_m <= FORMULA_MAX_ALT_M


def shape_coefficient(pitch_deg: float) -> float:
    """mu1 for a single-pitch or duopitch roof without snow guards (EN 1991-1-3 style)."""
    if not 0.0 <= pitch_deg <= 90.0:
        raise ValueError("roof pitch must be within 0..90 degrees")
    if pitch_deg <= 30.0:
        return 0.8
    if pitch_deg >= 60.0:
        return 0.0
    return 0.8 * (60.0 - pitch_deg) / 30.0


def exposure_coefficient(exposure: str) -> float:
    try:
        return EXPOSURE_CE[exposure.lower()]
    except KeyError:
        raise ValueError(f"exposure must be one of {sorted(EXPOSURE_CE)}") from None


def roof_design_load(altitude_m: float, z_offset_m: float = 0.0, pitch_deg: float = 0.0,
                     exposure: str = "normal", ct: float = 1.0) -> float:
    """Roof snow design load mu1 * Ce * Ct * s_k in kN/m2 (characteristic value)."""
    if ct <= 0:
        raise ValueError("Ct must be positive")
    return (shape_coefficient(pitch_deg) * exposure_coefficient(exposure) * ct
            * ground_snow_load_sk(altitude_m, z_offset_m))
