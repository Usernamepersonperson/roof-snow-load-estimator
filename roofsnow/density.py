"""Snow depth to snow water equivalent (SWE) density models.

All functions return bulk density in kg/m3 or SWE in mm (= kg/m2). 1 cm of snow at 100 kg/m3 is 1 mm SWE.

sturm_alpine : Sturm, Taras, Liston, Derksen, Jonas, Lea (2010), "Estimating snow water equivalent using
               snow depth data and climate classes", J. Hydrometeorology 11, 1380-1394.
               Eq. (6): rho = (rho_max - rho_0) * (1 - exp(-k1*HS - k2*DOY)) + rho_0, HS in cm.
               DOY is the paper's season day: -92 on 1 October, -1 on 31 December, +1 on 1 January,
               +181 on 30 June, no zero (paper p. 1388-1389: 1 February = 32, 15 November = -47).
               Alpine class (Table 4, checked against the paper 2026-11-09): rho_max 0.5975,
               rho_0 0.2237 g/cm3, k1 0.0012, k2 0.0038.
constant     : one fixed bulk density, the simplest engineering assumption (default 300 kg/m3).
layered      : settlement model. New snow (positive depth increments) enters as a layer at the
               temperature-dependent fresh density of Hedstrom & Pomeroy (1998), Hydrol. Process. 12,
               1611-1625: rho_new = 67.92 + 51.25 * exp(Ta / 2.59). Each layer densifies toward
               RHO_OLD with an e-folding time TAU_DAYS (assumption; no calibration), and depth losses not
               explained by settlement (melt, wind) remove mass from the top layers.
mass_balance : daily snow mass balance (precipitation split by temperature, degree-day melt), anchored to
               the measured depth so SWE stays within physical density bounds. Independent of the
               depth-density regression, so it is used as a cross-check.
               Degree-day factors for glacier-free snow are of order 2-6 mm/K/day (Hock 2003, J. Hydrol.
               282, 104-115). Rain-snow split by linear temperature interpolation is common practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

RHO_MIN = 50.0   # kg/m3, fresh, cold powder
RHO_MAX = 550.0  # kg/m3, ripe, old firn-like snow pack

STURM_ALPINE = {"rho_max": 597.5, "rho_0": 223.7, "k1": 0.0012, "k2": 0.0038}


def doy_from_oct1(date: pd.Timestamp) -> int:
    """Days since the most recent 1 October (0 on 1 Oct)."""
    year = date.year if date.month >= 10 else date.year - 1
    return (date.normalize() - pd.Timestamp(year=year, month=10, day=1)).days


STURM_DOY_MIN, STURM_DOY_MAX = -92, 181


def sturm_doy(date: pd.Timestamp) -> int:
    """Season day as defined by Sturm et al. (2010): -92 on 1 Oct ... -1 on 31 Dec, 1 on 1 Jan ... 181 on 30 Jun.

    Dates outside 1 Oct - 30 Jun are clamped to the model's range.
    """
    d = date.normalize()
    if d.month >= 10:
        doy = (d - pd.Timestamp(year=d.year + 1, month=1, day=1)).days
    else:
        doy = d.day_of_year
    return int(min(max(doy, STURM_DOY_MIN), STURM_DOY_MAX))


def sturm_density(hs_cm: float, doy: float, p: dict[str, float] | None = None) -> float:
    """Bulk density (kg/m3) from depth (cm) and the Sturm season day `doy` (see `sturm_doy`)."""
    p = p or STURM_ALPINE
    if hs_cm is None or not math.isfinite(hs_cm) or hs_cm < 0:
        return float("nan")
    rho = (p["rho_max"] - p["rho_0"]) * (1.0 - math.exp(-p["k1"] * hs_cm - p["k2"] * doy)) + p["rho_0"]
    return min(max(rho, RHO_MIN), RHO_MAX)


def swe_from_density(hs_cm: pd.Series, rho: pd.Series | float) -> pd.Series:
    """SWE in mm."""
    return hs_cm.clip(lower=0) * rho / 100.0


def swe_constant(hs_cm: pd.Series, rho: float = 300.0) -> pd.Series:
    if not RHO_MIN <= rho <= RHO_MAX:
        raise ValueError(f"density must be within {RHO_MIN}..{RHO_MAX} kg/m3")
    return swe_from_density(hs_cm, rho)


def swe_sturm(hs_cm: pd.Series, p: dict[str, float] | None = None) -> pd.Series:
    doy = pd.Series([sturm_doy(t) for t in hs_cm.index], index=hs_cm.index)
    rho = pd.Series([sturm_density(h, d, p) for h, d in zip(hs_cm, doy, strict=True)], index=hs_cm.index)
    return swe_from_density(hs_cm, rho)


@dataclass(frozen=True)
class MassBalance:
    ddf_mm_per_k_day: float = 4.0      # degree-day factor (assumption, Hock 2003 range 2-6)
    t_snow_c: float = 0.0              # all snow below this daily mean temperature
    t_rain_c: float = 2.0              # all rain above this daily mean temperature
    snow_catch: float = 1.15           # gauge undercatch correction for snowfall (assumption)


def snow_fraction(temp_c: float, t_snow: float = 0.0, t_rain: float = 2.0) -> float:
    if not math.isfinite(temp_c):
        return 0.0
    if temp_c <= t_snow:
        return 1.0
    if temp_c >= t_rain:
        return 0.0
    return (t_rain - temp_c) / (t_rain - t_snow)


def swe_mass_balance(hs_cm: pd.Series, precip_mm: pd.Series, temp_c: pd.Series,
                     p: MassBalance = MassBalance()) -> pd.Series:
    """Daily SWE (mm) from precipitation and degree-day melt, anchored to measured depth.

    The anchor enforces 0 SWE when the depth is 0 and keeps density in [RHO_MIN, RHO_MAX]; NaN depth
    days carry the previous SWE forward.
    """
    swe = 0.0
    out = []
    for hs, pr, t in zip(hs_cm.to_numpy(), precip_mm.to_numpy(), temp_c.to_numpy(), strict=True):
        pr = 0.0 if not math.isfinite(pr) else max(pr, 0.0)
        if math.isfinite(t):
            swe += pr * snow_fraction(t, p.t_snow_c, p.t_rain_c) * p.snow_catch
            if t > 0.0:
                swe -= min(swe, p.ddf_mm_per_k_day * t)
        if math.isfinite(hs):
            if hs <= 0:
                swe = 0.0
            else:
                swe = float(np.clip(swe, hs * RHO_MIN / 100.0, hs * RHO_MAX / 100.0))
        out.append(swe)
    return pd.Series(out, index=hs_cm.index)


RHO_OLD = 450.0   # kg/m3 asymptotic density of an aged layer (assumption)
TAU_DAYS = 30.0   # e-folding time of layer densification (assumption)


def fresh_snow_density(temp_c: float) -> float:
    """Hedstrom & Pomeroy (1998) new-snow density from air temperature (kg/m3)."""
    t = 0.0 if not math.isfinite(temp_c) else min(temp_c, 0.0)
    return 67.92 + 51.25 * math.exp(t / 2.59)


def swe_layered(hs_cm: pd.Series, temp_c: pd.Series, tau_days: float = TAU_DAYS,
                rho_old: float = RHO_OLD) -> pd.Series:
    """Layer settlement model; returns SWE (mm). NaN-depth days carry the previous SWE forward."""
    layers: list[list[float]] = []  # [thickness_cm, rho] from bottom to top
    decay = math.exp(-1.0 / tau_days)
    out = []
    for hs, t in zip(hs_cm.to_numpy(), temp_c.to_numpy(), strict=True):
        for lay in layers:
            mass = lay[0] * lay[1]
            lay[1] = min(rho_old - (rho_old - lay[1]) * decay, rho_old)
            lay[0] = mass / lay[1]  # settlement keeps mass: thickness shrinks as density rises
        if not math.isfinite(hs):
            out.append(sum(h * r for h, r in layers) / 100.0)
            continue
        if hs <= 0:
            layers = []
        else:
            diff = hs - sum(h for h, _ in layers)
            if diff > 0.5:
                layers.append([diff, fresh_snow_density(t)])
            elif diff < -0.01:
                cut = -diff
                while cut > 1e-9 and layers:
                    top = layers[-1]
                    if top[0] <= cut:
                        cut -= top[0]
                        layers.pop()
                    else:
                        top[0] -= cut
                        cut = 0.0
        out.append(sum(h * r for h, r in layers) / 100.0)
    return pd.Series(out, index=hs_cm.index)


def swe_by_model(name: str, hs_cm: pd.Series, precip_mm: pd.Series | None = None,
                 temp_c: pd.Series | None = None, const_rho: float = 300.0) -> pd.Series:
    if name == "constant":
        return swe_constant(hs_cm, const_rho)
    if name == "sturm":
        return swe_sturm(hs_cm)
    if name == "layered":
        if temp_c is None:
            raise ValueError("layered needs temperature")
        return swe_layered(hs_cm, temp_c)
    if name == "mass_balance":
        if precip_mm is None or temp_c is None:
            raise ValueError("mass_balance needs precipitation and temperature")
        return swe_mass_balance(hs_cm, precip_mm, temp_c)
    raise ValueError(f"unknown density model {name!r}; use constant, sturm, layered or mass_balance")


MODELS = ("constant", "sturm", "layered", "mass_balance")
