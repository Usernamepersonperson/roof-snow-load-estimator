"""Station data to roof snow load and utilisation of the design load."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from roofsnow import density, standard

MAX_GAP_DAYS = 3
MAX_TRANSFER_M = 500.0


@dataclass(frozen=True)
class Site:
    """Roof site and the assumptions used to move station data to it."""

    station_alt_m: float
    site_alt_m: float | None = None          # default: same altitude as the station
    z_offset_m: float = 0.0                  # SIA 261 map offset, read from the standard by the user
    pitch_deg: float = 0.0                   # 0 = flat roof
    exposure: str = "normal"                 # windswept | normal | sheltered
    ct: float = 1.0
    design_load_kn_m2: float | None = None   # override, e.g. the load the roof was actually built for
    gradient_per_100m: float = 0.06          # assumption: +6 % snow mass per 100 m altitude difference
    rain_capacity: float = 0.05              # liquid water the pack can hold, fraction of SWE (assumption)
    rain_drain_per_day: float = 0.5          # fraction of retained water that drains/refreezes per day
    rain_temp_c: float = 1.0                 # daily mean above this counts precipitation as rain

    @property
    def altitude(self) -> float:
        return self.station_alt_m if self.site_alt_m is None else self.site_alt_m

    def transfer_factor(self) -> float:
        """Station-to-site snow mass scaling (linear altitude gradient, bounded)."""
        dh = self.altitude - self.station_alt_m
        if abs(dh) > MAX_TRANSFER_M:
            raise ValueError(f"altitude transfer beyond {MAX_TRANSFER_M:.0f} m is not supported")
        return max(0.0, 1.0 + self.gradient_per_100m * dh / 100.0)

    def design_load(self) -> float:
        if self.design_load_kn_m2 is not None:
            if self.design_load_kn_m2 <= 0:
                raise ValueError("design load must be positive")
            return self.design_load_kn_m2
        d = standard.roof_design_load(self.altitude, self.z_offset_m, self.pitch_deg,
                                      self.exposure, self.ct)
        if d <= 0:
            raise ValueError("roof pitch >= 60 deg sheds snow (mu1 = 0): pass an explicit design load")
        return d

    def roof_factor(self) -> float:
        return (standard.shape_coefficient(self.pitch_deg)
                * standard.exposure_coefficient(self.exposure) * self.ct)


@dataclass(frozen=True)
class Result:
    table: pd.DataFrame
    design_kn_m2: float
    sk_kn_m2: float
    model: str
    site: Site = field(repr=False)


SPIKE_CM = 50.0


def despike(hs_cm: pd.Series, threshold_cm: float = SPIKE_CM) -> pd.Series:
    """Blank isolated 1-2 day depth outliers (sensor glitches): values more than ``threshold_cm`` away
    from the centred 5-day median. A real snowfall step persists and moves the median with it."""
    med = hs_cm.rolling(5, center=True, min_periods=3).median()
    bad = (hs_cm - med).abs() > threshold_cm
    return hs_cm.mask(bad)


def clean_depth(hs_cm: pd.Series) -> pd.Series:
    """Clip negatives, drop isolated spikes, interpolate short gaps (<= MAX_GAP_DAYS)."""
    s = despike(hs_cm.astype(float).clip(lower=0))
    return s.interpolate(limit=MAX_GAP_DAYS, limit_area="inside")


def rain_on_snow_mm(precip_mm: pd.Series, temp_c: pd.Series, swe_mm: pd.Series, site: Site) -> pd.Series:
    """Liquid water retained in the pack (mm), a state with capacity and daily drainage.

    Rain that falls on an existing pack (SWE > 0, daily mean above ``rain_temp_c``) is held up to
    ``rain_capacity`` x SWE; the retained amount decays by ``rain_drain_per_day`` each day. This adds
    to the load the depth gauge cannot see.
    """
    w = 0.0
    out = []
    for p, t, s in zip(precip_mm.to_numpy(), temp_c.to_numpy(), swe_mm.to_numpy(), strict=True):
        w *= 1.0 - site.rain_drain_per_day
        if pd.notna(s) and s > 0 and pd.notna(p) and pd.notna(t) and t > site.rain_temp_c and p > 0:
            w += p
        cap = site.rain_capacity * s if pd.notna(s) else 0.0
        w = min(w, cap)
        out.append(w)
    return pd.Series(out, index=swe_mm.index)


def compute(df: pd.DataFrame, site: Site, model: str = "sturm", const_rho: float = 300.0,
            rain_on_snow: bool = True) -> Result:
    """Compute the load timeline.

    df: daily frame with hs_cm, precip_mm, temp_c indexed by date (see roofsnow.data.select).
    Returned table columns: hs_cm, swe_mm, rho_kg_m3, rain_mm, ground_kn_m2 (at the station),
    site_kn_m2 (transferred), roof_kn_m2, utilisation, new_snow_cm.
    """
    if not {"hs_cm", "precip_mm", "temp_c"} <= set(df.columns):
        raise ValueError("frame needs hs_cm, precip_mm and temp_c columns")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("index must be unique and sorted")
    t = df.copy()
    t["hs_cm"] = clean_depth(t["hs_cm"])
    swe = density.swe_by_model(model, t["hs_cm"], t["precip_mm"], t["temp_c"], const_rho)
    out = pd.DataFrame(index=t.index)
    out["hs_cm"] = t["hs_cm"]
    out["swe_mm"] = swe
    out["rho_kg_m3"] = (swe * 100.0 / t["hs_cm"]).where(t["hs_cm"] > 0)
    out["new_snow_cm"] = t["hs_cm"].diff().clip(lower=0).fillna(0.0)
    if rain_on_snow:
        out["rain_mm"] = rain_on_snow_mm(t["precip_mm"], t["temp_c"], swe, site)
    else:
        out["rain_mm"] = 0.0
    out["ground_kn_m2"] = (out["swe_mm"] + out["rain_mm"]) * standard.G / 1000.0
    out["site_kn_m2"] = out["ground_kn_m2"] * site.transfer_factor()
    out["roof_kn_m2"] = out["site_kn_m2"] * site.roof_factor()
    design = site.design_load()
    out["utilisation"] = out["roof_kn_m2"] / design
    sk = standard.ground_snow_load_sk(site.altitude, site.z_offset_m)
    return Result(out, design, sk, model, site)
