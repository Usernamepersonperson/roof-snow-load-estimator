"""Run the model over stations and winters and summarise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from roofsnow import alerts, data, density, model

DEFAULT_MODEL = "layered"
ENVELOPE_MODELS = ("sturm", "layered", "mass_balance")


@dataclass(frozen=True)
class Run:
    station: str
    winter: str
    result: model.Result


def site_for(station: str, **kw) -> model.Site:
    return model.Site(station_alt_m=data.STATIONS[station][1], **kw)


def run_all(fx: pd.DataFrame, stations=None, winters=None, model_name: str = DEFAULT_MODEL,
            site_kw: dict | None = None, const_rho: float = 300.0) -> list[Run]:
    runs = []
    for st in stations or data.STATIONS:
        for w in winters or data.WINTERS:
            start, end = data.WINTERS[w]
            df = data.select(fx, st, start, end)
            res = model.compute(df, site_for(st, **(site_kw or {})), model_name, const_rho)
            runs.append(Run(st, w, res))
    return runs


def summarise(run: Run, thresholds=alerts.DEFAULT_THRESHOLDS, hysteresis: float = 0.05) -> dict:
    t = run.result.table
    u = t["utilisation"]
    tr = alerts.run_alerts(u, thresholds, hysteresis)
    peak_day = u.idxmax() if u.notna().any() else pd.NaT
    row = {
        "station": run.station,
        "winter": run.winter,
        "altitude_m": run.result.site.altitude,
        "design_kn_m2": round(run.result.design_kn_m2, 2),
        "max_hs_cm": float(t["hs_cm"].max()),
        "max_load_kn_m2": round(float(t["roof_kn_m2"].max()), 2),
        "max_util": round(float(u.max()), 3),
        "peak_day": peak_day.strftime("%Y-%m-%d") if pd.notna(peak_day) else None,
        "valid_days": int(u.notna().sum()),
        "alert_events": len(tr.events),
    }
    for th, n in alerts.days_above(u, thresholds).items():
        row[f"days_ge_{int(th * 100)}"] = n
    return row


def summary_table(runs: list[Run], **kw) -> pd.DataFrame:
    return pd.DataFrame([summarise(r, **kw) for r in runs])


def sensitivity(fx: pd.DataFrame, station: str, winter: str, site_kw: dict | None = None) -> pd.DataFrame:
    """Peak load and days >= 70 % for each density model (and constant-density bounds)."""
    start, end = data.WINTERS[winter]
    df = data.select(fx, station, start, end)
    site = site_for(station, **(site_kw or {}))
    rows = []
    variants = [("sturm", 300.0), ("layered", 300.0), ("mass_balance", 300.0), ("constant", 200.0),
                ("constant", 300.0), ("constant", 400.0)]
    for name, rho in variants:
        res = model.compute(df, site, name, rho)
        t = res.table
        label = f"constant {rho:.0f}" if name == "constant" else name
        rows.append({"station": station, "winter": winter, "model": label,
                     "max_swe_mm": round(float(t["swe_mm"].max()), 0),
                     "max_load_kn_m2": round(float(t["roof_kn_m2"].max()), 2),
                     "max_util": round(float(t["utilisation"].max()), 3),
                     "days_ge_70": int((t["utilisation"] >= 0.7).sum()),
                     "mean_rho_kg_m3": round(float(t["rho_kg_m3"].mean()), 0)})
    return pd.DataFrame(rows)


def crosscheck(fx: pd.DataFrame, stations=None, winters=None) -> pd.DataFrame:
    """Agreement between the layered settlement SWE (depth + temperature) and the mass-balance SWE
    (precipitation + temperature, anchored to depth only by density bounds).

    This is a consistency check between two models, NOT a validation against measured SWE.
    """
    rows = []
    for st in stations or data.STATIONS:
        for w in winters or data.WINTERS:
            start, end = data.WINTERS[w]
            df = data.select(fx, st, start, end)
            a = model.compute(df, site_for(st), "layered").table["swe_mm"]
            b = model.compute(df, site_for(st), "mass_balance").table["swe_mm"]
            m = a.notna() & b.notna() & (a > 20)
            if m.sum() < 5:
                continue
            rows.append({"station": st, "winter": w, "days": int(m.sum()),
                         "bias_pct": round(float((b[m] - a[m]).sum() / a[m].sum() * 100), 1),
                         "rmse_mm": round(float(np.sqrt(((b[m] - a[m]) ** 2).mean())), 0),
                         "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)})
    return pd.DataFrame(rows)


def density_summary(runs: list[Run]) -> dict:
    rho = pd.concat([r.result.table["rho_kg_m3"] for r in runs]).dropna()
    return {"min": float(rho.min()), "median": float(rho.median()), "max": float(rho.max())}


def reference_density_check() -> list[dict]:
    """Sturm model density at fixed depth/day versus values in the published class table (sanity only)."""
    out = []
    for hs, doy in [(0.0, 0), (50.0, 90), (100.0, 120), (200.0, 180)]:
        out.append({"hs_cm": hs, "doy": doy, "rho": round(density.sturm_density(hs, doy), 0)})
    return out


def envelope(df: pd.DataFrame, site: model.Site, models=ENVELOPE_MODELS) -> pd.DataFrame:
    """Per-day min / max roof load (kN/m2) and utilisation across the density models."""
    loads = pd.DataFrame({m: model.compute(df, site, m).table["roof_kn_m2"] for m in models})
    design = site.design_load()
    return pd.DataFrame({"load_min": loads.min(axis=1), "load_max": loads.max(axis=1),
                         "util_min": loads.min(axis=1) / design, "util_max": loads.max(axis=1) / design})
