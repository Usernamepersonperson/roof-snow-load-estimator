"""Result figures: docs/figure.png (1600x900) and docs/post-figure.png (960x700, dark)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from roofsnow import alerts, analysis, data, model  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
BAND_COLORS = (GREEN, AMBER, RED)
SOURCE = "Source: MeteoSwiss (SwissMetNet daily open data, CC BY)"

matplotlib.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def _style(ax, size=11):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=size)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _bands(ax, design, thresholds=alerts.DEFAULT_THRESHOLDS, label=True, size=11):
    """Shaded alert zones in kN/m2 on a load axis, plus the design line."""
    edges = (*thresholds, None)
    lows = (0.0, *thresholds)
    ymax = ax.get_ylim()[1]
    colors = (GREY, GREEN, AMBER, RED)
    for lo, hi, c in zip(lows, edges, colors, strict=True):
        top = ymax if hi is None else hi * design
        ax.axhspan(lo * design, top, color=c, alpha=0.10 if lo == 0 else 0.16, lw=0)
    for th, c in zip(thresholds, BAND_COLORS, strict=True):
        ax.axhline(th * design, color=c, lw=1.0, ls=":", alpha=0.9)
        if label:
            ax.text(1.005, th * design, f"{int(th * 100)} %", color=c, fontsize=size,
                    transform=ax.get_yaxis_transform(), va="center")
    ax.axhline(design, color=TEXT, lw=1.6, ls="--")


def post_figure(fx: pd.DataFrame, station: str, winter: str, path: str | Path,
                site_kw: dict | None = None) -> dict:
    start, end = data.WINTERS[winter]
    df = data.select(fx, station, start, end)
    site = analysis.site_for(station, **(site_kw or {}))
    res = model.compute(df, site, analysis.DEFAULT_MODEL)
    env = analysis.envelope(df, site)
    t = res.table
    name = data.STATIONS[station][0]
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.085, 0.13, 0.84, 0.70])
    _style(ax, 12)
    ymax = max(res.design_kn_m2 * 1.15, float(env["load_max"].max()) * 1.1)
    ax.set_ylim(0, ymax)
    _bands(ax, res.design_kn_m2, size=12)
    ax.fill_between(t.index, env["load_min"], env["load_max"], color=ACCENT, alpha=0.22, lw=0,
                    label="density-model range")
    ax.plot(t.index, t["roof_kn_m2"], color=ACCENT, lw=2.6, label="estimated roof load")
    ax.text(t.index[-3], res.design_kn_m2 * 1.02, f"design load {res.design_kn_m2:.1f} kN/m²",
            color=TEXT, fontsize=12, va="bottom", ha="right")
    ax.set_ylabel("roof snow load, kN/m²", color=MUTED, fontsize=12)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=12, facecolor=BG,
                    edgecolor=GRID, labelcolor=TEXT)
    leg.get_frame().set_alpha(0.9)
    summ = analysis.summarise(analysis.Run(station, winter, res))
    fig.text(0.04, 0.94, "Roof load vs design load over one winter", color=TEXT, fontsize=14.5,
             fontweight="bold")
    fig.text(0.04, 0.895, f"{name} ({site.altitude:.0f} m), winter {winter}, flat roof. "
             f"Alert bands at 50 / 70 / 90 % of design.", color=MUTED, fontsize=11.5)
    fig.text(0.04, 0.04, f"{SOURCE}. Model: snow depth to load, roofsnow.", color=MUTED, fontsize=11)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return summ


def main_figure(fx: pd.DataFrame, path: str | Path, winter: str = "2017/18",
                site_kw: dict | None = None) -> None:
    start, end = data.WINTERS[winter]
    stations = list(data.STATIONS)
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(len(stations), 2, width_ratios=[1.7, 1], left=0.06, right=0.97, top=0.86,
                          bottom=0.09, hspace=0.18, wspace=0.2)
    peaks: dict[str, dict[str, float]] = {}
    first = None
    for i, st in enumerate(stations):
        ax = fig.add_subplot(gs[i, 0], sharex=first)
        first = first or ax
        _style(ax, 11)
        df = data.select(fx, st, start, end)
        site = analysis.site_for(st, **(site_kw or {}))
        res = model.compute(df, site, analysis.DEFAULT_MODEL)
        env = analysis.envelope(df, site)
        u = res.table["utilisation"]
        ax.set_ylim(0, 1.3)
        for lo, hi, c in zip((0, *alerts.DEFAULT_THRESHOLDS), (*alerts.DEFAULT_THRESHOLDS, 1.3),
                             (GREY, GREEN, AMBER, RED), strict=True):
            ax.axhspan(lo, hi, color=c, alpha=0.10 if lo == 0 else 0.15, lw=0)
        ax.axhline(1.0, color=TEXT, lw=1.2, ls="--")
        ax.fill_between(u.index, env["util_min"], env["util_max"], color=ACCENT, alpha=0.22, lw=0)
        ax.plot(u.index, u, color=ACCENT, lw=1.8)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "50 %", "100 %"])
        ax.text(0.008, 0.9, f"{data.STATIONS[st][0]}  {site.altitude:.0f} m  (design "
                f"{res.design_kn_m2:.1f} kN/m²)", color=TEXT, fontsize=11.5, fontweight="bold",
                transform=ax.transAxes, va="center")
        if i < len(stations) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        peaks[st] = {m: float(model.compute(df, site, m).table["utilisation"].max())
                     for m in ("constant", "layered", "mass_balance", "sturm")}
    axb = fig.add_subplot(gs[:, 1])
    _style(axb, 11)
    axb.grid(True, axis="y", color=GRID)
    axb.grid(False, axis="x")
    models = [("mass_balance", "mass balance", VIOLET), ("constant", "constant 300", GREY),
              ("layered", "layered (default)", ACCENT), ("sturm", "Sturm regression", AMBER)]
    w = 0.2
    for k, (m, lab, c) in enumerate(models):
        vals = [peaks[s][m] * 100 for s in stations]
        axb.barh([j + (k - 1.5) * w for j in range(len(stations))], vals, height=w, color=c, label=lab)
    axb.set_yticks(range(len(stations)))
    axb.set_yticklabels([f"{data.STATIONS[s][0]}" for s in stations], color=TEXT, fontsize=11)
    axb.invert_yaxis()
    axb.axvline(100, color=TEXT, ls="--", lw=1.2)
    for th, c in zip(alerts.DEFAULT_THRESHOLDS, BAND_COLORS, strict=True):
        axb.axvline(th * 100, color=c, ls=":", lw=1)
    axb.set_xlabel("peak utilisation of design load, %", color=MUTED, fontsize=11)
    axb.set_xlim(0, 270)
    axb.legend(loc="lower right", fontsize=10.5, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    axb.set_title("Peak by density model", color=TEXT, fontsize=12.5, loc="left", fontweight="bold")
    fig.text(0.06, 0.94, f"Roof snow load vs design load, five stations, winter {winter}", color=TEXT,
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.905, "Utilisation = estimated flat-roof load / s_k from the public altitude formula "
             "(Z = 0). Blue band = density-model range. Bands: 50 / 70 / 90 / 100 %.",
             color=MUTED, fontsize=11.5)
    fig.text(0.06, 0.03, f"{SOURCE}, {start} to {end}. Station altitudes, not roof sites.",
             color=MUTED, fontsize=11)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
