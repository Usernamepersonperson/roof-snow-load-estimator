"""Command line: ``python -m roofsnow <command>``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from roofsnow import alerts, analysis, data, density, model, report, sensor, standard

app = typer.Typer(add_completion=False, help="Roof snow-load monitor from open Swiss snow data.")


def _fail(msg: str) -> typer.Exit:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=2)


@app.command()
def sk(altitude: float = typer.Argument(..., help="site altitude, m a.s.l."),
       z: float = typer.Option(0.0, help="location offset Z from the standard's map (user input)"),
       pitch: float = typer.Option(0.0, help="roof pitch, degrees"),
       exposure: str = typer.Option("normal", help="windswept | normal | sheltered")):
    """Characteristic ground and roof design load from the public altitude formula."""
    try:
        s = standard.ground_snow_load_sk(altitude, z)
        r = standard.roof_design_load(altitude, z, pitch, exposure)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"s_k = {s:.2f} kN/m2 (ground), roof design = {r:.2f} kN/m2")
    if not standard.altitude_in_validity(altitude):
        typer.secho("warning: formula is documented only up to 2000 m", fg=typer.colors.YELLOW)


@app.command()
def fetch(station: str, start: str, end: str, out: Path = typer.Option(None, help="CSV path")):
    """Download MeteoSwiss daily open data for one station."""
    try:
        df = data.fetch_smn(station, start, end)
    except data.DataError as exc:
        raise _fail(str(exc)) from exc
    if out:
        df.to_csv(out, index=False)
    typer.echo(f"{len(df)} days, snow depth present on {int(df['hs_cm'].notna().sum())}")


@app.command()
def analyze(station: str = typer.Argument(..., help="PIO, ENG, ULR, DAV or GRH (bundled data)"),
            winter: str = "2017/18",
            model_name: str = typer.Option("layered", "--model", help=" | ".join(density.MODELS)),
            site_alt: float = typer.Option(None, help="roof site altitude; moves station data to it"),
            z: float = typer.Option(0.0), pitch: float = typer.Option(0.0),
            exposure: str = typer.Option("normal"),
            design: float = typer.Option(None, help="design load kN/m2 (default: public formula)")):
    """Load timeline summary for a bundled station and winter."""
    try:
        fx = data.load_fixture()
        start, end = data.WINTERS[winter]
        df = data.select(fx, station, start, end)
        site = model.Site(station_alt_m=data.STATIONS[station.upper()][1], site_alt_m=site_alt,
                          z_offset_m=z, pitch_deg=pitch, exposure=exposure, design_load_kn_m2=design)
        res = model.compute(df, site, model_name)
    except (KeyError, ValueError, data.DataError) as exc:
        raise _fail(f"{exc}") from exc
    row = analysis.summarise(analysis.Run(station.upper(), winter, res))
    typer.echo(json.dumps(row, indent=1))


@app.command()
def live(station: str = typer.Argument("DAV2", help="IMIS code, e.g. DAV2"),
         design: float = typer.Option(None, help="design load kN/m2"),
         altitude: float = typer.Option(None, help="station altitude for the default design load")):
    """Current load from the SLF IMIS live API (last 24 h, 30 min values)."""
    try:
        obs = data.fetch_imis(station, days=1)
        if obs.empty:
            raise data.DataError("no observations returned")
        if design is None:
            if altitude is None:
                raise data.DataError("pass --design or --altitude")
            design = standard.roof_design_load(altitude)
        hs = obs["hs_cm"].dropna()
        if hs.empty:
            raise data.DataError("no valid snow depth in the response")
        last = hs.index[-1]
        rho = density.sturm_density(float(hs.iloc[-1]), density.sturm_doy(last.tz_localize(None)))
        load = standard.swe_mm_to_kn_m2(float(hs.iloc[-1]) * rho / 100.0)
    except (data.DataError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    util = load / design
    level = alerts.level_for(util)
    typer.echo(json.dumps({"station": station.upper(), "time": last.isoformat(), "hs_cm": float(hs.iloc[-1]),
                           "load_kn_m2": round(load, 3), "design_kn_m2": design,
                           "utilisation": round(util, 4), "level": alerts.LEVEL_NAMES[level],
                           "action": alerts.ACTIONS[level],
                           "note": "Sturm regression, no rain-on-snow term; conservative for fresh snow"}))


@app.command()
def listen(host: str = "localhost", topic: str = "roof/load", port: int = 1883,
           design: float = typer.Option(..., help="design load kN/m2")):
    """Subscribe to MQTT JSON from an on-roof sensor and print fused alerts (needs the mqtt extra)."""
    try:
        sensor.listen(host, topic, sensor.Monitor(design), port)
    except sensor.SensorError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def demo(out: Path = typer.Option(Path("docs"), help="output directory"),
         figures: bool = typer.Option(True, "--figures/--no-figures")):
    """Offline demo on the bundled MeteoSwiss winters: results tables and figures."""
    fx = data.load_fixture()
    res = report.build(fx)
    report.write(res, out)
    summ = pd.DataFrame(res["summary"])
    p = summ[summ.winter == res["primary_winter"]]
    typer.echo(p[["station", "altitude_m", "design_kn_m2", "max_load_kn_m2", "max_util", "peak_day",
                  "days_ge_50", "days_ge_70", "days_ge_90"]].to_string(index=False))
    typer.echo(f"headline: {json.dumps(res['headline'])}")
    if figures:
        from roofsnow import figures as fig
        out.mkdir(parents=True, exist_ok=True)
        fig.main_figure(fx, out / "figure.png", res["primary_winter"])
        fig.post_figure(fx, "ULR", "2017/18", out / "post-figure.png")
        typer.echo(f"wrote {out / 'figure.png'} and {out / 'post-figure.png'}")


if __name__ == "__main__":  # pragma: no cover
    app()
