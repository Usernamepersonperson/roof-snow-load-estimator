"""Data access.

* MeteoSwiss SwissMetNet daily open data (snow depth, precipitation, temperature), CC BY, archive
  reaches back to 2009 for snow depth. https://opendata.swiss / https://data.geo.admin.ch
* SLF IMIS public measurement API (live, 30 min, only the last 1/3/7 days are served).
  Terms of use: https://www.slf.ch/en/services-and-products/slf-data-service

Everything that parses is a pure function so tests run offline.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

SMN_BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
SMN_PARAMS = {"htoautd0": "hs_cm", "rre150d0": "precip_mm", "tre200d0": "temp_c"}
SMN_TIME_FMT = "%d.%m.%Y %H:%M"
IMIS_BASE = "https://measurement-api.slf.ch/public/api/imis"
USER_AGENT = "roof-snow-load-estimator/0.1 (open-source; research)"

# Stations used in the demo: altitude from the MeteoSwiss station metadata (m a.s.l.).
STATIONS: dict[str, tuple[str, float]] = {
    "PIO": ("Piotta", 990.0),
    "ENG": ("Engelberg", 1036.0),
    "ULR": ("Ulrichen", 1346.0),
    "DAV": ("Davos", 1594.0),
    "GRH": ("Grimsel Hospiz", 1980.0),
}

# Winters in the bundled fixture (inclusive dates).
WINTERS: dict[str, tuple[str, str]] = {
    "2017/18": ("2017-11-01", "2018-04-30"),
    "2018/19": ("2018-11-01", "2019-04-30"),
    "2020/21": ("2020-11-01", "2021-04-30"),
}

COLUMNS = ["station", "date", "hs_cm", "precip_mm", "temp_c"]


class DataError(RuntimeError):
    pass


def parse_smn_daily(text: str, station: str | None = None) -> pd.DataFrame:
    """Parse a MeteoSwiss ``ogd-smn_<abbr>_d_*.csv`` payload into tidy daily columns.

    Missing parameters become NaN columns. Raises DataError on malformed input.
    """
    try:
        raw = pd.read_csv(io.StringIO(text), sep=";")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"cannot parse MeteoSwiss CSV: {exc}") from exc
    if "reference_timestamp" not in raw.columns:
        raise DataError("missing reference_timestamp column")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw["reference_timestamp"], format=SMN_TIME_FMT, errors="coerce")
    for src, dst in SMN_PARAMS.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else float("nan")
    if station is None and "station_abbr" in raw.columns and len(raw):
        station = str(raw["station_abbr"].iloc[0])
    out.insert(0, "station", (station or "").upper())
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.normalize()
    return out.sort_values("date").reset_index(drop=True)


def _get(url: str, timeout: float = 60.0) -> str:
    try:
        r = httpx.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise DataError(f"request failed for {url}: {exc}") from exc
    return r.content.decode("latin1")


def fetch_smn(station: str, start: str, end: str, get: Callable[[str], str] = _get) -> pd.DataFrame:
    """Download historical (+ recent if needed) daily data for one station and trim to [start, end]."""
    abbr = station.lower()
    frames = []
    for kind in ("historical", "recent"):
        try:
            text = get(f"{SMN_BASE}/{abbr}/ogd-smn_{abbr}_d_{kind}.csv")
        except DataError:
            if kind == "historical":
                raise
            continue
        frames.append(parse_smn_daily(text, station))
    df = pd.concat(frames).drop_duplicates(subset=["date"]).sort_values("date")
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def load_fixture(path: str | Path | None = None) -> pd.DataFrame:
    """Bundled cached MeteoSwiss winters (5 stations, 3 winters)."""
    if path is None:
        text = resources.files("roofsnow").joinpath("data/smn_winters.csv").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    df = pd.read_csv(io.StringIO(text), parse_dates=["date"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise DataError(f"fixture lacks columns {sorted(missing)}")
    return df


def select(df: pd.DataFrame, station: str, start: str, end: str) -> pd.DataFrame:
    m = (df["station"] == station.upper()) & (df["date"] >= pd.Timestamp(start))
    m &= df["date"] <= pd.Timestamp(end)
    out = df.loc[m, COLUMNS[1:]].set_index("date").sort_index()
    if out.empty:
        raise DataError(f"no data for {station} {start}..{end}")
    return out


def build_fixture(path: str | Path, stations=STATIONS, winters=WINTERS,
                  get: Callable[[str], str] = _get) -> pd.DataFrame:
    """Download the demo winters and write the compact cache CSV."""
    parts = []
    for st in stations:
        for start, end in winters.values():
            parts.append(fetch_smn(st, start, end, get))
    df = pd.concat(parts)[COLUMNS]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.1f")
    return df


# ------------------------------------------------------------------ SLF IMIS (live)

IMIS_FIELDS = {"HS": "hs_cm", "TA_30MIN_MEAN": "temp_c"}


def parse_imis(records: Any) -> pd.DataFrame:
    """Parse IMIS ``/station/<code>/measurements`` JSON into a time-indexed frame (hs_cm, temp_c).

    Records without a valid timestamp are dropped; HS outside 0..1000 cm becomes NaN.
    """
    if not isinstance(records, list):
        raise DataError("unexpected IMIS response (expected a list)")
    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        t = pd.to_datetime(r.get("measure_date"), utc=True, errors="coerce")
        if pd.isna(t):
            continue
        row = {"time": t}
        for src, dst in IMIS_FIELDS.items():
            v = pd.to_numeric(r.get(src), errors="coerce")
            row[dst] = float(v) if pd.notna(v) else float("nan")
        rows.append(row)
    if not rows:
        idx = pd.DatetimeIndex([], tz="UTC", name="time")
        return pd.DataFrame({"hs_cm": [], "temp_c": []}, index=idx)
    df = pd.DataFrame(rows).drop_duplicates("time").set_index("time").sort_index()
    df.loc[(df["hs_cm"] < 0) | (df["hs_cm"] > 1000), "hs_cm"] = float("nan")
    return df


def fetch_imis(station: str, days: int = 7, timeout: float = 30.0,
               get_json: Callable[[str], Any] | None = None) -> pd.DataFrame:
    """Live IMIS measurements; the API only accepts period_in_days of 1, 3 or 7."""
    if days not in (1, 3, 7):
        raise ValueError("IMIS API serves only 1, 3 or 7 days")
    url = f"{IMIS_BASE}/station/{station.upper()}/measurements?period_in_days={days}"
    if get_json is None:
        try:
            r = httpx.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DataError(f"IMIS request failed: {exc}") from exc
    else:
        payload = get_json(url)
    return parse_imis(payload)
