import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from roofsnow import analysis, data, report
from roofsnow.cli import app

# Synthetic text in the MeteoSwiss open-data column layout (values are invented for the test).
SMN_TEXT = (
    "station_abbr;reference_timestamp;tre200d0;rre150d0;htoautd0\n"
    "TST;01.01.2019 00:00;-3.2;5.0;10\n"
    "TST;02.01.2019 00:00;-4.0;0.0;\n"
    "TST;bad date;1.0;1.0;1\n"
    "TST;03.01.2019 00:00;-1.0;12.5;35\n"
)

IMIS_JSON = [
    {"station_code": "TST2", "measure_date": "2026-01-10T10:00:00Z", "HS": 40.0, "TA_30MIN_MEAN": -3.1},
    {"station_code": "TST2", "measure_date": "2026-01-10T10:30:00Z", "HS": 41.5, "TA_30MIN_MEAN": None},
    {"station_code": "TST2", "measure_date": "garbage", "HS": 3.0},
    {"station_code": "TST2", "measure_date": "2026-01-10T11:00:00Z", "HS": 9999},
    "junk",
]


def test_parse_smn_daily_tidy_and_skips_bad_rows():
    df = data.parse_smn_daily(SMN_TEXT)
    assert list(df.columns) == ["station", "date", "hs_cm", "precip_mm", "temp_c"]
    assert len(df) == 3
    assert df["station"].iloc[0] == "TST"
    assert df["hs_cm"].isna().iloc[1]
    assert df["precip_mm"].iloc[2] == 12.5


def test_parse_smn_missing_parameter_columns_become_nan():
    df = data.parse_smn_daily("reference_timestamp;tre200d0\n01.01.2019 00:00;1.5\n", "xyz")
    assert df["hs_cm"].isna().all() and df["station"].iloc[0] == "XYZ"


def test_parse_smn_rejects_malformed():
    with pytest.raises(data.DataError):
        data.parse_smn_daily("a;b\n1;2\n")
    with pytest.raises(data.DataError):
        data.parse_smn_daily("")


def test_fetch_smn_with_injected_getter_trims_range():
    calls = []

    def get(url):
        calls.append(url)
        if url.endswith("recent.csv"):
            raise data.DataError("no recent")
        return SMN_TEXT

    df = data.fetch_smn("tst", "2019-01-02", "2019-01-31", get)
    assert len(df) == 2
    assert calls[0].endswith("/tst/ogd-smn_tst_d_historical.csv")


def test_fetch_smn_historical_failure_propagates():
    def get(url):
        raise data.DataError("down")

    with pytest.raises(data.DataError):
        data.fetch_smn("tst", "2019-01-01", "2019-01-31", get)


def test_parse_imis():
    df = data.parse_imis(IMIS_JSON)
    assert len(df) == 3
    assert df["hs_cm"].iloc[0] == 40.0
    assert pd.isna(df["temp_c"].iloc[1])
    assert pd.isna(df["hs_cm"].iloc[2])      # 9999 cm is rejected
    assert str(df.index.tz) == "UTC"


def test_parse_imis_rejects_non_list_and_handles_empty():
    with pytest.raises(data.DataError):
        data.parse_imis({"detail": "Not Found"})
    assert data.parse_imis([]).empty


def test_fetch_imis_period_validation_and_injected_json():
    with pytest.raises(ValueError):
        data.fetch_imis("DAV2", days=30)
    seen = []
    df = data.fetch_imis("dav2", 3, get_json=lambda u: seen.append(u) or IMIS_JSON)
    assert seen[0].endswith("/station/DAV2/measurements?period_in_days=3")
    assert len(df) == 3


def test_bundled_fixture_shape_and_select():
    fx = data.load_fixture()
    assert set(fx["station"]) == set(data.STATIONS)
    assert len(fx) == 5 * 3 * 181
    df = data.select(fx, "dav", *data.WINTERS["2018/19"])
    assert len(df) == 181
    with pytest.raises(data.DataError):
        data.select(fx, "XXX", "2019-01-01", "2019-02-01")


def test_fixture_reaches_known_snow_values():
    fx = data.load_fixture()
    df = data.select(fx, "GRH", *data.WINTERS["2017/18"])
    assert df["hs_cm"].max() == 430.0      # recorded in the MeteoSwiss file for that winter


def test_run_all_and_summary_table():
    fx = data.load_fixture()
    runs = analysis.run_all(fx, stations=["ENG", "GRH"], winters=["2018/19"])
    tab = analysis.summary_table(runs)
    assert list(tab["station"]) == ["ENG", "GRH"]
    assert (tab["days_ge_50"] >= tab["days_ge_70"]).all()
    assert (tab["days_ge_70"] >= tab["days_ge_90"]).all()
    assert tab.loc[1, "max_util"] > tab.loc[0, "max_util"]


def test_conservative_model_gives_higher_peaks():
    fx = data.load_fixture()
    a = analysis.summary_table(analysis.run_all(fx, ["DAV"], ["2017/18"], "layered"))
    b = analysis.summary_table(analysis.run_all(fx, ["DAV"], ["2017/18"], "sturm"))
    assert b.loc[0, "max_util"] > a.loc[0, "max_util"]


def test_sensitivity_and_envelope():
    fx = data.load_fixture()
    s = analysis.sensitivity(fx, "ULR", "2017/18")
    assert len(s) == 6
    const = s[s.model.str.startswith("constant")].sort_values("model")
    assert const["max_util"].is_monotonic_increasing
    df = data.select(fx, "ULR", *data.WINTERS["2017/18"])
    env = analysis.envelope(df, analysis.site_for("ULR"))
    assert (env["load_max"] >= env["load_min"]).all()


def test_crosscheck_has_expected_columns():
    fx = data.load_fixture()
    c = analysis.crosscheck(fx, ["DAV"], ["2018/19"])
    assert list(c.columns) == ["station", "winter", "days", "bias_pct", "rmse_mm", "corr"]
    assert c["corr"].iloc[0] > 0.8


def test_fusion_simulation_is_deterministic_and_improves_on_model():
    fx = data.load_fixture()
    a = report.fusion_simulation(fx)
    b = report.fusion_simulation(fx)
    assert a == b
    assert a["rmse_fused_kn_m2"] < a["rmse_model_kn_m2"]


def test_cli_sk_and_errors():
    r = CliRunner().invoke(app, ["sk", "1000"])
    assert r.exit_code == 0 and "3.67" in r.output
    bad = CliRunner().invoke(app, ["sk", "1000", "--z", "123"])
    assert bad.exit_code == 2
    warn = CliRunner().invoke(app, ["sk", "2500"])
    assert "2000 m" in warn.output


def test_cli_analyze_json():
    r = CliRunner().invoke(app, ["analyze", "ENG", "--winter", "2018/19"])
    assert r.exit_code == 0
    row = json.loads(r.output)
    assert row["station"] == "ENG" and row["max_hs_cm"] == 85.0


def test_cli_analyze_unknown_station():
    r = CliRunner().invoke(app, ["analyze", "NOPE"])
    assert r.exit_code == 2


def test_cli_demo_without_figures(tmp_path):
    r = CliRunner().invoke(app, ["demo", "--no-figures", "--out", str(tmp_path)])
    assert r.exit_code == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["headline"]["runs"] == 15
    assert (tmp_path / "results.md").exists()
