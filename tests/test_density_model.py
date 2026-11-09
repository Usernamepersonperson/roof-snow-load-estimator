import math

import pandas as pd
import pytest

from roofsnow import density, model


def series(vals, start="2019-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals)), dtype=float)


def test_doy_from_oct1():
    assert density.doy_from_oct1(pd.Timestamp("2018-10-01")) == 0
    assert density.doy_from_oct1(pd.Timestamp("2019-01-01")) == 92
    assert density.doy_from_oct1(pd.Timestamp("2018-12-31")) == 91


def test_sturm_doy_matches_paper_convention():
    # Sturm et al. (2010): DOY runs from -92 (1 October) to +181 (30 June), no zero;
    # 1 February is DOY 32 and 15 November is DOY -47.
    assert density.sturm_doy(pd.Timestamp("2018-10-01")) == -92
    assert density.sturm_doy(pd.Timestamp("2018-11-15")) == -47
    assert density.sturm_doy(pd.Timestamp("2018-12-31")) == -1
    assert density.sturm_doy(pd.Timestamp("2019-01-01")) == 1
    assert density.sturm_doy(pd.Timestamp("2019-02-01")) == 32
    assert density.sturm_doy(pd.Timestamp("2019-06-30")) == 181
    assert density.sturm_doy(pd.Timestamp("2019-08-15")) == 181  # clamped outside the season


def test_sturm_density_early_season_below_rho0():
    # A negative DOY lowers density below rho_0, as the paper's equation implies.
    assert density.sturm_density(50, -47) < density.STURM_ALPINE["rho_0"]


def test_sturm_density_hand_value():
    # HS 100 cm, DOY 120: 223.7 + 373.8 * (1 - exp(-0.12 - 0.456))
    expect = 223.7 + 373.8 * (1 - math.exp(-0.12 - 0.456))
    assert density.sturm_density(100, 120) == pytest.approx(expect, abs=0.05)


def test_sturm_density_bounds_and_monotone():
    assert density.sturm_density(0, 0) == pytest.approx(223.7)
    assert density.sturm_density(500, 300) <= density.RHO_MAX
    assert density.sturm_density(100, 60) < density.sturm_density(100, 160)
    assert math.isnan(density.sturm_density(-5, 10))
    assert math.isnan(density.sturm_density(float("nan"), 10))


def test_constant_swe_and_bounds():
    hs = series([0, 10, 100])
    swe = density.swe_constant(hs, 300)
    assert list(swe) == [0.0, 30.0, 300.0]  # 1 cm at 100 kg/m3 = 1 mm
    with pytest.raises(ValueError):
        density.swe_constant(hs, 700)


def test_snow_fraction():
    assert density.snow_fraction(-3) == 1.0
    assert density.snow_fraction(5) == 0.0
    assert density.snow_fraction(1.0) == pytest.approx(0.5)
    assert density.snow_fraction(float("nan")) == 0.0


def test_fresh_snow_density_hedstrom_pomeroy():
    assert density.fresh_snow_density(0) == pytest.approx(67.92 + 51.25)
    assert density.fresh_snow_density(-10) < density.fresh_snow_density(-1)
    assert density.fresh_snow_density(5) == density.fresh_snow_density(0)


def test_layered_fresh_snow_is_light_and_ages():
    hs = series([0, 30] + [30] * 59)
    temp = series([-5] * 61)
    swe = density.swe_layered(hs, temp)
    rho0 = swe.iloc[1] * 100 / 30
    rho_end = swe.iloc[-1] * 100 / 30
    assert 60 < rho0 < 120
    assert rho_end > rho0 + 100
    assert rho_end <= density.RHO_OLD


def test_layered_conserves_mass_without_depth_change():
    hs = series([0, 40, 40, 40])
    swe = density.swe_layered(hs, series([-4] * 4))
    # settlement raises density; layer thickness is recomputed to the observed depth, mass is kept
    assert swe.iloc[3] >= swe.iloc[1]


def test_layered_melt_removes_top_and_zero_depth_resets():
    hs = series([0, 50, 30, 0, 10])
    swe = density.swe_layered(hs, series([-3, -3, 2, 3, -3]))
    assert swe.iloc[2] < swe.iloc[1]
    assert swe.iloc[3] == 0.0
    assert swe.iloc[4] > 0.0


def test_layered_nan_depth_carries_forward():
    hs = series([0, 40, float("nan"), 40])
    swe = density.swe_layered(hs, series([-4] * 4))
    assert swe.iloc[2] == pytest.approx(swe.iloc[1] * 1.0, rel=0.05)


def test_mass_balance_accumulates_snow_and_melts():
    hs = series([50] * 6)
    precip = series([0, 10, 0, 0, 0, 0])
    temp = series([-5, -5, 3, 3, 3, 3])
    swe = density.swe_mass_balance(hs, precip, temp)
    assert swe.iloc[1] >= swe.iloc[0]
    assert swe.iloc[-1] <= swe.iloc[1]
    assert (swe >= 50 * density.RHO_MIN / 100 - 1e-9).all()


def test_mass_balance_resets_at_zero_depth():
    swe = density.swe_mass_balance(series([20, 0]), series([5, 0]), series([-3, -3]))
    assert swe.iloc[1] == 0.0


def test_swe_by_model_dispatch_and_errors():
    hs = series([10, 20])
    assert len(density.swe_by_model("sturm", hs)) == 2
    with pytest.raises(ValueError):
        density.swe_by_model("mass_balance", hs)
    with pytest.raises(ValueError):
        density.swe_by_model("layered", hs)
    with pytest.raises(ValueError):
        density.swe_by_model("magic", hs)


# ------------------------------------------------------------------ model

def frame(hs, precip=None, temp=None, start="2019-01-01"):
    n = len(hs)
    idx = pd.date_range(start, periods=n)
    return pd.DataFrame({"hs_cm": hs, "precip_mm": precip or [0.0] * n, "temp_c": temp or [-5.0] * n},
                        index=idx)


def test_compute_constant_density_numbers():
    site = model.Site(station_alt_m=1000)
    res = model.compute(frame([0, 100]), site, "constant", 300.0)
    t = res.table
    assert t["swe_mm"].iloc[1] == pytest.approx(300.0)
    assert t["ground_kn_m2"].iloc[1] == pytest.approx(300 * 9.81 / 1000)
    assert t["roof_kn_m2"].iloc[1] == pytest.approx(0.8 * 2.943)   # flat roof mu1 0.8, Ce 1, Ct 1
    assert res.design_kn_m2 == pytest.approx(0.8 * res.sk_kn_m2)
    assert t["utilisation"].iloc[1] == pytest.approx(2.3544 / res.design_kn_m2)


def test_utilisation_independent_of_roof_factors_without_override():
    f = frame([0, 80, 120])
    a = model.compute(f, model.Site(1200, exposure="normal"), "constant").table["utilisation"]
    b = model.compute(f, model.Site(1200, exposure="sheltered", pitch_deg=20), "constant").table["utilisation"]
    assert a.iloc[2] == pytest.approx(b.iloc[2])


def test_design_override_changes_utilisation():
    f = frame([0, 100])
    a = model.compute(f, model.Site(1000), "constant").table["utilisation"].iloc[1]
    b = model.compute(f, model.Site(1000, design_load_kn_m2=1.0), "constant").table["utilisation"].iloc[1]
    assert b > a
    with pytest.raises(ValueError):
        model.compute(f, model.Site(1000, design_load_kn_m2=-1), "constant")


def test_steep_roof_needs_explicit_design_load():
    with pytest.raises(ValueError):
        model.compute(frame([0, 50]), model.Site(1000, pitch_deg=65), "constant")


def test_transfer_factor_and_limit():
    assert model.Site(1000, site_alt_m=1200).transfer_factor() == pytest.approx(1.12)
    assert model.Site(1000, site_alt_m=900).transfer_factor() == pytest.approx(0.94)
    with pytest.raises(ValueError):
        model.Site(1000, site_alt_m=1800).transfer_factor()


def test_transfer_scales_load_up_with_altitude():
    f = frame([0, 100])
    same = model.compute(f, model.Site(1000), "constant").table["site_kn_m2"].iloc[1]
    higher = model.compute(f, model.Site(1000, site_alt_m=1300), "constant").table["site_kn_m2"].iloc[1]
    assert higher == pytest.approx(same * 1.18)


def test_rain_on_snow_adds_retained_water_capped_by_capacity():
    hs = [100, 100, 100, 100]
    f = frame(hs, precip=[0, 30, 0, 0], temp=[-2, 4, 4, 4])
    t = model.compute(f, model.Site(1000), "constant", 300.0).table
    assert t["rain_mm"].iloc[1] == pytest.approx(0.05 * 300)   # 30 mm of rain but capacity is 15 mm
    assert t["rain_mm"].iloc[3] < t["rain_mm"].iloc[1]         # drains
    off = model.compute(f, model.Site(1000), "constant", 300.0, rain_on_snow=False).table
    assert (off["rain_mm"] == 0).all()
    assert t["ground_kn_m2"].iloc[1] > off["ground_kn_m2"].iloc[1]


def test_no_rain_term_without_snow_or_when_cold():
    f = frame([0, 0, 50], precip=[20, 20, 20], temp=[5, 5, -5])
    t = model.compute(f, model.Site(1000), "constant").table
    assert (t["rain_mm"] == 0).all()


def test_despike_removes_isolated_outlier_but_keeps_step():
    hs = pd.Series([10, 12, 217, 14, 15, 16, 17.0])
    out = model.despike(hs)
    assert out.isna().iloc[2]
    step = pd.Series([0, 0, 0, 80, 85, 90, 92.0])
    assert not model.despike(step).isna().any()


def test_gap_interpolation_limit():
    hs = pd.Series([10, float("nan"), float("nan"), 16, float("nan"), float("nan"), float("nan"), float("nan"), 30.0])
    out = model.clean_depth(hs)
    assert out.iloc[1:3].notna().all()
    assert out.iloc[4:8].isna().sum() >= 1


def test_compute_validates_input():
    with pytest.raises(ValueError):
        model.compute(pd.DataFrame({"hs_cm": [1.0]}), model.Site(1000))
    f = frame([1, 2])
    with pytest.raises(ValueError):
        model.compute(f.iloc[::-1], model.Site(1000))
