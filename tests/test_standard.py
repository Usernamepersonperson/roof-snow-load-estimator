import math

import pytest

from roofsnow import standard


def test_sk_minimum_applies_at_low_altitude():
    assert standard.ground_snow_load_sk(0) == pytest.approx(0.9)
    assert standard.ground_snow_load_sk(300) == pytest.approx(0.9)


def test_sk_known_values():
    # (1 + (1000/350)^2) * 0.4 = 3.6653...
    assert standard.ground_snow_load_sk(1000) == pytest.approx(0.4 * (1 + (1000 / 350) ** 2))
    assert standard.ground_snow_load_sk(1594) == pytest.approx(8.70, abs=0.01)
    assert standard.ground_snow_load_sk(2000) == pytest.approx(0.4 * (1 + (2000 / 350) ** 2))


def test_sk_z_offset_shifts_reference_height():
    assert standard.ground_snow_load_sk(1000, 200) == pytest.approx(standard.ground_snow_load_sk(1200))
    assert standard.ground_snow_load_sk(1000, -200) < standard.ground_snow_load_sk(1000)


def test_sk_rejects_bad_z_and_nonfinite():
    with pytest.raises(ValueError):
        standard.ground_snow_load_sk(1000, 100)
    with pytest.raises(ValueError):
        standard.ground_snow_load_sk(math.nan)
    with pytest.raises(ValueError):
        standard.ground_snow_load_sk(math.inf)


def test_sk_monotonic_in_altitude():
    vals = [standard.ground_snow_load_sk(h) for h in range(400, 2001, 100)]
    assert vals == sorted(vals)


def test_validity_limit():
    assert standard.altitude_in_validity(2000)
    assert not standard.altitude_in_validity(2001)


@pytest.mark.parametrize("pitch,mu", [(0, 0.8), (30, 0.8), (45, 0.4), (60, 0.0), (80, 0.0)])
def test_shape_coefficient(pitch, mu):
    assert standard.shape_coefficient(pitch) == pytest.approx(mu)


def test_shape_coefficient_rejects_out_of_range():
    for bad in (-1, 91):
        with pytest.raises(ValueError):
            standard.shape_coefficient(bad)


def test_exposure_coefficient():
    assert standard.exposure_coefficient("Windswept") == 0.8
    assert standard.exposure_coefficient("sheltered") == 1.2
    with pytest.raises(ValueError):
        standard.exposure_coefficient("tropical")


def test_roof_design_load_combines_factors():
    base = standard.ground_snow_load_sk(1200)
    assert standard.roof_design_load(1200) == pytest.approx(0.8 * base)
    assert standard.roof_design_load(1200, exposure="sheltered") == pytest.approx(0.8 * 1.2 * base)
    with pytest.raises(ValueError):
        standard.roof_design_load(1200, ct=0)


def test_swe_to_load_roundtrip():
    assert standard.swe_mm_to_kn_m2(100) == pytest.approx(0.981)
    assert standard.kn_m2_to_swe_mm(standard.swe_mm_to_kn_m2(250)) == pytest.approx(250)
