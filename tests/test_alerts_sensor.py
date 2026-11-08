import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from roofsnow import alerts, sensor


def test_level_for_thresholds():
    assert alerts.level_for(0.1) == 0
    assert alerts.level_for(0.5) == 1
    assert alerts.level_for(0.7) == 2
    assert alerts.level_for(0.95) == 3
    assert alerts.level_for(float("nan")) == 0


def test_tracker_rises_immediately_and_falls_with_hysteresis():
    tr = alerts.AlertTracker()
    assert tr.update(0.55) == 1
    assert tr.update(0.48) == 1       # inside the 5-point band: stays
    assert tr.update(0.44) == 0
    assert tr.update(0.75) == 2
    assert tr.update(0.66) == 2       # 0.7 - 0.05 = 0.65 not yet crossed
    assert tr.update(0.64) == 1
    assert len(tr.events) == 4


def test_tracker_can_skip_levels_up_and_down():
    tr = alerts.AlertTracker()
    assert tr.update(0.95) == 3
    assert tr.update(0.1) == 0
    assert [(e["from"], e["to"]) for e in tr.events] == [(0, 3), (3, 0)]


def test_tracker_nan_keeps_level_and_event_has_action():
    tr = alerts.AlertTracker()
    tr.update(0.8, "t0")
    assert tr.update(float("nan")) == 2
    assert tr.events[0]["action"].startswith("Warning")


def test_tracker_validates_config():
    with pytest.raises(ValueError):
        alerts.AlertTracker(thresholds=(0.7, 0.5))
    with pytest.raises(ValueError):
        alerts.AlertTracker(thresholds=(0.5, 0.7), hysteresis=0.3)
    with pytest.raises(ValueError):
        alerts.AlertTracker(thresholds=(0.0, 0.5))


def test_hysteresis_suppresses_chatter():
    u = pd.Series([0.49, 0.51, 0.49, 0.51, 0.49, 0.51] * 3)
    with_h = alerts.run_alerts(u, hysteresis=0.05)
    without = alerts.run_alerts(u, hysteresis=0.0)
    assert len(with_h.events) == 1
    assert len(without.events) > 10


def test_days_above_and_rising_count():
    u = pd.Series([0.1, 0.6, 0.8, 0.95, 0.5, 0.2, float("nan")])
    d = alerts.days_above(u)
    assert d == {0.5: 4, 0.7: 2, 0.9: 1}
    tr = alerts.run_alerts(u)
    assert alerts.count_rising_alerts(tr, 1) == 1
    assert alerts.count_rising_alerts(tr, 3) == 1


# ------------------------------------------------------------------ sensor

NOW = datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc)


def test_parse_load_payloads():
    r = sensor.parse_payload(json.dumps({"load_kn_m2": 2.4, "ts": "2026-01-20T11:00:00Z"}))
    assert r.load_kn_m2 == pytest.approx(2.4)
    assert r.time == pd.Timestamp("2026-01-20T11:00:00Z")
    kg = sensor.parse_payload(b'{"load_kg_m2": 200}', NOW)
    assert kg.load_kn_m2 == pytest.approx(1.962)


def test_parse_depth_payload_uses_density_model():
    r = sensor.parse_payload('{"depth_cm": 100, "ts": "2026-01-20T12:00:00Z"}')
    assert 2.5 < r.load_kn_m2 < 4.5


@pytest.mark.parametrize("raw", [
    "not json", "[1,2]", "{}", '{"load_kn_m2": "x"}', '{"load_kn_m2": -1}', '{"load_kn_m2": 999}',
    '{"load_kn_m2": true}', '{"load_kn_m2": 1, "ts": "garbage"}', '{"load_kn_m2": 1, "sigma_rel": 0}',
    '{"depth_cm": NaN}',
])
def test_parse_rejects_bad_payloads(raw):
    with pytest.raises(sensor.SensorError):
        sensor.parse_payload(raw, NOW)


def test_fuse_is_between_inputs_and_tighter():
    f, s = sensor.fuse(2.0, 3.0, 0.3, 0.05)
    assert 2.0 < f < 3.0
    assert abs(f - 3.0) < abs(f - 2.0)     # sensor is more precise, so it dominates
    assert s < 0.15
    same, _ = sensor.fuse(2.0, 2.0)
    assert same == pytest.approx(2.0)


def test_monitor_model_only_then_fused_then_stale():
    m = sensor.Monitor(design_kn_m2=4.0, max_sensor_age_h=6)
    assert m.evaluate(NOW) is None
    r = m.update_model(NOW, 2.0)
    assert r["source"] == "model" and r["level"] == 1
    r = m.on_sensor(json.dumps({"load_kn_m2": 3.2, "ts": NOW.isoformat()}), NOW)
    assert r["source"] == "fused" and r["level"] == 2
    later = pd.Timestamp(NOW) + pd.Timedelta(hours=10)
    r = m.evaluate(later)
    assert r["source"] == "model"


def test_monitor_sensor_only_and_bad_design():
    m = sensor.Monitor(design_kn_m2=2.0)
    r = m.on_sensor('{"load_kn_m2": 1.9}', NOW)
    assert r["source"] == "sensor" and r["level"] == 3
    with pytest.raises(ValueError):
        sensor.Monitor(0)
