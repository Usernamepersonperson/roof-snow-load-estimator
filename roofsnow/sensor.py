"""Optional on-roof sensor input (load cell or snow-depth sensor) delivered as MQTT JSON, fused with
the station-based model by inverse-variance weighting.

Accepted payload (JSON object), one of:
    {"load_kn_m2": 2.4}            load cell, kN/m2
    {"load_kg_m2": 245}            load cell, kg/m2
    {"depth_cm": 80}               snow-depth sensor, converted with the Sturm density model
optional:  "ts": ISO 8601 timestamp (default: receive time),  "sigma_rel": relative 1-sigma error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from roofsnow import alerts, density, standard

MAX_LOAD_KN_M2 = 60.0
DEFAULT_SENSOR_SIGMA_REL = 0.05
DEFAULT_MODEL_SIGMA_REL = 0.30   # assumption: station model error on a roof
SIGMA_FLOOR_KN_M2 = 0.05


class SensorError(ValueError):
    pass


@dataclass(frozen=True)
class SensorReading:
    time: pd.Timestamp
    load_kn_m2: float
    sigma_rel: float


def parse_payload(raw: bytes | str, now: datetime | None = None) -> SensorReading:
    """Validate an MQTT payload and return a reading in kN/m2 (UTC time)."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SensorError(f"payload is not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SensorError("payload must be a JSON object")
    ts = obj.get("ts")
    if ts is None:
        t = pd.Timestamp(now or datetime.now(timezone.utc))
    else:
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t):
            raise SensorError("invalid ts")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    if "load_kn_m2" in obj:
        load = _num(obj["load_kn_m2"], "load_kn_m2")
    elif "load_kg_m2" in obj:
        load = standard.swe_mm_to_kn_m2(_num(obj["load_kg_m2"], "load_kg_m2"))
    elif "depth_cm" in obj:
        depth = _num(obj["depth_cm"], "depth_cm")
        rho = density.sturm_density(depth, density.sturm_doy(t.tz_convert("UTC").tz_localize(None)))
        load = standard.swe_mm_to_kn_m2(depth * rho / 100.0)
    else:
        raise SensorError("payload needs load_kn_m2, load_kg_m2 or depth_cm")
    if not 0.0 <= load <= MAX_LOAD_KN_M2:
        raise SensorError(f"load {load:.2f} kN/m2 outside 0..{MAX_LOAD_KN_M2}")
    sigma = _num(obj.get("sigma_rel", DEFAULT_SENSOR_SIGMA_REL), "sigma_rel")
    if not 0.0 < sigma <= 1.0:
        raise SensorError("sigma_rel must be in (0, 1]")
    return SensorReading(t, load, sigma)


def _num(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise SensorError(f"{name} must be a finite number")
    return float(v)


def fuse(model_load: float, sensor_load: float, model_sigma_rel: float = DEFAULT_MODEL_SIGMA_REL,
         sensor_sigma_rel: float = DEFAULT_SENSOR_SIGMA_REL) -> tuple[float, float]:
    """Inverse-variance fusion. Returns (fused load, fused 1-sigma in kN/m2)."""
    sm = max(model_sigma_rel * model_load, SIGMA_FLOOR_KN_M2)
    ss = max(sensor_sigma_rel * sensor_load, SIGMA_FLOOR_KN_M2)
    wm, ws = 1.0 / sm**2, 1.0 / ss**2
    return (wm * model_load + ws * sensor_load) / (wm + ws), math.sqrt(1.0 / (wm + ws))


class Monitor:
    """Holds the latest station-model roof load and sensor reading; yields utilisation and alerts."""

    def __init__(self, design_kn_m2: float, max_sensor_age_h: float = 6.0,
                 thresholds=alerts.DEFAULT_THRESHOLDS, hysteresis: float = 0.05):
        if design_kn_m2 <= 0:
            raise ValueError("design load must be positive")
        self.design = design_kn_m2
        self.max_age = pd.Timedelta(hours=max_sensor_age_h)
        self.tracker = alerts.AlertTracker(tuple(thresholds), hysteresis)
        self.model_load: float | None = None
        self.model_time: pd.Timestamp | None = None
        self.sensor: SensorReading | None = None

    def update_model(self, time, roof_load_kn_m2: float) -> dict | None:
        self.model_time = pd.Timestamp(time)
        self.model_load = float(roof_load_kn_m2)
        return self.evaluate(self.model_time)

    def on_sensor(self, raw: bytes | str, now: datetime | None = None) -> dict | None:
        self.sensor = parse_payload(raw, now)
        return self.evaluate(self.sensor.time)

    def evaluate(self, now) -> dict | None:
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        s = self.sensor
        fresh = s is not None and abs(now - s.time) <= self.max_age
        if fresh and self.model_load is not None:
            load, sigma = fuse(self.model_load, s.load_kn_m2, DEFAULT_MODEL_SIGMA_REL, s.sigma_rel)
            source = "fused"
        elif fresh:
            load, sigma, source = s.load_kn_m2, max(s.sigma_rel * s.load_kn_m2, SIGMA_FLOOR_KN_M2), "sensor"
        elif self.model_load is not None:
            load = self.model_load
            sigma = max(DEFAULT_MODEL_SIGMA_REL * load, SIGMA_FLOOR_KN_M2)
            source = "model"
        else:
            return None
        util = load / self.design
        level = self.tracker.update(util, now)
        return {"time": now.isoformat(), "load_kn_m2": round(load, 3), "sigma_kn_m2": round(sigma, 3),
                "utilisation": round(util, 4), "level": level, "level_name": alerts.LEVEL_NAMES[level],
                "source": source, "action": alerts.ACTIONS[level]}


def listen(host: str, topic: str, monitor: Monitor, port: int = 1883, on_alert=print) -> None:
    """Blocking MQTT subscriber (needs the optional ``mqtt`` extra: paho-mqtt)."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SensorError("install the mqtt extra: pip install 'roof-snow-load-estimator[mqtt]'") from exc

    def handle(_c, _u, msg):  # pragma: no cover - needs a broker
        try:
            res = monitor.on_sensor(msg.payload)
        except SensorError as exc:
            on_alert({"error": str(exc)})
            return
        if res:
            on_alert(json.dumps(res))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # pragma: no cover
    client.on_message = handle  # pragma: no cover
    client.connect(host, port)  # pragma: no cover
    client.subscribe(topic)  # pragma: no cover
    client.loop_forever()  # pragma: no cover
