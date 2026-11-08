"""Utilisation thresholds with hysteresis and recommended-action text."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9)
LEVEL_NAMES = ("normal", "watch", "warning", "critical")
ACTIONS = {
    0: "No action. Keep monitoring.",
    1: "Watch: check the forecast, inspect drains and access, plan clearing capacity.",
    2: ("Warning: arrange snow clearing, keep people away from the roof edge and below overhangs, "
        "check for sagging, drifts and blocked drains."),
    3: ("Critical: clear the roof now under a qualified plan, limit use of the building, consult a "
        "structural engineer, watch for deflection, creaking or sticking doors."),
}


def level_for(util: float, thresholds=DEFAULT_THRESHOLDS) -> int:
    """Level without hysteresis: number of thresholds met or exceeded."""
    if util is None or util != util:
        return 0
    return sum(util >= t for t in thresholds)


@dataclass
class AlertTracker:
    """Level state machine: rises immediately, falls only below threshold - hysteresis."""

    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    hysteresis: float = 0.05
    level: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if list(self.thresholds) != sorted(self.thresholds) or len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("thresholds must be strictly increasing")
        if not self.thresholds or self.thresholds[0] <= 0:
            raise ValueError("thresholds must be positive")
        if self.hysteresis < 0 or self.hysteresis >= min(
            b - a for a, b in zip((0.0, *self.thresholds), self.thresholds, strict=False)
        ):
            raise ValueError("hysteresis must be >= 0 and smaller than the gap between thresholds")

    def update(self, util: float, when=None) -> int:
        """Feed one utilisation value (NaN keeps the level). Returns the current level."""
        if util is None or util != util:
            return self.level
        new = self.level
        while new < len(self.thresholds) and util >= self.thresholds[new]:
            new += 1
        while new > 0 and util < self.thresholds[new - 1] - self.hysteresis:
            new -= 1
        if new != self.level:
            self.events.append({"time": when, "from": self.level, "to": new, "utilisation": float(util),
                                "action": ACTIONS[min(new, 3)]})
            self.level = new
        return self.level


def run_alerts(util: pd.Series, thresholds=DEFAULT_THRESHOLDS, hysteresis: float = 0.05) -> AlertTracker:
    tr = AlertTracker(tuple(thresholds), hysteresis)
    for t, u in util.items():
        tr.update(u, t)
    return tr


def days_above(util: pd.Series, thresholds=DEFAULT_THRESHOLDS) -> dict[float, int]:
    """Number of days with utilisation at or above each threshold (raw, no hysteresis)."""
    u = util.dropna()
    return {t: int((u >= t).sum()) for t in thresholds}


def count_rising_alerts(tracker: AlertTracker, level: int) -> int:
    """Number of times the tracker entered ``level`` or higher from below it."""
    return sum(1 for e in tracker.events if e["from"] < level <= e["to"])
