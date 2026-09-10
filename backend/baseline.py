"""Prototype climatological reference values for anomaly normalization.

These deterministic references are an extension point for a future ERA5-derived
30-year baseline. They are intentionally labelled as a prototype baseline.
"""

from __future__ import annotations

from datetime import datetime
from math import cos, radians


BASELINE_LABEL = "Prototype climatological baseline"


def reference_values(latitude: float, longitude: float, timestamp: str | None = None) -> dict[str, float | str]:
    """Return deterministic reference means and standard deviations for a point."""
    try:
        month = datetime.fromisoformat((timestamp or "2000-01-01").replace("Z", "+00:00")).month
    except ValueError:
        month = 1
    seasonal = cos((month - 1) * 2 * 3.141592653589793 / 12)
    latitude_effect = max(-8.0, min(8.0, (25.0 - abs(float(latitude))) * 0.08))
    return {
        "temperature": round(27.0 + latitude_effect + seasonal * 4.0, 2),
        "temperature_std": 3.5,
        "rainfall": round(6.0 + max(0.0, seasonal) * 5.0, 2),
        "rainfall_std": 8.0,
        "wind_speed": round(18.0 + abs(float(longitude)) * 0.01, 2),
        "wind_speed_std": 10.0,
        "label": BASELINE_LABEL,
    }


def z_score(value: float | None, mean: float, standard_deviation: float) -> float | None:
    if value is None:
        return None
    return round((float(value) - mean) / max(standard_deviation, 0.001), 3)