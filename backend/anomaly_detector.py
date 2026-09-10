"""Explainable anomaly detection for forecast values.

This is a configurable statistical reference model, not a trained ML model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import median
from typing import Any, Iterable

THRESHOLDS = {
    "temperature_reference_c": 30.0,
    "temperature_high_delta_c": 5.0,
    "temperature_low_delta_c": 10.0,
    "rainfall_reference_mm": 10.0,
    "wind_reference_kmh": 35.0,
}

SEVERITY_THRESHOLDS = {
    "SEVERE": 80.0,
    "HIGH": 60.0,
    "MODERATE": 40.0,
}


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def calculate_temperature_anomaly(value: Any) -> float | None:
    temperature = _number(value)
    if temperature is None:
        return None
    reference = THRESHOLDS["temperature_reference_c"]
    return temperature - reference


def calculate_rainfall_anomaly(value: Any) -> float | None:
    rainfall = _number(value)
    if rainfall is None:
        return None
    return rainfall - THRESHOLDS["rainfall_reference_mm"]


def calculate_wind_anomaly(value: Any) -> float | None:
    wind = _number(value)
    if wind is None:
        return None
    return wind - THRESHOLDS["wind_reference_kmh"]


def calculate_anomaly_score(point: dict[str, Any]) -> tuple[float, str, str, float | None]:
    """Return score, type, primary metric, and primary anomaly value."""
    candidates: list[tuple[float, str, str, float | None]] = []
    temperature = _number(point.get("temperature"))
    if temperature is not None:
        delta = calculate_temperature_anomaly(temperature)
        if delta is not None:
            high = max(0.0, delta) / THRESHOLDS["temperature_high_delta_c"] * 100
            low = max(0.0, -delta) / THRESHOLDS["temperature_low_delta_c"] * 100
            candidates.append((min(100.0, high), "Heat anomaly", "Temperature anomaly", delta))
            candidates.append((min(100.0, low), "Cold anomaly", "Temperature anomaly", delta))
    rainfall = _number(point.get("rainfall"))
    if rainfall is not None:
        anomaly = calculate_rainfall_anomaly(rainfall)
        candidates.append((min(100.0, max(0.0, rainfall) / 40 * 100), "Extreme rainfall", "Rainfall anomaly", anomaly))
    wind = _number(point.get("wind_speed"))
    if wind is not None:
        anomaly = calculate_wind_anomaly(wind)
        candidates.append((min(100.0, max(0.0, wind) / 80 * 100), "Extreme wind", "Wind anomaly", anomaly))
    if not candidates:
        return 0.0, "Weather observation", "Weather anomaly", None
    score, event_type, metric, anomaly = max(candidates, key=lambda item: item[0])
    return round(score, 2), event_type, metric, round(anomaly, 2) if anomaly is not None else None


def classify_severity(score: float) -> str:
    if score >= SEVERITY_THRESHOLDS["SEVERE"]:
        return "SEVERE"
    if score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= SEVERITY_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "NORMAL"


def calculate_detection_confidence(points: Iterable[dict[str, Any]], score: float) -> int:
    points = list(points)
    if not points:
        return 0
    complete = sum(1 for point in points if any(point.get(key) is not None for key in ("temperature", "rainfall", "wind_speed"))) / len(points)
    persistent = sum(1 for point in points if float(point.get("anomaly_score", 0)) >= 40) / len(points)
    magnitude = min(1.0, score / 100)
    return round((complete * 0.35 + persistent * 0.4 + magnitude * 0.25) * 100)


class AnomalyModel(ABC):
    @abstractmethod
    def analyse(self, weather: dict[str, Any], latitude: float, longitude: float) -> dict[str, Any]:
        raise NotImplementedError


class StatisticalAnomalyModel(AnomalyModel):
    """Default model using configurable statistical reference thresholds."""

    def analyse(self, weather: dict[str, Any], latitude: float, longitude: float) -> dict[str, Any]:
        hourly = weather.get("hourly", {})
        times = hourly.get("time", [])
        temperatures = hourly.get("temperature_2m", [])
        rainfall = hourly.get("precipitation", [])
        wind = hourly.get("wind_speed_10m", [])
        points: list[dict[str, Any]] = []
        for index, timestamp in enumerate(times):
            point = {
                "time": timestamp,
                "lat": latitude,
                "lon": longitude,
                "temperature": temperatures[index] if index < len(temperatures) else None,
                "rainfall": rainfall[index] if index < len(rainfall) else None,
                "wind_speed": wind[index] if index < len(wind) and _number(wind[index]) is not None else None,
            }
            score, event_type, metric, anomaly = calculate_anomaly_score(point)
            point.update({"anomaly_score": score, "severity": classify_severity(score), "event_type": event_type, "primary_metric": metric, "primary_anomaly": anomaly})
            points.append(point)
        sampled = points[::24] or points[:1]
        for point in sampled:
            point["wind_speed"] = round(point["wind_speed"], 2) if point["wind_speed"] is not None else None
        peak = max(points, key=lambda point: point["anomaly_score"], default={})
        score = float(peak.get("anomaly_score", 0))
        return {
            "points": sampled,
            "peak": peak,
            "anomaly_score": score,
            "severity": classify_severity(score),
            "confidence": calculate_detection_confidence(points, score),
            "baseline": "statistical reference thresholds",
        }


class MLAnomalyModel(AnomalyModel):
    """Extension point for a future trained model."""

    def analyse(self, weather: dict[str, Any], latitude: float, longitude: float) -> dict[str, Any]:
        raise NotImplementedError("A trained ML model is not configured")
