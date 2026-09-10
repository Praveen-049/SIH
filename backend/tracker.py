"""Explainable geometry and uncertainty helpers for event trajectories."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any


def distance_km(first: dict[str, Any], second: dict[str, Any]) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (float(first["lat"]), float(first["lon"]), float(second["lat"]), float(second["lon"])))
    value = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return round(6371.0 * 2 * asin(sqrt(max(0.0, min(1.0, value)))), 2)


def direction(first: dict[str, Any], last: dict[str, Any]) -> str:
    north = float(last["lat"]) - float(first["lat"])
    east = float(last["lon"]) - float(first["lon"])
    if abs(north) < 0.05 and abs(east) < 0.05:
        return "STATIONARY"
    return ("NORTH-" if north > 0.05 else "SOUTH-" if north < -0.05 else "") + ("EAST" if east > 0.05 else "WEST" if east < -0.05 else "")


def summarize(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    points = [point for point in trajectory if point.get("lat") is not None and point.get("lon") is not None]
    if len(points) < 2:
        return {"direction": "STATIONARY", "distance_km": 0, "speed_kmh": 0, "displacement_km": 0, "confidence": 0}
    total_distance = sum(distance_km(points[index - 1], points[index]) for index in range(1, len(points)))
    hours = max(1, float(points[-1].get("hour", 0)) - float(points[0].get("hour", 0)))
    intensity_start = float(points[0].get("anomaly_score", points[0].get("score", 0)) or 0)
    intensity_end = float(points[-1].get("anomaly_score", points[-1].get("score", 0)) or 0)
    confidence = round(max(0, min(100, 92 - max(0, len(points) - 4) * 3 - abs(intensity_end - intensity_start) * 0.1)))
    return {
        "direction": direction(points[0], points[-1]),
        "distance_km": round(total_distance, 1),
        "speed_kmh": round(total_distance / hours, 1),
        "displacement_km": distance_km(points[0], points[-1]),
        "confidence": confidence,
        "intensity_change_percent": round(((intensity_end - intensity_start) / max(1, abs(intensity_start))) * 100, 1),
    }


def uncertainty(point: dict[str, Any], index: int) -> dict[str, Any]:
    score = float(point.get("anomaly_score", point.get("score", 0)) or 0)
    spread = round(0.15 + index * 0.08, 2)
    return {"level": "HIGH" if index >= 3 else "MEDIUM" if index >= 2 else "LOW", "score_low": round(max(0, score * (1 - spread)), 1), "score_high": round(min(100, score * (1 + spread)), 1), "radius_km": round(15 + index * 18, 1)}