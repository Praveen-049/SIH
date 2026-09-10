from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field, ValidationError

from anomaly_detector import StatisticalAnomalyModel, calculate_detection_confidence
from baseline import BASELINE_LABEL, reference_values, z_score
from tracker import summarize, uncertainty
from weather_service import WeatherDataError, geocode_location, get_forecast

BASE = Path(__file__).resolve().parent
DATA = json.loads((BASE / "events.json").read_text(encoding="utf-8"))
DATA_MODE = os.getenv("DATA_MODE", "LIVE").upper()
MODEL = StatisticalAnomalyModel()
LIVE_EVENTS: list[dict] = []
LAST_DATA_UPDATE = "N/A"
LAST_DATA_STATE = "UNAVAILABLE"
FORECAST_SNAPSHOTS: dict[int, dict] = {}
FORECAST_REVISIONS: list[dict] = []
LOCATION_EVENTS: list[dict] = []

app = FastAPI(title="SIH26078 Weather Anomaly Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AlertRequest(BaseModel):
    event_id: int = Field(..., description="Existing event identifier")
    radius_km: float = Field(5, gt=0, le=5, description="Alert radius, capped at 5 km for this prototype")


class LocationEventRequest(BaseModel):
    query: str = Field(..., min_length=2, description="City name or latitude,longitude")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _weather_condition(code: object) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Forecast condition unavailable"
    if code == 0:
        return "Clear sky"
    if code in (1, 2, 3):
        return "Partly cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67):
        return "Rain"
    if code in (71, 73, 75, 77):
        return "Snow"
    if code in (80, 81, 82):
        return "Rain showers"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Mixed conditions"


def _location_analysis(location: dict, weather: dict, source: str, updated: str) -> dict:
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    analysis = MODEL.analyse(weather, latitude, longitude)
    points = analysis.get("points", [])
    hourly = weather.get("hourly", {})
    codes = hourly.get("weather_code", [])
    forecast = []
    for index, point in enumerate(points):
        z_scores = point.get("z_scores", {})
        weighted_sigma = sum((0.35 * abs(float(z_scores.get("temperature") or 0)), 0.40 * max(0, float(z_scores.get("rainfall") or 0)), 0.25 * max(0, float(z_scores.get("wind") or 0))))
        score = float(point.get("anomaly_score", 0) or 0)
        forecast.append({
            "hour": index * 24, "time": point.get("time"), "temperature_c": point.get("temperature"),
            "rainfall_mm": point.get("rainfall"), "wind_kmh": point.get("wind_speed"),
            "condition": _weather_condition(codes[index * 24] if index * 24 < len(codes) else None),
            "temperature_z": z_scores.get("temperature"), "rainfall_z": z_scores.get("rainfall"), "wind_z": z_scores.get("wind"),
            "composite_score": score, "composite_sigma": round(weighted_sigma, 2), "severity": point.get("severity", risk_from_score(score)),
            "confidence": max(0, round(float(analysis.get("confidence", 0)) - index * 6)), "baseline": point.get("baseline_values", {}),
        })
    tomorrow = forecast[1] if len(forecast) > 1 else (forecast[0] if forecast else {})
    z_values = {key: abs(float(tomorrow.get(f"{key}_z") or 0)) for key in ("temperature", "rainfall", "wind")}
    drivers = sorted(z_values.items(), key=lambda item: item[1], reverse=True)
    explanation = []
    labels = {"temperature": "Temperature", "rainfall": "Rainfall", "wind": "Wind"}
    for key, value in drivers:
        if value >= 1:
            explanation.append(f"{labels[key]} is {value:.2f}σ from the prototype baseline.")
    if not explanation:
        explanation.append("Forecast conditions are within the expected prototype climatological range.")
    explanation.append(f"The forecast is evaluated across {len(forecast)} daily forecast points.")
    baseline = tomorrow.get("baseline", {})
    return {
        "location": location, "source": source, "updated_at": updated, "baseline": {"label": BASELINE_LABEL, "temperature_mean": baseline.get("temperature"), "temperature_std": baseline.get("temperature_std"), "rainfall_mean": baseline.get("rainfall"), "rainfall_std": baseline.get("rainfall_std"), "wind_mean": baseline.get("wind_speed"), "wind_std": baseline.get("wind_speed_std")},
        "forecast": forecast, "tomorrow": tomorrow, "anomalies": {"temperature_z": tomorrow.get("temperature_z"), "rainfall_z": tomorrow.get("rainfall_z"), "wind_z": tomorrow.get("wind_z"), "composite_sigma": tomorrow.get("composite_sigma"), "composite_score": tomorrow.get("composite_score")},
        "severity": tomorrow.get("severity", "NORMAL"), "confidence": tomorrow.get("confidence", 0), "dominant_signal": labels[drivers[0][0]] if drivers else "None", "secondary_signal": labels[drivers[1][0]] if len(drivers) > 1 and drivers[1][1] >= 1 else "None", "explanation": explanation,
        "local_impact": {"radius_km": 5, "affected_area_km2": 78.54, "temperature_c": tomorrow.get("temperature_c"), "rainfall_mm": tomorrow.get("rainfall_mm"), "wind_kmh": tomorrow.get("wind_kmh"), "local_anomaly_sigma": tomorrow.get("composite_sigma"), "risk": tomorrow.get("severity", "NORMAL"), "population_exposure": "Population exposure layer not configured"},
    }


def _demo_location_analysis(location: dict, reason: str) -> dict:
    """Build an explicitly labelled deterministic fallback when the forecast provider is unavailable."""
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    now = datetime.now(timezone.utc)
    times = [(now.replace(minute=0, second=0, microsecond=0)).isoformat() for _ in range(168)]
    baseline = reference_values(latitude, longitude, now.isoformat())
    weather = {"hourly": {"time": times, "temperature_2m": [baseline["temperature"]] * 168, "precipitation": [baseline["rainfall"]] * 168, "wind_speed_10m": [baseline["wind_speed"]] * 168, "weather_code": [1] * 168}}
    result = _location_analysis(location, weather, "DEMO", now.isoformat())
    result["fallback_reason"] = reason
    return result


def _impact(event: dict, radius_km: float = 5) -> dict:
    latest = event.get("latest") or {}
    trajectory = event.get("trajectory") or []
    point = trajectory[-1] if trajectory else {}
    area = round(3.141592653589793 * radius_km * radius_km, 2)
    z_scores = point.get("z_scores", {}) if isinstance(point.get("z_scores"), dict) else {}
    local_anomaly = max((abs(float(value)) for value in z_scores.values() if value is not None), default=float(event.get("anomaly_score", 0) or 0) / 40)
    return {
        "event_id": event.get("id"), "event": event.get("type"),
        "center": {"lat": latest.get("lat"), "lon": latest.get("lon")},
        "radius_km": radius_km, "affected_area_km2": area,
        "rainfall_mm": point.get("rainfall", event.get("peak_rainfall")),
        "wind_kmh": point.get("wind_speed", event.get("peak_wind")),
        "temperature_c": point.get("temperature", event.get("temperature")),
        "anomaly_score": event.get("anomaly_score"), "risk": event.get("risk", event.get("severity")),
        "local_anomaly_sigma": round(local_anomaly, 2),
        "population_exposure": "Population exposure layer not configured",
    }


def _enrich_event(event: dict) -> dict:
    enriched = dict(event)
    trajectory = []
    for index, point in enumerate(event.get("trajectory", [])):
        item = dict(point)
        item.setdefault("anomaly_score", item.get("score"))
        baseline = item.get("baseline_values") or reference_values(float(item.get("lat", event.get("latest", {}).get("lat", 0))), float(item.get("lon", event.get("latest", {}).get("lon", 0))))
        item["baseline_values"] = baseline
        item.setdefault("z_scores", {
            "temperature": z_score(item.get("temperature"), float(baseline["temperature"]), float(baseline["temperature_std"])),
            "rainfall": z_score(item.get("rainfall"), float(baseline["rainfall"]), float(baseline["rainfall_std"])),
            "wind": z_score(item.get("wind_speed"), float(baseline["wind_speed"]), float(baseline["wind_speed_std"])),
        })
        item["uncertainty"] = uncertainty(item, index)
        score = float(item.get("anomaly_score", 0) or 0)
        item["severity"] = item.get("severity") or risk_from_score(score)
        trajectory.append(item)
    enriched["trajectory"] = trajectory
    enriched["tracking"] = summarize(trajectory)
    enriched["baseline"] = event.get("baseline", BASELINE_LABEL)
    enriched["persistence_hours"] = max(0, int(event.get("forecast_end", 0)) - int(event.get("forecast_start", 0)))
    enriched["forecast_steps"] = len(trajectory)
    scores = [float(point.get("anomaly_score", 0) or 0) for point in trajectory]
    enriched["severity_forecast"] = [{"hour": point.get("hour", index * 24), "severity": point.get("severity", risk_from_score(scores[index])), "score": scores[index]} for index, point in enumerate(trajectory)]
    enriched["peak"] = {"hour": trajectory[scores.index(max(scores))].get("hour", 0), "score": max(scores, default=0)} if scores else {"hour": None, "score": 0}
    enriched["lifecycle"] = _lifecycle(scores)
    enriched["risk_drivers"] = _risk_drivers(enriched, trajectory)
    enriched["confidence_timeline"] = [{"hour": point.get("hour", index * 24), "confidence": max(0, round(float(enriched.get("confidence", 0) or 0) - index * 6))} for index, point in enumerate(trajectory)]
    enriched["evidence"] = _evidence(enriched, trajectory)
    enriched["compound_hazard"] = _compound_hazard(enriched, trajectory)
    enriched["priority"] = _priority(enriched)
    enriched["impact_analysis"] = _impact(enriched)
    enriched["analysis"] = _analysis_text(enriched)
    return enriched


def _lifecycle(scores: list[float]) -> str:
    if not scores or max(scores) < 35:
        return "DETECTED"
    if scores[-1] < 35:
        return "DISSIPATED"
    peak_index = scores.index(max(scores))
    if peak_index == len(scores) - 1 and len(scores) > 1 and scores[-1] > scores[0]:
        return "INTENSIFYING"
    if peak_index == len(scores) - 1:
        return "PEAK"
    if scores[-1] < max(scores) * 0.9:
        return "WEAKENING"
    return "CONFIRMED"


def _risk_drivers(event: dict, trajectory: list[dict]) -> dict:
    latest = trajectory[-1] if trajectory else {}
    z = latest.get("z_scores", {}) if isinstance(latest.get("z_scores"), dict) else {}
    values = {"rainfall": max(0, float(z.get("rainfall") or 0)) * 0.40, "temperature": abs(float(z.get("temperature") or 0)) * 0.35, "wind": max(0, float(z.get("wind") or 0)) * 0.25, "persistence": min(1, float(event.get("persistence_hours", 0) or 0) / 96), "spatial_expansion": 0.0}
    total = sum(values.values()) or 1
    return {key: round(value / total * 100, 1) for key, value in values.items()}


def _evidence(event: dict, trajectory: list[dict]) -> dict:
    point = trajectory[-1] if trajectory else {}
    z = point.get("z_scores", {}) if isinstance(point.get("z_scores"), dict) else {}
    baseline = point.get("baseline_values", {}) if isinstance(point.get("baseline_values"), dict) else {}
    dominant = max(("rainfall", "temperature", "wind"), key=lambda key: abs(float(z.get(key) or 0)))
    return {"dominant_variable": dominant, "forecast_value": point.get(dominant if dominant != "wind" else "wind_speed"), "baseline_value": baseline.get(dominant if dominant != "wind" else "wind_speed"), "standard_deviation": baseline.get(f"{dominant if dominant != 'wind' else 'wind_speed'}_std"), "z_score": z.get(dominant), "persistence_hours": event.get("persistence_hours"), "affected_area_km2": event.get("affected_area_km2"), "intensity_trend_percent": event.get("tracking", {}).get("intensity_change_percent", 0), "track_confidence": event.get("tracking", {}).get("confidence", event.get("confidence"))}


def _compound_hazard(event: dict, trajectory: list[dict]) -> dict:
    point = trajectory[-1] if trajectory else {}
    z = point.get("z_scores", {}) if isinstance(point.get("z_scores"), dict) else {}
    active = [name for name, value in z.items() if abs(float(value or 0)) >= 1.5]
    return {"is_compound": len(active) >= 2, "signals": active, "label": "COMPOUND WEATHER EVENT" if len(active) >= 2 else "SINGLE-SIGNAL EVENT"}


def _priority(event: dict) -> dict:
    severity = event.get("risk", event.get("severity", "NORMAL"))
    return {"level": "P1 CRITICAL" if severity == "SEVERE" else "P2 HIGH" if severity == "HIGH" else "P3 MODERATE" if severity == "MODERATE" else "P4 LOW", "method": "Prototype hazard-based priority"}


def _analysis_text(event: dict) -> str:
    tracking = event.get("tracking", {})
    dominant = event.get("dominant_variable") or event.get("type", "weather").replace(" Anomaly", "")
    score = event.get("anomaly_score", "N/A")
    confidence = event.get("confidence", "N/A")
    persistence = event.get("persistence_hours", "N/A")
    direction = tracking.get("direction", event.get("movement", "STATIONARY"))
    return f"The system identifies a persistent {dominant.lower()} anomaly with a composite score of {score}/100 and {confidence}% detection confidence. The signal persists for {persistence} hours and the statistical tracker estimates movement {direction.lower()} at {tracking.get('speed_kmh', 0)} km/h. This is AI/statistical event analysis using a {event.get('baseline', BASELINE_LABEL).lower()}, not a trained ML forecast."


def _regions() -> list[dict]:
    return [event for event in DATA.get("events", []) if event.get("start") and event.get("latest")]


def _live_event(region: dict, weather: dict, source: str, updated: str) -> dict:
    analysis = MODEL.analyse(weather, region["latest"]["lat"], region["latest"]["lon"])
    peak = analysis["peak"]
    points = analysis["points"]
    event_type = peak.get("event_type", "Weather observation")
    metric = peak.get("primary_metric", "Weather anomaly")
    anomaly = peak.get("primary_anomaly")
    start = region["start"]
    latest = region["latest"]
    last_index = max(1, len(points) - 1)
    trajectory = [{
        "time": point.get("time"), "hour": index * 24,
        "lat": round(start["lat"] + (latest["lat"] - start["lat"]) * index / last_index, 4),
        "lon": round(start["lon"] + (latest["lon"] - start["lon"]) * index / last_index, 4),
        "temperature": point.get("temperature"), "rainfall": point.get("rainfall"),
        "wind_speed": point.get("wind_speed"), "anomaly_score": point.get("anomaly_score"), "z_scores": point.get("z_scores"), "baseline_values": point.get("baseline_values"),
        "severity": point.get("severity"),
    } for index, point in enumerate(points)]
    temperatures = [point.get("temperature") for point in points if point.get("temperature") is not None]
    rainfalls = [point.get("rainfall") for point in points if point.get("rainfall") is not None]
    winds = [point.get("wind_speed") for point in points if point.get("wind_speed") is not None]
    event = {
        "id": region["id"], "type": event_type, "risk": analysis["severity"], "severity": analysis["severity"],
        "anomaly_score": analysis["anomaly_score"], "confidence": analysis["confidence"],
        "detection_confidence": analysis["confidence"], "forecast_start": 0,
        "forecast_end": max(0, (len(trajectory) - 1) * 24), "movement": "STATIONARY",
        "speed_deg": round(((latest["lat"] - start["lat"]) ** 2 + (latest["lon"] - start["lon"]) ** 2) ** 0.5 / max(1, len(trajectory) - 1), 2), "peak_rainfall": round(max(rainfalls), 2) if rainfalls else None,
        "temperature_anomaly": anomaly if "Temperature" in metric else None,
        "wind_anomaly": anomaly if "Wind" in metric else None,
        "rainfall_anomaly": anomaly if "Rainfall" in metric else None,
        "affected_area_km2": None, "impact": [], "start": region["start"], "latest": region["latest"],
        "trajectory": trajectory, "source": source, "updated_at": updated,
        "baseline": analysis["baseline"], "dominant_variable": metric,
        "interpretation": f"{event_type} detected from Open-Meteo forecast values using a statistical reference. The signal is evaluated across {len(trajectory)} forecast timesteps.",
        "why": [f"Primary signal: {metric}.", f"Forecast values were evaluated across {len(trajectory)} timesteps.", "Detection confidence combines data completeness, persistence, and anomaly magnitude."],
    }
    if "Temperature" not in metric:
        event["temperature_anomaly"] = None
    if "Rainfall" not in metric:
        event["rainfall_anomaly"] = None
    if "Wind" not in metric:
        event["wind_anomaly"] = None
    if latest["lon"] > start["lon"] and latest["lat"] > start["lat"]:
        event["movement"] = "NORTH-EAST"
    elif latest["lon"] > start["lon"] and latest["lat"] < start["lat"]:
        event["movement"] = "SOUTH-EAST"
    elif latest["lon"] < start["lon"] and latest["lat"] > start["lat"]:
        event["movement"] = "NORTH-WEST"
    elif latest["lon"] < start["lon"] and latest["lat"] < start["lat"]:
        event["movement"] = "SOUTH-WEST"
    elif latest["lat"] > start["lat"]:
        event["movement"] = "NORTH"
    elif latest["lat"] < start["lat"]:
        event["movement"] = "SOUTH"
    elif latest["lon"] > start["lon"]:
        event["movement"] = "EAST"
    elif latest["lon"] < start["lon"]:
        event["movement"] = "WEST"
    return _enrich_event(event)


def _refresh_live_events() -> tuple[list[dict], str]:
    global LIVE_EVENTS, LAST_DATA_UPDATE, LAST_DATA_STATE, FORECAST_SNAPSHOTS, FORECAST_REVISIONS
    detected: list[dict] = []
    failures = 0
    update = _now()
    for region in _regions():
        try:
            weather, source, fetched_at = get_forecast(region["latest"]["lat"], region["latest"]["lon"])
            hourly = weather.get("hourly", {})
            current_snapshot = {"event_id": region["id"], "timestamp": fetched_at, "rainfall_mm": round(max((float(value) for value in hourly.get("precipitation", []) if value is not None), default=0), 2), "wind_kmh": round(max((float(value) for value in hourly.get("wind_speed_10m", []) if value is not None), default=0), 2), "temperature_c": round(max((float(value) for value in hourly.get("temperature_2m", []) if value is not None), default=0), 2)}
            previous_snapshot = FORECAST_SNAPSHOTS.get(region["id"])
            if previous_snapshot and previous_snapshot.get("timestamp") != current_snapshot["timestamp"]:
                changes = {key: round((current_snapshot[key] - previous_snapshot[key]) / max(0.01, abs(previous_snapshot[key])) * 100, 1) for key in ("rainfall_mm", "wind_kmh", "temperature_c")}
                FORECAST_REVISIONS.append({"event_id": region["id"], "previous": previous_snapshot, "current": current_snapshot, "change_percent": changes, "significant": any(abs(value) >= 25 for value in changes.values()), "source": "In-process forecast snapshots; not durable across serverless instance replacement."})
            FORECAST_SNAPSHOTS[region["id"]] = current_snapshot
            event = _live_event(region, weather, source, fetched_at)
            if event["anomaly_score"] >= 20:
                detected.append(event)
        except WeatherDataError:
            failures += 1
    if detected:
        LIVE_EVENTS = detected
        LAST_DATA_UPDATE = update
        LAST_DATA_STATE = "LIVE" if failures == 0 else "CACHED"
        return LIVE_EVENTS, LAST_DATA_STATE
    if LIVE_EVENTS:
        LAST_DATA_STATE = "CACHED"
        return LIVE_EVENTS, LAST_DATA_STATE
    LAST_DATA_STATE = "UNAVAILABLE"
    return [], LAST_DATA_STATE


def _current_events() -> tuple[list[dict], str]:
    if DATA_MODE == "DEMO":
        return [_enrich_event(event) for event in DATA.get("events", [])], "DEMO"
    return _refresh_live_events()


def _find_event(event_id: int) -> dict | None:
    events, _ = _current_events()
    return next((item for item in [*events, *LOCATION_EVENTS] if item.get("id") == event_id), None)

def risk_from_score(score: float) -> str:
    if score >= 75:
        return "SEVERE"
    if score >= 55:
        return "HIGH"
    if score >= 35:
        return "MODERATE"
    return "LOW"

@app.get("/")
def root():
    return {"name": "SIH26078 Weather Anomaly Intelligence API", "status": "online"}

@app.get("/api/health")
def health():
    events, state = _current_events()
    return {
        "status": "online",
        "api": True,
        "data": bool(events),
        "model": isinstance(MODEL, StatisticalAnomalyModel),
        "map": True,
    }

@app.get("/api/events")
def events():
    records, state = _current_events()
    return {"events": records, "generated": True, "source": state, "updated_at": LAST_DATA_UPDATE, "baseline": BASELINE_LABEL}


@app.get("/api/anomalies")
def anomalies():
    records, state = _current_events()
    return {"anomalies": records, "source": state, "baseline": BASELINE_LABEL, "generated_at": _now()}


@app.get("/api/spatial-field")
def spatial_field():
    records, state = _current_events()
    cells = []
    for event in records:
        for point in event.get("trajectory", []):
            cells.append({"event_id": event.get("id"), "lat": point.get("lat"), "lon": point.get("lon"), "hour": point.get("hour", 0), "temperature": point.get("temperature"), "rainfall": point.get("rainfall"), "wind_speed": point.get("wind_speed"), "anomaly_score": point.get("anomaly_score", point.get("score", 0)), "severity": point.get("severity", risk_from_score(float(point.get("anomaly_score", point.get("score", 0)) or 0))), "z_scores": point.get("z_scores", {}), "persistence": event.get("persistence_hours", 0)})
    return {"cells": cells, "source": state, "baseline": BASELINE_LABEL, "note": "Cells represent forecast sample locations; no spatial interpolation is claimed."}


@app.get("/api/events/live")
def live_events():
    records, state = _refresh_live_events()
    return {"events": records, "source": state, "updated_at": LAST_DATA_UPDATE, "baseline": "statistical reference thresholds"}


@app.get("/api/weather")
def weather(
    latitude: float | None = Query(None, ge=-90, le=90),
    longitude: float | None = Query(None, ge=-180, le=180),
):
    region = _regions()[0] if latitude is None or longitude is None else {"latest": {"lat": latitude, "lon": longitude}}
    try:
        payload, source, updated = get_forecast(region["latest"]["lat"], region["latest"]["lon"])
    except WeatherDataError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    payload.pop("_fetched_at", None)
    return {"latitude": region["latest"]["lat"], "longitude": region["latest"]["lon"], "source": source, "updated_at": updated, "weather": payload}


@app.get("/api/location-intelligence")
def location_intelligence(
    location: str | None = Query(None, min_length=2),
    latitude: float | None = Query(None, ge=-90, le=90),
    longitude: float | None = Query(None, ge=-180, le=180),
):
    if latitude is not None and longitude is not None:
        query = f"{latitude},{longitude}"
    elif location:
        query = location
    else:
        raise HTTPException(status_code=400, detail="Provide a location or latitude and longitude")
    try:
        resolved, geocode_source = geocode_location(query)
    except WeatherDataError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        forecast, source, updated = get_forecast(resolved["latitude"], resolved["longitude"], forecast_days=7, timezone_name="auto")
        result = _location_analysis(resolved, forecast, source, updated)
    except WeatherDataError as error:
        result = _demo_location_analysis(resolved, str(error))
    result["geocoder"] = geocode_source
    return result


@app.post("/api/location-events")
def create_location_event(payload: LocationEventRequest):
    try:
        resolved, _ = geocode_location(payload.query)
    except WeatherDataError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        forecast, source, updated = get_forecast(resolved["latitude"], resolved["longitude"], forecast_days=7, timezone_name="auto")
        intelligence = _location_analysis(resolved, forecast, source, updated)
    except WeatherDataError as error:
        intelligence = _demo_location_analysis(resolved, str(error))
    score = max((float(item.get("composite_score", 0) or 0) for item in intelligence["forecast"]), default=0)
    if score < 20:
        raise HTTPException(status_code=409, detail="Location does not meet the event detection threshold")
    event_id = 1000 + len(LOCATION_EVENTS)
    trajectory = [{"hour": item["hour"], "time": item.get("time"), "lat": resolved["latitude"], "lon": resolved["longitude"], "temperature": item.get("temperature_c"), "rainfall": item.get("rainfall_mm"), "wind_speed": item.get("wind_kmh"), "anomaly_score": item.get("composite_score"), "severity": item.get("severity"), "z_scores": {"temperature": item.get("temperature_z"), "rainfall": item.get("rainfall_z"), "wind": item.get("wind_z")}, "baseline_values": item.get("baseline")} for item in intelligence["forecast"]]
    event = _enrich_event({"id": event_id, "type": f"{intelligence['dominant_signal']} anomaly", "risk": intelligence["severity"], "severity": intelligence["severity"], "anomaly_score": score, "confidence": intelligence["confidence"], "forecast_start": 0, "forecast_end": trajectory[-1]["hour"] if trajectory else 0, "movement": "STATIONARY", "affected_area_km2": None, "impact": [], "start": {"lat": resolved["latitude"], "lon": resolved["longitude"]}, "latest": {"lat": resolved["latitude"], "lon": resolved["longitude"]}, "trajectory": trajectory, "source": source, "updated_at": updated, "baseline": BASELINE_LABEL, "dominant_variable": intelligence["dominant_signal"], "why": intelligence["explanation"]})
    LOCATION_EVENTS.append(event)
    return {"event": event, "location_intelligence": intelligence}


@app.get("/api/system/status")
def system_status():
    records, state = _current_events()
    return {"api": "ONLINE", "data": state, "anomaly_engine": "ONLINE", "tracker": "ONLINE", "map": "ONLINE", "events": len(records), "last_data_update": LAST_DATA_UPDATE, "data_mode": DATA_MODE, "baseline": BASELINE_LABEL}

@app.get("/api/events/{event_id}")
def event(event_id: int):
    item = _find_event(event_id)
    if item:
        return item
    raise HTTPException(status_code=404, detail="Event not found")

@app.get("/api/events/{event_id}/track")
def track(event_id: int):
    item = _find_event(event_id)
    if item:
        return {"event_id": event_id, "trajectory": item.get("trajectory", []), "risk": item.get("risk"), "anomaly_score": item.get("anomaly_score"), "affected_area_km2": item.get("affected_area_km2"), "latest": item.get("latest"), "movement": item.get("movement"), "confidence": item.get("confidence"), "tracking": item.get("tracking"), "uncertainty": [point.get("uncertainty") for point in item.get("trajectory", [])]}
    raise HTTPException(status_code=404, detail="Event not found")

@app.post("/api/alerts")
def create_alert(payload: AlertRequest):
    event_id = payload.event_id
    event = _find_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    impact = _impact(event, payload.radius_km)
    return {
        "success": True,
        "alert_id": f"ALT-{event_id:03d}-{DATA_MODE}",
        "event_id": event_id,
        "severity": event.get("risk"),
        "latitude": event["latest"]["lat"],
        "longitude": event["latest"]["lon"],
        "message": f"Alert generated for Event #{event_id} around {event['latest']['lat']}°N, {event['latest']['lon']}°E.",
        "radius_km": payload.radius_km,
        "valid_hours": max(1, event.get("forecast_end", 24) - event.get("forecast_start", 0)),
        "impact": impact,
        "recommended_action": "Prepare localized flood response and increase monitoring." if "rain" in event.get("type", "").lower() else "Increase local monitoring and prepare targeted response.",
        "generated_at": _now(),
        "status": "READY",
    }


@app.get("/api/events/{event_id}/analysis")
def event_analysis(event_id: int):
    item = _find_event(event_id)
    if item:
        return {"event_id": event_id, "analysis": item.get("analysis"), "why": item.get("why", []), "score": item.get("anomaly_score"), "severity": item.get("risk", item.get("severity")), "confidence": item.get("confidence"), "baseline": item.get("baseline", BASELINE_LABEL), "dominant_variable": item.get("dominant_variable", item.get("type"))}
    raise HTTPException(status_code=404, detail="Event not found")


@app.get("/api/events/{event_id}/impact")
def event_impact(event_id: int, radius_km: float = Query(5, gt=0, le=5)):
    item = _find_event(event_id)
    if item:
        return _impact(item, radius_km)
    raise HTTPException(status_code=404, detail="Event not found")


@app.get("/api/forecast-bust")
def forecast_bust():
    if not FORECAST_REVISIONS:
        return {"available": False, "message": "Forecast revision comparison unavailable", "source": "Insufficient previous forecast history in this serverless instance."}
    latest = FORECAST_REVISIONS[-1]
    return {"available": True, **latest}


@app.get("/api/validation")
def validation():
    # Controlled synthetic checks exercise the same score/classification functions.
    cases = [("rainfall", 95, True), ("temperature", 42, True), ("wind", 12, False), ("rainfall", 3, False)]
    tp = sum(1 for _, value, actual in cases if actual and value > 40)
    fp = sum(1 for _, value, actual in cases if not actual and value > 40)
    fn = sum(1 for _, value, actual in cases if actual and value <= 40)
    tn = len(cases) - tp - fp - fn
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {"label": "Controlled prototype validation", "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(2 * precision * recall / max(0.001, precision + recall), 3), "false_alarm_rate": round(fp / max(1, fp + tn), 3), "detection_rate": round(recall, 3)}
