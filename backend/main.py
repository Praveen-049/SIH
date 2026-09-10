from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import os
from pathlib import Path
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from anomaly_detector import StatisticalAnomalyModel
from weather_service import WeatherDataError, get_forecast

BASE = Path(__file__).resolve().parent
DATA = json.loads((BASE / "events.json").read_text(encoding="utf-8"))
DATA_MODE = os.getenv("DATA_MODE", "LIVE").upper()
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5501,http://localhost:5501",
).split(",") if origin.strip()]
MODEL = StatisticalAnomalyModel()
LIVE_EVENTS: list[dict] = []
LAST_DATA_UPDATE = "N/A"
LAST_DATA_STATE = "UNAVAILABLE"

app = FastAPI(title="SIH26078 Weather Anomaly Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AlertRequest(BaseModel):
    event_id: int = Field(..., description="Existing event identifier")
    radius_km: float = Field(5, gt=0, le=5, description="Alert radius, capped at 5 km for this prototype")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regions() -> list[dict]:
    return [event for event in DATA.get("events", []) if event.get("start") and event.get("latest")]


def _demo_events() -> list[dict]:
    records = []
    for source in DATA.get("events", []):
        event = dict(source)
        event["data_kind"] = "demo_simulated"
        event["tracking_method"] = "manual_demo_trajectory"
        event["uncertainty_type"] = "demo_heuristic_score"
        event["impact_zone_type"] = "demo_metadata_not_model_output"
        if event.get("start") and event.get("latest"):
            event["path_distance_km"] = round(_geodesic_distance_km(event["start"], event["latest"]), 1)
            duration_hours = max(1, int(event.get("forecast_end", 0)) - int(event.get("forecast_start", 0)))
            event["speed_kmh"] = round(event["path_distance_km"] / duration_hours, 2)
        event["trajectory"] = [
            {**point, "anomaly_score": point.get("anomaly_score", point.get("score"))}
            for point in source.get("trajectory", [])
        ]
        records.append(event)
    return records


def _geodesic_distance_km(start: dict, latest: dict) -> float:
    earth_radius_km = 6371.0088
    lat1, lat2 = math.radians(float(start["lat"])), math.radians(float(latest["lat"]))
    delta_lat = lat2 - lat1
    delta_lon = math.radians(float(latest["lon"]) - float(start["lon"]))
    haversine = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return earth_radius_km * 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))


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
    path_distance_km = _geodesic_distance_km(start, latest)
    forecast_hours = max(1, (len(points) - 1) * 24)
    trajectory = [{
        "time": point.get("time"), "hour": index * 24,
        "lat": round(start["lat"] + (latest["lat"] - start["lat"]) * index / last_index, 4),
        "lon": round(start["lon"] + (latest["lon"] - start["lon"]) * index / last_index, 4),
        "temperature": point.get("temperature"), "rainfall": point.get("rainfall"),
        "wind_speed": point.get("wind_speed"), "anomaly_score": point.get("anomaly_score"),
        "severity": point.get("severity"),
    } for index, point in enumerate(points)]
    temperatures = [point.get("temperature") for point in points if point.get("temperature") is not None]
    rainfalls = [point.get("rainfall") for point in points if point.get("rainfall") is not None]
    winds = [point.get("wind_speed") for point in points if point.get("wind_speed") is not None]
    event = {
        "id": region["id"], "type": event_type, "risk": analysis["severity"], "severity": analysis["severity"],
        "data_kind": "live_point_forecast",
        "anomaly_score": analysis["anomaly_score"], "confidence": analysis["confidence"],
        "detection_confidence": analysis["confidence"], "forecast_start": 0,
        "forecast_end": max(0, (len(trajectory) - 1) * 24), "movement": "STATIONARY",
        "path_distance_km": round(path_distance_km, 1), "speed_kmh": round(path_distance_km / forecast_hours, 2),
        "tracking_method": "endpoint_interpolation_visualization",
        "uncertainty_type": "heuristic_evidence_score",
        "impact_zone_type": "not_computed_for_point_forecast",
        "peak_rainfall": round(max(rainfalls), 2) if rainfalls else None,
        "temperature_anomaly": anomaly if "Temperature" in metric else None,
        "wind_anomaly": anomaly if "Wind" in metric else None,
        "rainfall_anomaly": anomaly if "Rainfall" in metric else None,
        "affected_area_km2": None, "impact": [], "start": region["start"], "latest": region["latest"],
        "trajectory": trajectory, "source": source, "updated_at": updated,
        "baseline": analysis["baseline"],
        "interpretation": f"{event_type} detected from Open-Meteo point-forecast values using fixed statistical reference thresholds. The path is an endpoint interpolation across {len(trajectory)} forecast timesteps; it is not object tracking.",
        "why": [f"Primary signal: {metric}.", f"Forecast values were evaluated across {len(trajectory)} timesteps.", "The evidence score combines data completeness, persistence, and anomaly magnitude; it is not probabilistic confidence."],
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
    return event


def _refresh_live_events() -> tuple[list[dict], str]:
    global LIVE_EVENTS, LAST_DATA_UPDATE, LAST_DATA_STATE
    detected: list[dict] = []
    failures = 0
    update = _now()
    regions = _regions()
    with ThreadPoolExecutor(max_workers=min(6, len(regions) or 1)) as executor:
        requests = {
            executor.submit(get_forecast, region["latest"]["lat"], region["latest"]["lon"]): region
            for region in regions
        }
        for request in as_completed(requests):
            region = requests[request]
            try:
                weather, source, fetched_at = request.result()
                event = _live_event(region, weather, source, fetched_at)
                if event["anomaly_score"] >= 20:
                    detected.append(event)
            except WeatherDataError:
                failures += 1
    detected.sort(key=lambda event: event.get("id", 0))
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
        return _demo_events(), "DEMO"
    return _refresh_live_events()


def _find_event(event_id: int) -> dict | None:
    events, _ = _current_events()
    return next((item for item in events if item.get("id") == event_id), None)

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
        "model_type": "statistical_threshold_prototype",
        "map": True,
    }

@app.get("/api/events")
def events():
    records, state = _current_events()
    return {"events": records, "generated": True, "source": state, "updated_at": LAST_DATA_UPDATE, "baseline": "statistical reference thresholds"}


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
    return {"latitude": region["latest"]["lat"], "longitude": region["latest"]["lon"], "source": source, "updated_at": updated, "timezone": payload.get("timezone", "UTC"), "units": payload.get("hourly_units", {}), "temporal_resolution_hours": 1, "precipitation_accumulation": "provider hourly accumulation", "weather": payload}


@app.get("/api/system/status")
def system_status():
    records, state = _current_events()
    return {"api": "ONLINE", "data": state, "anomaly_engine": "ONLINE", "model_type": "statistical_threshold_prototype", "map": "ONLINE", "events": len(records), "last_data_update": LAST_DATA_UPDATE, "data_mode": DATA_MODE}

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
        return {"event_id": event_id, "trajectory": item.get("trajectory", []), "risk": item.get("risk"), "anomaly_score": item.get("anomaly_score"), "affected_area_km2": item.get("affected_area_km2"), "latest": item.get("latest"), "movement": item.get("movement"), "confidence": item.get("confidence")}
    raise HTTPException(status_code=404, detail="Event not found")

@app.post("/api/alerts")
def create_alert(payload: AlertRequest):
    event_id = payload.event_id
    event = _find_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return {
        "success": True,
        "alert_id": f"ALT-{event_id:03d}-{DATA_MODE}",
        "event_id": event_id,
        "severity": event.get("risk"),
        "latitude": event["latest"]["lat"],
        "longitude": event["latest"]["lon"],
        "message": f"Alert generated for Event #{event_id} around {event['latest']['lat']}°N, {event['latest']['lon']}°E.",
        "radius_km": payload.radius_km,
        "generated_at": _now(),
        "status": "READY",
    }
