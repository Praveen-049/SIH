"""Open-Meteo access with a small in-memory TTL cache."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 900

_cache: dict[str, tuple[float, dict[str, Any]]] = {}

EXPECTED_HOURLY_UNITS = {
    "temperature_2m": "°C", "precipitation": "mm", "wind_speed_10m": "km/h",
    "wind_direction_10m": "°", "surface_pressure": "hPa", "relative_humidity_2m": "%",
}


def _validate_payload(payload: dict[str, Any]) -> None:
    """Validate forecast arrays before the anomaly pipeline consumes them."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise WeatherDataError("Weather provider response is missing hourly time data")
    times = hourly["time"]
    units = payload.get("hourly_units", {})
    if not isinstance(units, dict):
        raise WeatherDataError("Weather provider response has invalid hourly units")
    for variable, expected_unit in EXPECTED_HOURLY_UNITS.items():
        if variable in units and units[variable] != expected_unit:
            raise WeatherDataError(f"Unexpected unit for {variable}: {units[variable]}")
        values = hourly.get(variable)
        if values is not None and (not isinstance(values, list) or len(values) != len(times)):
            raise WeatherDataError(f"Hourly data length mismatch for {variable}")


class WeatherDataError(RuntimeError):
    """Raised when a weather provider response cannot be used."""


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise WeatherDataError("Coordinates are outside the valid latitude/longitude range")


def _cache_key(latitude: float, longitude: float, forecast_days: int, timezone_name: str) -> str:
    return f"{latitude:.4f}:{longitude:.4f}:{forecast_days}:{timezone_name}"


def _request_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "SIH26078-weather-prototype/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise WeatherDataError(f"Weather provider unavailable: {error}") from error
    if isinstance(payload, dict) and payload.get("error"):
        raise WeatherDataError(str(payload.get("reason", "Weather provider returned an invalid response")))
    if not isinstance(payload, (dict, list)):
        raise WeatherDataError("Weather provider returned an invalid response")
    return payload


def get_forecast(latitude: float, longitude: float, forecast_days: int = 7, timezone_name: str = "UTC") -> tuple[dict[str, Any], str, str]:
    """Return forecast JSON, source state, and update timestamp.

    The cache is deliberately short-lived and never manufactures weather values.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    forecast_days = max(1, min(int(forecast_days), 10))
    _validate_coordinates(latitude, longitude)
    timezone_name = timezone_name or "UTC"
    key = _cache_key(latitude, longitude, forecast_days, timezone_name)
    cached = _cache.get(key)
    if cached and monotonic() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1], "CACHED", cached[1].get("_fetched_at", "N/A")

    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,relative_humidity_2m,weather_code",
        "forecast_days": forecast_days,
        "timezone": timezone_name,
    })
    payload = _request_json(f"{OPEN_METEO_URL}?{query}")
    _validate_payload(payload)
    fetched_at = datetime.now(timezone.utc).isoformat()
    payload["_fetched_at"] = fetched_at
    _cache[key] = (monotonic(), payload)
    return payload, "LIVE", fetched_at


def cache_size() -> int:
    return len(_cache)


def geocode_location(query: str) -> tuple[dict[str, Any], str]:
    """Resolve a city or coordinate query through the public Nominatim service."""
    query = str(query or "").strip()
    if not query:
        raise WeatherDataError("Location query is empty")
    try:
        latitude, longitude = (float(value.strip()) for value in query.split(",", 1))
        _validate_coordinates(latitude, longitude)
        return {"name": f"{latitude:.4f}, {longitude:.4f}", "latitude": latitude, "longitude": longitude, "country": "", "region": "", "display_name": f"{latitude:.4f}, {longitude:.4f}"}, "LIVE"
    except (ValueError, WeatherDataError):
        pass
    url = "https://nominatim.openstreetmap.org/search?" + urlencode({"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1})
    payload = _request_json(url)
    results = payload if isinstance(payload, list) else []
    if not results:
        raise WeatherDataError("Location not found")
    result = results[0]
    address = result.get("address", {})
    try:
        latitude = float(result["lat"])
        longitude = float(result["lon"])
    except (KeyError, TypeError, ValueError) as error:
        raise WeatherDataError("Geocoder returned invalid coordinates") from error
    return {
        "name": address.get("city") or address.get("town") or address.get("village") or result.get("name") or query,
        "latitude": latitude, "longitude": longitude,
        "country": address.get("country", ""), "region": address.get("state", ""),
        "display_name": result.get("display_name", query),
    }, "LIVE"
