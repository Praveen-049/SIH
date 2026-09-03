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


class WeatherDataError(RuntimeError):
    """Raised when a weather provider response cannot be used."""


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise WeatherDataError("Coordinates are outside the valid latitude/longitude range")


def _cache_key(latitude: float, longitude: float, forecast_days: int) -> str:
    return f"{latitude:.4f}:{longitude:.4f}:{forecast_days}"


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "SIH26078-weather-prototype/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise WeatherDataError(f"Weather provider unavailable: {error}") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise WeatherDataError(str(payload.get("reason", "Weather provider returned an invalid response")))
    return payload


def get_forecast(latitude: float, longitude: float, forecast_days: int = 7) -> tuple[dict[str, Any], str, str]:
    """Return forecast JSON, source state, and update timestamp.

    The cache is deliberately short-lived and never manufactures weather values.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    forecast_days = max(1, min(int(forecast_days), 10))
    _validate_coordinates(latitude, longitude)
    key = _cache_key(latitude, longitude, forecast_days)
    cached = _cache.get(key)
    if cached and monotonic() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1], "CACHED", cached[1].get("_fetched_at", "N/A")

    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,relative_humidity_2m,weather_code",
        "forecast_days": forecast_days,
        "timezone": "UTC",
    })
    payload = _request_json(f"{OPEN_METEO_URL}?{query}")
    fetched_at = datetime.now(timezone.utc).isoformat()
    payload["_fetched_at"] = fetched_at
    _cache[key] = (monotonic(), payload)
    return payload, "LIVE", fetched_at


def cache_size() -> int:
    return len(_cache)
