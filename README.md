# AI Weather Intelligence

SIH26078 prototype for **AI-driven spatio-temporal tracking of extreme weather anomalies in medium-range forecasts**.

The dashboard demonstrates the operational chain:

`DETECT -> TRACK -> PREDICT -> EXPLAIN -> LOCALIZE -> ALERT`

## What Is Implemented

- Open-Meteo medium-range forecast ingestion with a 15-minute in-memory cache.
- Deterministic demo mode using `backend/events.json` when explicitly selected.
- Prototype climatological baseline with variable-specific means and standard deviations.
- Normalized temperature, rainfall, and wind z-scores combined into a transparent weighted anomaly score.
- Severity, persistence, confidence, dominant-variable, baseline, and forecast evidence fields.
- Explainable trajectory geometry: direction, distance, speed, displacement, intensity change, and confidence.
- Statistical uncertainty bands that widen with forecast horizon.
- Backend-derived spatial forecast sample field with Leaflet layer controls.
- Event lifecycle, severity forecast, expected peak, scientific evidence, risk drivers, compound-hazard flag, and prototype hazard priority.
- Controlled prototype validation metrics for precision, recall, F1, false-alarm rate, and detection rate.
- Guided 90-120 second Judge Mode for the internal demonstration.
- 5 km local impact geometry and structured alert generation.
- Interactive Leaflet map, forecast playback, selected-event analysis, trajectory, and alert zone.
- Same-origin Vercel deployment through `api/index.py`.
- Arbitrary city/coordinate Location Intelligence using public geocoding and Open-Meteo forecasts.
- Tomorrow and multi-day forecast comparison against the prototype baseline, with z-scores, composite score, risk, explanation, and 5 km point analysis.

The current model is statistical. It is not a trained GNN, an official warning system, or a complete 30-year ERA5 climatology.

## Data And Status Honesty

Live requests use the public Open-Meteo forecast API. The UI reports `LIVE`, `CACHED`, `DEMO`, or unavailable status from the backend. Cached or demo records are never labelled live. The baseline is explicitly labelled **Prototype climatological baseline** and can later be replaced by an ERA5-derived archive.

## Methodology

For each forecast point:

```text
temperature_z = (temperature - temperature_baseline) / temperature_std
rainfall_z    = (rainfall - rainfall_baseline) / rainfall_std
wind_z        = (wind - wind_baseline) / wind_std
composite     = 0.35 * abs(temperature_z)
              + 0.40 * max(rainfall_z, 0)
              + 0.25 * max(wind_z, 0)
```

The composite is normalized to a 0-100 display score. Persistence is measured across forecast steps, and confidence combines data completeness, persistence, and magnitude. Trajectory speed uses great-circle distance divided by forecast time. The uncertainty corridor is a prototype statistical estimate, not an ensemble forecast.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Basic service health |
| GET | `/api/system/status` | API, data, model, tracker, map, and event status |
| GET | `/api/events` | Current live, cached, or demo events |
| GET | `/api/anomalies` | Anomaly records and baseline metadata |
| GET | `/api/events/{id}` | One enriched event |
| GET | `/api/events/{id}/track` | Trajectory and uncertainty data |
| GET | `/api/events/{id}/analysis` | Generated evidence-based event analysis |
| GET | `/api/events/{id}/impact` | Calculated impact zone, weather values, and risk |
| POST | `/api/alerts` | Structured 5 km alert response |
| GET | `/api/forecast-bust` | Calculated revision status when an in-process prior snapshot exists |
| GET | `/api/validation` | Controlled prototype validation metrics |
| GET | `/api/spatial-field` | Forecast sample cells for map anomaly layers |
| GET | `/api/location-intelligence?location=Chennai` | Geocode, forecast, baseline comparison, and local intelligence for any location |
| POST | `/api/location-events` | Create a trackable event when a searched location crosses the detection threshold |

## Local Setup

Backend:

```powershell
cd backend
py -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Frontend: open `frontend/index.html` with VS Code Live Server, or run `run_frontend.bat` and open `http://127.0.0.1:5500`.

For deterministic judging:

```powershell
$env:DATA_MODE="DEMO"
```

For Open-Meteo data, use `LIVE` or omit `DATA_MODE`.

## Vercel Deployment

The repository includes `api/index.py`, `vercel.json`, and a root `requirements.txt` for the Python function. Frontend requests use same-origin `/api/...` paths after deployment.

```powershell
npx vercel --prod
```

The current production deployment is `https://sih-lac-three.vercel.app`.

Forecast revision snapshots are held in process memory only. A fresh or replaced Vercel serverless instance correctly reports insufficient history rather than fabricating a comparison.

## Demo Script

1. Open the dashboard and show API, data source, tracker, model, and map status.
2. Select `Extreme Rainfall` and click `ANALYZE EVENT`.
3. Explain the dominant signal, prototype baseline, score, confidence, and persistence.
4. Click `TRACK WEATHER EVENT` and show direction, speed, displacement, and uncertainty.
5. Press `PLAY FORECAST` to move through the forecast horizon.
6. Click `GENERATE 5 KM ALERT` and show the calculated zone, area, valid window, and action.
7. Explain that forecast-bust comparison becomes available after a previous forecast snapshot is retained.

Judge Mode provides the same flow as a guided sequence: detect, inspect scientific evidence, explain, track, play the forecast, show uncertainty, generate impact, and generate the alert.

Location Intelligence provides a second workflow: search a city or `latitude,longitude`, inspect tomorrow's real Open-Meteo forecast, compare it with the prototype baseline, and view the calculated 5 km local analysis. Nominatim is used for public geocoding. If the forecast provider fails after geocoding succeeds, the response is explicitly labelled `DEMO` and includes the fallback reason; it is never labelled live.

## Future Production Extension

The defensible evolution is:

`NWP / NEPS-G + 30-year ERA5 baseline + spherical grid + GNN event tracking + physics constraints + diffusion downscaling + operational warning integration`

Those components are future work, not claims about this prototype. Population exposure, official warning dissemination, persistent forecast revision storage, and trained ML inference are also not configured yet.

## Prototype Validation

`GET /api/validation` evaluates a small controlled synthetic dataset through a transparent threshold check. Its precision, recall, F1, false-alarm rate, and detection rate are software-demonstration metrics only; they are not operational accuracy or real-world validation.

## SIH Relevance

The prototype makes the problem legible within a short demonstration: it identifies **what** is anomalous, **where** the event is moving, **when** it persists in the medium-range forecast, **why** the system detected it, and **what** a localized 5 km response could look like.
