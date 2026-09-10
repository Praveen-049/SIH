# SIH26078 Weather Anomaly Intelligence Prototype

## What this prototype demonstrates

1. Forecast anomaly detection results.
2. Severity classification: Severe / High / Moderate.
3. Geographic anomaly locations on an interactive map.
4. Spatio-temporal tracking from +24h to +96h.
5. Event trajectory and forecast evolution.
6. Statistical anomaly score and heuristic evidence score.
7. Explainable statistical interpretation.
8. Impact assessment.
9. 5 km alert generation through a REST API.

The default `DATA_MODE=LIVE` path retrieves forecast data from Open-Meteo and applies the statistical anomaly engine. Records in `backend/events.json` are retained only for explicitly selected `DATA_MODE=DEMO` runs.

## Data Sources

- **Provider:** Open-Meteo public forecast API, accessed server-side by `backend/weather_service.py`.
- **Variables:** 2 m temperature, precipitation, 10 m wind speed and direction, surface pressure, relative humidity, and weather code.
- **Update behavior:** Responses are cached in memory for 15 minutes. Provider failures use a still-valid cache where available; otherwise live data is reported unavailable.
- **Anomaly methodology:** `backend/anomaly_detector.py` uses configurable statistical reference thresholds and produces a 0-100 score with Normal, Moderate, High, and Severe bands.
- **Baseline:** The interface labels the current method as **statistical reference thresholds**, not a 30-year climatology.
- **Detection confidence:** This is not ML confidence. It combines data completeness (35%), persistence above threshold (40%), and anomaly magnitude (25%).
- **Limitations:** Live mode uses point forecasts at six configured region coordinates. Point forecasts do not provide affected-area extent, so that value may be `N/A`. Cyclone detection is not a dedicated cyclone-track product, and detections are not official warnings.

## Data Modes

Live mode is the default. Use demo records only explicitly:

```powershell
$env:DATA_MODE="LIVE"
$env:DATA_MODE="DEMO"
```

## Run on Windows

Open PowerShell in the `backend` folder.

If Python is installed:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

If PowerShell blocks activation, you do NOT need to activate the environment. Run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn main:app --reload
```

Then open `frontend/index.html` with Live Server in VS Code.

The frontend expects the API at:

`http://127.0.0.1:8000`

## Recommended demo sequence

1. Open the dashboard.
2. Select Event #10.
3. Change +24h → +48h → +72h → +96h.
4. Click TRACK WEATHER EVENT.
5. Explain that the orange trajectory represents the evolving anomaly footprint.
6. Show anomaly score, confidence and impact assessment.
7. Click GENERATE 5 KM ALERT.
8. Explain that the current backend uses public forecast data and a statistical detector; it is not a trained ML warning system.

## Architecture

Public forecast data
        ↓
Data preprocessing
        ↓
Statistical anomaly detection
        ↓
Spatio-temporal tracking
        ↓
Risk + confidence
        ↓
Interactive geospatial dashboard
        ↓
5 km alert API

The intended research architecture can later replace the statistical model with the proposed GNN, diffusion, or physics-informed approach. This prototype keeps that replacement behind the `AnomalyModel` interface.

## Scientific scope and limitations

This repository is a technically honest prototype, not an implementation of the complete SIH research architecture.

| Capability | Status |
| --- | --- |
| Open-Meteo surface point forecast ingestion | Implemented |
| Hourly unit and array validation | Implemented |
| Statistical reference-threshold score | Implemented prototype |
| Historical climatology or 30-year baseline | Not implemented |
| NEPS-G, NCUM, ERA5, or IMDAA ingestion | Not implemented |
| EPS ensemble members and spread | Not implemented |
| EFI | Not implemented |
| Spatial anomaly fields and object tracking | Not implemented |
| Icosahedral mesh or spherical GNN | Planned, not implemented |
| Diffusion downscaling from 12 km to 5 km | Planned, not implemented |
| Physics-informed loss or constraints | Planned, not implemented |
| Historical-event validation and skill scores | Not implemented |

Live trajectories are endpoint-interpolation visualizations between configured region coordinates. Demo trajectories in `backend/events.json` are manually authored and are marked by the API as `demo_simulated`. The displayed evidence score is a deterministic heuristic based on completeness, persistence, and score magnitude; it is not probabilistic confidence or ensemble uncertainty.

The alert feature generates a REST response and displays a 5 km radius around the selected point. It is not a 5 km-resolution impact model, does not calculate an affected-area field, and does not send an external warning. Live precipitation is interpreted using the provider's hourly accumulation and the API exposes the returned units and one-hour temporal resolution.

Run the focused scientific contract checks with:

```powershell
py -m unittest backend.test_scientific_contract
```
