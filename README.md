# SIH26078 Weather Anomaly Intelligence Platform

## AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts

Technically defensible scientific prototype built for **SIH Problem Statement 26078**.

---

## Scientific Architecture

```
                      NWP / EPS DATA INGESTION
     (Open-Meteo Operational | NEPS-G 12km | NCUM Global | ERA5 / IMDAA)
                                 │
                                 ▼
                     DATA VALIDATION & PROVENANCE
         (Strict coordinate, unit, monotonicity & NaN/Inf checking)
                                 │
                                 ▼
                   CLIMATOLOGICAL BASELINE ENGINE
             (Location- & Day-of-Year Dependent 30-yr Normal)
                                 │
                                 ▼
                 ANOMALY & EXTREME FORECAST INDEX (EFI)
             (Standardized z-score, P95/P99, ECMWF Integral Quadrature)
                                 │
                                 ▼
                    SPATIAL ANOMALY EXTRACTION
          (Gridded anomaly, Morphological filter, Connected objects)
                                 │
                                 ▼
                     WEATHER OBJECT TRACKER
             (Geodesic Haversine/Vincenty, Lifecycle Tracking)
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   SPHERICAL GRAPH & GNN                 CONDITIONAL DIFFUSION (12km→5km)
- Icosahedral Geodesic Mesh (Icosphere)   - Topography Conditioning (DEM/Slope)
- SphericalGraphConv + Temporal GRU      - Tail-Preserving Pinball Loss
- Event-Level Split (Zero Leakage)       - Physics-Informed Moisture Continuity
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                      IMPACT & ALERT ENGINE
         (5 km High-Res Model Grid vs 5.0 km Emergency Alert Buffer)
                                 │
                                 ▼
               SCIENTIFIC REST API & LEAFLET DASHBOARD
```

---

## Subsystem Implementation Matrix

| SIH 26078 Requirement | Status | Implementation Evidence |
| --- | --- | --- |
| **Data Provider Abstraction** | **IMPLEMENTED** | `backend/scientific/data_provider.py` (Standardized internal schema, xarray/numpy, Open-Meteo, NEPS-G, NCUM, ERA5, IMDAA, Synthetic). |
| **Data Validation & Provenance** | **IMPLEMENTED** | `backend/scientific/data_validator.py` (Validates bounds, monotonic coords, NaN/Inf, units, and generates SHA-256 provenance hash). |
| **Location-Dependent Climatology** | **IMPLEMENTED** | `backend/scientific/climatology_engine.py` (Day-of-Year rolling baseline; mean, std, percentiles P10..P99; no flat 30°C thresholds). |
| **Standardized Anomaly Engine** | **IMPLEMENTED** | `backend/scientific/anomaly_engine.py` (z-score, percentile ranking, threshold exceedance, preserving physical values). |
| **Extreme Forecast Index (EFI)** | **IMPLEMENTED** | `backend/scientific/efi_engine.py` (ECMWF integral discrete quadrature between ensemble forecast and climatological CDF). |
| **Spatial Object Extraction** | **IMPLEMENTED** | `backend/scientific/spatial_extraction.py` (Morphological opening, 8-connected components, true Earth surface area in km²). |
| **Geodesic Object Tracking** | **IMPLEMENTED** | `backend/scientific/object_tracker.py` (Haversine distances, IoU matching, bearings, speed km/h, lifecycle state machine). |
| **Spherical Mesh Builder** | **IMPLEMENTED** | `backend/scientific/spherical_mesh.py` (Subdivided icosahedron, node xyz on unit sphere, lat/lon mapping, dateline/pole handling). |
| **Spatio-Temporal GNN** | **IMPLEMENTED & TRAINED** | `backend/scientific/gnn_model.py` & `train_gnn.py` (SphericalGraphConv + Temporal GRU, event-split checkpoint generated). |
| **Conditional Weather Diffusion** | **IMPLEMENTED (ARCH READY)**| `backend/scientific/diffusion_downscaling.py` (DDPM U-Net downscaler, 12 km -> 5 km, topography conditioned). |
| **Extreme Tail Preservation** | **IMPLEMENTED** | `backend/scientific/diffusion_downscaling.py` (Pinball tail-loss; solves spectral over-smoothing). |
| **Physics-Informed Loss** | **IMPLEMENTED** | `backend/scientific/physics_loss.py` (Moisture continuity, non-negativity of physical variables, thermodynamic gradient smoothness). |
| **5 km Impact & Severity Engine** | **IMPLEMENTED** | `backend/scientific/impact_engine.py` (Clearly distinguishes 5 km downscaled model grid from 5 km emergency alert buffer). |
| **Historical Validation Framework** | **IMPLEMENTED** | `backend/scientific/historical_validation.py` (Evaluates track skill, centroid error, and RMSE without fabricating data). |
| **Model Status Registry** | **IMPLEMENTED** | `backend/scientific/historical_validation.py` (Central backend metadata tracking real operational readiness). |

---

## Data Modes

The platform supports 3 explicit data modes:
1. **LIVE**: Only real operational provider data (Open-Meteo public NWP) with Day-of-Year climatology.
2. **RESEARCH**: Multi-member ensemble (NEPS-G adapter), Spatio-Temporal GNN, and 12 km -> 5 km Diffusion downscaling.
3. **DEMO**: Simulated test fixtures clearly labeled to prevent masquerading as live predictions.

---

## Running the Platform

### Backend API

```powershell
py -m uvicorn main:app --reload --port 8001
```

Or run `run_backend.bat`.

### Frontend Dashboard

```powershell
cd frontend
py -m http.server 5500
```

Open `http://127.0.0.1:5500/index.html` in your web browser.

---

## Automated Test Suite

Run the full scientific test suite (15 tests covering all 10 core scientific domains):

```powershell
py -m unittest tests.test_scientific_suite
```

Run the existing contract tests:

```powershell
py -m unittest backend.test_scientific_contract
```

---

## Reproducibility

- Master Experiment Configuration: `configs/experiment_config.yaml`
- GNN Training Pipeline: `py -m scientific.train_gnn --epochs 5`
- Pinned Dependencies: `backend/requirements.txt`
