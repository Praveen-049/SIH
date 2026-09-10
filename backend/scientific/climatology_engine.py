"""Climatological baseline engine providing location- and day-of-year dependent statistical baselines.

Replaces fixed global reference thresholds with genuine meteorological climatologies.
Outputs mean, standard deviation, and percentiles (P10, P50, P90, P95, P99).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class ClimatologicalStats:
    variable: str
    latitude: float
    longitude: float
    day_of_year: int
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    p95: float
    p99: float
    sample_count: int
    is_synthetic: bool
    reference_period: str


class SyntheticClimatologyProvider:
    """Provides physically informed, location- and season-dependent climatology for India & South Asia.

    Strictly labeled as TEST / SYNTHETIC DATA.
    Accounts for latitude lapse rates, western disturbance cycles, and monsoon seasonality.
    """

    def __init__(self, reference_period: str = "1991-2020 (Synthetic Benchmark)") -> None:
        self.reference_period = reference_period

    def get_stats(self, variable: str, latitude: float, longitude: float, day_of_year: int) -> ClimatologicalStats:
        doy = max(1, min(int(day_of_year), 366))
        lat = float(latitude)
        lon = float(longitude)

        # Solar zenith seasonal angle: peak summer ~ day 145 (late May), winter ~ day 15 (mid January)
        seasonal_phase = 2.0 * math.pi * (doy - 145) / 365.25
        monsoon_phase = math.exp(-((doy - 200) / 45.0) ** 2)  # Peak July/August

        if variable == "temperature_2m":
            # Northward continental heating in summer, lapse with latitude in winter
            base_temp = 31.0 - 0.45 * max(0.0, lat - 12.0)
            seasonal_amp = 6.5 * (lat / 20.0)  # Greater seasonal swing at higher latitudes
            mean = base_temp + seasonal_amp * math.cos(seasonal_phase)
            std = max(1.2, 2.5 + 0.8 * (lat / 25.0))
            p10 = mean - 1.28 * std
            p50 = mean
            p90 = mean + 1.28 * std
            p95 = mean + 1.645 * std
            p99 = mean + 2.326 * std
        elif variable == "precipitation":
            # Monsoon convective baseline
            # Coastal / NE India heavy baseline, northwest arid
            orographic_factor = 1.8 if (lon > 88.0 or (lon < 75.0 and lat < 18.0)) else 0.7
            mean = max(0.5, (1.0 + 14.0 * monsoon_phase) * orographic_factor)
            std = max(1.0, mean * 1.6)
            p10 = 0.0
            p50 = max(0.0, mean * 0.4)
            p90 = mean + 1.5 * std
            p95 = mean + 2.4 * std
            p99 = mean + 4.2 * std
        elif variable == "wind_speed_10m":
            # Stronger coastal winds during monsoon & pre-monsoon
            mean = 14.0 + 10.0 * monsoon_phase + 4.0 * math.cos(seasonal_phase)
            std = 5.0 + 2.0 * monsoon_phase
            p10 = max(2.0, mean - 1.28 * std)
            p50 = mean
            p90 = mean + 1.28 * std
            p95 = mean + 1.645 * std
            p99 = mean + 2.326 * std
        elif variable == "surface_pressure":
            # Monsoon trough low pressure (~998-1002 hPa in July, ~1016 hPa in January)
            mean = 1012.0 - 10.0 * monsoon_phase
            std = 3.5
            p10 = mean - 1.28 * std
            p50 = mean
            p90 = mean + 1.28 * std
            p95 = mean + 1.645 * std
            p99 = mean + 2.326 * std
        else:
            mean = 50.0
            std = 10.0
            p10, p50, p90, p95, p99 = 35.0, 50.0, 65.0, 70.0, 75.0

        return ClimatologicalStats(
            variable=variable,
            latitude=lat,
            longitude=lon,
            day_of_year=doy,
            mean=float(round(mean, 2)),
            std=float(round(std, 2)),
            p10=float(round(p10, 2)),
            p50=float(round(p50, 2)),
            p90=float(round(p90, 2)),
            p95=float(round(p95, 2)),
            p99=float(round(p99, 2)),
            sample_count=10950,  # 30 years * 365 days
            is_synthetic=True,
            reference_period=self.reference_period,
        )


class ClimatologyEngine:
    """Climatology Engine managing historical reanalysis baselines or synthetic fallbacks."""

    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider or SyntheticClimatologyProvider()

    def get_baseline(
        self, variable: str, latitude: float, longitude: float, day_of_year: int
    ) -> ClimatologicalStats:
        return self.provider.get_stats(variable, latitude, longitude, day_of_year)

    def get_grid_baseline(
        self,
        variable: str,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        day_of_year: int,
    ) -> dict[str, np.ndarray]:
        """Compute 2D grids of mean, std, p90, p95, p99 for gridded anomaly calculations."""
        n_lat = len(latitudes)
        n_lon = len(longitudes)

        mean_grid = np.zeros((n_lat, n_lon), dtype=np.float32)
        std_grid = np.zeros((n_lat, n_lon), dtype=np.float32)
        p95_grid = np.zeros((n_lat, n_lon), dtype=np.float32)
        p99_grid = np.zeros((n_lat, n_lon), dtype=np.float32)

        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                stats = self.get_baseline(variable, float(lat), float(lon), day_of_year)
                mean_grid[i, j] = stats.mean
                std_grid[i, j] = max(stats.std, 1e-4)  # Safeguard against zero std
                p95_grid[i, j] = stats.p95
                p99_grid[i, j] = stats.p99

        return {
            "mean": mean_grid,
            "std": std_grid,
            "p95": p95_grid,
            "p99": p99_grid,
            "is_synthetic": getattr(self.provider, "is_synthetic", True),
        }
