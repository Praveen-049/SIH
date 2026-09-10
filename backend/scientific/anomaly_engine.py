"""Anomaly engine computing standardized anomalies (z-scores), percentiles, and exceedances.

Never destroys raw physical values. Safeguards against division by zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import stats

from .climatology_engine import ClimatologyEngine, ClimatologicalStats


@dataclass
class AnomalyPointResult:
    variable: str
    raw_value: float
    climatological_mean: float
    climatological_std: float
    anomaly: float  # raw - mean
    z_score: float  # (raw - mean) / std
    percentile: float  # 0 to 100
    p95_threshold: float
    p99_threshold: float
    exceeds_p95: bool
    exceeds_p99: bool
    severity: str  # NORMAL, MODERATE, HIGH, SEVERE
    is_synthetic_baseline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "raw_value": round(self.raw_value, 2),
            "climatological_mean": round(self.climatological_mean, 2),
            "climatological_std": round(self.climatological_std, 2),
            "anomaly": round(self.anomaly, 2),
            "z_score": round(self.z_score, 2),
            "percentile": round(self.percentile, 1),
            "p95_threshold": round(self.p95_threshold, 2),
            "p99_threshold": round(self.p99_threshold, 2),
            "exceeds_p95": self.exceeds_p95,
            "exceeds_p99": self.exceeds_p99,
            "severity": self.severity,
            "is_synthetic_baseline": self.is_synthetic_baseline,
        }


class AnomalyEngine:
    """Scientific Anomaly Engine calculating meteorological standardized anomalies."""

    EPSILON = 1e-5

    def __init__(self, climatology_engine: ClimatologyEngine | None = None) -> None:
        self.climatology = climatology_engine or ClimatologyEngine()

    def compute_point_anomaly(
        self,
        variable: str,
        value: float,
        latitude: float,
        longitude: float,
        day_of_year: int,
    ) -> AnomalyPointResult:
        clim = self.climatology.get_baseline(variable, latitude, longitude, day_of_year)
        val = float(value)
        std = max(clim.std, self.EPSILON)

        anomaly = val - clim.mean
        z_score = anomaly / std

        # Percentile calculation under normal approximation or empirical bounds
        # z = -3.0 -> ~0.1%, z = 0 -> 50%, z = +3.0 -> 99.9%
        cdf = stats.norm.cdf(z_score)
        percentile = float(np.clip(cdf * 100.0, 0.01, 99.99))

        exceeds_p95 = val >= clim.p95
        exceeds_p99 = val >= clim.p99

        # Meteorological severity based on standard deviations and percentile exceedances
        if z_score >= 3.0 or exceeds_p99:
            severity = "SEVERE"
        elif z_score >= 2.0 or exceeds_p95:
            severity = "HIGH"
        elif z_score >= 1.28 or percentile >= 90.0:  # Top decile
            severity = "MODERATE"
        else:
            severity = "NORMAL"

        return AnomalyPointResult(
            variable=variable,
            raw_value=val,
            climatological_mean=clim.mean,
            climatological_std=clim.std,
            anomaly=anomaly,
            z_score=z_score,
            percentile=percentile,
            p95_threshold=clim.p95,
            p99_threshold=clim.p99,
            exceeds_p95=exceeds_p95,
            exceeds_p99=exceeds_p99,
            severity=severity,
            is_synthetic_baseline=clim.is_synthetic,
        )

    def compute_gridded_anomaly(
        self,
        variable: str,
        data_grid: np.ndarray,  # 2D (lat, lon)
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        day_of_year: int,
    ) -> dict[str, np.ndarray]:
        """Compute 2D grids of anomaly, z-score, and extreme binary mask."""
        clim_grids = self.climatology.get_grid_baseline(variable, latitudes, longitudes, day_of_year)
        mean_grid = clim_grids["mean"]
        std_grid = np.maximum(clim_grids["std"], self.EPSILON)
        p95_grid = clim_grids["p95"]

        anomaly_grid = data_grid - mean_grid
        z_grid = anomaly_grid / std_grid
        extreme_mask = (data_grid >= p95_grid) | (z_grid >= 2.0)

        return {
            "anomaly": anomaly_grid,
            "z_score": z_grid,
            "extreme_mask": extreme_mask,
            "climatological_mean": mean_grid,
            "climatological_std": std_grid,
            "p95_threshold": p95_grid,
            "is_synthetic": clim_grids["is_synthetic"],
        }
