"""Extreme Forecast Index (EFI) Engine.

Implements the standard ECMWF / WMO formulation:
    EFI = (2 / pi) * Integral_0^1 [ (p - F_f(Q_c(p))) / sqrt(p * (1 - p)) ] dp

Calculates tail-weighted divergence between ensemble forecast CDF and climatological CDF.
EFI is bounded in [-1.0, 1.0].
Values > 0.5 denote high probability of an extreme anomaly; > 0.8 denotes very extreme.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class EFIResult:
    efi: float
    variable: str
    valid_time: str
    reference_period: str
    ensemble_size: int
    method: str
    is_synthetic: bool
    sot_90: float | None = None  # Shift of Tails (SOT) 90th percentile index

    def to_dict(self) -> dict[str, Any]:
        return {
            "efi": round(self.efi, 3),
            "variable": self.variable,
            "valid_time": self.valid_time,
            "reference_period": self.reference_period,
            "ensemble_size": self.ensemble_size,
            "method": self.method,
            "is_synthetic": self.is_synthetic,
            "sot_90": round(self.sot_90, 3) if self.sot_90 is not None else None,
        }


class EFIEngine:
    """Extreme Forecast Index engine using discrete numerical quadrature."""

    def __init__(self, quadrature_steps: int = 100) -> None:
        self.quadrature_steps = quadrature_steps
        # Integration points away from exact 0 and 1 to prevent division by zero in sqrt(p(1-p))
        eps = 1e-4
        self.p_points = np.linspace(eps, 1.0 - eps, quadrature_steps)
        self.weights = 1.0 / np.sqrt(self.p_points * (1.0 - self.p_points))
        self.dp = (1.0 - 2 * eps) / (quadrature_steps - 1)

    def calculate_efi(
        self,
        ensemble_forecast_values: np.ndarray | list[float],
        climatological_samples: np.ndarray | list[float] | None = None,
        clim_mean: float | None = None,
        clim_std: float | None = None,
        variable: str = "precipitation",
        valid_time: str = "T+72h",
        reference_period: str = "1991-2020",
        is_synthetic: bool = False,
    ) -> EFIResult:
        """Calculate EFI from an ensemble forecast sample.

        If climatological samples are provided, uses empirical quantile function Q_c(p).
        Otherwise uses Gaussian parametric approximation with clim_mean and clim_std.
        """
        ens = np.asarray(ensemble_forecast_values, dtype=np.float64)
        ens = ens[~np.isnan(ens)]
        if len(ens) == 0:
            return EFIResult(
                efi=0.0,
                variable=variable,
                valid_time=valid_time,
                reference_period=reference_period,
                ensemble_size=0,
                method="undefined_no_members",
                is_synthetic=is_synthetic,
            )

        n_ens = len(ens)

        if climatological_samples is not None and len(climatological_samples) > 10:
            clim = np.asarray(climatological_samples, dtype=np.float64)
            clim = clim[~np.isnan(clim)]
            # Quantiles Q_c(p) from historical distribution
            q_c = np.percentile(clim, self.p_points * 100.0)
        else:
            # Parametric Gaussian quantile approximation
            m = clim_mean if clim_mean is not None else float(np.mean(ens))
            s = max(clim_std if clim_std is not None else float(np.std(ens)), 1e-4)
            from scipy.stats import norm
            q_c = norm.ppf(self.p_points, loc=m, scale=s)

        # Compute empirical forecast CDF F_f(Q_c(p))
        # F_f(x) = count(ens <= x) / n_ens
        # Broadcasting: ens shape (N, 1), q_c shape (1, Steps)
        f_f = np.mean(ens[:, None] <= q_c[None, :], axis=0)

        # Quadrature integration
        integrand = (self.p_points - f_f) * self.weights
        integral = float(np.sum(integrand) * self.dp)
        efi_raw = (2.0 / math.pi) * integral

        # Bounding to [-1, 1]
        efi = float(np.clip(efi_raw, -1.0, 1.0))

        # Shift of Tails (SOT) index at 90th percentile: (Q_ens(90) - Q_clim(90)) / (Q_clim(99) - Q_clim(90))
        q_ens_90 = float(np.percentile(ens, 90.0))
        q_clim_90 = float(q_c[int(self.quadrature_steps * 0.90)])
        q_clim_99 = float(q_c[min(int(self.quadrature_steps * 0.99), self.quadrature_steps - 1)])
        denom = max(q_clim_99 - q_clim_90, 1e-4)
        sot_90 = (q_ens_90 - q_clim_90) / denom

        return EFIResult(
            efi=efi,
            variable=variable,
            valid_time=valid_time,
            reference_period=reference_period,
            ensemble_size=n_ens,
            method="ecmwf_integral_quadrature",
            is_synthetic=is_synthetic,
            sot_90=sot_90,
        )

    def calculate_gridded_efi(
        self,
        ensemble_grid: np.ndarray,  # (n_ens, n_lat, n_lon)
        clim_mean_grid: np.ndarray,  # (n_lat, n_lon)
        clim_std_grid: np.ndarray,   # (n_lat, n_lon)
        variable: str = "precipitation",
        valid_time: str = "T+72h",
        is_synthetic: bool = False,
    ) -> np.ndarray:
        """Compute 2D spatial grid of EFI values across ensemble members."""
        n_ens, n_lat, n_lon = ensemble_grid.shape
        efi_grid = np.zeros((n_lat, n_lon), dtype=np.float32)

        for i in range(n_lat):
            for j in range(n_lon):
                ens_members = ensemble_grid[:, i, j]
                res = self.calculate_efi(
                    ensemble_forecast_values=ens_members,
                    clim_mean=float(clim_mean_grid[i, j]),
                    clim_std=float(clim_std_grid[i, j]),
                    variable=variable,
                    valid_time=valid_time,
                    is_synthetic=is_synthetic,
                )
                efi_grid[i, j] = res.efi

        return efi_grid
