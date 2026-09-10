"""Data validation module ensuring physical, spatial, and temporal consistency.

Strictly rejects malformed data, coordinates, NaN/inf violations, or out-of-order dimensions.
Attaches cryptographic-like provenance metadata to every dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .data_provider import StandardizedDataset


class ValidationError(ValueError):
    """Raised when meteorological data fails validation standards."""


STANDARD_VARIABLE_UNITS = {
    "temperature_2m": {"°C", "K"},
    "precipitation": {"mm", "kg m-2", "mm/h"},
    "wind_speed_10m": {"km/h", "m/s", "knots"},
    "wind_direction_10m": {"°", "deg", "degrees"},
    "surface_pressure": {"hPa", "Pa", "bar"},
    "relative_humidity_2m": {"%", "fraction"},
    "geopotential_500hPa": {"m2 s-2", "gpm", "m"},
}

VARIABLE_PHYSICAL_BOUNDS = {
    "temperature_2m": (-90.0, 65.0),       # °C
    "precipitation": (0.0, 2000.0),        # mm
    "wind_speed_10m": (0.0, 450.0),        # km/h
    "wind_direction_10m": (0.0, 360.0),    # degrees
    "surface_pressure": (800.0, 1100.0),   # hPa
    "relative_humidity_2m": (0.0, 100.0),  # %
}


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    provenance: dict[str, Any]


class DataValidator:
    """Scientific validator for meteorological datasets."""

    @classmethod
    def validate(cls, dataset: StandardizedDataset) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Coordinate checks
        lats = dataset.latitudes
        lons = dataset.longitudes

        if len(lats) == 0:
            errors.append("Latitude array is empty.")
        elif np.any(np.isnan(lats)) or np.any(np.isinf(lats)):
            errors.append("Latitude coordinates contain NaN or Inf values.")
        elif np.any(lats < -90.0) or np.any(lats > 90.0):
            errors.append(f"Latitude out of bounds [-90, 90]: min={lats.min()}, max={lats.max()}")

        if len(lons) == 0:
            errors.append("Longitude array is empty.")
        elif np.any(np.isnan(lons)) or np.any(np.isinf(lons)):
            errors.append("Longitude coordinates contain NaN or Inf values.")
        elif np.any(lons < -180.0) or np.any(lons > 180.0):
            # Check if 0..360 convention
            if np.all(lons >= 0.0) and np.all(lons <= 360.0):
                warnings.append("Longitudes use 0..360 convention instead of -180..180.")
            else:
                errors.append(f"Longitude out of bounds [-180, 180]: min={lons.min()}, max={lons.max()}")

        # Monotonicity check
        if len(lats) > 1 and not (np.all(np.diff(lats) > 0) or np.all(np.diff(lats) < 0)):
            warnings.append("Latitudes are not strictly monotonic.")
        if len(lons) > 1 and not np.all(np.diff(lons) > 0):
            warnings.append("Longitudes are not strictly monotonically increasing.")

        # 2. Temporal checks
        leads = dataset.forecast_lead_times_hours
        if len(leads) == 0:
            errors.append("Forecast lead times list is empty.")
        if len(leads) != len(set(leads)):
            errors.append("Duplicate forecast lead times found.")
        if leads != sorted(leads):
            errors.append("Forecast lead times are not strictly ascending.")

        # 3. Variable dimensions and physical validity
        for var_name, data in dataset.variables.items():
            if not isinstance(data, np.ndarray):
                errors.append(f"Variable {var_name} must be a numpy.ndarray.")
                continue

            if np.any(np.isnan(data)):
                errors.append(f"Variable {var_name} contains NaN values.")
            if np.any(np.isinf(data)):
                errors.append(f"Variable {var_name} contains Inf values.")

            # Check dimensions match
            if data.shape[0] != len(leads):
                errors.append(f"Variable {var_name} lead-time dimension {data.shape[0]} != {len(leads)}")

            if data.ndim >= 3:
                # Lat/lon should be last two dimensions
                if data.shape[-2] != len(lats) or data.shape[-1] != len(lons):
                    errors.append(
                        f"Variable {var_name} spatial shape ({data.shape[-2]}, {data.shape[-1]}) "
                        f"does not match coordinates ({len(lats)}, {len(lons)})"
                    )

            # Check units
            unit = dataset.units.get(var_name)
            if not unit:
                warnings.append(f"Missing unit specification for variable {var_name}")
            elif var_name in STANDARD_VARIABLE_UNITS:
                valid_units = STANDARD_VARIABLE_UNITS[var_name]
                if unit not in valid_units:
                    errors.append(f"Variable {var_name} unit '{unit}' not in accepted {valid_units}")

            # Physical bounds check
            if var_name in VARIABLE_PHYSICAL_BOUNDS:
                lower, upper = VARIABLE_PHYSICAL_BOUNDS[var_name]
                d_min, d_max = float(np.min(data)), float(np.max(data))
                if d_min < lower or d_max > upper:
                    warnings.append(
                        f"Variable {var_name} values [{d_min:.2f}, {d_max:.2f}] exceed standard meteorological bounds [{lower}, {upper}]"
                    )

        # 4. Generate provenance
        provenance_str = (
            f"{dataset.provider}|{dataset.dataset_name}|{dataset.version}|"
            f"{dataset.initialization_time}|{dataset.data_kind}|{len(dataset.variables)}"
        )
        checksum = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()[:16]

        provenance = {
            "provider": dataset.provider,
            "dataset": dataset.dataset_name,
            "version": dataset.version,
            "data_kind": dataset.data_kind,
            "initialization_time": dataset.initialization_time,
            "forecast_horizon_hours": max(leads) if leads else 0,
            "spatial_resolution_deg": float(np.round(abs(lats[1] - lats[0]), 4)) if len(lats) > 1 else 0.25,
            "ensemble_members": dataset.ensemble_size,
            "variables": list(dataset.variables.keys()),
            "units": dataset.units,
            "provenance_hash": checksum,
            "validated": len(errors) == 0,
        }

        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            provenance=provenance,
        )
