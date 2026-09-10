"""Spatial event extraction module.

Extracts connected meteorological anomaly objects from 2D gridded fields.
Calculates geodesic centroids, true spherical surface areas (km²), bounding boxes,
and intensity profiles using morphological connected component analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy import ndimage


EARTH_RADIUS_KM = 6371.0088


@dataclass
class WeatherObject:
    """Individual extracted spatial meteorological event object."""
    object_id: int
    timestep_hour: int
    timestamp_iso: str
    variable: str
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    max_intensity: float
    mean_intensity: float
    p95_intensity: float
    bounding_box: dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    polygon_coords: list[list[float]] = field(default_factory=list)  # [[lon, lat], ...] for GeoJSON
    pixel_count: int = 0
    raw_metric: str = "mm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "timestep_hour": self.timestep_hour,
            "timestamp_iso": self.timestamp_iso,
            "variable": self.variable,
            "centroid": {
                "lat": round(self.centroid_lat, 4),
                "lon": round(self.centroid_lon, 4),
            },
            "area_km2": round(self.area_km2, 1),
            "max_intensity": round(self.max_intensity, 2),
            "mean_intensity": round(self.mean_intensity, 2),
            "p95_intensity": round(self.p95_intensity, 2),
            "bounding_box": self.bounding_box,
            "polygon_coords": self.polygon_coords,
            "pixel_count": self.pixel_count,
            "raw_metric": self.raw_metric,
        }


class SpatialAnomalyExtractor:
    """Extracts coherent spatial weather objects from gridded anomaly fields."""

    def __init__(self, min_area_pixels: int = 3) -> None:
        self.min_area_pixels = min_area_pixels

    @staticmethod
    def _cell_area_km2(lat_deg: float, dlat_deg: float, dlon_deg: float) -> float:
        """Calculate spherical surface area of a grid cell on Earth."""
        lat_rad = math.radians(lat_deg)
        dlat_rad = math.radians(dlat_deg)
        dlon_rad = math.radians(dlon_deg)
        return (EARTH_RADIUS_KM ** 2) * math.cos(lat_rad) * dlat_rad * dlon_rad

    def extract_objects(
        self,
        grid_data: np.ndarray,       # (lat, lon) physical values
        anomaly_grid: np.ndarray,    # (lat, lon) anomaly values
        extreme_mask: np.ndarray,    # (lat, lon) boolean
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        timestep_hour: int,
        timestamp_iso: str,
        variable: str = "precipitation",
        metric_unit: str = "mm",
    ) -> list[WeatherObject]:
        """Run morphological connected components and extract weather objects."""
        # 1. Morphological opening to remove 1-pixel isolated noise spikes
        clean_mask = ndimage.binary_opening(extreme_mask, structure=np.ones((2, 2)))

        # 2. Connected component labeling (8-connectivity)
        structure = np.ones((3, 3), dtype=int)
        labeled_grid, num_features = ndimage.label(clean_mask, structure=structure)

        if num_features == 0:
            return []

        dlat = abs(float(latitudes[1] - latitudes[0])) if len(latitudes) > 1 else 0.25
        dlon = abs(float(longitudes[1] - longitudes[0])) if len(longitudes) > 1 else 0.25

        objects: list[WeatherObject] = []

        for feature_id in range(1, num_features + 1):
            coords = np.argwhere(labeled_grid == feature_id)
            pixel_count = len(coords)
            if pixel_count < self.min_area_pixels:
                continue

            lat_indices = coords[:, 0]
            lon_indices = coords[:, 1]

            obj_lats = latitudes[lat_indices]
            obj_lons = longitudes[lon_indices]
            obj_intensities = grid_data[lat_indices, lon_indices]

            # Weighted centroid by intensity to track meteorological center of mass
            weights = np.maximum(obj_intensities, 0.01)
            total_weight = float(np.sum(weights))
            centroid_lat = float(np.sum(obj_lats * weights) / total_weight)
            centroid_lon = float(np.sum(obj_lons * weights) / total_weight)

            # Spherical Area in km²
            total_area_km2 = sum(
                self._cell_area_km2(float(lat), dlat, dlon) for lat in obj_lats
            )

            # Bounding box
            bbox = {
                "min_lat": float(round(obj_lats.min(), 4)),
                "max_lat": float(round(obj_lats.max(), 4)),
                "min_lon": float(round(obj_lons.min(), 4)),
                "max_lon": float(round(obj_lons.max(), 4)),
            }

            # Convex / rough contour polygon around the object footprint
            # Generate simple boundary coordinates for Leaflet display
            min_l, max_l = bbox["min_lat"], bbox["max_lat"]
            min_o, max_o = bbox["min_lon"], bbox["max_lon"]
            polygon_coords = [
                [min_o, min_l],
                [max_o, min_l],
                [max_o, max_l],
                [min_o, max_l],
                [min_o, min_l],
            ]

            objects.append(
                WeatherObject(
                    object_id=feature_id,
                    timestep_hour=timestep_hour,
                    timestamp_iso=timestamp_iso,
                    variable=variable,
                    centroid_lat=centroid_lat,
                    centroid_lon=centroid_lon,
                    area_km2=total_area_km2,
                    max_intensity=float(np.max(obj_intensities)),
                    mean_intensity=float(np.mean(obj_intensities)),
                    p95_intensity=float(np.percentile(obj_intensities, 95.0)),
                    bounding_box=bbox,
                    polygon_coords=polygon_coords,
                    pixel_count=pixel_count,
                    raw_metric=metric_unit,
                )
            )

        # Sort largest area first
        objects.sort(key=lambda o: o.area_km2, reverse=True)
        return objects
