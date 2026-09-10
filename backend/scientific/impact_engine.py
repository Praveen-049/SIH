"""Impact zone modeling and authoritative severity engine.

Clearly distinguishes 5 KM MODEL GRID RESOLUTION from a 5 KM ALERT RADIUS.
Provides exposure adapters (population, infrastructure, agriculture) and unified multi-factorial severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .spatial_extraction import EARTH_RADIUS_KM


@dataclass
class ExposureAssessment:
    estimated_population: int
    critical_infrastructure: list[str]
    agricultural_risk_level: str
    transport_disruption_risk: str
    data_source: str = "SYNTHETIC_EXPOSURE_ADAPTER"


@dataclass
class ImpactZoneResult:
    event_id: str
    severity: str
    lead_time_hours: int
    grid_resolution_km: float  # 5.0 km
    impact_area_km2: float
    impact_radius_km: float  # Equivalent circle radius for alert zone
    centroid: dict[str, float]
    uncertainty: dict[str, Any]
    multi_factorial_score: float  # 0 to 100
    exceedance_probability: float  # 0 to 1.0
    exposure: ExposureAssessment
    impact_polygons: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "severity": self.severity,
            "lead_time_hours": self.lead_time_hours,
            "grid_resolution_km": self.grid_resolution_km,
            "impact_area_km2": round(self.impact_area_km2, 1),
            "impact_radius_km": round(self.impact_radius_km, 1),
            "centroid": self.centroid,
            "uncertainty": self.uncertainty,
            "multi_factorial_score": round(self.multi_factorial_score, 1),
            "exceedance_probability": round(self.exceedance_probability, 3),
            "exposure": {
                "estimated_population": self.exposure.estimated_population,
                "critical_infrastructure": self.exposure.critical_infrastructure,
                "agricultural_risk_level": self.exposure.agricultural_risk_level,
                "transport_disruption_risk": self.exposure.transport_disruption_risk,
                "data_source": self.exposure.data_source,
            },
            "impact_polygons": self.impact_polygons,
        }


class AuthoritativeSeverityEngine:
    """Authoritative unified severity engine combining physical, statistical, and ensemble signals."""

    @classmethod
    def evaluate_severity(
        cls,
        physical_intensity: float,
        anomaly_z_score: float,
        efi_value: float | None = None,
        exceedance_prob: float | None = None,
        lead_time_hours: int = 48,
        affected_area_km2: float = 1000.0,
    ) -> tuple[str, float, float]:
        """Compute authoritative severity band, composite score (0-100), and probability (0-1)."""
        # 1. Anomaly component (0 to 40 pts)
        z_norm = max(0.0, min(float(anomaly_z_score), 4.5)) / 4.5
        score_anomaly = z_norm * 40.0

        # 2. EFI component (0 to 25 pts)
        if efi_value is not None:
            efi_clamped = max(0.0, min(float(efi_value), 1.0))
            score_efi = efi_clamped * 25.0
        else:
            score_efi = z_norm * 25.0  # Proxy if EFI unavailable

        # 3. Exceedance Probability component (0 to 20 pts)
        prob = (
            float(exceedance_prob)
            if exceedance_prob is not None
            else min(1.0, max(0.0, z_norm * 0.9))
        )
        score_prob = prob * 20.0

        # 4. Lead-time decay / urgency factor (0 to 15 pts)
        # Shorter lead times have higher immediate operational urgency
        urgency = max(0.2, 1.0 - (min(lead_time_hours, 120) / 140.0))
        score_urgency = urgency * 15.0

        composite_score = float(score_anomaly + score_efi + score_prob + score_urgency)

        # Scale for large spatial extent
        if affected_area_km2 > 25000:
            composite_score = min(100.0, composite_score * 1.1)

        # Authoritative classification thresholds
        if composite_score >= 75.0 or (anomaly_z_score >= 3.5 and prob >= 0.7):
            severity = "SEVERE"
        elif composite_score >= 55.0 or (anomaly_z_score >= 2.0 and prob >= 0.5):
            severity = "HIGH"
        elif composite_score >= 35.0:
            severity = "MODERATE"
        else:
            severity = "NORMAL"

        return severity, round(composite_score, 1), round(prob, 3)


class ImpactZoneEngine:
    """Calculates 5 km impact grids, exposure overlays, and authoritative alert parameters."""

    def __init__(self) -> None:
        self.severity_engine = AuthoritativeSeverityEngine()

    def generate_impact_zone(
        self,
        event_id: str,
        centroid_lat: float,
        centroid_lon: float,
        physical_intensity: float,
        anomaly_z_score: float,
        affected_area_km2: float,
        lead_time_hours: int = 48,
        efi_value: float | None = None,
        exceedance_prob: float | None = None,
        track_spread_km: float = 15.0,
    ) -> ImpactZoneResult:
        severity, score, prob = self.severity_engine.evaluate_severity(
            physical_intensity=physical_intensity,
            anomaly_z_score=anomaly_z_score,
            efi_value=efi_value,
            exceedance_prob=exceedance_prob,
            lead_time_hours=lead_time_hours,
            affected_area_km2=affected_area_km2,
        )

        # Equivalent circular alert radius (km)
        equiv_radius_km = math.sqrt(max(affected_area_km2, 78.5) / math.pi)

        # Exposure proxy assessment (clearly labeled as adapter)
        pop_density_proxy = 350.0  # persons/km² approx mean for rural/semi-urban South Asia
        est_pop = int(affected_area_km2 * pop_density_proxy)

        infra: list[str] = []
        if severity in ("HIGH", "SEVERE"):
            infra.extend(["Substations at Flood Risk", "State Highway Corridors", "District Hospitals"])
        if severity == "SEVERE":
            infra.extend(["Railway Bridge Scour Watch", "Cell Tower Backup Power Needed"])

        exposure = ExposureAssessment(
            estimated_population=est_pop,
            critical_infrastructure=infra,
            agricultural_risk_level="HIGH" if severity in ("HIGH", "SEVERE") else "MODERATE",
            transport_disruption_risk="SEVERE" if severity == "SEVERE" else "MODERATE",
            data_source="SYNTHETIC_EXPOSURE_ADAPTER (Requires official Census/Bhuvan integration)",
        )

        uncertainty = {
            "centroid_uncertainty_km": round(track_spread_km, 1),
            "lead_time_hours": lead_time_hours,
            "confidence_band": "HIGH" if lead_time_hours <= 48 else "MODERATE" if lead_time_hours <= 96 else "LOW",
            "ensemble_spread_detected": efi_value is not None,
        }

        # 5 km high-resolution impact polygon ring (approx circular footprint)
        num_pts = 16
        poly_coords: list[list[float]] = []
        for p in range(num_pts + 1):
            theta = 2.0 * math.pi * (p % num_pts) / num_pts
            # 1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(lat)
            dlat = (equiv_radius_km * math.sin(theta)) / 111.0
            dlon = (equiv_radius_km * math.cos(theta)) / (111.0 * math.cos(math.radians(centroid_lat)))
            poly_coords.append([round(centroid_lon + dlon, 4), round(centroid_lat + dlat, 4)])

        return ImpactZoneResult(
            event_id=event_id,
            severity=severity,
            lead_time_hours=lead_time_hours,
            grid_resolution_km=5.0,  # 5 km grid resolution
            impact_area_km2=affected_area_km2,
            impact_radius_km=equiv_radius_km,
            centroid={"lat": round(centroid_lat, 4), "lon": round(centroid_lon, 4)},
            uncertainty=uncertainty,
            multi_factorial_score=score,
            exceedance_probability=prob,
            exposure=exposure,
            impact_polygons=[{
                "type": "Polygon",
                "coordinates": [poly_coords],
                "properties": {
                    "grid_resolution_km": 5.0,
                    "severity": severity,
                    "radius_km": round(equiv_radius_km, 1),
                },
            }],
        )
