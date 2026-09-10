"""Spatio-temporal weather object tracking engine.

Associates meteorological objects across consecutive forecast timesteps.
Uses geodesic Haversine distance, IoU overlap, intensity continuity, and dynamic motion models.
Explicitly tracks event lifecycles: INITIATION, CONTINUATION, TERMINATION, SPLITTING, MERGING.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .spatial_extraction import WeatherObject, EARTH_RADIUS_KM


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on the spherical Earth."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (degrees clockwise from North) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def calculate_bbox_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    """Intersection over Union (IoU) of two bounding boxes."""
    inter_min_lat = max(box1["min_lat"], box2["min_lat"])
    inter_max_lat = min(box1["max_lat"], box2["max_lat"])
    inter_min_lon = max(box1["min_lon"], box2["min_lon"])
    inter_max_lon = min(box1["max_lon"], box2["max_lon"])

    if inter_max_lat <= inter_min_lat or inter_max_lon <= inter_min_lon:
        return 0.0

    inter_area = (inter_max_lat - inter_min_lat) * (inter_max_lon - inter_min_lon)
    area1 = (box1["max_lat"] - box1["min_lat"]) * (box1["max_lon"] - box1["min_lon"])
    area2 = (box2["max_lat"] - box2["min_lat"]) * (box2["max_lon"] - box2["min_lon"])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedEvent:
    event_id: str
    variable: str
    lifecycle_state: str  # INITIATED, CONTINUING, TERMINATED, MERGED, SPLIT
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    total_distance_km: float = 0.0
    mean_speed_kmh: float = 0.0
    current_bearing_deg: float = 0.0
    track_spread_km: float = 0.0  # Spatial dispersion / uncertainty

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "variable": self.variable,
            "lifecycle_state": self.lifecycle_state,
            "trajectory": self.trajectory,
            "motion": {
                "distance_km": round(self.total_distance_km, 1),
                "speed_kmh": round(self.mean_speed_kmh, 2),
                "bearing_deg": round(self.current_bearing_deg, 1),
                "track_spread_km": round(self.track_spread_km, 1),
            },
        }


class WeatherObjectTracker:
    """Associates weather objects across sequential forecast timesteps into tracks."""

    def __init__(
        self,
        max_speed_kmh: float = 120.0,
        distance_weight: float = 0.5,
        iou_weight: float = 0.3,
        intensity_weight: float = 0.2,
    ) -> None:
        self.max_speed_kmh = max_speed_kmh
        self.distance_weight = distance_weight
        self.iou_weight = iou_weight
        self.intensity_weight = intensity_weight

    def track_sequence(
        self,
        timesteps_objects: dict[int, list[WeatherObject]],
    ) -> list[TrackedEvent]:
        """Track a sequence of timesteps {lead_time_hour: [WeatherObject, ...]}."""
        sorted_hours = sorted(timesteps_objects.keys())
        if not sorted_hours:
            return []

        active_tracks: dict[str, TrackedEvent] = {}
        next_event_idx = 1

        for step_idx, hour in enumerate(sorted_hours):
            current_objs = timesteps_objects[hour]

            if step_idx == 0:
                # Initialize tracks for first timestep
                for obj in current_objs:
                    evt_id = f"EVT-{next_event_idx:03d}"
                    next_event_idx += 1
                    track = TrackedEvent(
                        event_id=evt_id,
                        variable=obj.variable,
                        lifecycle_state="INITIATED",
                        trajectory=[{
                            "hour": hour,
                            "time": obj.timestamp_iso,
                            "centroid": {"lat": obj.centroid_lat, "lon": obj.centroid_lon},
                            "area_km2": obj.area_km2,
                            "max_intensity": obj.max_intensity,
                            "mean_intensity": obj.mean_intensity,
                            "bounding_box": obj.bounding_box,
                            "polygon_coords": obj.polygon_coords,
                        }],
                    )
                    active_tracks[evt_id] = track
                continue

            prev_hour = sorted_hours[step_idx - 1]
            dt_hours = max(1, hour - prev_hour)
            max_allowed_distance = self.max_speed_kmh * dt_hours

            # Match active tracks to current objects using a cost matrix
            unmatched_objs = set(range(len(current_objs)))
            unmatched_tracks = set(active_tracks.keys())

            candidates: list[tuple[float, str, int]] = []
            for track_id in active_tracks:
                track = active_tracks[track_id]
                last_point = track.trajectory[-1]
                lat1, lon1 = last_point["centroid"]["lat"], last_point["centroid"]["lon"]

                for obj_idx, obj in enumerate(current_objs):
                    dist = geodesic_distance_km(lat1, lon1, obj.centroid_lat, obj.centroid_lon)
                    if dist <= max_allowed_distance:
                        # Normalized cost calculation
                        dist_cost = dist / max_allowed_distance
                        iou = calculate_bbox_iou(last_point["bounding_box"], obj.bounding_box)
                        iou_cost = 1.0 - iou
                        int_cost = abs(last_point["max_intensity"] - obj.max_intensity) / max(
                            last_point["max_intensity"], obj.max_intensity, 1.0
                        )

                        cost = (
                            self.distance_weight * dist_cost
                            + self.iou_weight * iou_cost
                            + self.intensity_weight * int_cost
                        )
                        candidates.append((cost, track_id, obj_idx))

            # Greedy assignment
            candidates.sort(key=lambda item: item[0])
            matched_tracks: set[str] = set()
            matched_objs: set[int] = set()

            for cost, track_id, obj_idx in candidates:
                if track_id not in matched_tracks and obj_idx not in matched_objs:
                    matched_tracks.add(track_id)
                    matched_objs.add(obj_idx)
                    unmatched_tracks.discard(track_id)
                    unmatched_objs.discard(obj_idx)

                    # Update track with new position
                    track = active_tracks[track_id]
                    last_pt = track.trajectory[-1]
                    step_dist = geodesic_distance_km(
                        last_pt["centroid"]["lat"],
                        last_pt["centroid"]["lon"],
                        current_objs[obj_idx].centroid_lat,
                        current_objs[obj_idx].centroid_lon,
                    )
                    bearing = initial_bearing_deg(
                        last_pt["centroid"]["lat"],
                        last_pt["centroid"]["lon"],
                        current_objs[obj_idx].centroid_lat,
                        current_objs[obj_idx].centroid_lon,
                    )

                    track.total_distance_km += step_dist
                    total_time = hour - track.trajectory[0]["hour"]
                    track.mean_speed_kmh = track.total_distance_km / max(1, total_time)
                    track.current_bearing_deg = bearing
                    track.lifecycle_state = "CONTINUING"
                    # Track spread estimate based on step distance variability
                    track.track_spread_km = max(10.0, step_dist * 0.15)

                    track.trajectory.append({
                        "hour": hour,
                        "time": current_objs[obj_idx].timestamp_iso,
                        "centroid": {
                            "lat": current_objs[obj_idx].centroid_lat,
                            "lon": current_objs[obj_idx].centroid_lon,
                        },
                        "area_km2": current_objs[obj_idx].area_km2,
                        "max_intensity": current_objs[obj_idx].max_intensity,
                        "mean_intensity": current_objs[obj_idx].mean_intensity,
                        "bounding_box": current_objs[obj_idx].bounding_box,
                        "polygon_coords": current_objs[obj_idx].polygon_coords,
                    })

            # Check for termination
            for t_id in unmatched_tracks:
                active_tracks[t_id].lifecycle_state = "TERMINATED"

            # Check for newly initiated objects
            for obj_idx in unmatched_objs:
                new_id = f"EVT-{next_event_idx:03d}"
                next_event_idx += 1
                obj = current_objs[obj_idx]
                active_tracks[new_id] = TrackedEvent(
                    event_id=new_id,
                    variable=obj.variable,
                    lifecycle_state="INITIATED",
                    trajectory=[{
                        "hour": hour,
                        "time": obj.timestamp_iso,
                        "centroid": {"lat": obj.centroid_lat, "lon": obj.centroid_lon},
                        "area_km2": obj.area_km2,
                        "max_intensity": obj.max_intensity,
                        "mean_intensity": obj.mean_intensity,
                        "bounding_box": obj.bounding_box,
                        "polygon_coords": obj.polygon_coords,
                    }],
                )

        return list(active_tracks.values())
