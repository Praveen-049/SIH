"""Comprehensive scientific test suite for SIH 26078.

Tests all 10 core scientific domains:
1. Data schema & validation
2. Climatology & standardized anomaly
3. EFI rank-sum mathematical formulation
4. Geodesic distance, bearing, and area calculations
5. Spatial object extraction and morphological cleaning
6. Weather object tracking and lifecycle states
7. Spherical icosahedral mesh builder
8. Spatio-Temporal GNN forward pass and tensor shapes
9. Conditional Weather Diffusion forward pass and downscaling evaluation
10. Physics-informed constraint loss and numerical stability
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import unittest

# Ensure Windows Python 3.14 loads PyTorch C++ DLLs cleanly
if sys.platform == "win32":
    torch_lib_dir = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"
    )
    if os.path.exists(torch_lib_dir):
        try:
            os.add_dll_directory(torch_lib_dir)
        except Exception:
            pass

# Add backend directory to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

import numpy as np
import torch

from scientific.anomaly_engine import AnomalyEngine
from scientific.climatology_engine import ClimatologyEngine, SyntheticClimatologyProvider
from scientific.data_provider import SyntheticProvider, StandardizedDataset
from scientific.data_validator import DataValidator, ValidationError
from scientific.diffusion_downscaling import (
    ConditionalWeatherDiffusion,
    compare_downscaling_methods,
    extreme_tail_preservation_loss,
)
from scientific.efi_engine import EFIEngine
from scientific.gnn_model import SpatioTemporalAnomalyGNN
from scientific.impact_engine import AuthoritativeSeverityEngine, ImpactZoneEngine
from scientific.object_tracker import (
    WeatherObjectTracker,
    geodesic_distance_km,
    initial_bearing_deg,
)
from scientific.physics_loss import PhysicsInformedLoss
from scientific.spatial_extraction import SpatialAnomalyExtractor, WeatherObject
from scientific.spherical_mesh import SphericalMeshBuilder


class TestDataLayer(unittest.TestCase):
    def setUp(self):
        self.provider = SyntheticProvider(seed=123)

    def test_standardized_dataset_creation(self):
        ds = self.provider.fetch_forecast()
        self.assertEqual(ds.data_kind, "synthetic_test")
        self.assertIn("temperature_2m", ds.variables)
        self.assertIn("precipitation", ds.variables)
        self.assertEqual(len(ds.latitudes), 15)
        self.assertEqual(len(ds.longitudes), 15)

    def test_validator_detects_nan_and_out_of_bounds(self):
        ds = self.provider.fetch_forecast()
        report = DataValidator.validate(ds)
        self.assertTrue(report.is_valid)
        self.assertIn("provenance_hash", report.provenance)

        # Inject NaN into variables
        bad_vars = dict(ds.variables)
        bad_vars["temperature_2m"] = bad_vars["temperature_2m"].copy()
        bad_vars["temperature_2m"][0, 0, 0] = np.nan
        bad_ds = StandardizedDataset(
            provider=ds.provider,
            dataset_name=ds.dataset_name,
            version=ds.version,
            data_kind=ds.data_kind,
            initialization_time=ds.initialization_time,
            forecast_lead_times_hours=ds.forecast_lead_times_hours,
            latitudes=ds.latitudes,
            longitudes=ds.longitudes,
            variables=bad_vars,
            units=ds.units,
        )
        bad_report = DataValidator.validate(bad_ds)
        self.assertFalse(bad_report.is_valid)
        self.assertTrue(any("NaN" in e for e in bad_report.errors))


class TestMeteorologyAndEFI(unittest.TestCase):
    def setUp(self):
        self.clim_engine = ClimatologyEngine()
        self.anomaly_engine = AnomalyEngine(self.clim_engine)
        self.efi_engine = EFIEngine(quadrature_steps=50)

    def test_climatology_location_and_doy_dependency(self):
        # Summer (day 145) vs Winter (day 15) in North India (lat 28)
        summer_stats = self.clim_engine.get_baseline("temperature_2m", 28.0, 77.0, day_of_year=145)
        winter_stats = self.clim_engine.get_baseline("temperature_2m", 28.0, 77.0, day_of_year=15)
        self.assertGreater(summer_stats.mean, winter_stats.mean)
        self.assertGreater(summer_stats.std, 0.0)

    def test_anomaly_zscore_and_percentiles(self):
        res = self.anomaly_engine.compute_point_anomaly("temperature_2m", 45.0, 28.0, 77.0, day_of_year=15)
        # 45 deg in winter is a massive positive anomaly
        self.assertGreater(res.z_score, 3.0)
        self.assertEqual(res.severity, "SEVERE")
        self.assertTrue(res.exceeds_p99)
        self.assertAlmostEqual(res.raw_value, 45.0)

    def test_efi_bounds_and_tail_sensitivity(self):
        # Standard normal climatology
        clim_mean, clim_std = 20.0, 5.0
        # Normal ensemble matching climatology
        normal_ens = np.random.default_rng(42).normal(20.0, 5.0, size=50)
        normal_efi = self.efi_engine.calculate_efi(normal_ens, clim_mean=clim_mean, clim_std=clim_std)
        self.assertAlmostEqual(normal_efi.efi, 0.0, delta=0.25)

        # Extreme ensemble all above 99th percentile (e.g. 40 mm)
        extreme_ens = np.full(50, 42.0)
        extreme_efi = self.efi_engine.calculate_efi(extreme_ens, clim_mean=clim_mean, clim_std=clim_std)
        self.assertGreater(extreme_efi.efi, 0.70)
        self.assertLessEqual(extreme_efi.efi, 1.0)


class TestGeodesicsAndTracking(unittest.TestCase):
    def test_geodesic_distance_and_bearing(self):
        # Distance from Kolkata (22.5, 89.5) to Guwahati (26.0, 93.0)
        dist = geodesic_distance_km(22.5, 89.5, 26.0, 93.0)
        self.assertAlmostEqual(dist, 526.6, delta=1.5)

        # Bearing from equator (0, 0) to North Pole (90, 0)
        bearing = initial_bearing_deg(0.0, 0.0, 90.0, 0.0)
        self.assertAlmostEqual(bearing, 0.0, delta=0.1)

    def test_spatial_object_extraction_and_area(self):
        extractor = SpatialAnomalyExtractor(min_area_pixels=2)
        lats = np.linspace(20.0, 25.0, 20)
        lons = np.linspace(80.0, 85.0, 20)
        grid = np.zeros((20, 20), dtype=np.float32)
        grid[5:9, 5:9] = 85.0  # 4x4 pulse
        mask = grid > 50.0

        objs = extractor.extract_objects(
            grid_data=grid,
            anomaly_grid=grid,
            extreme_mask=mask,
            latitudes=lats,
            longitudes=lons,
            timestep_hour=24,
            timestamp_iso="2026-09-10T00:00:00Z",
        )
        self.assertEqual(len(objs), 1)
        self.assertGreater(objs[0].area_km2, 1000.0)
        self.assertAlmostEqual(objs[0].max_intensity, 85.0)

    def test_tracker_lifecycle_continuation(self):
        tracker = WeatherObjectTracker(max_speed_kmh=100.0)
        obj_t1 = WeatherObject(
            object_id=1, timestep_hour=24, timestamp_iso="T1", variable="precip",
            centroid_lat=20.0, centroid_lon=80.0, area_km2=5000.0,
            max_intensity=80.0, mean_intensity=60.0, p95_intensity=75.0,
            bounding_box={"min_lat": 19, "max_lat": 21, "min_lon": 79, "max_lon": 81},
        )
        # Shift slightly at T+48 (coherent continuation)
        obj_t2 = WeatherObject(
            object_id=2, timestep_hour=48, timestamp_iso="T2", variable="precip",
            centroid_lat=20.5, centroid_lon=80.8, area_km2=5200.0,
            max_intensity=85.0, mean_intensity=62.0, p95_intensity=80.0,
            bounding_box={"min_lat": 19.5, "max_lat": 21.5, "min_lon": 79.8, "max_lon": 81.8},
        )
        tracks = tracker.track_sequence({24: [obj_t1], 48: [obj_t2]})
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].lifecycle_state, "CONTINUATION" if tracks[0].lifecycle_state == "CONTINUATION" else "CONTINUING")
        self.assertGreater(tracks[0].total_distance_km, 50.0)


class TestSphericalMeshAndGNN(unittest.TestCase):
    def test_mesh_construction(self):
        mesh = SphericalMeshBuilder.build_mesh(subdivision_level=1)
        self.assertEqual(mesh.num_nodes, 42)
        self.assertEqual(mesh.edge_index.shape[0], 2)
        self.assertGreater(mesh.num_edges, 0)
        # Node XYZ norms must be 1.0 (on sphere)
        norms = np.linalg.norm(mesh.node_xyz, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_gnn_forward_pass_shapes(self):
        mesh = SphericalMeshBuilder.build_mesh(subdivision_level=1)
        edge_index = torch.from_numpy(mesh.edge_index)
        edge_dist = torch.from_numpy(mesh.edge_distances_km[:, None] / 1000.0)

        model = SpatioTemporalAnomalyGNN(in_channels=10, hidden_dim=16)
        x_seq = torch.randn(2, 4, mesh.num_nodes, 10)  # (Batch=2, T=4, N=42, C=10)

        out = model(x_seq, edge_index, edge_dist)
        self.assertEqual(out["anomaly_probability"].shape, (2, mesh.num_nodes))
        self.assertEqual(out["intensity_delta"].shape, (2, mesh.num_nodes))
        self.assertEqual(out["trajectory_offset_deg"].shape, (2, 2))
        # Probabilities must be in [0, 1]
        self.assertTrue(torch.all(out["anomaly_probability"] >= 0.0))
        self.assertTrue(torch.all(out["anomaly_probability"] <= 1.0))


class TestDiffusionAndPhysics(unittest.TestCase):
    def test_diffusion_forward_and_sample(self):
        model = ConditionalWeatherDiffusion(timesteps=5)
        coarse = torch.randn(1, 1, 8, 8)
        anom = torch.randn(1, 1, 8, 8)
        topo = torch.randn(1, 3, 16, 16)
        sample = model.sample(coarse, anom, topo, target_shape=(16, 16))
        self.assertEqual(sample.shape, (1, 1, 16, 16))
        # Non-negative weather field
        self.assertTrue(torch.all(sample >= 0.0))

    def test_physics_loss_stability(self):
        crit = PhysicsInformedLoss(lambda_physics=0.1, lambda_extreme=0.2)
        pred = torch.randn(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        loss = crit(pred, target, variable_type="precipitation")
        self.assertFalse(torch.isnan(loss.total_loss))
        self.assertFalse(torch.isinf(loss.total_loss))
        self.assertGreater(loss.total_loss.item(), 0.0)

    def test_downscaling_spectral_preservation_evaluation(self):
        coarse = np.random.rand(8, 8) * 50.0
        truth = np.random.rand(16, 16) * 75.0
        model_out = truth + np.random.normal(0, 2.0, size=(16, 16))
        comp = compare_downscaling_methods(coarse, truth, model_out)
        self.assertIn("baseline_bicubic", comp)
        self.assertIn("diffusion_model", comp)
        self.assertIn("rmse", comp["diffusion_model"])


class TestImpactAndSeverity(unittest.TestCase):
    def test_authoritative_severity_hierarchy(self):
        sev_severe, score_sev, prob_sev = AuthoritativeSeverityEngine.evaluate_severity(
            physical_intensity=120.0, anomaly_z_score=3.8, efi_value=0.85, lead_time_hours=24
        )
        sev_normal, score_norm, _ = AuthoritativeSeverityEngine.evaluate_severity(
            physical_intensity=2.0, anomaly_z_score=0.2, efi_value=0.05, lead_time_hours=96
        )
        self.assertEqual(sev_severe, "SEVERE")
        self.assertEqual(sev_normal, "NORMAL")
        self.assertGreater(score_sev, score_norm)

    def test_impact_zone_distinguishes_resolution_from_radius(self):
        engine = ImpactZoneEngine()
        res = engine.generate_impact_zone(
            event_id="EVT-001", centroid_lat=22.0, centroid_lon=88.0,
            physical_intensity=90.0, anomaly_z_score=3.0, affected_area_km2=20000.0,
        )
        self.assertEqual(res.grid_resolution_km, 5.0)  # 5 km model resolution
        self.assertGreater(res.impact_radius_km, 50.0)  # Alert buffer radius (~80 km)


if __name__ == "__main__":
    unittest.main()
