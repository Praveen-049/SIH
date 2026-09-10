import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anomaly_detector import classify_severity
from main import _geodesic_distance_km
from weather_service import WeatherDataError, _validate_payload


class ScientificContractTests(unittest.TestCase):
    def test_severity_thresholds_are_authoritative(self):
        self.assertEqual(classify_severity(39.99), "NORMAL")
        self.assertEqual(classify_severity(40), "MODERATE")
        self.assertEqual(classify_severity(60), "HIGH")
        self.assertEqual(classify_severity(80), "SEVERE")

    def test_geodesic_distance_uses_latitude(self):
        distance = _geodesic_distance_km({"lat": 22.5, "lon": 89.5}, {"lat": 26, "lon": 93})
        self.assertAlmostEqual(distance, 526.6, delta=0.2)

    def test_provider_payload_requires_consistent_hourly_arrays(self):
        payload = {
            "hourly": {
                "time": ["t1", "t2"],
                "temperature_2m": [1, 2],
                "precipitation": [0, 1],
            },
            "hourly_units": {"temperature_2m": "°C", "precipitation": "mm"},
        }
        _validate_payload(payload)
        payload["hourly"]["precipitation"] = [0]
        with self.assertRaises(WeatherDataError):
            _validate_payload(payload)

    def test_provider_payload_rejects_unexpected_units(self):
        payload = {
            "hourly": {"time": ["t1"], "precipitation": [1]},
            "hourly_units": {"precipitation": "mm/3h"},
        }
        with self.assertRaises(WeatherDataError):
            _validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
