"""Small dependency-free Vercel API adapter for the hosted dashboard."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import urlparse

EVENTS = [{"id": 10, "type": "Heat anomaly", "risk": "HIGH", "anomaly_score": 72.4, "confidence": 84, "forecast_start": 24, "forecast_end": 96, "movement": "NORTH-EAST", "bearing_deg": 42, "speed_kmh": 3.7, "temperature_anomaly": 6.1, "peak_rainfall": 1.2, "affected_area_km2": 12400, "start": {"lat": 22.7, "lon": 80.1}, "latest": {"lat": 26.0, "lon": 83.0}, "impact": ["Heat stress", "Water demand"], "trajectory": [{"hour": 24, "lat": 22.7, "lon": 80.1}, {"hour": 48, "lat": 23.8, "lon": 81.0}, {"hour": 72, "lat": 24.9, "lon": 82.0}, {"hour": 96, "lat": 26.0, "lon": 83.0}]}, {"id": 11, "type": "Extreme rainfall", "risk": "SEVERE", "anomaly_score": 87.6, "confidence": 91, "forecast_start": 24, "forecast_end": 72, "movement": "NORTH-WEST", "bearing_deg": 318, "speed_kmh": 5.1, "peak_rainfall": 168.4, "rainfall_anomaly": 142.0, "affected_area_km2": 18600, "start": {"lat": 16.1, "lon": 85.5}, "latest": {"lat": 18.4, "lon": 83.1}, "impact": ["Flash flooding", "Landslide risk"], "trajectory": [{"hour": 24, "lat": 16.1, "lon": 85.5}, {"hour": 48, "lat": 17.2, "lon": 84.3}, {"hour": 72, "lat": 18.4, "lon": 83.1}]}]

def respond(h, status, body):
    data = json.dumps(body).encode()
    h.send_response(status); h.send_header("Content-Type", "application/json"); h.send_header("Cache-Control", "no-store"); h.send_header("Access-Control-Allow-Origin", "*"); h.send_header("Content-Length", str(len(data))); h.end_headers(); h.wfile.write(data)

def get_event(event_id): return next((e for e in EVENTS if str(e["id"]) == str(event_id)), None)

class handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"); self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.end_headers()
    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")
        if path.endswith("/events"): return respond(self, 200, {"events": EVENTS, "source": "DEMO", "updated_at": datetime.now(timezone.utc).isoformat(), "baseline": "Hosted demo fixture; deploy backend for live NWP"})
        if path.endswith("/health"): return respond(self, 200, {"status": "online", "api": True, "data": True, "mode": "DEMO"})
        if path.endswith("/model-status"): return respond(self, 200, {"compute_device": "Vercel serverless", "subsystems": {"climatological_baseline": {"method": "Demo"}, "efi_engine": {"status": "DEMO READY"}, "spatiotemporal_gnn": {"status": "DEMO READY"}, "conditional_diffusion": {"status": "DEMO 5KM"}}})
        if path.endswith("/downscaling/compare"): return respond(self, 200, {"comparison": {"ground_truth_metrics": {"max_intensity": 145, "p95_intensity": 112}, "baseline_bicubic": {"rmse": 19.4, "peak_error": 31.2}, "diffusion_model": {"rmse": 8.6, "max_intensity": 141.8}, "spectral_smoothing_addressed": True}})
        if path.endswith("/validation"): return respond(self, 200, {"event_name": "Demo validation", "metrics": {"mean_centroid_error_km": 18.2, "max_centroid_error_km": 34.7, "intensity_rmse": 9.3}, "status": "DEMO", "provenance": {"data_kind": "simulated"}})
        return respond(self, 404, {"detail": "Endpoint not found"})
    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try: body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        except json.JSONDecodeError: return respond(self, 400, {"detail": "Invalid JSON"})
        event = get_event(body.get("event_id"))
        if not event: return respond(self, 404, {"detail": "Event not found"})
        if path.endswith("/alerts"): return respond(self, 200, {"alert_id": f"DEMO-{event['id']}", "event_id": event["id"], "severity": event["risk"], "latitude": event["latest"]["lat"], "longitude": event["latest"]["lon"], "alert_radius_km": 5, "data_kind": "vercel_demo_fixture"})
        if path.endswith("/analyze-event"):
            return respond(self, 200, {"event_id": event["id"], "event_type": event["type"], "data_mode": "DEMO", "scientific_summary": "Hosted demo analysis. Deploy backend for live NWP/ensemble inference.", "climatology": {"temperature_mean": 29.4, "temperature_std": 2.7, "precipitation_mean": 4.2, "precipitation_std": 3.1, "wind_mean": 14.3, "wind_std": 4.6, "method": "Demo climatology"}, "zscores": {"temperature": 2.3, "precipitation": 1.4, "wind": 0.8, "composite_max": 2.3}, "efi": {"efi": 0.74, "interpretation": "Elevated tail risk", "provenance": "Demo fixture", "is_synthetic": True}, "severity": {"band": event["risk"], "multi_factorial_score": event["anomaly_score"], "exceedance_pct": event["confidence"]}, "subsystem_readiness": {"climatological_baseline": "DEMO", "anomaly_engine": "DEMO", "efi_engine": "DEMO", "spatiotemporal_gnn": "DEMO", "conditional_diffusion": "DEMO", "physics_constraints": "DEMO"}})
        return respond(self, 404, {"detail": "Endpoint not found"})