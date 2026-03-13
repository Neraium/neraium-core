import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from lead_time_engine import HybridSIIDetector

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

events = []
scenario = "normal"
paused = False

MAX_EVENTS = 300

detector = HybridSIIDetector()

SITES = [
    "alpha-water-grid",
    "reservoir-east",
    "north-loop"
]

ASSETS = [
    "pump-station-1",
    "district-main-b",
    "distribution-node-7"
]

SENSOR_NAMES = (
    "pressure_inlet",
    "pressure_outlet",
    "flow_rate",
    "tank_level",
    "quality_index",
    "pump_vibration",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def add_event(event):
    events.append(event)
    if len(events) > MAX_EVENTS:
        events.pop(0)


def build_sensor_values():
    if scenario == "normal":
        return (
            round(random.uniform(60, 65), 2),   # pressure_inlet
            round(random.uniform(56, 61), 2),   # pressure_outlet
            round(random.uniform(118, 136), 2), # flow_rate
            round(random.uniform(68, 78), 2),   # tank_level
            round(random.uniform(94, 99), 2),   # quality_index
            round(random.uniform(0.16, 0.30), 3) # pump_vibration
        )

    if scenario == "degrading":
        return (
            round(random.uniform(48, 60), 2),
            round(random.uniform(42, 56), 2),
            round(random.uniform(95, 124), 2),
            round(random.uniform(56, 73), 2),
            round(random.uniform(82, 95), 2),
            round(random.uniform(0.22, 0.48), 3)
        )

    return (
        round(random.uniform(34, 52), 2),
        round(random.uniform(28, 47), 2),
        round(random.uniform(70, 108), 2),
        round(random.uniform(46, 66), 2),
        round(random.uniform(64, 84), 2),
        round(random.uniform(0.35, 0.82), 3)
    )


def predicted_impact_from_state(state, lead_time_hours):
    if state == "ALERT":
        if lead_time_hours is None:
            return "Critical structural instability detected."
        return f"Critical instability detected. Estimated intervention window: {round(lead_time_hours, 1)}h."
    if state == "WATCH":
        if lead_time_hours is None:
            return "Early degradation detected. Monitoring recommended."
        return f"Early structural drift detected. Estimated warning horizon: {round(lead_time_hours, 1)}h."
    return "No near term operational disruption expected."


def event_type_from_state(state):
    if state == "ALERT":
        return random.choice([
            "relational_break",
            "structural_drift_spike",
            "instability_escalation",
            "cross_sensor_divergence"
        ])
    if state == "WATCH":
        return random.choice([
            "gradual_drift",
            "correlation_shift",
            "relational_variance",
            "stability_decay"
        ])
    return random.choice([
        "baseline_structure",
        "stable_correlation",
        "normal_telemetry_frame",
        "relational_observation"
    ])


def generate_event():
    site = random.choice(SITES)
    asset = random.choice(ASSETS)
    timestamp = now()
    sensor_values = build_sensor_values()

    result = detector.update(
        site_id=site,
        asset_id=asset,
        timestamp=timestamp,
        sensor_names=SENSOR_NAMES,
        sensor_values=sensor_values,
        missing_fraction=0.0,
    )

    event = {
        "id": len(events) + 1,
        "event_type": event_type_from_state(result.state),
        "site_id": site,
        "asset_id": asset,
        "zone": site,  # compatibility with older frontend code
        "timestamp": timestamp,
        "state": result.state,
        "confidence": round(max(0.88, result.lead_time_confidence), 2),

        # SII fields
        "structural_drift_score": result.structural_drift_score,
        "smoothed_drift_score": result.smoothed_drift_score,
        "drift_velocity": result.drift_velocity,
        "drift_acceleration": result.drift_acceleration,
        "relational_stability_score": result.relational_stability_score,
        "lead_time_hours": result.lead_time_hours,
        "lead_time_lower_hours": result.lead_time_lower_hours,
        "lead_time_upper_hours": result.lead_time_upper_hours,
        "lead_time_confidence": result.lead_time_confidence,
        "structural_driver": result.structural_driver,

        # compatibility fields for existing UI
        "network_drift_score": result.structural_drift_score,
        "quality_persistence_score": result.relational_stability_score,
        "early_warning_horizon_hours": (
            None if result.lead_time_hours is None else int(round(result.lead_time_hours))
        ),

        # raw telemetry
        "sensor_names": list(SENSOR_NAMES),
        "sensor_values": {
            SENSOR_NAMES[i]: sensor_values[i] for i in range(len(SENSOR_NAMES))
        },

        # older water metrics for compatibility
        "flow_rate": sensor_values[2],
        "tank_level": sensor_values[3],
        "water_quality_index": sensor_values[4],
        "line_pressure": sensor_values[1],
        "pump_vibration": sensor_values[5],
        "predicted_impact": predicted_impact_from_state(result.state, result.lead_time_hours),
        "explanation": (
            "SII is detecting active structural instability across the sensor network."
            if result.state == "ALERT"
            else "SII is detecting gradual relational drift before conventional threshold failure."
            if result.state == "WATCH"
            else "SII is observing stable structural relationships across the monitored system."
        ),
    }

    add_event(event)


def telemetry_loop():
    while True:
        if not paused:
            generate_event()
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_static(self, path):
        file_path = STATIC_DIR / path

        if not file_path.exists():
            self.send_error(404)
            return

        self.send_response(200)

        if file_path.suffix == ".js":
            self.send_header("Content-Type", "application/javascript")
        elif file_path.suffix == ".css":
            self.send_header("Content-Type", "text/css")
        else:
            self.send_header("Content-Type", "text/html")

        data = file_path.read_bytes()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global scenario, paused

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return self.serve_static("index.html")

        if path.startswith("/static/"):
            return self.serve_static(path.replace("/static/", ""))

        if path == "/api/status":
            latest = events[-1] if events else None

            if not latest:
                return self.send_json({
                    "connected": True,
                    "scenario": scenario,
                    "paused": paused,
                    "events_tracked": 0,
                    "state": "UNKNOWN",
                    "site_id": "-",
                    "asset_id": "-",
                    "zone": "-",
                    "confidence": 0,
                    "structural_drift_score": 0,
                    "relational_stability_score": 0,
                    "lead_time_hours": None,
                    "lead_time_confidence": 0,
                    "structural_driver": "-",
                    "drift_velocity": 0,
                    "drift_acceleration": 0,
                    "network_drift_score": 0,
                    "quality_persistence_score": 0,
                    "early_warning_horizon_hours": None,
                    "predicted_impact": "",
                    "explanation": "",
                    "last_timestamp": ""
                })

            response = {
                "connected": True,
                "scenario": scenario,
                "paused": paused,
                "events_tracked": len(events),
                "state": latest.get("state", "UNKNOWN"),
                "site_id": latest.get("site_id", "-"),
                "asset_id": latest.get("asset_id", "-"),
                "zone": latest.get("zone", "-"),
                "confidence": latest.get("confidence", 0),

                # new SII fields
                "structural_drift_score": latest.get("structural_drift_score", 0),
                "relational_stability_score": latest.get("relational_stability_score", 0),
                "lead_time_hours": latest.get("lead_time_hours"),
                "lead_time_confidence": latest.get("lead_time_confidence", 0),
                "structural_driver": latest.get("structural_driver", "-"),
                "drift_velocity": latest.get("drift_velocity", 0),
                "drift_acceleration": latest.get("drift_acceleration", 0),

                # compatibility fields
                "network_drift_score": latest.get("network_drift_score", 0),
                "quality_persistence_score": latest.get("quality_persistence_score", 0),
                "early_warning_horizon_hours": latest.get("early_warning_horizon_hours"),

                # existing extras
                "predicted_impact": latest.get("predicted_impact", ""),
                "explanation": latest.get("explanation", ""),
                "last_timestamp": latest.get("timestamp", "")
            }

            return self.send_json(response)

        if path == "/api/events":
            return self.send_json(events)

        if path == "/api/pause":
            paused = True
            return self.send_json({"status": "ok", "paused": True})

        if path == "/api/resume":
            paused = False
            return self.send_json({"status": "ok", "paused": False})

        if path == "/api/reset":
            events.clear()
            return self.send_json({"status": "ok", "reset": True})

        if path == "/api/scenario":
            params = parse_qs(parsed.query)
            scenario = params.get("mode", ["normal"])[0]
            return self.send_json({"status": "ok", "scenario": scenario})

        self.send_error(404)

    def log_message(self, format, *args):
        return


def run():
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Server running on http://0.0.0.0:8000")

    thread = threading.Thread(target=telemetry_loop, daemon=True)
    thread.start()

    server.serve_forever()


if __name__ == "__main__":
    run()