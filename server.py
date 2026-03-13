import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

events = []
scenario = "normal"
paused = False

MAX_EVENTS = 300


def now():
    return datetime.now(timezone.utc).isoformat()


def add_event(event):
    events.append(event)
    if len(events) > MAX_EVENTS:
        events.pop(0)


def calculate_early_warning_horizon(drift):
    drift = max(0.0, min(1.0, float(drift)))
    return max(0, int((1 - drift) * 72))


def generate_event():
    global scenario

    zone = random.choice([
        "Reservoir East",
        "West Feed Main",
        "North Loop",
        "South Basin"
    ])

    if scenario == "normal":
        state = "STABLE"
        drift = round(random.uniform(0.05, 0.25), 2)
        persistence = round(random.uniform(0.80, 0.96), 2)
        leak_risk = round(random.uniform(0.03, 0.15), 2)
        flow_rate = round(random.uniform(118, 136), 1)
        line_pressure = round(random.uniform(58, 66), 1)
        water_quality_index = round(random.uniform(94, 99), 1)
        tank_level = round(random.uniform(66, 80), 1)
        predicted_impact = "No near term service disruption expected."

    elif scenario == "degrading":
        state = "WATCH"
        drift = round(random.uniform(0.25, 0.65), 2)
        persistence = round(random.uniform(0.30, 0.75), 2)
        leak_risk = round(random.uniform(0.15, 0.45), 2)
        flow_rate = round(random.uniform(88, 118), 1)
        line_pressure = round(random.uniform(45, 58), 1)
        water_quality_index = round(random.uniform(74, 90), 1)
        tank_level = round(random.uniform(55, 72), 1)
        predicted_impact = "Early degradation detected. Maintenance window recommended."

    else:
        state = "ALERT"
        drift = round(random.uniform(0.65, 0.98), 2)
        persistence = round(random.uniform(0.70, 0.95), 2)
        leak_risk = round(random.uniform(0.50, 0.95), 2)
        flow_rate = round(random.uniform(60, 95), 1)
        line_pressure = round(random.uniform(28, 45), 1)
        water_quality_index = round(random.uniform(58, 78), 1)
        tank_level = round(random.uniform(42, 65), 1)
        predicted_impact = "Potential localized service disruption within 1 to 2 hours."

    early_warning_horizon_hours = calculate_early_warning_horizon(drift)

    event = {
        "id": len(events) + 1,
        "type": random.choice([
            "flow_observation",
            "quality_observation",
            "pressure_frame",
            "telemetry_frame",
            "leak_signature"
        ]),
        "zone": zone,
        "timestamp": now(),
        "state": state,
        "confidence": round(random.uniform(0.88, 0.98), 2),
        "network_drift_score": drift,
        "quality_persistence_score": persistence,
        "early_warning_horizon_hours": early_warning_horizon_hours,
        "leak_risk": leak_risk,
        "flow_rate": flow_rate,
        "line_pressure": line_pressure,
        "water_quality_index": water_quality_index,
        "tank_level": tank_level,
        "predicted_impact": predicted_impact
    }

    add_event(event)


def telemetry_loop():
    while True:
        if not paused:
            generate_event()
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
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
                    "zone": "-",
                    "confidence": 0,
                    "network_drift_score": 0,
                    "quality_persistence_score": 0,
                    "early_warning_horizon_hours": 0,
                    "leak_risk": 0,
                    "flow_rate": 0,
                    "line_pressure": 0,
                    "water_quality_index": 0,
                    "tank_level": 0,
                    "predicted_impact": "",
                    "last_timestamp": ""
                })

            drift = latest.get("network_drift_score", 0)
            early_warning_horizon_hours = latest.get(
                "early_warning_horizon_hours",
                calculate_early_warning_horizon(drift)
            )

            response = {
                "connected": True,
                "scenario": scenario,
                "paused": paused,
                "events_tracked": len(events),
                "state": latest.get("state", "UNKNOWN"),
                "zone": latest.get("zone", "-"),
                "confidence": latest.get("confidence", 0),
                "network_drift_score": latest.get("network_drift_score", 0),
                "quality_persistence_score": latest.get("quality_persistence_score", 0),
                "early_warning_horizon_hours": early_warning_horizon_hours,
                "leak_risk": latest.get("leak_risk", 0),
                "flow_rate": latest.get("flow_rate", 0),
                "line_pressure": latest.get("line_pressure", 0),
                "water_quality_index": latest.get("water_quality_index", 0),
                "tank_level": latest.get("tank_level", 0),
                "predicted_impact": latest.get("predicted_impact", ""),
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