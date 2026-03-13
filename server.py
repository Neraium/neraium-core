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


def generate_event():
    global scenario

    zone = random.choice([
        "Reservoir East",
        "West Feed Main",
        "North Loop",
        "South Basin"
    ])

    drift = round(random.uniform(0.05, 0.98), 2)
    persistence = round(random.uniform(0.05, 0.95), 2)

    if scenario == "normal":
        state = "STABLE"
        drift = random.uniform(0.05, 0.25)
    elif scenario == "degrading":
        state = "WATCH"
        drift = random.uniform(0.25, 0.65)
    else:
        state = "ALERT"
        drift = random.uniform(0.65, 0.98)

    drift = round(drift, 2)

    # Early warning calculation
    early_warning = max(0, int((1 - drift) * 72))

    event = {
        "id": len(events) + 1,
        "type": random.choice([
            "flow_observation",
            "quality_observation",
            "pressure_frame",
            "telemetry_frame"
        ]),
        "zone": zone,
        "timestamp": now(),
        "state": state,
        "confidence": round(random.uniform(0.9, 0.98), 2),
        "network_drift_score": drift,
        "quality_persistence_score": persistence,
        "early_warning_horizon_hours": early_warning,
        "flow_rate": round(random.uniform(60, 140), 1),
        "line_pressure": round(random.uniform(55, 70), 1),
        "water_quality_index": round(random.uniform(94, 99), 1),
        "tank_level": round(random.uniform(60, 80), 1),
        "predicted_impact": (
            "No near term service disruption expected."
            if state == "STABLE"
            else "Early instability developing."
            if state == "WATCH"
            else "Potential localized service disruption within 1 to 2 hours."
        )
    }

    add_event(event)


def telemetry_loop():
    while True:
        if not paused:
            generate_event()
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

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

        self.end_headers()
        self.wfile.write(file_path.read_bytes())

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
                self.send_json({
                    "connected": True,
                    "scenario": scenario,
                    "paused": paused,
                    "events_tracked": 0,
                    "state": "UNKNOWN"
                })
                return

            response = {
                "connected": True,
                "scenario": scenario,
                "paused": paused,
                "events_tracked": len(events),
                "state": latest["state"],
                "zone": latest["zone"],
                "confidence": latest["confidence"],
                "network_drift_score": latest["network_drift_score"],
                "quality_persistence_score": latest["quality_persistence_score"],
                "early_warning_horizon_hours": latest["early_warning_horizon_hours"],
                "flow_rate": latest["flow_rate"],
                "line_pressure": latest["line_pressure"],
                "water_quality_index": latest["water_quality_index"],
                "tank_level": latest["tank_level"],
                "predicted_impact": latest["predicted_impact"],
                "last_timestamp": latest["timestamp"]
            }

            return self.send_json(response)

        if path == "/api/events":
            return self.send_json(events)

        if path == "/api/pause":
            paused = True
            return self.send_json({"paused": True})

        if path == "/api/resume":
            paused = False
            return self.send_json({"paused": False})

        if path == "/api/reset":
            events.clear()
            return self.send_json({"reset": True})

        if path == "/api/scenario":
            params = parse_qs(parsed.query)
            scenario = params.get("mode", ["normal"])[0]
            return self.send_json({"scenario": scenario})

        self.send_error(404)


def run():
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Server running on http://0.0.0.0:8000")

    thread = threading.Thread(target=telemetry_loop, daemon=True)
    thread.start()

    server.serve_forever()


if __name__ == "__main__":
    run()