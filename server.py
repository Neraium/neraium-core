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
MAX_EVENTS = 200
last_drift = None


def now():
    return datetime.now(timezone.utc).isoformat()


def structural_drift(values):
    base = [60.0, 125.0, 75.0, 97.0, 0.2]
    total = 0.0
    for i, v in enumerate(values):
        total += abs(v - base[i]) / abs(base[i])
    return total


def relational_stability(values):
    pressure, flow, tank, quality, vibration = values
    score = 1 - (
        abs((pressure / flow) - 0.5) * 0.4 +
        abs((tank / pressure) - 1.2) * 0.3 +
        vibration * 0.3
    )
    return max(0.0, min(1.0, score))


def lead_time(drift, velocity):
    boundary = 4.0
    if velocity <= 0:
        return None
    hours = (boundary - drift) / velocity
    if hours < 0:
        return 0.0
    return hours


def build_sensor_values():
    if scenario == "normal":
        return (
            random.uniform(58, 64),
            random.uniform(118, 134),
            random.uniform(70, 78),
            random.uniform(95, 99),
            random.uniform(0.15, 0.25),
        )
    if scenario == "degrading":
        return (
            random.uniform(48, 60),
            random.uniform(100, 125),
            random.uniform(60, 75),
            random.uniform(85, 96),
            random.uniform(0.20, 0.45),
        )
    return (
        random.uniform(35, 52),
        random.uniform(80, 105),
        random.uniform(50, 65),
        random.uniform(70, 85),
        random.uniform(0.30, 0.70),
    )


def generate_event():
    global last_drift

    site = random.choice(["Reservoir East", "North Loop", "South Basin", "West Feed Main"])
    asset = random.choice(["Pump Station 1", "District Main B", "Distribution Node 7"])
    timestamp = now()
    values = build_sensor_values()

    drift = structural_drift(values)
    stability = relational_stability(values)

    velocity = 0.0
    if last_drift is not None:
        velocity = drift - last_drift
    last_drift = drift

    lt = lead_time(drift, velocity)

    if drift > 3.0:
        state = "ALERT"
    elif drift > 1.5:
        state = "WATCH"
    else:
        state = "STABLE"

    event = {
        "id": len(events) + 1,
        "event_type": random.choice([
            "baseline_structure",
            "stable_correlation",
            "relational_observation",
            "gradual_drift",
            "correlation_shift",
            "instability_escalation",
        ]),
        "timestamp": timestamp,
        "site_id": site,
        "asset_id": asset,
        "state": state,
        "confidence": round(random.uniform(0.90, 0.99), 2),
        "structural_drift_score": round(drift, 3),
        "relational_stability_score": round(stability, 3),
        "drift_velocity": round(velocity, 3),
        "lead_time_hours": None if lt is None else round(lt, 1),
        "lead_time_confidence": round(random.uniform(0.70, 0.95), 2),
        "structural_driver": "pressure-flow imbalance",
        "predicted_impact": (
            "No near term operational disruption expected."
            if state == "STABLE"
            else "Early instability developing."
            if state == "WATCH"
            else "Potential localized service disruption within 1 to 2 hours."
        ),
        "explanation": "SII analyzing structural geometry of the sensor network.",
    }

    events.append(event)
    if len(events) > MAX_EVENTS:
        events.pop(0)


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

    def serve_static(self, filename):
        path = STATIC_DIR / filename
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return

        data = path.read_bytes()
        self.send_response(200)
        if filename.endswith(".js"):
            self.send_header("Content-Type", "application/javascript")
        elif filename.endswith(".css"):
            self.send_header("Content-Type", "text/css")
        else:
            self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global scenario, paused

        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            return self.serve_static("index.html")

        if path.startswith("/static/"):
            return self.serve_static(path.replace("/static/", ""))

        if path == "/api/events":
            return self.send_json(events)

        if path == "/api/status":
            latest = events[-1] if events else {}
            return self.send_json({
                "state": latest.get("state", "UNKNOWN"),
                "site_id": latest.get("site_id", "-"),
                "asset_id": latest.get("asset_id", "-"),
                "confidence": latest.get("confidence", 0),
                "structural_drift_score": latest.get("structural_drift_score", 0),
                "relational_stability_score": latest.get("relational_stability_score", 0),
                "lead_time_hours": latest.get("lead_time_hours"),
                "lead_time_confidence": latest.get("lead_time_confidence", 0),
                "drift_velocity": latest.get("drift_velocity", 0),
                "structural_driver": latest.get("structural_driver", "-"),
                "events_tracked": len(events),
                "last_timestamp": latest.get("timestamp", ""),
                "predicted_impact": latest.get("predicted_impact", ""),
                "explanation": latest.get("explanation", ""),
                "paused": paused,
                "scenario": scenario,
            })

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
            scenario = params.get("mode", ["normal"])[0]
            return self.send_json({"status": "ok", "scenario": scenario})

        self.send_error(404)

    def log_message(self, format, *args):
        return


def run():
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Server running on http://localhost:8000")
    threading.Thread(target=telemetry_loop, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    run()