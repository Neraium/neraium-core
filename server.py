import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent

events = []
paused = False
scenario = "normal"
MAX_EVENTS = 300
last_drift = None
lock = threading.Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def structural_drift(values):
    base = [60.0, 125.0, 75.0, 97.0, 0.20]
    total = 0.0
    for i, v in enumerate(values):
        total += abs(v - base[i]) / abs(base[i])
    return total


def relational_stability(values):
    pressure, flow, tank, quality, vibration = values
    score = 1 - (
        abs((pressure / max(flow, 1e-6)) - 0.5) * 0.4
        + abs((tank / max(pressure, 1e-6)) - 1.2) * 0.3
        + vibration * 0.3
    )
    return max(0.0, min(1.0, score))


def lead_time(drift, velocity):
    boundary = 4.0
    if velocity <= 0:
        return None
    hours = (boundary - drift) / velocity
    return max(0.0, round(hours, 1))


def build_sensor_values():
    global scenario

    base_pressure = 60
    base_flow = 125
    base_tank = 75
    base_quality = 97
    base_vibration = 0.20

    if scenario == "normal":
        return (
            random.uniform(base_pressure - 3, base_pressure + 3),
            random.uniform(base_flow - 4, base_flow + 4),
            random.uniform(base_tank - 3, base_tank + 3),
            random.uniform(base_quality - 1.5, base_quality + 1.5),
            random.uniform(0.18, 0.24),
        )

    if scenario == "degrading":
        drift = random.uniform(0.5, 1.5)
        return (
            base_pressure - drift * 6 + random.uniform(-2, 2),
            base_flow - drift * 10 + random.uniform(-3, 3),
            base_tank - drift * 6 + random.uniform(-2, 2),
            base_quality - drift * 4 + random.uniform(-1, 1),
            base_vibration + drift * 0.35,
        )

    spike = random.uniform(2.5, 4.0)
    return (
        base_pressure - spike * 10 + random.uniform(-4, 4),
        base_flow - spike * 15 + random.uniform(-6, 6),
        base_tank - spike * 8 + random.uniform(-4, 4),
        base_quality - spike * 6 + random.uniform(-2, 2),
        base_vibration + spike * 0.5,
    )


def generate_event():
    global last_drift

    site = random.choice(["Reservoir East", "North Loop", "South Basin", "West Feed Main"])
    asset = random.choice(["Pump Station 1", "District Main B", "Distribution Node 7"])
    timestamp = now_iso()

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
        event_type = "instability_escalation"
        predicted_impact = "Potential localized service disruption within 1 to 2 hours."
    elif drift > 1.5:
        state = "WATCH"
        event_type = "gradual_drift"
        predicted_impact = "Early degradation detected. Maintenance window recommended."
    else:
        state = "STABLE"
        event_type = "baseline_structure"
        predicted_impact = "No near term operational disruption expected."

    event = {
        "id": len(events) + 1,
        "event_type": event_type,
        "timestamp": timestamp,
        "site_id": site,
        "asset_id": asset,
        "state": state,
        "confidence": round(random.uniform(0.90, 0.99), 2),
        "structural_drift_score": round(drift, 3),
        "relational_stability_score": round(stability, 3),
        "drift_velocity": round(velocity, 3),
        "lead_time_hours": lt,
        "lead_time_confidence": round(random.uniform(0.70, 0.95), 2),
        "structural_driver": "pressure-flow imbalance",
        "predicted_impact": predicted_impact,
        "explanation": "SII analyzing structural geometry of the sensor network.",
        "pressure": round(values[0], 2),
        "flow": round(values[1], 2),
        "tank_level": round(values[2], 2),
        "quality": round(values[3], 2),
        "vibration": round(values[4], 3),
    }

    with lock:
        events.append(event)
        if len(events) > MAX_EVENTS:
            events.pop(0)


def telemetry_loop():
    generate_event()
    while True:
        time.sleep(2)
        if not paused:
            generate_event()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def serve_file(self, relative_path, content_type):
        path = BASE_DIR / relative_path
        if not path.exists():
            self.send_error(404, "File not found.")
            return

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global paused, scenario, last_drift

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/dashboard":
            return self.serve_file("static/dashboard.html", "text/html; charset=utf-8")

        if path == "/static/app.js":
            return self.serve_file("static/app.js", "application/javascript; charset=utf-8")

        if path == "/static/styles.css":
            return self.serve_file("static/styles.css", "text/css; charset=utf-8")

        if path == "/api/status":
            with lock:
                latest = events[-1] if events else {
                    "state": "UNKNOWN",
                    "timestamp": "-",
                    "site_id": "—",
                    "asset_id": "—",
                    "structural_drift_score": 0.0,
                    "relational_stability_score": 0.0,
                    "lead_time_hours": None,
                    "lead_time_confidence": 0.0,
                    "drift_velocity": 0.0,
                    "structural_driver": "—",
                    "predicted_impact": "—",
                    "explanation": "Initializing structural telemetry...",
                }
                out = dict(latest)
                out["events_tracked"] = len(events)
                out["paused"] = paused
                out["scenario"] = scenario
                out["connected"] = True

            return self.send_json(out)

        if path == "/api/events":
            with lock:
                return self.send_json(list(events))

        if path == "/api/pause":
            paused = True
            return self.send_json({"ok": True, "paused": True})

        if path == "/api/resume":
            paused = False
            return self.send_json({"ok": True, "paused": False})

        if path == "/api/reset":
            with lock:
                events.clear()
            last_drift = None
            return self.send_json({"ok": True, "reset": True})

        if path == "/api/scenario/normal":
            scenario = "normal"
            return self.send_json({"ok": True, "scenario": scenario})

        if path == "/api/scenario/degrading":
            scenario = "degrading"
            return self.send_json({"ok": True, "scenario": scenario})

        if path == "/api/scenario/incident":
            scenario = "incident"
            return self.send_json({"ok": True, "scenario": scenario})

        self.send_error(404, "File not found.")

    def log_message(self, format, *args):
        return


def run():
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("NERAIUM WATER PLATFORM DEMO ACTIVE")
    print("Server running at http://0.0.0.0:8000")
    threading.Thread(target=telemetry_loop, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    run()