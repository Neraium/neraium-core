import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from structural_engine import StructuralEngine

BASE_DIR = Path(__file__).resolve().parent

events   = []
paused   = False
scenario = "normal"
MAX_EVENTS = 300
engine   = StructuralEngine(baseline_window=24, recent_window=8)
lock     = threading.Lock()

SITES  = ["Reservoir East", "North Loop", "South Basin", "West Feed Main"]
ASSETS = ["Pump Station 1", "District Main B", "Distribution Node 7"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_sensor_values():
    base_pressure  = 60
    base_flow      = 125
    base_tank      = 75
    base_quality   = 97
    base_vibration = 0.20

    if scenario == "normal":
        return {
            "pressure":   random.uniform(base_pressure - 3,  base_pressure + 3),
            "flow":       random.uniform(base_flow - 4,      base_flow + 4),
            "tank_level": random.uniform(base_tank - 3,      base_tank + 3),
            "quality":    random.uniform(base_quality - 1.5, base_quality + 1.5),
            "vibration":  random.uniform(0.18, 0.24),
        }

    if scenario == "degrading":
        drift = random.uniform(0.5, 1.5)
        return {
            "pressure":   base_pressure  - drift * 6  + random.uniform(-2, 2),
            "flow":       base_flow      - drift * 10 + random.uniform(-3, 3),
            "tank_level": base_tank      - drift * 6  + random.uniform(-2, 2),
            "quality":    base_quality   - drift * 4  + random.uniform(-1, 1),
            "vibration":  base_vibration + drift * 0.35,
        }

    spike = random.uniform(2.5, 4.0)
    return {
        "pressure":   base_pressure  - spike * 10 + random.uniform(-4, 4),
        "flow":       base_flow      - spike * 15 + random.uniform(-6, 6),
        "tank_level": base_tank      - spike * 8  + random.uniform(-4, 4),
        "quality":    base_quality   - spike * 6  + random.uniform(-2, 2),
        "vibration":  base_vibration + spike * 0.5,
    }


def generate_event():
    frame = {
        "timestamp":     now_iso(),
        "site_id":       random.choice(SITES),
        "asset_id":      random.choice(ASSETS),
        "sensor_values": build_sensor_values(),
    }

    result = engine.process_frame(frame)

    with lock:
        result["id"] = len(events) + 1
        for k, v in frame["sensor_values"].items():
            result[k] = round(v, 3)
        events.append(result)
        if len(events) > MAX_EVENTS:
            events.pop(0)


def telemetry_loop():
    generate_event()
    while True:
        time.sleep(2)
        with lock:
            is_paused = paused
        if not is_paused:
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
        global paused, scenario

        parsed = urlparse(self.path)
        path   = parsed.path

        if path in ("/", "/dashboard"):
            return self.serve_file("static/dashboard.html", "text/html; charset=utf-8")
        if path == "/static/app.js":
            return self.serve_file("static/app.js", "application/javascript; charset=utf-8")
        if path == "/static/styles.css":
            return self.serve_file("static/styles.css", "text/css; charset=utf-8")

        if path == "/api/status":
            with lock:
                if events:
                    out = dict(events[-1])
                else:
                    out = {
                        "state": "UNKNOWN",
                        "timestamp": "-",
                        "site_id": "—",
                        "asset_id": "—",
                        "structural_drift_score": 0.0,
                        "relational_stability_score": 0.0,
                        "system_health": 100,
                        "lead_time_hours": None,
                        "lead_time_confidence": 0.0,
                        "drift_velocity": 0.0,
                        "structural_driver": "—",
                        "predicted_impact": "—",
                        "explanation": "Initializing structural telemetry...",
                    }
                out["events_tracked"] = len(events)
                out["paused"]    = paused
                out["scenario"]  = scenario
                out["connected"] = True
            return self.send_json(out)

        if path == "/api/events":
            with lock:
                return self.send_json(list(events))

        if path == "/api/pause":
            with lock:
                paused = True
            return self.send_json({"ok": True, "paused": True})

        if path == "/api/resume":
            with lock:
                paused = False
            return self.send_json({"ok": True, "paused": False})

        if path == "/api/reset":
            with lock:
                events.clear()
                paused   = False
                scenario = "normal"
            engine.frames.clear()
            engine.prev_drift    = None
            engine.latest_result = None
            engine.sensor_order  = []
            return self.send_json({"ok": True, "reset": True, "scenario": "normal"})

        if path == "/api/scenario/normal":
            with lock:
                scenario = "normal"
            return self.send_json({"ok": True, "scenario": scenario})

        if path == "/api/scenario/degrading":
            with lock:
                scenario = "degrading"
            return self.send_json({"ok": True, "scenario": scenario})

        if path == "/api/scenario/incident":
            with lock:
                scenario = "incident"
            return self.send_json({"ok": True, "scenario": scenario})

        self.send_error(404, "File not found.")

    def log_message(self, format, *args):
        return


def run():
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("NERAIUM WATER PLATFORM DEMO ACTIVE")
    print("Server: http://0.0.0.0:8000")
    threading.Thread(target=telemetry_loop, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    run()
