import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from engine import StructuralEngine
from ingest import normalize_rest_payload, parse_csv_text, now_iso

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

engine = StructuralEngine()
events = []
paused = False
scenario = "normal"


def add_event(result):
    result = dict(result)
    result["id"] = len(events) + 1
    events.append(result)
    if len(events) > 300:
        events.pop(0)
    engine.latest_result = result


def make_demo_frame():
    global scenario

    if scenario == "incident":
        pressure = random.uniform(20, 40)
        flow = random.uniform(55, 85)
        tank_level = random.uniform(35, 55)
        quality = random.uniform(55, 75)
        vibration = random.uniform(0.55, 0.9)
        site = "West Feed Main"
        asset = "Distribution Line B"
    elif scenario == "degrading":
        pressure = random.uniform(42, 55)
        flow = random.uniform(88, 110)
        tank_level = random.uniform(55, 68)
        quality = random.uniform(76, 90)
        vibration = random.uniform(0.28, 0.48)
        site = "Reservoir East"
        asset = "Distribution Line A"
    else:
        pressure = random.uniform(58, 66)
        flow = random.uniform(118, 132)
        tank_level = random.uniform(71, 79)
        quality = random.uniform(94, 99)
        vibration = random.uniform(0.08, 0.18)
        site = random.choice(["North Loop", "Reservoir East", "South Basin"])
        asset = random.choice(["Distribution Line A", "Distribution Line B", "Pump Station 1"])

    return {
        "timestamp": now_iso(),
        "site_id": site,
        "asset_id": asset,
        "sensor_values": {
            "pressure": pressure,
            "flow": flow,
            "tank_level": tank_level,
            "quality": quality,
            "vibration": vibration,
        },
    }


def telemetry_loop():
    while True:
        if not paused:
            frame = make_demo_frame()
            result = engine.process_frame(frame)
            add_event(result)
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def read_text_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        return raw.decode("utf-8")

    def serve_static(self, filename):
        path = STATIC_DIR / filename
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return

        data = path.read_bytes()

        if filename.endswith(".js"):
            content_type = "application/javascript"
        elif filename.endswith(".css"):
            content_type = "text/css"
        else:
            content_type = "text/html"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global paused, scenario

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/dashboard":
            return self.serve_static("dashboard.html")

        if path.startswith("/static/"):
            return self.serve_static(path.replace("/static/", ""))

        if path == "/api/status":
            latest = engine.latest_result or {
                "state": "UNKNOWN",
                "structural_drift_score": 0.0,
                "relational_stability_score": 0.0,
                "system_health": 100,
                "drift_alert": False,
                "site_id": "-",
                "asset_id": "-",
                "timestamp": "-",
                "lead_time_hours": None,
                "lead_time_confidence": 0.0,
                "drift_velocity": 0.0,
                "structural_driver": "-",
                "predicted_impact": "—",
                "explanation": "Initializing structural telemetry...",
            }
            status = dict(latest)
            status["events_tracked"] = len(events)
            status["paused"] = paused
            status["scenario"] = scenario
            return self.send_json(status)

        if path == "/api/events":
            return self.send_json(events)

        if path == "/api/pause":
            paused = True
            return self.send_json({"ok": True, "paused": True})

        if path == "/api/resume":
            paused = False
            return self.send_json({"ok": True, "paused": False})

        if path == "/api/reset":
            events.clear()
            engine.frames.clear()
            engine.latest_result = None
            engine.prev_drift = None
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

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/ingest":
                payload = self.read_json_body()
                frame = normalize_rest_payload(payload)
                result = engine.process_frame(frame)
                add_event(result)
                return self.send_json({"ok": True, "result": result})

            if path == "/api/load-csv":
                csv_text = self.read_text_body()
                frames = parse_csv_text(csv_text)

                results = []
                for frame in frames:
                    result = engine.process_frame(frame)
                    add_event(result)
                    results.append(result)

                return self.send_json({
                    "ok": True,
                    "frames_loaded": len(frames),
                    "latest_result": results[-1] if results else None,
                })

            self.send_error(404)

        except Exception as e:
            return self.send_json({"ok": False, "error": str(e)}, status=400)

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