import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

EVENTS = []
MAX_EVENTS = 240
SCENARIO = "normal"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def clamp(value, low, high):
    return max(low, min(high, value))


def next_id():
    return len(EVENTS) + 1


def add_event(event):
    EVENTS.append(event)
    if len(EVENTS) > MAX_EVENTS:
        del EVENTS[0:len(EVENTS) - MAX_EVENTS]


def latest_event():
    return EVENTS[-1] if EVENTS else None


def current_scenario():
    return SCENARIO


def set_scenario(value):
    global SCENARIO
    allowed = {"normal", "degrading", "incident"}
    SCENARIO = value if value in allowed else "normal"


def make_event(payload):
    return {
        "id": payload.get("id", next_id()),
        "type": payload.get("type", "telemetry"),
        "timestamp": payload.get("timestamp", now_iso()),
        "state": str(payload.get("state", "UNKNOWN")).upper(),
        "confidence": float(payload.get("confidence", 0)),
        "sii_score": float(payload.get("sii_score", 0)),
        "ewma_score": float(payload.get("ewma_score", 0)),
        "velocity": float(payload.get("velocity", 0)),
        "drift_vector": float(payload.get("drift_vector", 0)),
        "cpu_usage": float(payload.get("cpu_usage", 0)),
        "memory_usage": float(payload.get("memory_usage", 0)),
    }


def generate_metrics(prev, scenario):
    sii = float(prev.get("sii_score", 0.20))
    ewma = float(prev.get("ewma_score", 0.12))
    velocity = float(prev.get("velocity", 0.08))
    drift_vector = float(prev.get("drift_vector", 0.06))
    cpu = float(prev.get("cpu_usage", 35.0))
    memory = float(prev.get("memory_usage", 42.0))

    if scenario == "normal":
        sii = clamp(sii + random.uniform(-0.02, 0.02), 0.08, 0.28)
        ewma = clamp(ewma + random.uniform(-0.015, 0.015), 0.05, 0.22)
        velocity = clamp(velocity + random.uniform(-0.015, 0.015), 0.03, 0.18)
        drift_vector = clamp(drift_vector + random.uniform(-0.015, 0.015), 0.02, 0.16)
        cpu = clamp(cpu + random.uniform(-4, 4), 18, 58)
        memory = clamp(memory + random.uniform(-3, 3), 25, 66)
        confidence = clamp(random.uniform(0.88, 0.96), 0, 1)
        state = "STABLE"

    elif scenario == "degrading":
        sii = clamp(sii + random.uniform(0.01, 0.045), 0.22, 0.58)
        ewma = clamp(ewma + random.uniform(0.01, 0.04), 0.16, 0.46)
        velocity = clamp(velocity + random.uniform(0.005, 0.03), 0.08, 0.34)
        drift_vector = clamp(drift_vector + random.uniform(0.005, 0.03), 0.07, 0.30)
        cpu = clamp(cpu + random.uniform(1, 6), 35, 82)
        memory = clamp(memory + random.uniform(1, 5), 40, 82)
        confidence = clamp(random.uniform(0.80, 0.91), 0, 1)
        state = "WATCH" if sii < 0.52 else "ALERT"

    else:  # incident
        sii = clamp(sii + random.uniform(0.02, 0.08), 0.55, 0.98)
        ewma = clamp(ewma + random.uniform(0.02, 0.07), 0.40, 0.92)
        velocity = clamp(velocity + random.uniform(0.01, 0.06), 0.22, 0.75)
        drift_vector = clamp(drift_vector + random.uniform(0.01, 0.06), 0.18, 0.72)
        cpu = clamp(cpu + random.uniform(2, 10), 60, 98)
        memory = clamp(memory + random.uniform(2, 9), 58, 98)
        confidence = clamp(random.uniform(0.84, 0.97), 0, 1)
        state = "ALERT"

    return {
        "state": state,
        "confidence": round(confidence, 2),
        "sii_score": round(sii, 2),
        "ewma_score": round(ewma, 2),
        "velocity": round(velocity, 2),
        "drift_vector": round(drift_vector, 2),
        "cpu_usage": round(cpu, 1),
        "memory_usage": round(memory, 1),
    }


def simulator_loop():
    while True:
        prev = latest_event() or {
            "sii_score": 0.20,
            "ewma_score": 0.12,
            "velocity": 0.08,
            "drift_vector": 0.06,
            "cpu_usage": 35.0,
            "memory_usage": 42.0,
        }

        metrics = generate_metrics(prev, current_scenario())

        add_event(make_event({
            "type": "telemetry",
            "timestamp": now_iso(),
            **metrics,
        }))

        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data, status: int = 200):
        payload = json.dumps(data).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status)

    def _send_file(self, path: Path, content_type: str):
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return
        self._send_bytes(path.read_bytes(), content_type)

    def log_message(self, format, *args):
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")

        if path == "/static/styles.css":
            return self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")

        if path == "/static/app.js":
            return self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")

        if path == "/api/events":
            return self._send_json(EVENTS)

        if path == "/api/status":
            latest = latest_event()
            payload = {
                "connected": True,
                "scenario": current_scenario(),
                "events_tracked": len(EVENTS),
                "state": latest.get("state", "UNKNOWN") if latest else "UNKNOWN",
                "confidence": latest.get("confidence", 0) if latest else 0,
                "sii_score": latest.get("sii_score", 0) if latest else 0,
                "ewma_score": latest.get("ewma_score", 0) if latest else 0,
                "velocity": latest.get("velocity", 0) if latest else 0,
                "drift_vector": latest.get("drift_vector", 0) if latest else 0,
                "cpu_usage": latest.get("cpu_usage", 0) if latest else 0,
                "memory_usage": latest.get("memory_usage", 0) if latest else 0,
                "last_timestamp": latest.get("timestamp", "None") if latest else "None",
            }
            return self._send_json(payload)

        if path == "/api/scenario":
            scenario = query.get("mode", ["normal"])[0].lower()
            set_scenario(scenario)
            return self._send_json({"status": "ok", "scenario": current_scenario()})

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/ingest":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    return self._send_json({"error": "JSON body must be an object"}, 400)

                event = make_event(payload)
                add_event(event)
                return self._send_json({"status": "ok", "event": event}, 200)

            except json.JSONDecodeError:
                return self._send_json({"error": "Invalid JSON"}, 400)

            except Exception as e:
                return self._send_json({"error": str(e)}, 400)

        self.send_error(404, "Not found")


if __name__ == "__main__":
    add_event(make_event({
        "type": "startup",
        "timestamp": now_iso(),
        "state": "STABLE",
        "confidence": 0.91,
        "sii_score": 0.22,
        "ewma_score": 0.14,
        "velocity": 0.09,
        "drift_vector": 0.07,
        "cpu_usage": 34.0,
        "memory_usage": 41.0,
    }))

    sim_thread = threading.Thread(target=simulator_loop, daemon=True)
    sim_thread.start()

    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("NERAIUM INVESTOR DEMO ACTIVE")
    print("Server running at http://0.0.0.0:8000")
    print("Scenarios: normal | degrading | incident")
    server.serve_forever()