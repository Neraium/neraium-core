import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

EVENTS = []
MAX_EVENTS = 200


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


def simulator_loop():
    phase = "stable"

    while True:
        prev = latest_event() or {
            "sii_score": 0.20,
            "ewma_score": 0.12,
            "velocity": 0.08,
            "drift_vector": 0.06,
            "cpu_usage": 35.0,
            "memory_usage": 42.0,
        }

        sii = float(prev.get("sii_score", 0.20))
        ewma = float(prev.get("ewma_score", 0.12))
        velocity = float(prev.get("velocity", 0.08))
        drift_vector = float(prev.get("drift_vector", 0.06))
        cpu = float(prev.get("cpu_usage", 35.0))
        memory = float(prev.get("memory_usage", 42.0))

        roll = random.random()
        if phase == "stable" and roll < 0.12:
            phase = "watch"
        elif phase == "watch" and roll < 0.18:
            phase = "alert"
        elif phase == "alert" and roll < 0.35:
            phase = "watch"
        elif phase == "watch" and roll < 0.22:
            phase = "stable"

        if phase == "stable":
            sii = clamp(sii + random.uniform(-0.03, 0.03), 0.08, 0.35)
            ewma = clamp(ewma + random.uniform(-0.02, 0.02), 0.05, 0.25)
            velocity = clamp(velocity + random.uniform(-0.02, 0.02), 0.03, 0.22)
            drift_vector = clamp(drift_vector + random.uniform(-0.02, 0.02), 0.02, 0.20)
            cpu = clamp(cpu + random.uniform(-5, 5), 18, 65)
            memory = clamp(memory + random.uniform(-4, 4), 25, 70)
            confidence = clamp(random.uniform(0.87, 0.96), 0, 1)
            state = "STABLE"
        elif phase == "watch":
            sii = clamp(sii + random.uniform(0.01, 0.06), 0.30, 0.62)
            ewma = clamp(ewma + random.uniform(0.01, 0.05), 0.20, 0.50)
            velocity = clamp(velocity + random.uniform(0.01, 0.04), 0.12, 0.40)
            drift_vector = clamp(drift_vector + random.uniform(0.01, 0.04), 0.10, 0.34)
            cpu = clamp(cpu + random.uniform(2, 8), 40, 82)
            memory = clamp(memory + random.uniform(2, 7), 40, 84)
            confidence = clamp(random.uniform(0.78, 0.90), 0, 1)
            state = "WATCH"
        else:
            sii = clamp(sii + random.uniform(0.04, 0.10), 0.60, 0.98)
            ewma = clamp(ewma + random.uniform(0.03, 0.08), 0.45, 0.90)
            velocity = clamp(velocity + random.uniform(0.02, 0.07), 0.25, 0.75)
            drift_vector = clamp(drift_vector + random.uniform(0.02, 0.07), 0.20, 0.70)
            cpu = clamp(cpu + random.uniform(4, 10), 65, 98)
            memory = clamp(memory + random.uniform(4, 9), 60, 98)
            confidence = clamp(random.uniform(0.82, 0.97), 0, 1)
            state = "ALERT"

        event = make_event({
            "type": "telemetry",
            "timestamp": now_iso(),
            "state": state,
            "confidence": round(confidence, 2),
            "sii_score": round(sii, 2),
            "ewma_score": round(ewma, 2),
            "velocity": round(velocity, 2),
            "drift_vector": round(drift_vector, 2),
            "cpu_usage": round(cpu, 1),
            "memory_usage": round(memory, 1),
        })

        add_event(event)
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
        path = self.path.split("?", 1)[0]

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

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]

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
    print("Server running at http://0.0.0.0:8000")
    print("Fake telemetry simulator feeding data every 2 seconds")
    server.serve_forever()
    server = HTTPServer(("0.0.0.0", 8000), Handler)