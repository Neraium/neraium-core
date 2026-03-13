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
PAUSED = False

SITES = ["alpha-water-grid", "reservoir-east", "north-loop"]
ASSETS = ["pump-station-1", "district-main-b", "distribution-node-7"]

SENSOR_NAMES = [
    "pressure_inlet",
    "pressure_outlet",
    "flow_rate",
    "tank_level",
    "quality_index",
    "pump_vibration",
]

BASE_SENSOR_VALUES = {
    "pressure_inlet": 62.0,
    "pressure_outlet": 58.0,
    "flow_rate": 128.0,
    "tank_level": 73.0,
    "quality_index": 96.0,
    "pump_vibration": 0.22,
}


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


def set_scenario(value):
    global SCENARIO
    if value in {"normal", "degrading", "incident"}:
        SCENARIO = value
    else:
        SCENARIO = "normal"


def reset_demo():
    EVENTS.clear()
    add_event(make_startup_event())


def make_startup_event():
    return {
        "id": 1,
        "event_type": "startup",
        "site_id": "alpha-water-grid",
        "asset_id": "pump-station-1",
        "timestamp": now_iso(),
        "state": "STABLE",
        "confidence": 0.93,
        "structural_drift_score": 0.14,
        "relational_stability_score": 0.92,
        "early_warning_horizon_hours": 72,
        "predicted_impact": "No near term operational disruption expected.",
        "sensor_names": SENSOR_NAMES,
        "sensor_values": {
            "pressure_inlet": 62.0,
            "pressure_outlet": 58.0,
            "flow_rate": 128.0,
            "tank_level": 73.0,
            "quality_index": 96.0,
            "pump_vibration": 0.22,
        },
        "sensor_quality_flags": {name: "ok" for name in SENSOR_NAMES},
        "explanation": "Baseline structural relationships remain stable across the monitored sensor network.",
    }


def event_type_for_state(state, scenario):
    if state == "ALERT":
        return random.choice([
            "relational_break",
            "structural_drift_spike",
            "instability_escalation",
            "cross_sensor_divergence",
        ])
    if scenario == "degrading":
        return random.choice([
            "gradual_drift",
            "correlation_shift",
            "relational_variance",
            "stability_decay",
        ])
    return random.choice([
        "baseline_structure",
        "stable_correlation",
        "normal_telemetry_frame",
        "relational_observation",
    ])


def explanation_for_state(state):
    if state == "ALERT":
        return "SII is detecting a material change in structural relationships between sensors, indicating active system instability."
    if state == "WATCH":
        return "SII is detecting gradual relational drift before threshold-level failure."
    return "Sensor relationships remain structurally consistent with healthy operating behavior."


def predicted_impact(state, early_warning_horizon_hours):
    if state == "ALERT":
        return "Intervention likely required if relational drift continues to compound."
    if state == "WATCH":
        return f"Early instability detected with approximately {early_warning_horizon_hours} hours of warning."
    return "No near term operational disruption expected."


def build_sensor_values(prev_values, scenario):
    vals = dict(prev_values)

    if scenario == "normal":
        vals["pressure_inlet"] = clamp(vals["pressure_inlet"] + random.uniform(-1.0, 1.0), 59, 65)
        vals["pressure_outlet"] = clamp(vals["pressure_outlet"] + random.uniform(-1.0, 1.0), 55, 61)
        vals["flow_rate"] = clamp(vals["flow_rate"] + random.uniform(-5, 5), 118, 136)
        vals["tank_level"] = clamp(vals["tank_level"] + random.uniform(-1.2, 1.2), 68, 77)
        vals["quality_index"] = clamp(vals["quality_index"] + random.uniform(-0.8, 0.8), 93, 99)
        vals["pump_vibration"] = clamp(vals["pump_vibration"] + random.uniform(-0.03, 0.03), 0.16, 0.30)

    elif scenario == "degrading":
        vals["pressure_inlet"] = clamp(vals["pressure_inlet"] + random.uniform(-2.2, 0.3), 50, 63)
        vals["pressure_outlet"] = clamp(vals["pressure_outlet"] + random.uniform(-2.5, 0.2), 44, 59)
        vals["flow_rate"] = clamp(vals["flow_rate"] + random.uniform(-8, 2), 98, 132)
        vals["tank_level"] = clamp(vals["tank_level"] + random.uniform(-2.2, 0.4), 58, 76)
        vals["quality_index"] = clamp(vals["quality_index"] + random.uniform(-2.8, 0.2), 82, 97)
        vals["pump_vibration"] = clamp(vals["pump_vibration"] + random.uniform(0.00, 0.06), 0.20, 0.48)

    else:
        vals["pressure_inlet"] = clamp(vals["pressure_inlet"] + random.uniform(-4.5, -1.0), 35, 58)
        vals["pressure_outlet"] = clamp(vals["pressure_outlet"] + random.uniform(-5.0, -1.5), 28, 54)
        vals["flow_rate"] = clamp(vals["flow_rate"] + random.uniform(-15, -2), 72, 120)
        vals["tank_level"] = clamp(vals["tank_level"] + random.uniform(-3.0, 0.2), 48, 72)
        vals["quality_index"] = clamp(vals["quality_index"] + random.uniform(-5.0, -0.8), 64, 90)
        vals["pump_vibration"] = clamp(vals["pump_vibration"] + random.uniform(0.03, 0.10), 0.28, 0.85)

    return vals


def generate_event(prev):
    prev_drift = float(prev.get("structural_drift_score", 0.14))
    prev_stability = float(prev.get("relational_stability_score", 0.92))
    prev_warning = int(prev.get("early_warning_horizon_hours", 72))
    prev_values = prev.get("sensor_values", BASE_SENSOR_VALUES)

    sensor_values = build_sensor_values(prev_values, SCENARIO)

    if SCENARIO == "normal":
        drift = clamp(prev_drift + random.uniform(-0.02, 0.02), 0.08, 0.24)
        stability = clamp(prev_stability + random.uniform(-0.02, 0.02), 0.86, 0.97)
        warning = clamp(prev_warning + random.randint(-4, 4), 48, 120)
        confidence = round(random.uniform(0.90, 0.97), 2)
        state = "STABLE"

    elif SCENARIO == "degrading":
        drift = clamp(prev_drift + random.uniform(0.01, 0.05), 0.20, 0.62)
        stability = clamp(prev_stability + random.uniform(-0.05, -0.01), 0.52, 0.86)
        warning = clamp(prev_warning + random.randint(-8, -2), 18, 72)
        confidence = round(random.uniform(0.84, 0.94), 2)
        state = "WATCH" if drift < 0.50 else "ALERT"

    else:
        drift = clamp(prev_drift + random.uniform(0.03, 0.08), 0.55, 0.98)
        stability = clamp(prev_stability + random.uniform(-0.08, -0.03), 0.20, 0.62)
        warning = clamp(prev_warning + random.randint(-12, -4), 4, 36)
        confidence = round(random.uniform(0.87, 0.98), 2)
        state = "ALERT"

    return {
        "id": next_id(),
        "event_type": event_type_for_state(state, SCENARIO),
        "site_id": random.choice(SITES),
        "asset_id": random.choice(ASSETS),
        "timestamp": now_iso(),
        "state": state,
        "confidence": confidence,
        "structural_drift_score": round(drift, 2),
        "relational_stability_score": round(stability, 2),
        "early_warning_horizon_hours": int(warning),
        "predicted_impact": predicted_impact(state, int(warning)),
        "sensor_names": SENSOR_NAMES,
        "sensor_values": sensor_values,
        "sensor_quality_flags": {name: "ok" for name in SENSOR_NAMES},
        "explanation": explanation_for_state(state),
    }


def simulator_loop():
    while True:
        try:
            if not PAUSED:
                prev = latest_event() or {}
                add_event(generate_event(prev))
            time.sleep(2)
        except Exception as e:
            print("SIMULATOR ERROR:", e)
            time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, path, content_type):
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global PAUSED

        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            return self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")

        if path == "/static/styles.css":
            return self.send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")

        if path == "/static/app.js":
            return self.send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")

        if path == "/api/events":
            return self.send_json(EVENTS)

        if path == "/api/status":
            latest = latest_event() or {}
            return self.send_json({
                "connected": True,
                "scenario": SCENARIO,
                "paused": PAUSED,
                "events_tracked": len(EVENTS),
                "state": latest.get("state", "UNKNOWN"),
                "site_id": latest.get("site_id", "-"),
                "asset_id": latest.get("asset_id", "-"),
                "confidence": latest.get("confidence", 0),
                "structural_drift_score": latest.get("structural_drift_score", 0),
                "relational_stability_score": latest.get("relational_stability_score", 0),
                "early_warning_horizon_hours": latest.get("early_warning_horizon_hours", 0),
                "predicted_impact": latest.get("predicted_impact", ""),
                "explanation": latest.get("explanation", ""),
                "last_timestamp": latest.get("timestamp", "None"),
            })

        if path == "/api/scenario":
            mode = query.get("mode", ["normal"])[0].lower()
            set_scenario(mode)
            return self.send_json({"status": "ok", "scenario": SCENARIO})

        if path == "/api/pause":
            PAUSED = True
            return self.send_json({"status": "ok", "paused": True})

        if path == "/api/resume":
            PAUSED = False
            return self.send_json({"status": "ok", "paused": False})

        if path == "/api/reset":
            reset_demo()
            return self.send_json({"status": "ok"})

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        self.send_error(404, "Not found")

    def log_message(self, format, *args):
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


if __name__ == "__main__":
    reset_demo()
    threading.Thread(target=simulator_loop, daemon=True).start()
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("NERAIUM SII DEMO ACTIVE")
    print("Server running at http://0.0.0.0:8000")
    server.serve_forever()