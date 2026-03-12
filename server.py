import json
import math
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from db_store import init_db, insert
from neraium_core.pipeline import TelemetryPipeline
from neraium_core.store import EventStore
from neraium_core.telemetry import TelemetryPayload

pipeline = TelemetryPipeline()
store = EventStore()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def utc_now():
    return datetime.now(timezone.utc)


def iso_now():
    return utc_now().isoformat().replace("+00:00", "Z")


def clamp(value, low, high):
    return max(low, min(high, value))


def avg(values):
    return sum(values) / len(values) if values else 0.0


class SIIMemory:

    def __init__(self, lookback_events=180):
        self.lookback_events = lookback_events
        self.event_timeline = deque(maxlen=lookback_events)
        self.subsystem_trajectories = {}

    def record_event(self, event):
        self.event_timeline.append(event)
        for sub in event.get("subsystem", []):
            self.subsystem_trajectories.setdefault(sub, deque(maxlen=self.lookback_events))
            self.subsystem_trajectories[sub].append(event)

    def infer_degradation_state(self, subsystem):
        history = list(self.subsystem_trajectories.get(subsystem, []))
        if not history:
            return {
                "state": "unknown",
                "confidence": 0.0,
                "evidence": [],
                "avg_score": 0.0,
                "score_trend": 0.0,
                "event_count": 0,
                "recent_topology_changes": 0,
            }

        scores = [float(e.get("ewma_score", 0.0)) for e in history]
        velocities = [float(e.get("drift_velocity", 0.0)) for e in history]
        topo = [float(e.get("topology_drift", 0.0)) for e in history]

        recent_scores = scores[-7:] if len(scores) > 7 else scores
        recent_velocities = velocities[-7:] if len(velocities) > 7 else velocities

        avg_score = avg(recent_scores)
        avg_velocity = avg(recent_velocities)
        topo_events = sum(1 for t in topo[-14:] if t > 0.10)

        if len(recent_scores) > 1:
            score_trend = avg([recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))])
        else:
            score_trend = 0.0

        evidence = []

        if avg_score > 80 or avg_velocity > 30:
            state = "critical"
            confidence = 0.90
            evidence.append("High anomaly score " + str(round(avg_score, 1)))
        elif avg_score > 50 or avg_velocity > 10 or topo_events > 2:
            state = "stressed"
            confidence = 0.75
            evidence.append("Moderate anomaly score " + str(round(avg_score, 1)))
            if topo_events > 2:
                evidence.append("Topology instability " + str(topo_events) + " events")
        elif score_trend > 0.5 or len(history) > 5:
            state = "drifting"
            confidence = 0.65
            evidence.append("Sustained trend detected")
            if score_trend > 0:
                evidence.append("Positive score trend " + str(round(score_trend, 2)))
        slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores)) / denom
        current_score = scores[-1]
        critical = 90.0

        if slope <= 0.1:
            return {
                "days_to_failure": None,
                "confidence": 0.5,
                "method": "stable",
                "current_score": round(current_score, 2),
                "trend_slope": round(slope, 3),
            }

        if current_score >= critical:
            days_to_failure = 0.0
            confidence = 0.95
        else:
            days_to_failure = max(1.0, (critical - current_score) / slope)
            confidence = max(0.3, 0.8 - (days_to_failure / 90.0))

        return {
            "days_to_failure": round(days_to_failure, 1),
            "confidence": round(confidence, 2),
            "method": "linear_extrapolation",
            "current_score": round(current_score, 2),
            "trend_slope": round(slope, 3),
        }

    def detect_pattern_recurrence(self, event_type, subsystem):
        history = list(self.subsystem_trajectories.get(subsystem, []))
        matching = [e for e in history if e.get("water_event_type") == event_type]

        if not matching:
            return {
                "recurrence_count": 0,
                "is_chronic": False,
                "average_gap_events": None,
                "last_occurrence": None,
            }

        if len(matching) >= 3:
            positions = [i for i, e in enumerate(history) if e.get("water_event_type") == event_type]
            gaps = [positions[i] - positions[i-1] for i in range(1, len(positions))]
            avg_gap = avg(gaps) if gaps else None
            is_chronic = avg_gap is None or avg_gap < 12
        else:
            avg_gap = None
            is_chronic = False

        return {
            "recurrence_count": len(matching),
            "is_chronic": is_chronic,
            "average_gap_events": round(avg_gap, 2) if avg_gap is not None else None,
            "last_occurrence": matching[-1].get("timestamp"),
        }

    def subsystem_summary(self):
        return {
            sub: {
                "degradation_state": self.infer_degradation_state(sub),
                "failure_window": self.predict_failure_window(sub),
            }
            for sub in self.subsystem_trajectories
        }

    def summary_report(self):
        if not self.subsystem_trajectories:
            return "No events recorded yet."

        lines = ["SII SYSTEM MEMORY REPORT", ""]
        for sub in sorted(self.subsystem_trajectories):
            state = self.infer_degradation_state(sub)
            pred = self.predict_failure_window(sub)
            lines.append("Subsystem: " + sub)
            lines.append("State: " + state["state"].upper() + " confidence: " + str(round(state["confidence"] * 100)) + "%")
            for ev in state["evidence"]:
                lines.append(" - " + ev)
            if pred["days_to_failure"] is not None:
                lines.append("Failure window: " + str(pred["days_to_failure"]) + " days")
            lines.append("")

        return "\n".join(lines)


memory = SIIMemory()

timeline_scores = deque([0.0] * 24, maxlen=24)
timeline_ewma = deque([0.0] * 24, maxlen=24)
timeline_velocity = deque([0.0] * 24, maxlen=24)
timeline_topology = deque([0.0] * 24, maxlen=24)

recent_logs = deque(["[boot] SII console initialized", "[info] Waiting for telemetry"], maxlen=40)

_prev_ewma = 0.0
_prev_score = None
_prev_vector = None
_event_counter = 0


def infer_water_event(top_sensors, score, velocity, topology):
    blob = " ".join(top_sensors).lower()
    if any(k in blob for k in ["chlorine", "turbidity", "ph", "quality"]):
        return "contamination"
    if any(k in blob for k in ["pump", "motor"]):
        return "pump_fault"
    if topology > 0.18 and score < 50:
        return "topology_change"
    if any(k in blob for k in ["pressure", "flow"]) and velocity > 7:
        return "leak_signature"
    if score > 4.0:
        return "hydraulic_anomaly"
    return "unknown"


def build_root_cause(top_sensor, subsystem, causal_links, water_event_type, score):
    upstream = []
    downstream = []

    for link in causal_links:
        if abs(link["strength"]) > 0.5:
            if link["source"] == top_sensor:
                downstream.append(link["target"])
            elif link["target"] == top_sensor:
                upstream.append(link["source"])

    explanation = "Structural drift detected in " + top_sensor + "."
    if water_event_type == "leak_signature":
        explanation += " Pattern consistent with pressure loss and flow redistribution."
    elif water_event_type == "contamination":
        explanation += " Water quality degradation signature is dominant."
    elif water_event_type == "pump_fault":
        explanation += " Mechanical or electrical pumping anomaly likely."
    elif water_event_type == "topology_change":
        explanation += " Correlation structure changed without a large point anomaly."
    elif water_event_type == "hydraulic_anomaly":
        explanation += " Geometric deviation exceeds normal operational range."

    if upstream:
        explanation += " Upstream influence from " + ", ".join(upstream[:2]) + "."
    if downstream:
        explanation += " Affecting downstream: " + ", ".join(downstream[:2]) + "."

    return {
        "primary_sensor": top_sensor,
        "subsystem": subsystem,
        "explanation": explanation,
        "upstream_sources": upstream,
        "downstream_effects": downstream,
        "confidence": round(clamp(score / 10.0, 0.1, 0.95), 2),
    }


def enrich_with_sii(pipeline_event, cpu, mem):
    global _prev_ewma, _prev_score, _prev_vector, _event_counter

    _event_counter += 1
    raw_score = float(pipeline_event.get("score", 0.0))
    ewma = round(0.2 * raw_score + 0.8 * _prev_ewma, 4)

    if _prev_score is None:
        velocity = 0.0
    else:
        last_vel = list(timeline_velocity)[-1] if timeline_velocity else 0.0
        velocity = round(0.3 * (raw_score - _prev_score) + 0.7 * last_vel, 4)

    current_vector = [cpu, mem, (cpu + mem) / 2.0, abs(cpu - mem)]

    if _prev_vector is None:
        topology_drift = 0.0
    else:
        diffs = [abs(a - b) for a, b in zip(current_vector, _prev_vector)]
        topology_drift = round((sum(diffs) / len(diffs)) / 100.0, 4)

    cluster_id = 1 if cpu < 40 else 2 if cpu < 70 else 3

    sensor_contributions = {
        "pressure_main": round(abs(cpu - 50) * 0.9 + abs(mem - 55) * 0.2, 3),
        "flow_north": round(abs(mem - 55) * 0.8 + abs(cpu - 50) * 0.3, 3),
        "pump_vibration_2": round(max(0.0, cpu - 60) * 0.7, 3),
        "chlorine_residual": round(max(0.0, mem - 80) * 0.5, 3),
        "zone_valve_feedback": round(topology_drift * 100.0 * 0.8, 3),
    }

    ranked = sorted(sensor_contributions.items(), key=lambda x: x[1], reverse=True)
    top_sensors = [name for name, _ in ranked[:4]]
    sensor_contribs = {k: v for k, v in ranked[:4]}

    subsystem = []
    if "chlorine_residual" in top_sensors:
        subsystem.append("quality_zone_alpha")
    if "pump_vibration_2" in top_sensors:
        subsystem.append("pump_station_west")
    if any(s in top_sensors for s in ["pressure_main", "flow_north", "zone_valve_feedback"]):
        subsystem.append("distribution_zone_a")
    if not subsystem:
        subsystem.append("distribution_zone_a")

    causal_links = []
    if raw_score > 2.0:
        causal_links = [
            {"source": "pressure_main", "target": "flow_north", "lag": 2, "strength": 0.71},
            {"source": "flow_north", "target": "zone_valve_feedback", "lag": 1, "strength": 0.52},
        ]
        if "pump_vibration_2" in top_sensors:
            causal_links.append({"source": "pump_vibration_2", "target": "pressure_main", "lag": 1, "strength": -0.44})

    water_event_type = infer_water_event(top_sensors, raw_score, velocity, topology_drift)

    is_anomaly = (pipeline_event.get("status") == "anomaly" or topology_drift > 0.10 or abs(velocity) > 6.0)
    is_persistent = ewma > 3.0 or (is_anomaly and _event_counter > 3)

    root_cause = build_root_cause(
        top_sensor=top_sensors[0] if top_sensors else "unknown",
        subsystem=subsystem,
        causal_links=causal_links,
        water_event_type=water_event_type,
        score=raw_score,
    )

    detection = {
        "timestamp": pipeline_event.get("timestamp", iso_now()),
        "cluster_id": cluster_id,
        "score": round(raw_score, 4),
        "ewma_score": ewma,
        "drift_velocity": velocity,
        "topology_drift": topology_drift,
        "is_anomaly": is_anomaly,
        "is_persistent": is_persistent,
        "top_sensors": top_sensors,
        "sensor_contributions": sensor_contribs,
        "subsystem": subsystem,
        "causal_links": causal_links,
        "water_event_type": water_event_type,
        "root_cause": root_cause,
    }

    _prev_score = raw_score
    _prev_ewma = ewma
    _prev_vector = current_vector

    timeline_scores.append(raw_score)
    timeline_ewma.append(ewma)
    timeline_velocity.append(velocity)
    timeline_topology.append(topology_drift)

    memory.record_event(detection)

    recent_logs.append("[info] score=" + str(round(raw_score, 3)) + " ewma=" + str(round(ewma, 3)) + " vel=" + str(round(velocity, 3)) + " topo=" + str(round(topology_drift, 4)))
    if is_anomaly:
        recent_logs.append("[warn] anomaly cluster=" + str(cluster_id) + " " + water_event_type + " persistent=" + str(is_persistent))

    return detection


class NeraiumHandler(BaseHTTPRequestHandler):

    def send_json(self, code, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, filename):
        path = STATIC_DIR / filename
        if not path.exists():
            self.send_json(404, {"error": filename + " not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def build_dashboard_payload(self):
        events = store.all() or []
        latest = events[-1] if events else None

        if latest:
            latest_ts = latest.get("timestamp", utc_now())
            if isinstance(latest_ts, str):
                latest_ts = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
            data_age_seconds = max(0, int((utc_now() - latest_ts).total_seconds()))
            signals = latest.get("signals", {}) or {}
            cpu = float(signals.get("cpu_usage", 0.0))
            mem = float(signals.get("memory_usage", 0.0))
        else:
            data_age_seconds = 999
            cpu = 0.0
            mem = 0.0

        class _P:
            pass

        _P.signals = {"cpu_usage": cpu, "memory_usage": mem}

        pipeline_event = pipeline.process(_P())
        detection = enrich_with_sii(pipeline_event, cpu, mem)

        subsystem_states = memory.subsystem_summary()
        state_priority = {"critical": 4, "stressed": 3, "drifting": 2, "healthy": 1, "unknown": 0}
        worst_state = "healthy"
        worst_conf = 0.85

        for sub_data in subsystem_states.values():
            s = sub_data["degradation_state"]["state"]
            if state_priority.get(s, 0) > state_priority.get(worst_state, 0):
                worst_state = s
                worst_conf = sub_data["degradation_state"]["confidence"]

        nearest_failure = None
        for sub_data in subsystem_states.values():
            fw = sub_data["failure_window"]
            dtf = fw.get("days_to_failure")
            if dtf is not None:
                if nearest_failure is None or dtf < nearest_failure["days_to_failure"]:
                    nearest_failure = fw

        return {
            "timestamp": iso_now(),
            "data_age_seconds": data_age_seconds,
            "system_state": {
                "state": worst_state,
                "confidence": round(worst_conf, 2),
            },
            "failure_window": nearest_failure,
            "current_detection": detection,
            "subsystem_states": subsystem_states,
            "timelines": {
                "score": list(timeline_scores),
                "ewma": list(timeline_ewma),
                "velocity": list(timeline_velocity),
                "topology": list(timeline_topology),
            },
            "memory_report": memory.summary_report(),
            "recent_logs": list(recent_logs),
            "event_count": len(memory.event_timeline),
        }

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path in ("", "/"):
            self.send_html("index.html")
        elif path == "/api/dashboard":
            self.send_json(200, self.build_dashboard_payload())
        elif path == "/api/events":
            events = store.all() or []
            self.send_json(200, {"events": events, "count": len(events)})
        elif path == "/api/memory":
            self.send_json(200, {
                "subsystem_states": memory.subsystem_summary(),
                "report": memory.summary_report(),
                "event_count": len(memory.event_timeline),
            })
        elif path == "/api/health":
            self.send_json(200, {
                "status": "ok",
                "timestamp": iso_now(),
                "event_count": len(memory.event_timeline),
            })
        else:
            self.send_json(404, {"error": "Unknown route: " + path})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/api/ingest":
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self.send_json(400, {"error": "Empty body"})
                return

            try:
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.send_json(400, {"error": "Invalid JSON: " + str(e)})
                return

            try:
                payload = TelemetryPayload(**body)
                insert(payload)
                store.add(payload.__dict__)

                pipeline_event = pipeline.process(payload)
                cpu = float(payload.signals.get("cpu_usage", 0.0))
                mem = float(payload.signals.get("memory_usage", 0.0))
                enrich_with_sii(pipeline_event, cpu, mem)

                signals = body.get("signals", {})
                recent_logs.append("[ingest] " + str(payload.timestamp) + " cpu=" + str(signals.get("cpu_usage", "?")) + " mem=" + str(signals.get("memory_usage", "?")))
                self.send_json(200, {"status": "accepted", "timestamp": iso_now()})
            except Exception as e:
                self.send_json(422, {"error": "Payload error: " + str(e)})
        else:
            self.send_json(404, {"error": "Unknown POST route: " + path})

    def log_message(self, fmt, *args):
        pass


def run(host="0.0.0.0", port=8000):
    init_db()
    server = HTTPServer((host, port), NeraiumHandler)
    print("[boot] Neraium SII server running on http://" + host + ":" + str(port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[shutdown] Server stopped.")


if __name__ == "__main__":
    run()