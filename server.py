#!/usr/bin/env python3
“””
NERAIUM: Systemic Infrastructure Intelligence Platform
Telemetry generation and dashboard server.
“””

import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import urlparse

# ============================================================================

# CONFIGURATION

# ============================================================================

class SystemState(str, Enum):
STABLE = “STABLE”
WATCH = “WATCH”
ALERT = “ALERT”
UNKNOWN = “UNKNOWN”

class EventType(str, Enum):
BASELINE_STRUCTURE = “baseline_structure”
GRADUAL_DRIFT = “gradual_drift”
INSTABILITY_ESCALATION = “instability_escalation”

class Scenario(str, Enum):
NORMAL = “normal”
DEGRADING = “degrading”
INCIDENT = “incident”

@dataclass
class SensorValues:
“”“Tuple of (pressure, flow, tank, quality, vibration)”””

```
pressure: float
flow: float
tank: float
quality: float
vibration: float

def as_tuple(self) -> tuple:
    return (self.pressure, self.flow, self.tank, self.quality, self.vibration)
```

@dataclass
class Event:
“”“Infrastructure monitoring event.”””

```
id: int
event_type: EventType
timestamp: str
site_id: str
asset_id: str
state: SystemState
confidence: float
structural_drift_score: float
relational_stability_score: float
drift_velocity: float
lead_time_hours: Optional[float]
lead_time_confidence: float
structural_driver: str
predicted_impact: str
explanation: str
```

@dataclass
class GlobalState:
“”“Thread-safe global state.”””

```
events: list[Event] = field(default_factory=list)
paused: bool = False
scenario: Scenario = Scenario.NORMAL
last_drift: Optional[float] = None
lock: threading.Lock = field(default_factory=threading.Lock)

# Configuration
max_events: int = 300
event_generation_interval: float = 2.0
```

# Globals

STATE = GlobalState()

# Base sensor values (system operational baseline)

BASE_VALUES = SensorValues(
pressure=60.0,
flow=125.0,
tank=75.0,
quality=97.0,
vibration=0.20,
)

# Degradation thresholds

DRIFT_THRESHOLD_WATCH = 1.5
DRIFT_THRESHOLD_ALERT = 3.0
LEAD_TIME_BOUNDARY = 4.0

# ============================================================================

# UTILITY FUNCTIONS

# ============================================================================

def now_iso() -> str:
“”“Current timestamp in ISO 8601 format.”””
return datetime.now(timezone.utc).isoformat()

def structural_drift(values: SensorValues) -> float:
“””
Calculate structural drift as normalized deviation from baseline.
Higher values indicate greater deviation from operational norms.
“””
vals = values.as_tuple()
base = BASE_VALUES.as_tuple()
total = 0.0
for v, b in zip(vals, base):
if b != 0:
total += abs(v - b) / abs(b)
return total

def relational_stability(values: SensorValues) -> float:
“””
Calculate relational stability score (0.0 to 1.0).
Based on pressure-flow ratio, tank-pressure ratio, and vibration.
“””
pressure, flow, tank, quality, vibration = values.as_tuple()

```
# Relational constraints
score = 1.0 - (
    abs((pressure / max(flow, 1e-6)) - 0.5) * 0.4
    + abs((tank / max(pressure, 1e-6)) - 1.2) * 0.3
    + vibration * 0.3
)
return max(0.0, min(1.0, score))
```

def lead_time(drift: float, velocity: float) -> Optional[float]:
“””
Calculate estimated hours until crossing alert boundary.
Returns None if velocity <= 0 (no forward progress toward boundary).
“””
if velocity <= 0:
return None
hours = (LEAD_TIME_BOUNDARY - drift) / velocity
return max(0.0, round(hours, 1))

# ============================================================================

# SENSOR DATA GENERATION

# ============================================================================

def build_sensor_values() -> SensorValues:
“”“Generate sensor readings based on current scenario.”””
scenario = STATE.scenario
base = BASE_VALUES.as_tuple()

```
if scenario == Scenario.NORMAL:
    # Stable system — small noise
    return SensorValues(
        pressure=random.uniform(base[0] - 2, base[0] + 2),
        flow=random.uniform(base[1] - 3, base[1] + 3),
        tank=random.uniform(base[2] - 2, base[2] + 2),
        quality=random.uniform(base[3] - 1, base[3] + 1),
        vibration=random.uniform(0.18, 0.22),
    )

elif scenario == Scenario.DEGRADING:
    # Slow structural drift toward alarm
    drift = random.uniform(0.1, 0.8)
    return SensorValues(
        pressure=base[0] - drift * 5 + random.uniform(-2, 2),
        flow=base[1] - drift * 8 + random.uniform(-3, 3),
        tank=base[2] - drift * 4 + random.uniform(-2, 2),
        quality=base[3] - drift * 3 + random.uniform(-1, 1),
        vibration=base[4] + drift * 0.25,
    )

elif scenario == Scenario.INCIDENT:
    # Severe instability spike
    spike = random.uniform(1.5, 3.5)
    return SensorValues(
        pressure=base[0] - spike * 10 + random.uniform(-5, 5),
        flow=base[1] - spike * 15 + random.uniform(-8, 8),
        tank=base[2] - spike * 8 + random.uniform(-4, 4),
        quality=base[3] - spike * 6 + random.uniform(-2, 2),
        vibration=base[4] + spike * 0.4,
    )

return SensorValues(*base)
```

# ============================================================================

# EVENT GENERATION

# ============================================================================

def generate_event() -> None:
“”“Generate and store a telemetry event.”””
site = random.choice([“Reservoir East”, “North Loop”, “South Basin”, “West Feed Main”])
asset = random.choice(
[“Pump Station 1”, “District Main B”, “Distribution Node 7”]
)
timestamp = now_iso()

```
values = build_sensor_values()
drift = structural_drift(values)
stability = relational_stability(values)

# Calculate velocity (change in drift since last event)
velocity = 0.0
with STATE.lock:
    if STATE.last_drift is not None:
        velocity = drift - STATE.last_drift
    STATE.last_drift = drift

# Determine event state and impact
lt = lead_time(drift, velocity)

if drift > DRIFT_THRESHOLD_ALERT:
    state = SystemState.ALERT
    event_type = EventType.INSTABILITY_ESCALATION
    predicted_impact = "Potential localized service disruption within 1 to 2 hours."
elif drift > DRIFT_THRESHOLD_WATCH:
    state = SystemState.WATCH
    event_type = EventType.GRADUAL_DRIFT
    predicted_impact = "Early degradation detected. Maintenance window recommended."
else:
    state = SystemState.STABLE
    event_type = EventType.BASELINE_STRUCTURE
    predicted_impact = "No near term operational disruption expected."

event = Event(
    id=0,  # Will be set by append
    event_type=event_type,
    timestamp=timestamp,
    site_id=site,
    asset_id=asset,
    state=state,
    confidence=round(random.uniform(0.90, 0.99), 2),
    structural_drift_score=round(drift, 3),
    relational_stability_score=round(stability, 3),
    drift_velocity=round(velocity, 3),
    lead_time_hours=lt,
    lead_time_confidence=round(random.uniform(0.70, 0.95), 2),
    structural_driver="pressure-flow imbalance",
    predicted_impact=predicted_impact,
    explanation="SII analyzing structural geometry of the sensor network.",
)

# Append with thread safety
with STATE.lock:
    event.id = len(STATE.events) + 1
    STATE.events.append(event)
    if len(STATE.events) > STATE.max_events:
        STATE.events.pop(0)
```

def telemetry_loop() -> None:
“”“Continuously generate events at configured interval.”””
generate_event()  # First event immediately
while True:
time.sleep(STATE.event_generation_interval)
if not STATE.paused:
generate_event()

# ============================================================================

# HTTP REQUEST HANDLER

# ============================================================================

class RequestHandler(BaseHTTPRequestHandler):
“”“HTTP request handler for dashboard and API.”””

```
def send_json(self, data: dict, status: int = 200) -> None:
    """Send JSON response."""
    payload = json.dumps(data).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

def send_html(self, html: str, status: int = 200) -> None:
    """Send HTML response."""
    payload = html.encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

def do_GET(self) -> None:
    """Handle GET requests."""
    parsed = urlparse(self.path)
    path = parsed.path

    # Route to appropriate handler
    route_handlers = {
        "/": self._handle_dashboard,
        "/dashboard": self._handle_dashboard,
        "/api/status": self._handle_status,
        "/api/events": self._handle_events,
        "/api/pause": self._handle_pause,
        "/api/resume": self._handle_resume,
        "/api/reset": self._handle_reset,
        "/api/scenario/normal": self._handle_scenario_normal,
        "/api/scenario/degrading": self._handle_scenario_degrading,
        "/api/scenario/incident": self._handle_scenario_incident,
    }

    handler = route_handlers.get(path)
    if handler:
        handler()
    else:
        self.send_error(404)

def _handle_dashboard(self) -> None:
    """Serve dashboard HTML."""
    self.send_html(DASHBOARD_HTML)

def _handle_status(self) -> None:
    """Return current system status."""
    with STATE.lock:
        latest = (
            asdict(STATE.events[-1])
            if STATE.events
            else {
                "state": SystemState.UNKNOWN.value,
                "timestamp": "-",
                "site_id": "-",
                "asset_id": "-",
                "structural_drift_score": 0.0,
                "relational_stability_score": 0.0,
                "lead_time_hours": None,
                "lead_time_confidence": 0.0,
                "drift_velocity": 0.0,
                "structural_driver": "-",
                "predicted_impact": "—",
                "explanation": "Initializing structural telemetry...",
            }
        )
        out = {**latest}
        out["events_tracked"] = len(STATE.events)
        out["paused"] = STATE.paused
        out["scenario"] = STATE.scenario.value

    self.send_json(out)

def _handle_events(self) -> None:
    """Return all events."""
    with STATE.lock:
        events_list = [asdict(e) for e in STATE.events]
    self.send_json(events_list)

def _handle_pause(self) -> None:
    """Pause event generation."""
    STATE.paused = True
    self.send_json({"ok": True, "paused": True})

def _handle_resume(self) -> None:
    """Resume event generation."""
    STATE.paused = False
    self.send_json({"ok": True, "paused": False})

def _handle_reset(self) -> None:
    """Clear all events."""
    with STATE.lock:
        STATE.events.clear()
    STATE.last_drift = None
    self.send_json({"ok": True, "reset": True})

def _handle_scenario_normal(self) -> None:
    """Switch to normal scenario."""
    STATE.scenario = Scenario.NORMAL
    self.send_json({"ok": True, "scenario": STATE.scenario.value})

def _handle_scenario_degrading(self) -> None:
    """Switch to degrading scenario."""
    STATE.scenario = Scenario.DEGRADING
    self.send_json({"ok": True, "scenario": STATE.scenario.value})

def _handle_scenario_incident(self) -> None:
    """Switch to incident scenario."""
    STATE.scenario = Scenario.INCIDENT
    self.send_json({"ok": True, "scenario": STATE.scenario.value})

def log_message(self, format, *args):
    """Suppress default HTTP logging."""
    return
```

# ============================================================================

# DASHBOARD HTML

# ============================================================================

DASHBOARD_HTML = r”””<!doctype html>

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NERAIUM // SYSTEMIC INFRASTRUCTURE INTELLIGENCE</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; overflow-x: hidden; }
    body {
      font-family: 'IBM Plex Mono', monospace;
      background: #000000;
      color: #00FF41;
      font-size: 13px;
      line-height: 1.6;
      letter-spacing: 0.05em;
      background-image: 
        repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(0, 255, 65, 0.03) 1px, rgba(0, 255, 65, 0.03) 2px),
        repeating-linear-gradient(90deg, transparent, transparent 1px, rgba(0, 255, 65, 0.03) 1px, rgba(0, 255, 65, 0.03) 2px);
      background-size: 40px 40px;
      position: relative;
    }
    body::before {
      content: '';
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      pointer-events: none;
      background-image: repeating-linear-gradient(0deg, transparent 0px, transparent 2px, rgba(0, 255, 65, 0.02) 2px, rgba(0, 255, 65, 0.02) 4px);
      animation: scanlines 8s linear infinite;
      z-index: 1;
    }
    @keyframes scanlines { 0% { transform: translateY(0); } 100% { transform: translateY(10px); } }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 40px; position: relative; z-index: 2; border: 1px solid #00FF41; border-width: 1px 0; min-height: 100vh; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #00FF41; }
    .header-title { font-size: 16px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
    .header-stamp { font-size: 10px; text-align: right; letter-spacing: 1px; opacity: 0.6; }
    .hero { display: grid; grid-template-columns: 2fr 1fr; gap: 40px; margin-bottom: 40px; border: 1px solid #00FF41; padding: 30px; background: rgba(0, 255, 65, 0.01); }
    .hero-state { font-size: 96px; font-weight: 700; line-height: 1; margin-bottom: 20px; letter-spacing: -0.02em; }
    .state-stable { color: #00FF41; } .state-watch { color: #FFFF00; } .state-alert { color: #FF4141; }
    .hero-meta { font-size: 12px; opacity: 0.8; margin-bottom: 12px; }
    .hero-value { font-size: 28px; font-weight: 600; margin-bottom: 24px; color: #00FF41; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; margin-bottom: 40px; background: #00FF41; padding: 1px; }
    .metric-card { background: #000000; padding: 20px; border: 1px solid #00FF41; min-height: 120px; display: flex; flex-direction: column; justify-content: center; }
    .metric-label { font-size: 11px; letter-spacing: 1.5px; opacity: 0.7; text-transform: uppercase; margin-bottom: 12px; }
    .metric-value { font-size: 36px; font-weight: 700; color: #00FF41; font-variant-numeric: tabular-nums; }
    .controls { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; margin-bottom: 40px; background: #00FF41; padding: 1px; }
    button { background: #000000; border: 1px solid #00FF41; color: #00FF41; padding: 12px 16px; cursor: pointer; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; font-weight: 600; transition: all 100ms linear; }
    button:hover { background: #00FF41; color: #000000; }
    button:active { transform: scale(0.98); }
    .charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 40px; margin-bottom: 40px; }
    .chart-card { border: 1px solid #00FF41; padding: 20px; background: rgba(0, 255, 65, 0.01); }
    .chart-title { font-size: 12px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 20px; opacity: 0.8; }
    .chart-wrap { position: relative; height: 280px; width: 100%; }
    .events-card { border: 1px solid #00FF41; padding: 20px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { text-align: left; padding: 12px 8px; border-bottom: 1px solid #00FF41; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; font-weight: 600; opacity: 0.8; }
    td { padding: 12px 8px; border-bottom: 1px solid rgba(0, 255, 65, 0.3); }
    tr:last-child td { border-bottom: none; }
    .status-badge { display: inline-block; padding: 4px 8px; border: 1px solid currentColor; font-size: 10px; letter-spacing: 1px; }
    .status-stable { color: #00FF41; border-color: #00FF41; }
    .status-watch { color: #FFFF00; border-color: #FFFF00; }
    .status-alert { color: #FF4141; border-color: #FF4141; }
    .connection-indicator { display: inline-block; width: 8px; height: 8px; background: #00FF41; margin-right: 8px; animation: pulse 2s infinite; }
    .connection-indicator.disconnected { background: #FF4141; animation: none; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    @media (max-width: 1200px) { .charts-grid { grid-template-columns: 1fr; } .hero { grid-template-columns: 1fr; } }
    @media (max-width: 768px) { .metrics { grid-template-columns: repeat(2, 1fr); } .controls { grid-template-columns: repeat(2, 1fr); } .wrap { padding: 20px; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="header-title">◇ NERAIUM SII DASHBOARD ◇</div>
      <div class="header-stamp"><div>CLASSIFIED</div><div style="font-size: 9px; margin-top: 4px;">TELEMETRY ANALYSIS</div></div>
    </div>
    <div class="hero">
      <div>
        <div class="hero-meta">▸ SYSTEM STATE</div>
        <div id="systemState" class="hero-state state-stable">UNKNOWN</div>
        <div style="font-size: 12px; margin-bottom: 20px;">
          <div id="siteLine" style="margin-bottom: 8px;">Site: —</div>
          <div id="assetLine" style="margin-bottom: 8px;">Asset: —</div>
          <div id="lastTimestamp">Updated: —</div>
        </div>
        <div id="impactText" style="opacity: 0.8; font-size: 13px;">—</div>
      </div>
      <div>
        <div class="hero-meta">▸ CONFIDENCE</div>
        <div class="hero-value" id="confidenceValue">—</div>
        <div class="hero-meta" style="margin-top: 20px;">▸ EVENTS TRACKED</div>
        <div class="hero-value" id="eventsTracked">0</div>
        <div style="display: flex; align-items: center; font-size: 12px;">
          <div class="connection-indicator" id="connectionDot"></div>
          <span id="connectionText">Initializing...</span>
        </div>
      </div>
    </div>
    <div class="metrics">
      <div class="metric-card"><div class="metric-label">Structural Drift</div><div class="metric-value" id="driftValue">0.000</div></div>
      <div class="metric-card"><div class="metric-label">Relational Stability</div><div class="metric-value" id="stabilityValue">0.000</div></div>
      <div class="metric-card"><div class="metric-label">Drift Velocity</div><div class="metric-value" id="velocityValue">0.000</div></div>
      <div class="metric-card"><div class="metric-label">Warning Horizon</div><div class="metric-value" id="leadValue">—</div></div>
    </div>
    <div class="controls">
      <button onclick="hit('/api/pause')">⏸ Pause</button>
      <button onclick="hit('/api/resume')">▶ Resume</button>
      <button onclick="hit('/api/reset')">↻ Reset</button>
      <button onclick="hit('/api/scenario/normal')">⬜ Normal</button>
      <button onclick="hit('/api/scenario/degrading')">🟡 Degrade</button>
      <button onclick="hit('/api/scenario/incident')">🔴 Incident</button>
    </div>
    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">▸ Structural Drift Trend</div>
        <div class="chart-wrap"><canvas id="driftChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">▸ Relational Stability Trend</div>
        <div class="chart-wrap"><canvas id="stabilityChart"></canvas></div>
      </div>
    </div>
    <div class="events-card">
      <div class="chart-title">▸ Recent Structural Events</div>
      <table>
        <thead><tr><th>ID</th><th>Type</th><th>Timestamp</th><th>Site</th><th>Asset</th><th>State</th></tr></thead>
        <tbody id="eventsTableBody"><tr><td colspan="6" style="text-align: center; padding: 20px;">Loading telemetry...</td></tr></tbody>
      </table>
    </div>
  </div>
  <script>
    let driftChart = null;
    let stabilityChart = null;

```
function stateClass(state) {
  const s = String(state || "").toUpperCase();
  return s === "ALERT" ? "state-alert" : s === "WATCH" ? "state-watch" : "state-stable";
}

function statusClass(state) {
  const s = String(state || "").toUpperCase();
  return s === "ALERT" ? "status-alert" : s === "WATCH" ? "status-watch" : "status-stable";
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function pct(v) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n * 100) + "%" : "—";
}

function fmt(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function lead(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(1) + "h" : "—";
}

async function api(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

async function hit(url) {
  await fetch(url, { cache: "no-store" });
  await new Promise(r => setTimeout(r, 100));
  await refresh();
}

function buildCharts() {
  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false }, filler: { propagate: false } },
    scales: { x: { display: false }, y: { display: false, grid: { display: false } } }
  };

  driftChart = new Chart(document.getElementById("driftChart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: "#00FF41",
        backgroundColor: "rgba(0, 255, 65, 0.1)",
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: true
      }]
    },
    options: chartOpts
  });

  stabilityChart = new Chart(document.getElementById("stabilityChart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: "#00FF41",
        backgroundColor: "rgba(0, 255, 65, 0.1)",
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: true
      }]
    },
    options: { ...chartOpts, scales: { ...chartOpts.scales, y: { min: 0, max: 1, display: false, grid: { display: false } } } }
  });
}

async function refresh() {
  try {
    const [status, eventList] = await Promise.all([api("/api/status"), api("/api/events")]);

    const latest = eventList.length ? eventList[eventList.length - 1] : null;
    const stateEl = document.getElementById("systemState");
    stateEl.textContent = status.state || "UNKNOWN";
    stateEl.className = "hero-state " + stateClass(status.state);

    setText("siteLine", "Site: " + (status.site_id || "—"));
    setText("assetLine", "Asset: " + (status.asset_id || "—"));
    setText("lastTimestamp", "Updated: " + (status.timestamp ? status.timestamp.slice(11, 19) : "—"));
    setText("impactText", status.predicted_impact || "—");
    setText("confidenceValue", pct(status.lead_time_confidence));
    setText("eventsTracked", String(eventList.length));

    const dot = document.getElementById("connectionDot");
    const connText = document.getElementById("connectionText");
    if (status.paused) {
      dot.classList.add("disconnected");
      connText.textContent = "PAUSED";
    } else {
      dot.classList.remove("disconnected");
      connText.textContent = "CONNECTED";
    }

    setText("driftValue", fmt(status.structural_drift_score));
    setText("stabilityValue", fmt(status.relational_stability_score));
    setText("velocityValue", fmt(status.drift_velocity));
    setText("leadValue", lead(status.lead_time_hours));

    const recent = eventList.slice(-40);
    if (driftChart) {
      driftChart.data.labels = recent.map((_, i) => i + 1);
      driftChart.data.datasets[0].data = recent.map(e => Number(e.structural_drift_score || 0));
      driftChart.update("none");
    }

    if (stabilityChart) {
      stabilityChart.data.labels = recent.map((_, i) => i + 1);
      stabilityChart.data.datasets[0].data = recent.map(e => Number(e.relational_stability_score || 0));
      stabilityChart.update("none");
    }

    const tbody = document.getElementById("eventsTableBody");
    tbody.innerHTML = "";
    eventList.slice(-10).reverse().forEach((e) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${e.id}</td><td>${e.event_type}</td><td>${e.timestamp.slice(11, 19)}</td><td>${e.site_id}</td><td>${e.asset_id}</td><td><div class='status-badge ${statusClass(e.state)}'>${e.state}</div></td>`;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
    const dot = document.getElementById("connectionDot");
    dot.classList.add("disconnected");
    setText("connectionText", "DISCONNECTED");
  }
}

window.addEventListener("load", () => {
  buildCharts();
  refresh();
  setInterval(refresh, 2000);
});
```

  </script>
</body>
</html>
"""

# ============================================================================

# SERVER STARTUP

# ============================================================================

def run_server(host: str = “0.0.0.0”, port: int = 8000) -> None:
“”“Start the HTTP server and telemetry loop.”””
print(“╔════════════════════════════════════════════════════════════╗”)
print(“║  NERAIUM: SYSTEMIC INFRASTRUCTURE INTELLIGENCE PLATFORM    ║”)
print(“║  Dashboard: http://localhost:8000                         ║”)
print(“║  Status:    http://localhost:8000/api/status              ║”)
print(“║  Events:    http://localhost:8000/api/events              ║”)
print(“╚════════════════════════════════════════════════════════════╝”)

```
server = HTTPServer((host, port), RequestHandler)

# Start telemetry thread
telemetry_thread = threading.Thread(target=telemetry_loop, daemon=True)
telemetry_thread.start()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n✕ Server stopped.")
    server.server_close()
```

if **name** == “**main**”:
run_server()