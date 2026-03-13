import json
import math
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

events = []
paused = False
scenario = “normal”
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
abs((pressure / max(flow, 1e-6)) - 0.5) * 0.4 +
abs((tank / max(pressure, 1e-6)) - 1.2) * 0.3 +
vibration * 0.3
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
if scenario == “normal”:
return (
random.uniform(58, 64),
random.uniform(118, 134),
random.uniform(70, 78),
random.uniform(95, 99),
random.uniform(0.15, 0.25),
)

```
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
```

def generate_event():
global last_drift

```
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
    "explanation": "SII analyzing structural geometry of the sensor network."
}

with lock:
    events.append(event)
    if len(events) > MAX_EVENTS:
        events.pop(0)
```

def telemetry_loop():
global paused
while True:
with lock:
if not paused:
pass
if not paused:
generate_event()
time.sleep(2)

DASHBOARD_HTML = r”””<!doctype html>

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NERAIUM // SYSTEMIC INFRASTRUCTURE INTELLIGENCE</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

```
html, body {
  width: 100%;
  height: 100%;
  overflow-x: hidden;
}

body {
  font-family: 'IBM Plex Mono', monospace;
  background: #000000;
  color: #00FF41;
  font-size: 13px;
  line-height: 1.6;
  letter-spacing: 0.05em;
  background-image: 
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 1px,
      rgba(0, 255, 65, 0.03) 1px,
      rgba(0, 255, 65, 0.03) 2px
    ),
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 1px,
      rgba(0, 255, 65, 0.03) 1px,
      rgba(0, 255, 65, 0.03) 2px
    );
  background-size: 40px 40px;
  position: relative;
}

body::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  background-image: 
    repeating-linear-gradient(
      0deg,
      transparent 0px,
      transparent 2px,
      rgba(0, 255, 65, 0.02) 2px,
      rgba(0, 255, 65, 0.02) 4px
    );
  animation: scanlines 8s linear infinite;
  z-index: 1;
}

@keyframes scanlines {
  0% { transform: translateY(0); }
  100% { transform: translateY(10px); }
}

.wrap {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px;
  position: relative;
  z-index: 2;
  border: 1px solid #00FF41;
  border-width: 1px 0;
  min-height: 100vh;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40px;
  padding-bottom: 20px;
  border-bottom: 1px solid #00FF41;
}

.header-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.header-stamp {
  font-size: 10px;
  text-align: right;
  letter-spacing: 1px;
  opacity: 0.6;
}

.hero {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 40px;
  margin-bottom: 40px;
  border: 1px solid #00FF41;
  padding: 30px;
  background: rgba(0, 255, 65, 0.01);
}

.hero-state {
  font-size: 96px;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 20px;
  letter-spacing: -0.02em;
}

.state-stable { color: #00FF41; }
.state-watch { color: #FFFF00; }
.state-alert { color: #FF4141; }

.hero-meta {
  font-size: 12px;
  opacity: 0.8;
  margin-bottom: 12px;
}

.hero-value {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 24px;
  color: #00FF41;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  margin-bottom: 40px;
  background: #00FF41;
  padding: 1px;
}

.metric-card {
  background: #000000;
  padding: 20px;
  border: 1px solid #00FF41;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.metric-label {
  font-size: 11px;
  letter-spacing: 1.5px;
  opacity: 0.7;
  text-transform: uppercase;
  margin-bottom: 12px;
}

.metric-value {
  font-size: 36px;
  font-weight: 700;
  color: #00FF41;
  font-variant-numeric: tabular-nums;
}

.controls {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  margin-bottom: 40px;
  background: #00FF41;
  padding: 1px;
}

button {
  background: #000000;
  border: 1px solid #00FF41;
  color: #00FF41;
  padding: 12px 16px;
  cursor: pointer;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-weight: 600;
  transition: all 100ms linear;
  position: relative;
}

button:hover {
  background: #00FF41;
  color: #000000;
}

button:active {
  transform: scale(0.98);
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 40px;
  margin-bottom: 40px;
}

.chart-card {
  border: 1px solid #00FF41;
  padding: 20px;
  background: rgba(0, 255, 65, 0.01);
}

.chart-title {
  font-size: 12px;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 20px;
  opacity: 0.8;
}

.chart-wrap {
  position: relative;
  height: 280px;
  width: 100%;
}

.events-card {
  border: 1px solid #00FF41;
  padding: 20px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

th {
  text-align: left;
  padding: 12px 8px;
  border-bottom: 1px solid #00FF41;
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-weight: 600;
  opacity: 0.8;
}

td {
  padding: 12px 8px;
  border-bottom: 1px solid rgba(0, 255, 65, 0.3);
}

tr:last-child td {
  border-bottom: none;
}

.status-badge {
  display: inline-block;
  padding: 4px 8px;
  border: 1px solid currentColor;
  font-size: 10px;
  letter-spacing: 1px;
}

.status-stable {
  color: #00FF41;
  border-color: #00FF41;
}

.status-watch {
  color: #FFFF00;
  border-color: #FFFF00;
}

.status-alert {
  color: #FF4141;
  border-color: #FF4141;
}

.connection-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: #00FF41;
  margin-right: 8px;
  animation: pulse 2s infinite;
}

.connection-indicator.disconnected {
  background: #FF4141;
  animation: none;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

@media (max-width: 1200px) {
  .charts-grid { grid-template-columns: 1fr; }
  .hero { grid-template-columns: 1fr; }
}

@media (max-width: 768px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .controls { grid-template-columns: repeat(2, 1fr); }
  .wrap { padding: 20px; }
}
```

  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="header-title">◇ NERAIUM SII DASHBOARD ◇</div>
      <div class="header-stamp">
        <div>CLASSIFIED</div>
        <div style="font-size: 9px; margin-top: 4px;">TELEMETRY ANALYSIS</div>
      </div>
    </div>

```
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
  <div class="metric-card">
    <div class="metric-label">Structural Drift</div>
    <div class="metric-value" id="driftValue">0.000</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Relational Stability</div>
    <div class="metric-value" id="stabilityValue">0.000</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Drift Velocity</div>
    <div class="metric-value" id="velocityValue">0.000</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Warning Horizon</div>
    <div class="metric-value" id="leadValue">—</div>
  </div>
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
    <thead>
      <tr>
        <th>ID</th>
        <th>Type</th>
        <th>Timestamp</th>
        <th>Site</th>
        <th>Asset</th>
        <th>State</th>
      </tr>
    </thead>
    <tbody id="eventsTableBody">
      <tr><td colspan="6" style="text-align: center; padding: 20px;">Loading telemetry...</td></tr>
    </tbody>
  </table>
</div>
```

  </div>

<script>
let driftChart = null;
let stabilityChart = null;

function stateClass(state) {
  const s = String(state || "").toUpperCase();
  if (s === "ALERT") return "state-alert";
  if (s === "WATCH") return "state-watch";
  return "state-stable";
}

function statusClass(state) {
  const s = String(state || "").toUpperCase();
  if (s === "ALERT") return "status-alert";
  if (s === "WATCH") return "status-watch";
  return "status-stable";
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
    plugins: {
      legend: { display: false },
      filler: { propagate: false }
    },
    scales: {
      x: { display: false },
      y: { display: false, grid: { display: false } }
    }
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
    const [status, eventList] = await Promise.all([
      api("/api/status"),
      api("/api/events")
    ]);

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
      tr.innerHTML =
        "<td>" + e.id + "</td>" +
        "<td>" + e.event_type + "</td>" +
        "<td>" + e.timestamp.slice(11, 19) + "</td>" +
        "<td>" + e.site_id + "</td>" +
        "<td>" + e.asset_id + "</td>" +
        "<td><div class='status-badge " + statusClass(e.state) + "'>" + e.state + "</div></td>";
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
</script>

</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
def send_json(self, data, status=200):
payload = json.dumps(data).encode(“utf-8”)
self.send_response(status)
self.send_header(“Content-Type”, “application/json”)
self.send_header(“Content-Length”, str(len(payload)))
self.end_headers()
self.wfile.write(payload)

```
def send_html(self, html, status=200):
    payload = html.encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

def do_GET(self):
    global paused, scenario, last_drift, events

    parsed = urlparse(self.path)
    path = parsed.path

    if path == "/" or path == "/dashboard":
        return self.send_html(DASHBOARD_HTML)

    if path == "/api/status":
        with lock:
            latest = events[-1] if events else {
                "state": "UNKNOWN",
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
                "explanation": "Initializing structural telemetry..."
            }
            out = dict(latest)
            out["events_tracked"] = len(events)
            out["paused"] = paused
            out["scenario"] = scenario
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

    self.send_error(404)

def log_message(self, format, *args):
    return
```

def run():
server = HTTPServer((“0.0.0.0”, 8000), Handler)
print(“NERAIUM WATER PLATFORM DEMO ACTIVE”)
print(“Server running at http://0.0.0.0:8000”)
threading.Thread(target=telemetry_loop, daemon=True).start()
server.serve_forever()

if **name** == “**main**”:
run()