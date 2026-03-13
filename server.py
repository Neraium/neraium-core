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

events.append(event)
if len(events) > MAX_EVENTS:
    events.pop(0)
```

def telemetry_loop():
while True:
if not paused:
generate_event()
time.sleep(2)

DASHBOARD_HTML = r”””<!doctype html>

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Neraium Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 20px;
      font-family: Arial, sans-serif;
      background: #08111f;
      color: #e8f1ff;
    }
    .wrap { max-width: 1200px; margin: 0 auto; }
    .card {
      background: #10203a;
      border: 1px solid #223b63;
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 16px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      gap: 20px;
    }
    .hero-state {
      font-size: 72px;
      font-weight: 800;
      margin: 8px 0;
    }
    .state-stable { color: #ffffff; }
    .state-watch { color: #ffd56a; }
    .state-alert { color: #ff6c86; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-bottom: 16px;
    }
    .metric-value {
      font-size: 38px;
      font-weight: 800;
      color: #8cf0ff;
      margin-top: 8px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }
    button {
      border: 1px solid #35537e;
      background: #0e1c33;
      color: white;
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      font-size: 16px;
    }
    .charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    .chart-wrap {
      position: relative;
      height: 280px;
      width: 100%;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 10px;
      border-bottom: 1px solid #223b63;
      font-size: 14px;
    }
    th {
      color: #9fb4d3;
      font-size: 12px;
      text-transform: uppercase;
    }
    .pill {
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #35537e;
      margin-top: 12px;
    }
    @media (max-width: 900px) {
      .metric-grid { grid-template-columns: 1fr 1fr; }
      .charts { grid-template-columns: 1fr; }
      .hero { flex-direction: column; }
      .hero-state { font-size: 52px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card hero">
      <div>
        <div style="font-size:12px;color:#9fb4d3;letter-spacing:2px;text-transform:uppercase;">System State</div>
        <div id="systemState" class="hero-state">UNKNOWN</div>
        <div id="siteLine">Site: -</div>
        <div id="lastTimestamp" style="margin-top:8px;">Last update: none</div>
        <div id="storyText" style="margin-top:14px;font-size:20px;">Initializing structural telemetry...</div>
        <div id="impactText" style="margin-top:8px;color:#8cf0ff;">Predicted impact: —</div>
      </div>

```
  <div>
    <div style="font-size:12px;color:#9fb4d3;letter-spacing:2px;text-transform:uppercase;">Confidence</div>
    <div id="confidenceValue" class="metric-value">—</div>
    <div style="font-size:12px;color:#9fb4d3;letter-spacing:2px;text-transform:uppercase;margin-top:20px;">Events Tracked</div>
    <div id="eventsTracked" class="metric-value">0</div>
    <div id="assetBadge" class="pill">Asset: -</div>
    <div id="connectionText" style="margin-top:12px;">Connecting...</div>
  </div>
</div>

<div class="controls">
  <button onclick="hit('/api/pause')">Pause Feed</button>
  <button onclick="hit('/api/resume')">Resume Feed</button>
  <button onclick="hit('/api/reset')">Reset Demo</button>
  <button onclick="hit('/api/scenario/normal')">Normal</button>
  <button onclick="hit('/api/scenario/degrading')">Degrading</button>
  <button onclick="hit('/api/scenario/incident')">Incident</button>
</div>

<div class="metric-grid">
  <div class="card">
    <div>Structural Drift</div>
    <div id="driftValue" class="metric-value">0.00</div>
  </div>
  <div class="card">
    <div>Relational Stability</div>
    <div id="stabilityValue" class="metric-value">0.00</div>
  </div>
  <div class="card">
    <div>Early Warning Horizon</div>
    <div id="leadValue" class="metric-value">—</div>
  </div>
  <div class="card">
    <div>Latest Event Type</div>
    <div id="eventTypeValue" class="metric-value" style="font-size:26px;">-</div>
  </div>
</div>

<div class="charts">
  <div class="card">
    <h3>Structural Drift Trend</h3>
    <div class="chart-wrap"><canvas id="driftChart"></canvas></div>
  </div>
  <div class="card">
    <h3>Relational Stability Trend</h3>
    <div class="chart-wrap"><canvas id="stabilityChart"></canvas></div>
  </div>
</div>

<div class="card">
  <h3>Recent Structural Events</h3>
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
      <tr><td colspan="6">Loading...</td></tr>
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
  return Number.isFinite(n) ? n.toFixed(2) : "—";
}

function lead(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(1) + "h" : "—";
}

async function api(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
  return res.json();
}

async function hit(url) {
  await fetch(url, { cache: "no-store" });
  await refresh();
}

function buildCharts() {
  driftChart = new Chart(document.getElementById("driftChart"), {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderWidth: 2, tension: 0.35, pointRadius: 0, fill: false }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false }
  });

  stabilityChart = new Chart(document.getElementById("stabilityChart"), {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderWidth: 2, tension: 0.35, pointRadius: 0, fill: false }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: { y: { min: 0, max: 1 } }
    }
  });
}

async function refresh() {
  try {
    const [status, events] = await Promise.all([
      api("/api/status"),
      api("/api/events")
    ]);

    const latest = events.length ? events[events.length - 1] : null;

    const stateEl = document.getElementById("systemState");
    stateEl.textContent = status.state || "UNKNOWN";
    stateEl.className = "hero-state " + stateClass(status.state);

    setText("siteLine", "Site: " + (status.site_id || "-"));
    setText("assetBadge", "Asset: " + (status.asset_id || "-"));
    setText("lastTimestamp", "Last update: " + (status.timestamp || "-"));
    setText("storyText", status.explanation || "Telemetry connected.");
    setText("impactText", "Predicted impact: " + (status.predicted_impact || "—"));
    setText("confidenceValue", pct(status.lead_time_confidence));
    setText("eventsTracked", String(events.length));
    setText("connectionText", status.paused ? "Paused" : "Connected");

    setText("driftValue", fmt(status.structural_drift_score));
    setText("stabilityValue", fmt(status.relational_stability_score));
    setText("leadValue", lead(status.lead_time_hours));
    setText("eventTypeValue", latest ? (latest.event_type || "-") : "-");

    const recent = events.slice(-40);
    driftChart.data.labels = recent.map((_, i) => i + 1);
    driftChart.data.datasets[0].data = recent.map(e => Number(e.structural_drift_score || 0));
    driftChart.update("none");

    stabilityChart.data.labels = recent.map((_, i) => i + 1);
    stabilityChart.data.datasets[0].data = recent.map(e => Number(e.relational_stability_score || 0));
    stabilityChart.update("none");

    const tbody = document.getElementById("eventsTableBody");
    tbody.innerHTML = "";
    events.slice(-10).reverse().forEach((e) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + e.id + "</td>" +
        "<td>" + e.event_type + "</td>" +
        "<td>" + e.timestamp + "</td>" +
        "<td>" + e.site_id + "</td>" +
        "<td>" + e.asset_id + "</td>" +
        "<td class='" + stateClass(e.state) + "'>" + e.state + "</td>";
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
    setText("connectionText", "Disconnected");
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
    global paused, scenario, last_drift

    parsed = urlparse(self.path)
    path = parsed.path

    if path == "/" or path == "/dashboard":
        return self.send_html(DASHBOARD_HTML)

    if path == "/api/status":
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
        return self.send_json(events)

    if path == "/api/pause":
        paused = True
        return self.send_json({"ok": True, "paused": True})

    if path == "/api/resume":
        paused = False
        return self.send_json({"ok": True, "paused": False})

    if path == "/api/reset":
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