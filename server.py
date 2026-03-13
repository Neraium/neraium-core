import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

events = []
scenario = "normal"
paused = False

MAX_EVENTS = 200

def now():
    return datetime.now(timezone.utc).isoformat()

SITES = ["Reservoir East","North Loop","South Basin"]
ASSETS = ["Pump Station 1","District Main B","Distribution Node 7"]

SENSOR_NAMES = (
    "pressure",
    "flow",
    "tank",
    "quality",
    "vibration",
)

def build_sensor_values():

    if scenario == "normal":
        return (
            random.uniform(58,64),
            random.uniform(118,134),
            random.uniform(70,78),
            random.uniform(95,99),
            random.uniform(0.15,0.25)
        )

    if scenario == "degrading":
        return (
            random.uniform(48,60),
            random.uniform(100,125),
            random.uniform(60,75),
            random.uniform(85,96),
            random.uniform(0.20,0.45)
        )

    return (
        random.uniform(35,52),
        random.uniform(80,105),
        random.uniform(50,65),
        random.uniform(70,85),
        random.uniform(0.30,0.70)
    )

def structural_drift(values):

    base = [60,125,75,97,0.2]

    drift = 0

    for i,v in enumerate(values):
        drift += abs(v-base[i])/abs(base[i])

    return drift

def relational_stability(values):

    pressure,flow,tank,quality,vibration = values

    stability = 1 - (
        abs((pressure/flow)-0.5)*0.4 +
        abs((tank/pressure)-1.2)*0.3 +
        vibration*0.3
    )

    return max(0,min(1,stability))

def lead_time(drift,velocity):

    boundary = 4.0

    if velocity <= 0:
        return None

    return (boundary-drift)/velocity

last_drift = None

def generate_event():

    global last_drift

    site = random.choice(SITES)
    asset = random.choice(ASSETS)

    timestamp = now()

    values = build_sensor_values()

    drift = structural_drift(values)

    stability = relational_stability(values)

    velocity = 0

    if last_drift is not None:
        velocity = drift-last_drift

    last_drift = drift

    lt = lead_time(drift,velocity)

    state = "STABLE"

    if drift > 3:
        state = "ALERT"
    elif drift > 1.5:
        state = "WATCH"

    event = {

        "id": len(events)+1,
        "timestamp": timestamp,

        "site_id": site,
        "asset_id": asset,

        "state": state,
        "confidence": round(random.uniform(0.9,0.99),2),

        "structural_drift_score": round(drift,3),
        "relational_stability_score": round(stability,3),

        "drift_velocity": round(velocity,3),

        "lead_time_hours": None if lt is None else round(lt,1),
        "lead_time_confidence": round(random.uniform(0.7,0.95),2),

        "structural_driver": "pressure-flow imbalance",

        "sensor_values":{
            "pressure":values[0],
            "flow":values[1],
            "tank":values[2],
            "quality":values[3],
            "vibration":values[4],
        },

        "explanation":"SII analyzing structural geometry of the sensor network",
        "predicted_impact":"Potential instability developing in system"
    }

    events.append(event)

    if len(events)>MAX_EVENTS:
        events.pop(0)

def telemetry_loop():

    while True:

        if not paused:
            generate_event()

        time.sleep(2)

class Handler(BaseHTTPRequestHandler):

    def send_json(self,data):

        payload=json.dumps(data).encode()

        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(payload)))
        self.end_headers()

        self.wfile.write(payload)

    def serve_static(self,file):

        path=STATIC_DIR/file

        if not path.exists():
            self.send_error(404)
            return

        data=path.read_bytes()

        self.send_response(200)

        if file.endswith(".js"):
            self.send_header("Content-Type","application/javascript")
        elif file.endswith(".css"):
            self.send_header("Content-Type","text/css")
        else:
            self.send_header("Content-Type","text/html")

        self.send_header("Content-Length",str(len(data)))
        self.end_headers()

        self.wfile.write(data)

    def do_GET(self):

        if self.path=="/":
            return self.serve_static("index.html")

        if self.path.startswith("/static/"):
            return self.serve_static(self.path.replace("/static/",""))

        if self.path=="/api/events":
            return self.send_json(events)

        if self.path=="/api/status":

            latest=events[-1] if events else {}

            return self.send_json({

                "state":latest.get("state"),
                "site_id":latest.get("site_id"),
                "asset_id":latest.get("asset_id"),

                "confidence":latest.get("confidence"),

                "structural_drift_score":latest.get("structural_drift_score"),
                "relational_stability_score":latest.get("relational_stability_score"),

                "lead_time_hours":latest.get("lead_time_hours"),
                "lead_time_confidence":latest.get("lead_time_confidence"),

                "drift_velocity":latest.get("drift_velocity"),
                "structural_driver":latest.get("structural_driver"),

                "events_tracked":len(events),

                "last_timestamp":latest.get("timestamp"),

                "predicted_impact":latest.get("predicted_impact"),
                "explanation":latest.get("explanation"),
            })

        self.send_error(404)

def run():

    server=HTTPServer(("0.0.0.0",8000),Handler)

    print("Server running http://localhost:8000")

    thread=threading.Thread(target=telemetry_loop,daemon=True)
    thread.start()

    server.serve_forever()

if __name__=="__main__":
    run()