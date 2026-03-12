import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from neraium_core.pipeline import TelemetryPipeline
from neraium_core.store import EventStore
from neraium_core.telemetry import TelemetryPayload


pipeline = TelemetryPipeline()
store = EventStore()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class NeraiumHandler(BaseHTTPRequestHandler):
    def send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, filename):
        path = STATIC_DIR / filename
        if not path.exists():
            self.send_json(404, {"error": f"{filename} not found"})
            return

        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/dashboard")
            self.end_headers()
            return

        if self.path == "/health":
            self.send_json(200, {"status": "ok", "service": "neraium"})
            return

        if self.path == "/dashboard":
            self.send_html("dashboard.html")
            return

        if self.path == "/events/recent":
            self.send_json(200, {"events": store.all()[-100:]})
            return

        if self.path == "/events":
            self.send_json(200, {"events": store.all()})
            return

        if self.path == "/events/latest":
            latest = store.latest()
            if latest is None:
                self.send_json(404, {"error": "no events"})
            else:
                self.send_json(200, latest)
            return

        if self.path == "/events/anomalies":
            self.send_json(200, {"events": store.anomalies()})
            return

        if self.path == "/structural/summary":
            self.send_json(200, store.structural_summary())
            return

        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/telemetry":
            self.send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))

            payload = TelemetryPayload(
                timestamp=datetime.now(timezone.utc),
                signals={
                    "cpu_usage": float(data["cpu_usage"]),
                    "memory_usage": float(data["memory_usage"]),
                },
            )

            result = pipeline.process(payload)
            store.add(result)

            self.send_json(200, result)

        except Exception as e:
            self.send_json(400, {"error": str(e)})


def main():
    server = HTTPServer(("127.0.0.1", 8000), NeraiumHandler)

    print("Neraium running at http://127.0.0.1:8000")
    print("GET  /")
    print("GET  /dashboard")
    print("GET  /health")
    print("GET  /events")
    print("GET  /events/recent")
    print("GET  /events/latest")
    print("GET  /events/anomalies")
    print("GET  /structural/summary")
    print("POST /telemetry")

    server.serve_forever()


if __name__ == "__main__":
    main()

