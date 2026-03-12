import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

from neraium_core.pipeline import TelemetryPipeline
from neraium_core.store import EventStore
from neraium_core.telemetry import TelemetryPayload


pipeline = TelemetryPipeline()
store = EventStore()


class NeraiumHandler(BaseHTTPRequestHandler):

    def _send_json(self, code, payload):

        body = json.dumps(payload).encode("utf-8")

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):

        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "neraium"})
            return

        if self.path == "/events":
            self._send_json(200, {"events": store.all()})
            return

        if self.path == "/events/latest":
            latest = store.latest()

            if latest is None:
                self._send_json(404, {"error": "no events"})
            else:
                self._send_json(200, latest)

            return

        if self.path == "/events/anomalies":
            self._send_json(200, {"events": store.anomalies()})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):

        if self.path != "/telemetry":
            self._send_json(404, {"error": "not found"})
            return

        try:

            length = int(self.headers.get("Content-Length", 0))

            body = self.rfile.read(length)

            data = json.loads(body.decode())

            payload = TelemetryPayload(
                timestamp=datetime.now(timezone.utc),
                signals={
                    "cpu_usage": float(data["cpu_usage"]),
                    "memory_usage": float(data["memory_usage"])
                }
            )

            result = pipeline.process(payload)

            store.add(result)

            self._send_json(200, result)

        except Exception as e:

            self._send_json(400, {"error": str(e)})


def main():

    server = HTTPServer(("127.0.0.1", 8000), NeraiumHandler)

    print("Neraium server running on http://127.0.0.1:8000")
    print("GET /health")
    print("GET /events")
    print("GET /events/latest")
    print("GET /events/anomalies")
    print("POST /telemetry")

    server.serve_forever()


if __name__ == "__main__":
    main()
