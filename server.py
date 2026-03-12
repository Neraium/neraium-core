import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

from neraium_core.pipeline import TelemetryPipeline
from neraium_core.telemetry import TelemetryPayload


pipeline = TelemetryPipeline()


class NeraiumHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "neraium"})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/telemetry":
            self._send_json(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body.decode("utf-8"))

            payload = TelemetryPayload(
                timestamp=datetime.now(timezone.utc),
                signals={
                    "cpu_usage": float(data["cpu_usage"]),
                    "memory_usage": float(data["memory_usage"]),
                },
            )

            result = pipeline.process(payload)
            self._send_json(200, result)

        except KeyError as exc:
            self._send_json(400, {"error": f"missing field: {exc.args[0]}"})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


def main():
    server = HTTPServer(("127.0.0.1", 8000), NeraiumHandler)
    print("Neraium server running on http://127.0.0.1:8000")
    print("Health:  GET  /health")
    print("Ingest:  POST /telemetry")
    server.serve_forever()


if __name__ == "__main__":
    main()