import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from engine import StructuralEngine
from ingest import normalize_rest_payload, parse_csv_text

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

engine = StructuralEngine()
events = []


def add_event(result):
    events.append(result)
    if len(events) > 300:
        events.pop(0)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text, status=200, content_type="text/plain"):
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def read_text_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        return raw.decode("utf-8")

    def serve_static(self, filename):
        path = STATIC_DIR / filename
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return

        data = path.read_bytes()

        if filename.endswith(".js"):
            content_type = "application/javascript"
        elif filename.endswith(".css"):
            content_type = "text/css"
        else:
            content_type = "text/html"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return self.serve_static("index.html")

        if path.startswith("/static/"):
            return self.serve_static(path.replace("/static/", ""))

        if path == "/api/status":
            latest = engine.latest_result or {
                "state": "STABLE",
                "structural_drift_score": 0.0,
                "relational_stability_score": 1.0,
                "system_health": 100,
                "drift_alert": False,
                "site_id": "-",
                "asset_id": "-",
                "timestamp": "-",
            }
            return self.send_json(latest)

        if path == "/api/events":
            return self.send_json(events)

        self.send_error(404)

def do_POST(self):
    parsed = urlparse(self.path)
    path = parsed.path

    if path == "/telemetry":
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(400)
            return

        # normalize incoming telemetry
        normalized = normalize_rest_payload(payload)

        # process through engine
        result = engine.process(normalized)

        # store event
        add_event(result)

        self.send_json({"status": "ok"})
        return

    self.send_error(404)
    
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/ingest":
                payload = self.read_json_body()
                frame = normalize_rest_payload(payload)
                result = engine.process_frame(frame)
                add_event(result)
                return self.send_json({"ok": True, "result": result})

            if path == "/api/load-csv":
                csv_text = self.read_text_body()
                frames = parse_csv_text(csv_text)

                results = []
                for frame in frames:
                    result = engine.process_frame(frame)
                    add_event(result)
                    results.append(result)

                return self.send_json({
                    "ok": True,
                    "frames_loaded": len(frames),
                    "latest_result": results[-1] if results else None,
                })

            self.send_error(404)

        except Exception as e:
            return self.send_json({"ok": False, "error": str(e)}, status=400)

    def log_message(self, format, *args):
        return


def run():
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Neraium MVP running at http://0.0.0.0:8000")
    server.serve_forever()


if __name__ == "__main__":
    run()