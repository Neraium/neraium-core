import io
import json
from urllib import error

from neraium_core.integrations.gal2_client import GAL2Client


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_gal2_client_success(monkeypatch):
    monkeypatch.setenv("GAL2_KEY", "test-key")

    payload = {
        "gal2_time": "2026-03-25T12:00:00.000Z",
        "drift_ms": 2.5,
        "wobble_ms": 1.2,
        "live_ms": 12,
        "fractal_factor": 0.99,
    }

    def fake_urlopen(_req, timeout):
        assert timeout == 3.0
        return _FakeResponse(payload)

    monkeypatch.setattr("neraium_core.integrations.gal2_client.request.urlopen", fake_urlopen)

    client = GAL2Client(cache_ms=250, sleep_fn=lambda _x: None)
    first = client.get_time()
    second = client.get_time()

    assert first["available"] is True
    assert first["gal2_time"] == payload["gal2_time"]
    assert first["drift_ms"] == payload["drift_ms"]
    assert second == first


def test_gal2_client_retries_and_falls_back(monkeypatch):
    monkeypatch.setenv("GAL2_KEY", "test-key")

    attempts = {"n": 0}

    def fake_urlopen(_req, timeout):
        attempts["n"] += 1
        raise error.HTTPError(
            url="https://api-v2.gal-2.com/time",
            code=503,
            msg="service unavailable",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr("neraium_core.integrations.gal2_client.request.urlopen", fake_urlopen)

    client = GAL2Client(max_attempts=3, sleep_fn=lambda _x: None)
    payload = client.get_time()

    assert attempts["n"] == 3
    assert payload["available"] is False
    assert payload["reason"] == "http_error_503"
