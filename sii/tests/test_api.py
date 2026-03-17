from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from api.server import app


def test_api_endpoints() -> None:
    client = TestClient(app)
    now = datetime.utcnow()
    points = []
    for i in range(60):
        points.append(
            {
                "timestamp": (now + timedelta(seconds=i)).isoformat(),
                "signals": {"a": float(i), "b": float(i + 1), "c": float(60 - i)},
            }
        )
    res = client.post("/telemetry", json={"points": points})
    assert res.status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/windows/latest").status_code == 200
    assert client.get("/observables/latest").status_code == 200
    assert client.get("/drift/latest").status_code == 200
    assert client.get("/regime/latest").status_code == 200
    assert client.get("/forecast/latest").status_code == 200
