from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from apps.api.main import create_app
from fastapi.testclient import TestClient


def test_create_app_falls_back_when_configured_db_path_unwritable(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_DB_PATH", "/proc/neraium.db")

    app = create_app()
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["persistence_available"] is True
    assert body["db_path"] == "/tmp/neraium.db"
