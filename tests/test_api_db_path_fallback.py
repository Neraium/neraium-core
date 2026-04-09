from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from apps.api.bootstrap import config as boot_config
from apps.api.main import create_app
from fastapi.testclient import TestClient


def test_create_app_falls_back_when_configured_db_path_unwritable(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_DB_PATH", "/proc/neraium.db")

    import apps.api.main as api_main

    def fake_resolve(configured: str) -> tuple[str, bool]:
        if str(configured).strip() == "/proc/neraium.db":
            return "/tmp/neraium.db", True
        return boot_config.resolve_db_path(configured)

    monkeypatch.setattr(api_main, "resolve_db_path", fake_resolve)

    app = create_app()
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["persistence_available"] is True
    assert body["runtime_state_diagnostics"]["db_path"] == "/tmp/neraium.db"


def test_create_app_production_mode_requires_persistence(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_RUNTIME_MODE", "production")
    monkeypatch.delenv("NERAIUM_RUNTIME_ENV", raising=False)

    import apps.api.main as api_main

    monkeypatch.setattr(api_main, "resolve_db_path", lambda _configured: ("/tmp/neraium.db", False))

    with pytest.raises(RuntimeError, match="persistence is unavailable"):
        create_app()


def test_create_app_runtime_mode_env_precedence_over_runtime_env(monkeypatch) -> None:
    # NERAIUM_RUNTIME_MODE should drive startup mode even when NERAIUM_RUNTIME_ENV disagrees.
    monkeypatch.setenv("NERAIUM_RUNTIME_MODE", "production")
    monkeypatch.setenv("NERAIUM_RUNTIME_ENV", "development")

    import apps.api.main as api_main

    monkeypatch.setattr(api_main, "resolve_db_path", lambda _configured: ("/tmp/neraium.db", False))

    with pytest.raises(RuntimeError, match="persistence is unavailable"):
        create_app()
