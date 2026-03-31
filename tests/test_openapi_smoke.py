from __future__ import annotations

from apps.api.main import create_app


def test_create_app_openapi_smoke() -> None:
    app = create_app()
    schema = app.openapi()
    assert isinstance(schema, dict)
    assert schema.get("openapi")
    assert isinstance(schema.get("paths"), dict)
