from __future__ import annotations

from apps.api.main import create_app


def test_create_app_openapi_smoke() -> None:
    app = create_app()
    schema = app.openapi()
    assert isinstance(schema, dict)
    assert schema.get("openapi")
    assert isinstance(schema.get("paths"), dict)


def test_openapi_includes_v2_decision_contract_examples() -> None:
    app = create_app()
    schema = app.openapi()
    state_v2 = schema["paths"]["/v2/state"]["get"]
    example = state_v2["responses"]["200"]["content"]["application/json"]["example"]
    assert "state" in example
    assert example["state"]["contract_version"] == "decision-contract.v2"
