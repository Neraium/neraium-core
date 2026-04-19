from __future__ import annotations

import json

import pytest

from apps.api.integration import (
    IntegrationMappingError,
    apply_integration_mapping,
    load_integration_config,
)


def test_load_integration_config_defaults_when_missing_file(tmp_path) -> None:
    cfg = load_integration_config(str(tmp_path / "missing.json"))
    assert cfg == {"default": {}, "customers": {}}


def test_load_integration_config_rejects_invalid_json_shape(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not_customers": {}}), encoding="utf-8")
    cfg = load_integration_config(str(bad))
    assert cfg["customers"] == {}
    assert cfg["default"] == {}


def test_apply_integration_mapping_supports_alias_and_nested_payload() -> None:
    cfg = {
        "customers": {
            "cust-a": {
                "mapping": {
                    "payload_path": "data.events",
                    "items_path": "",
                    "field_aliases": {
                        "timestamp": ["time_stamp"],
                        "site_id": ["SITE ID"],
                        "asset_id": ["asset id"],
                        "sensor_values": ["SENSOR_VALUES"],
                    },
                },
            }
        }
    }
    payload = {
        "data": {
            "events": [
                {
                    "time_stamp": "2026-01-01T00:00:00+00:00",
                    "SITE ID": "site-1",
                    "asset id": "asset-1",
                    "SENSOR_VALUES": {"pressure": 51.2},
                }
            ]
        }
    }
    rows = apply_integration_mapping(payload, customer_id="cust-a", config=cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert row["site_id"] == "site-1"
    assert row["asset_id"] == "asset-1"
    assert row["sensor_values"]["pressure"] == 51.2
    assert row["customer_id"] == "cust-a"


def test_apply_integration_mapping_rejects_missing_required_fields() -> None:
    payload = {"timestamp": "2026-01-01T00:00:00+00:00", "site_id": "site-1"}
    with pytest.raises(IntegrationMappingError) as exc:
        apply_integration_mapping(payload, customer_id="cust-b", config={})
    assert "missing required fields" in str(exc.value)
