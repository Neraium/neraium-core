from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _make_service(tmp_path, *, baseline_window: int = 5, recent_window: int = 3) -> StructuralMonitoringService:
    engine = StructuralEngine(
        baseline_window=baseline_window,
        recent_window=recent_window,
        window_stride=1,
        regime_store_path=str(tmp_path / "regimes.json"),
    )
    store = ResultStore(db_path=str(tmp_path / "test.db"))
    return StructuralMonitoringService(engine=engine, store=store)


def _pilot_keys(result: dict[str, Any]) -> None:
    required = {"timestamp", "signals", "score", "status", "aligned", "anomaly"}
    assert required.issubset(result.keys())
    assert isinstance(result["timestamp"], str)
    assert isinstance(result["signals"], dict)
    assert isinstance(result["score"], float)
    assert isinstance(result["status"], str)
    assert isinstance(result["aligned"], bool)
    assert isinstance(result["anomaly"], bool)
    assert result["anomaly"] == (result["status"] in {"WATCH", "ALERT"})


def test_pilot_rejects_non_numeric_sensor_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"pressure": "abc"},
    }

    with pytest.raises(ValueError, match="Invalid signal value"):
        service.ingest_payload(payload)


def test_pilot_rejects_non_numeric_sensor_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"pressure": {"bad": "type"}},
    }

    with pytest.raises(ValueError, match="Invalid signal type"):
        service.ingest_payload(payload)


def test_pilot_missing_timestamp_normalizes_and_schema_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"pressure": 12.3, "vibration": 0.4},
    }

    result = service.ingest_payload(payload)
    _pilot_keys(result)

    # ISO-8601 parse check.
    dt = datetime.fromisoformat(result["timestamp"])
    assert dt.tzinfo is not None


def test_pilot_missing_sensor_values_schema_signals_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {"timestamp": "2026-01-01T00:00:00+00:00", "site_id": "s1", "asset_id": "a1"}
    result = service.ingest_payload(payload)

    _pilot_keys(result)
    assert result["signals"] == {}


def test_pilot_rejects_sensor_values_not_an_object(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": "not-an-object",
    }

    with pytest.raises(ValueError, match="sensor_values must be an object"):
        service.ingest_payload(payload)


def test_pilot_invalid_timestamp_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {
        "timestamp": "not-a-real-timestamp",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"pressure": 12.3},
    }

    with pytest.raises(ValueError, match="Invalid timestamp"):
        service.ingest_payload(payload)


def test_pilot_schema_consistency_and_nan_handling(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"s1": 1.0, "s2": float("nan"), "s3": None},
    }
    result = service.ingest_payload(payload)
    _pilot_keys(result)
    assert set(result["signals"].keys()) == {"s1", "s2", "s3"}
    assert result["signals"]["s2"] is None
    assert result["signals"]["s3"] is None


def test_pilot_does_not_change_existing_decision_fields(tmp_path, monkeypatch) -> None:
    """
    Regression guard:
    pilot hardening should not change the decision-layer outputs for valid numeric payloads.
    """

    payload = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"pressure": 50.0, "vibration": 0.2, "temp": 10.0},
    }

    # Pilot OFF
    monkeypatch.delenv("NERAIUM_PILOT_HARDENING", raising=False)
    service_off = _make_service(tmp_path / "off")
    result_off = service_off.ingest_payload(payload)

    # Pilot ON
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service_on = _make_service(tmp_path / "on")
    result_on = service_on.ingest_payload(payload)

    assert result_off["risk_level"] == result_on["risk_level"]
    assert result_off["action_state"] == result_on["action_state"]
    assert result_off["operator_message"] == result_on["operator_message"]

    _pilot_keys(result_on)

