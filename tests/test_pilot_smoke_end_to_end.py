from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _make_service(tmp_path) -> StructuralMonitoringService:
    engine = StructuralEngine(
        baseline_window=5,
        recent_window=3,
        window_stride=1,
        regime_store_path=str(tmp_path / "regimes.json"),
    )
    store = ResultStore(db_path=str(tmp_path / "test.db"))
    return StructuralMonitoringService(engine=engine, store=store)


def test_pilot_smoke_short_sequence_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_PILOT_HARDENING", "1")
    service = _make_service(tmp_path)

    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(10):
        ts = (start + timedelta(seconds=i)).isoformat()

        # Realistic-ish: smooth oscillation + controlled missingness.
        s1 = math.sin(i / 2.0)
        s2 = math.cos(i / 2.0)
        s3: float | None
        if i == 3:
            s3 = None
        elif i == 5:
            s3 = float("nan")
        else:
            s3 = 0.5 + 0.1 * i

        payload: dict[str, Any] = {
            "timestamp": ts,
            "site_id": "site-1",
            "asset_id": "asset-1",
            "sensor_values": {"s1": s1, "s2": s2, "s3": s3},
        }

        result = service.ingest_payload(payload)

        required = {"timestamp", "signals", "score", "status", "aligned", "anomaly"}
        assert required.issubset(result.keys())
        assert isinstance(result["timestamp"], str)
        assert isinstance(result["signals"], dict)
        assert isinstance(result["score"], float)
        assert isinstance(result["status"], str)
        assert isinstance(result["aligned"], bool)
        assert isinstance(result["anomaly"], bool)
        assert result["anomaly"] == (result["status"] in {"WATCH", "ALERT"})

