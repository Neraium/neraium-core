from __future__ import annotations

from apps.api.services.demo_jobs import DemoJobsManager
from apps.api.services.state_store import RuntimeStateStore


class _DummyService:
    pass


def _build_manager() -> DemoJobsManager:
    return DemoJobsManager(
        store=RuntimeStateStore(),
        service_instance=_DummyService(),
        resolve_customer_id=lambda customer_id: str(customer_id or "default"),
        utc_now_iso=lambda: "2026-01-01T00:00:00+00:00",
        log_structured=lambda *args, **kwargs: None,
        summarize_exception_for_logs=lambda exc: str(exc),
        logger=__import__("logging").getLogger("test"),
    )


def test_greenhouse_demo_subset_output_shape_and_order() -> None:
    manager = _build_manager()
    rows = manager.load_greenhouse_demo_subset(60)
    assert len(rows) == 60
    timestamps = [str(row["timestamp"]) for row in rows]
    assert timestamps == sorted(timestamps)
    for row in rows:
        assert "timestamp" in row
        assert "site_id" in row
        assert "asset_id" in row
        assert isinstance(row.get("sensor_values"), dict)
        assert row["sensor_values"]


def test_greenhouse_demo_subset_uses_cache_for_same_frame_count() -> None:
    manager = _build_manager()
    first = manager.load_greenhouse_demo_subset(45)
    second = manager.load_greenhouse_demo_subset(45)
    assert len(first) == len(second) == 45
    assert first == second
