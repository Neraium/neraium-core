from __future__ import annotations

from neraium_core.alignment import StructuralEngine
from neraium_core.engine_registry import EngineRegistry
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _frame(*, asset_id: str, customer_id: str = "cust-a", site_id: str = "site-a") -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "customer_id": customer_id,
        "site_id": site_id,
        "asset_id": asset_id,
        "sensor_values": {"pressure": 1.0, "temperature": 2.0},
    }


def test_registry_creates_engine_for_new_asset_key() -> None:
    template = StructuralEngine(baseline_window=8, recent_window=4)
    registry = EngineRegistry(template)

    engine = registry.get_or_create(_frame(asset_id="asset-1"))

    assert isinstance(engine, StructuralEngine)
    assert len(registry.engines) == 1


def test_registry_reuses_engine_for_same_asset_key() -> None:
    template = StructuralEngine(baseline_window=8, recent_window=4)
    registry = EngineRegistry(template)

    first = registry.get_or_create(_frame(asset_id="asset-1"))
    second = registry.get_or_create(_frame(asset_id="asset-1"))

    assert first is second
    assert len(registry.engines) == 1


def test_registry_separates_engines_by_asset_key() -> None:
    template = StructuralEngine(baseline_window=8, recent_window=4)
    registry = EngineRegistry(template)

    left = registry.get_or_create(_frame(asset_id="asset-1"))
    right = registry.get_or_create(_frame(asset_id="asset-2"))

    assert left is not right
    assert len(registry.engines) == 2


def test_registry_tracks_last_access_when_enabled() -> None:
    template = StructuralEngine(baseline_window=8, recent_window=4)
    registry = EngineRegistry(template, track_last_access=True)

    key = ("cust-a", "site-a", "asset-1")
    registry.get_or_create(_frame(asset_id="asset-1"))
    initial = registry.metadata_for_key(key)
    registry.get_or_create(_frame(asset_id="asset-1"))
    later = registry.metadata_for_key(key)

    assert initial["last_access"] is not None
    assert later["last_access"] is not None
    assert int(later["last_access"]) > int(initial["last_access"])


def test_service_facade_preserves_engine_reuse_across_entrypoints(tmp_path) -> None:
    service = StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3),
        store=ResultStore(db_path=str(tmp_path / "registry_service.db")),
    )

    payload = _frame(asset_id="asset-1")
    service.ingest_payload(dict(payload), run_id="run-a", customer_id="cust-a")
    service.ingest_batch([dict(payload)], run_id="run-a", customer_id="cust-a")

    assert len(service._engine_registry.engines) == 1
    key = ("cust-a", "site-a", "asset-1")
    assert service._engine_registry.get_existing(key) is not None
