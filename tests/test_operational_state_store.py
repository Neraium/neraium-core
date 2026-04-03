from __future__ import annotations

from neraium_core.store import ResultStore


def test_operational_state_roundtrip_and_prefix_listing(tmp_path) -> None:
    store = ResultStore(db_path=str(tmp_path / "state.db"))
    store.upsert_operational_state(
        state_key="alerts:customer-a",
        state={"alerts": [{"id": "a1"}]},
        customer_id="customer-a",
        run_id="run-1",
    )
    store.upsert_operational_state(
        state_key="ingest_job:job-1",
        state={"job_id": "job-1", "status": "completed"},
        customer_id="customer-a",
        run_id="run-1",
    )

    alert_state = store.get_operational_state(state_key="alerts:customer-a")
    assert alert_state is not None
    assert alert_state["customer_id"] == "customer-a"
    assert alert_state["state"]["alerts"][0]["id"] == "a1"

    prefixed = store.list_operational_state(key_prefix="ingest_job:")
    assert len(prefixed) == 1
    assert prefixed[0]["state_key"] == "ingest_job:job-1"

    store.delete_operational_state(state_key="ingest_job:job-1")
    assert store.get_operational_state(state_key="ingest_job:job-1") is None
