from neraium_core.store import EventStore


def test_store_add_and_all():
    store = EventStore()

    event = {"status": "normal", "anomaly": {"anomaly": False}}
    store.add(event)

    assert store.all() == [event]


def test_store_latest():
    store = EventStore()

    first = {"status": "normal", "anomaly": {"anomaly": False}}
    second = {"status": "anomaly", "anomaly": {"anomaly": True}}

    store.add(first)
    store.add(second)

    assert store.latest() == second


def test_store_anomalies():
    store = EventStore()

    normal = {"status": "normal", "anomaly": {"anomaly": False}}
    anomaly = {"status": "anomaly", "anomaly": {"anomaly": True}}

    store.add(normal)
    store.add(anomaly)

    assert store.anomalies() == [anomaly]


def test_structural_summary_empty():
    store = EventStore()

    summary = store.structural_summary()

    assert summary["total_events"] == 0
    assert summary["total_structural_anomalies"] == 0
    assert summary["latest_status"] == "no data"


def test_structural_summary_populated():
    store = EventStore()

    event = {
        "status": "anomaly",
        "score": 44,
        "aligned": [11, 43],
        "signals": {"cpu_usage": 11, "memory_usage": 43},
        "anomaly": {"anomaly": True},
    }

    store.add(event)

    summary = store.structural_summary()

    assert summary["total_events"] == 1
    assert summary["total_structural_anomalies"] == 1
    assert summary["latest_status"] == "anomaly"
    assert summary["latest_drift_score"] == 44
