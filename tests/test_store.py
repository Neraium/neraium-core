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
