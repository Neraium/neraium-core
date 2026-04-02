from neraium_core.data_connectors import normalize_records


def test_normalize_records_casts_values_to_float() -> None:
    rows = [{"cpu": 1, "mem": "2.5"}, {"cpu": 3.2, "mem": 4}]

    out = normalize_records(rows)

    assert out == [{"cpu": 1.0, "mem": 2.5}, {"cpu": 3.2, "mem": 4.0}]
