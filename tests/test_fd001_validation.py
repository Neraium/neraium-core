from neraium_core.fd001_validation import (
    Fd001Row,
    fd001_row_to_payload,
    flatten_validation_result,
    group_rows_by_unit,
    load_fd001_dataset,
    replay_fd001_units,
)


def _sample_row(unit: int, cycle: int, sensor_base: float = 1.0) -> str:
    operating = [0.1, 0.2, 0.3]
    sensors = [sensor_base + i * 0.01 for i in range(21)]
    values = [unit, cycle, *operating, *sensors]
    return " ".join(str(v) for v in values)


def test_load_fd001_dataset_parses_rows(tmp_path):
    source = tmp_path / "test_FD001.txt"
    source.write_text(f"{_sample_row(1, 1)}\n{_sample_row(1, 2)}\n", encoding="utf-8")

    rows = load_fd001_dataset(source)

    assert len(rows) == 2
    assert rows[0].unit_id == 1
    assert rows[0].cycle == 1
    assert rows[0].operating_settings == (0.1, 0.2, 0.3)
    assert len(rows[0].sensors) == 21


def test_group_rows_by_unit_keeps_timelines_separate_and_sorted():
    rows = [
        Fd001Row(unit_id=2, cycle=3, operating_settings=(0.0, 0.1, 0.2), sensors=tuple(float(i) for i in range(21))),
        Fd001Row(unit_id=1, cycle=2, operating_settings=(0.0, 0.1, 0.2), sensors=tuple(float(i) for i in range(21))),
        Fd001Row(unit_id=1, cycle=1, operating_settings=(0.0, 0.1, 0.2), sensors=tuple(float(i) for i in range(21))),
    ]

    grouped = group_rows_by_unit(rows)

    assert sorted(grouped) == [1, 2]
    assert [r.cycle for r in grouped[1]] == [1, 2]
    assert [r.cycle for r in grouped[2]] == [3]


def test_fd001_row_to_payload_mapping_shape():
    row = Fd001Row(
        unit_id=7,
        cycle=3,
        operating_settings=(0.11, 0.22, 0.33),
        sensors=tuple(float(i) for i in range(1, 22)),
    )

    payload = fd001_row_to_payload(row)

    assert payload["asset_id"] == "fd001_unit_007"
    assert payload["timestamp"] == "3"
    assert payload["sensor_values"]["setting_1"] == 0.11
    assert payload["sensor_values"]["setting_2"] == 0.22
    assert payload["sensor_values"]["setting_3"] == 0.33
    assert payload["sensor_values"]["s1"] == 1.0
    assert payload["sensor_values"]["s21"] == 21.0


def test_replay_fd001_units_smoke():
    rows = [
        Fd001Row(unit_id=1, cycle=1, operating_settings=(0.0, 0.1, 0.2), sensors=tuple(float(i) for i in range(21))),
        Fd001Row(unit_id=1, cycle=2, operating_settings=(0.0, 0.1, 0.2), sensors=tuple(float(i + 1) for i in range(21))),
    ]
    grouped = group_rows_by_unit(rows)

    full, summary = replay_fd001_units(grouped, unit_ids=[1], max_cycles=2)

    assert len(full) == 2
    assert len(summary) == 2
    assert full[0]["unit_id"] == 1
    assert full[0]["cycle"] == 1
    assert full[0]["row_index"] == 1
    assert full[0]["ingest_metadata"] == {"unit_id": 1, "cycle": 1, "row_index": 1}
    assert "decision" in full[0]
    assert "risk_assessment" in full[0]

    flat = flatten_validation_result(full[0], unit_id=1, cycle=1, row_index=1)
    assert set(flat.keys()) == {
        "unit_id",
        "cycle",
        "row_index",
        "decision_action",
        "decision_confidence",
        "risk_current_level",
        "risk_trend",
        "risk_trend_direction",
        "top_hypothesis_id",
        "top_hypothesis_confidence",
        "top_attribution_driver",
        "first_risk_cycle",
        "first_decision_cycle",
    }

    assert "first_risk_cycle" in summary[0]
    assert "first_decision_cycle" in summary[0]
