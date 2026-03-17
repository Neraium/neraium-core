from neraium_core.fd004_real import (
    Fd004Row,
    fd004_row_to_sii_record,
    load_fd004_dataset,
    run_fd004_real_evaluation,
)


def _sample_row(unit: int, cycle: int, sensor_base: float = 1.0) -> str:
    operating = [0.1, 0.2, 0.3]
    sensors = [sensor_base + i * 0.01 for i in range(21)]
    values = [unit, cycle, *operating, *sensors]
    return " ".join(str(v) for v in values)


def test_load_fd004_dataset_parses_rows(tmp_path):
    source = tmp_path / "train_FD004.txt"
    source.write_text(f"{_sample_row(1, 1)}\n{_sample_row(1, 2)}\n", encoding="utf-8")

    rows = load_fd004_dataset(source)

    assert len(rows) == 2
    assert rows[0].unit == 1
    assert rows[0].time == 1
    assert len(rows[0].sensors) == 21


def test_fd004_row_to_sii_record_pads_to_24_sensors():
    row = Fd004Row(
        unit=7,
        time=3,
        operating_settings=(0.0, 0.1, 0.2),
        sensors=tuple(float(i) for i in range(1, 22)),
    )

    record = fd004_row_to_sii_record(row)

    assert record["asset_id"] == "unit_007"
    assert len(record["sensor_values"]) == 24
    assert record["sensor_values"]["s21"] == 21.0
    assert record["sensor_values"]["s22"] == 0.0
    assert record["sensor_values"]["s24"] == 0.0


def test_run_fd004_real_evaluation_generates_outputs(tmp_path):
    train = tmp_path / "train_FD004.txt"
    test = tmp_path / "test_FD004.txt"
    rul = tmp_path / "RUL_FD004.txt"

    train.write_text(
        "\n".join([_sample_row(1, 1), _sample_row(1, 2), _sample_row(2, 1)]),
        encoding="utf-8",
    )
    test.write_text(_sample_row(1, 1), encoding="utf-8")
    rul.write_text("123\n", encoding="utf-8")

    report = run_fd004_real_evaluation(
        train_path=str(train),
        test_path=str(test),
        rul_path=str(rul),
        output_dir=str(tmp_path),
    )

    assert report["overall_summary"]["units_total"] == 2
    assert report["overall_summary"]["rows_processed"] == 3
    assert report["rul_by_unit"]["unit_001"] == 123
    assert (tmp_path / "fd004_real_report.json").exists()
    assert (tmp_path / "fd004_real_timeseries.csv").exists()
