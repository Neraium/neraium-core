from neraium_core.fd004_synthetic import generate_fd004_synthetic_dataset, run_fd004_evaluation


def test_generate_fd004_dataset_shape():
    frames, configs = generate_fd004_synthetic_dataset(
        num_units=3,
        num_steps=30,
        num_sensors=20,
        num_regimes=3,
        seed=11,
    )

    assert len(frames) == 90
    assert len(configs) == 3
    sample = frames[0]
    assert {"timestamp", "site_id", "asset_id", "sensor_values"}.issubset(sample.keys())
    assert len(sample["sensor_values"]) == 20


def test_run_fd004_evaluation_report(tmp_path):
    report = run_fd004_evaluation(num_units=4, num_steps=40, seed=3, output_dir=str(tmp_path))

    assert report["overall_summary"]["units_total"] == 4
    assert len(report["unit_summaries"]) == 4
    assert (tmp_path / "fd004_synthetic_report.json").exists()
    assert (tmp_path / "fd004_synthetic_timeseries.csv").exists()
