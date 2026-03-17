from pathlib import Path

from neraium_core.fd004_plotting import generate_fd004_subset_plots, select_representative_units


def test_select_representative_units_uses_low_median_high_peak_instability():
    summaries = [
        {"asset_id": "unit_001", "peak_instability": 0.1},
        {"asset_id": "unit_002", "peak_instability": 0.5},
        {"asset_id": "unit_003", "peak_instability": 0.9},
        {"asset_id": "unit_004", "peak_instability": 0.6},
    ]

    selected = select_representative_units(summaries, max_units=3)

    assert selected == ["unit_001", "unit_004", "unit_003"]


def test_generate_fd004_subset_plots_writes_selected_units(monkeypatch, tmp_path):
    timeseries = [
        {"asset_id": "unit_001", "cycle": 1, "phase": "stable"},
        {"asset_id": "unit_002", "cycle": 1, "phase": "drift"},
        {"asset_id": "unit_003", "cycle": 1, "phase": "unstable"},
    ]
    unit_summaries = [
        {"asset_id": "unit_001", "peak_instability": 0.1},
        {"asset_id": "unit_002", "peak_instability": 0.5},
        {"asset_id": "unit_003", "peak_instability": 1.0},
    ]

    def fake_plot(rows, output_path, include_rul_curve=True):  # noqa: ANN001, FBT002
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"rows={len(rows)} rul={include_rul_curve}", encoding="utf-8")

    monkeypatch.setattr("neraium_core.fd004_plotting.plot_fd004_unit_timeseries", fake_plot)

    generated = generate_fd004_subset_plots(
        timeseries,
        unit_summaries,
        output_dir=tmp_path,
        max_units=3,
        include_rul_curve=True,
    )

    assert len(generated) == 3
    assert all(path.exists() for path in generated)
    assert generated[0] == tmp_path / "plots" / "unit_001_sii_plot.png"
