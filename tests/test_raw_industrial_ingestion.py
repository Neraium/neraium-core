from __future__ import annotations

import json

import pytest

from neraium_core.adapters.raw_telemetry_adapter import ingest_raw_industrial_input


def test_ingest_raw_industrial_tabular_schema_discovery(tmp_path) -> None:
    src = tmp_path / "telemetry.csv"
    src.write_text(
        "recorded_at,machine_tag,plant,temp_c,vibration\n"
        "2026-01-01T00:00:00+00:00,M-1,P-1,42.5,0.18\n"
        "2026-01-01T00:01:00+00:00,M-1,P-1,43.2,0.22\n"
    )

    out = ingest_raw_industrial_input(src)

    assert out.diagnostics.detected_input_type == "tabular_timeseries"
    assert out.diagnostics.timestep_count == 2
    assert out.diagnostics.sensor_count == 2
    assert out.diagnostics.has_valid_signals is True
    assert "temp_c" in out.diagnostics.sensor_columns
    assert "plant" in out.diagnostics.metadata_columns


def test_ingest_raw_industrial_directory_waveform_with_validation_mode(tmp_path) -> None:
    raw_dir = tmp_path / "raw_blocks"
    raw_dir.mkdir()
    (raw_dir / "001.csv").write_text("0.1,0.2\n0.3,0.4\n0.5,0.6\n")
    (raw_dir / "002.csv").write_text("0.2,0.1\n0.4,0.2\n0.7,0.3\n")

    out = ingest_raw_industrial_input(raw_dir, sample_size=1)

    assert out.diagnostics.detected_input_type == "directory_signal_blocks"
    assert out.diagnostics.preprocessing_mode == "waveform_features"
    assert out.diagnostics.timestep_count == 1
    assert out.diagnostics.sensor_count > 0
    assert any("Validation mode enabled" in w for w in out.diagnostics.warnings)


def test_ingest_raw_industrial_directory_timestep_files(tmp_path) -> None:
    raw_dir = tmp_path / "dir_rows"
    raw_dir.mkdir()
    (raw_dir / "0001.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "asset_id": "A-9",
                "site_id": "S-9",
                "amp_x": 0.21,
                "amp_y": 0.33,
                "mode": "startup",
            }
        )
    )
    (raw_dir / "0002.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:01:00+00:00",
                "asset_id": "A-9",
                "site_id": "S-9",
                "amp_x": 0.27,
                "amp_y": 0.36,
                "mode": "steady",
            }
        )
    )

    out = ingest_raw_industrial_input(raw_dir)

    assert out.diagnostics.detected_input_type == "tabular_timeseries"
    assert out.diagnostics.timestep_count == 2
    assert out.diagnostics.sensor_count == 2
    assert out.frames[0]["asset_id"] == "A-9"
    assert out.frames[0]["site_id"] == "S-9"
    assert "mode" in out.diagnostics.metadata_columns


def test_ingest_raw_industrial_rejects_invalid_preprocessing_mode(tmp_path) -> None:
    src = tmp_path / "telemetry.csv"
    src.write_text("timestamp,temp\n2026-01-01T00:00:00+00:00,1.0\n")

    with pytest.raises(ValueError):
        ingest_raw_industrial_input(src, preprocessing_mode="something-else")


def test_service_ingest_raw_industrial_data_runtime_path(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from neraium_core.service import StructuralMonitoringService

    src = tmp_path / "telemetry.json"
    src.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "asset": "A-1",
                        "site": "S-1",
                        "pressure": 10.5,
                    },
                    {
                        "timestamp": "2026-01-01T00:01:00+00:00",
                        "asset": "A-1",
                        "site": "S-1",
                        "pressure": 10.8,
                    },
                ]
            }
        )
    )

    service = StructuralMonitoringService()
    out = service.ingest_raw_industrial_data(str(src), customer_id="c1", site_id="fallback-site")

    assert out["diagnostics"]["detected_input_type"] == "tabular_timeseries"
    assert out["diagnostics"]["has_valid_signals"] is True
    assert len(out["results"]) == 2
    assert out["results"][0]["asset_id"] == "A-1"
