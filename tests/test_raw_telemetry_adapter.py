from __future__ import annotations

import json

from neraium_core.adapters.raw_telemetry_adapter import (
    convert_raw_telemetry_to_structural_rows,
    ingest_raw_industrial_input,
)


def test_convert_raw_telemetry_to_structural_rows_json(tmp_path) -> None:
    src = tmp_path / "windows.json"
    src.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "unit": "u-1",
                        "cycle": 1,
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "channel_names": ["x", "y"],
                        "values": [[0.1, 0.2], [0.2, 0.4], [0.3, 0.5]],
                    }
                ]
            }
        )
    )

    rows = convert_raw_telemetry_to_structural_rows(src)
    assert len(rows) == 1
    row = rows[0]
    assert row["unit"] == "u-1"
    assert row["cycle"] == 1
    assert any(k.startswith("sensor__x__") for k in row.keys())
    assert any(k.startswith("sensor__cross__") for k in row.keys())


def test_ingest_raw_industrial_accepts_single_json_object_row(tmp_path) -> None:
    src = tmp_path / "single_row.json"
    src.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "asset_id": "pump-1",
                "site_id": "site-1",
                "pressure": 12.5,
                "temperature": "41.7",
            }
        )
    )

    result = ingest_raw_industrial_input(src)
    assert result.diagnostics.detected_input_type == "tabular_timeseries"
    assert result.diagnostics.timestep_count == 1
    assert result.frames[0]["asset_id"] == "pump-1"
    assert result.frames[0]["sensor_values"]["pressure"] == 12.5
    assert result.frames[0]["sensor_values"]["temperature"] == 41.7


def test_ingest_directory_surfaces_skipped_file_warnings(tmp_path) -> None:
    broken = tmp_path / "broken.npy"
    broken.write_text("not-a-valid-npy")

    result = ingest_raw_industrial_input(tmp_path)
    assert result.diagnostics.detected_input_type == "directory_signal_blocks"
    assert any("Skipped broken.npy" in warning for warning in result.diagnostics.warnings)
