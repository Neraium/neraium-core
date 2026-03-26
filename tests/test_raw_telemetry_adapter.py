from __future__ import annotations

import json

from neraium_core.adapters.raw_telemetry_adapter import convert_raw_telemetry_to_structural_rows


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
