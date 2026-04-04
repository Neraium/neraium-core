from __future__ import annotations

import json

from neraium_core.sii import cli


def _sample_payloads() -> list[dict[str, object]]:
    return [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.0, "pressure": 100.0},
        },
        {
            "timestamp": "2026-01-01T00:01:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.2, "pressure": 99.7},
        },
    ]


def test_cli_runs_pipeline_and_writes_output_file(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(_sample_payloads()), encoding="utf-8")

    code = cli.main(["--input", str(input_path), "--output", str(output_path)])

    assert code == 0
    stdout = capsys.readouterr().out
    parsed_stdout = json.loads(stdout)
    parsed_file = json.loads(output_path.read_text(encoding="utf-8"))

    assert parsed_stdout == parsed_file
    assert set(parsed_stdout.keys()) >= {
        "structural_state",
        "validation_results",
        "signal_ranking",
        "decision_output",
    }


def test_cli_rejects_invalid_json_shape(tmp_path, capsys) -> None:
    input_path = tmp_path / "bad.json"
    input_path.write_text('{"not":"a-list"}', encoding="utf-8")

    code = cli.main(["--input", str(input_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert "JSON input must be a list of record objects" in captured.err
