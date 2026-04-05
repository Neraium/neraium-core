from __future__ import annotations

import json

from neraium_core.sii import cli
from neraium_core import sii_cli as hardened_cli


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


class _DummyLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None


class _DummyEngine:
    def close(self) -> None:
        return None


class _DummyApp:
    def __init__(self) -> None:
        self.engine = _DummyEngine()

    def run_payloads(self, payloads):
        return [{"frame_id": i} for i, _ in enumerate(payloads)]

    def write_output_file(self, output_path, outputs) -> None:
        output_path.write_text(json.dumps(outputs), encoding="utf-8")


def test_cli_rejects_invalid_json_shape(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "bad.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"temperature": 10.0},
                },
                "not-an-object",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hardened_cli, "configure_structured_logging", lambda _level: _DummyLogger())
    monkeypatch.setattr(hardened_cli.SIIApplication, "from_config", lambda _config: _DummyApp())
    monkeypatch.setattr(
        hardened_cli,
        "run_structural_pipeline",
        lambda _payloads, config=None: {"validation_results": {}},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sii_cli", "--input", str(input_path), "--output", str(output_path)],
    )

    code = hardened_cli.main()
    summary = json.loads(capsys.readouterr().out)

    assert code == 1
    assert summary["frames_succeeded"] == 1
    assert summary["frames_failed"] == 1
    assert summary["partial_success"] is True
    assert summary["all_failed"] is False
    assert summary["ingest_errors"]


def test_cli_prints_human_report_when_enabled(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_sample_payloads()), encoding="utf-8")

    code = cli.main(["--input", str(input_path), "--report"])

    assert code == 0
    stdout = capsys.readouterr().out
    assert "==== Structural Risk Summary ====" in stdout
    assert "\"decision_output\"" in stdout


def test_cli_writes_report_file_when_requested(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    report_path = tmp_path / "report.txt"
    input_path.write_text(json.dumps(_sample_payloads()), encoding="utf-8")

    code = cli.main(["--input", str(input_path), "--report-file", str(report_path)])

    assert code == 0
    report_text = report_path.read_text(encoding="utf-8")
    assert "==== Structural Risk Summary ====" in report_text
