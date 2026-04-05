from __future__ import annotations

import json
from pathlib import Path

import pytest

from neraium_core import sii_cli
from neraium_core.sii import canonical_records_from_payloads_isolated
from neraium_core.sii.errors import SIIValidationError
from neraium_core.sii.ingestion import load_frames_from_csv, load_frames_from_json


def _valid_payload(ts: str = "2026-01-01T00:00:00Z") -> dict[str, object]:
    return {
        "timestamp": ts,
        "site_id": "site-a",
        "asset_id": "asset-a",
        "sensor_values": {"temperature": 10.0},
    }


def test_canonical_records_isolated_is_publicly_exported() -> None:
    records, errors = canonical_records_from_payloads_isolated([_valid_payload(), 1])
    assert len(records) == 1
    assert len(errors) == 1
    assert errors[0]["code"] == "invalid_payload"


def test_csv_isolation_one_bad_row_preserves_valid_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text(
        (
            "timestamp,site_id,asset_id,temp,metadata\n"
            "2026-01-01T00:00:00Z,site-a,asset-a,1.0,\"{\"\"ok\"\": true}\"\n"
            "2026-01-01T00:01:00Z,site-a,asset-a,2.0,{bad-json\n"
            "2026-01-01T00:02:00Z,site-a,asset-a,3.0,\"{\"\"ok\"\": true}\"\n"
        ),
        encoding="utf-8",
    )

    payloads, ingest_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)
    assert len(payloads) == 2
    assert len(ingest_errors) == 1
    assert ingest_errors[0]["message"].startswith("Invalid CSV row 3:")


def test_csv_isolation_multiple_bad_rows_collects_errors(tmp_path: Path) -> None:
    csv_path = tmp_path / "multi_bad.csv"
    csv_path.write_text(
        (
            "timestamp,site_id,asset_id,temp,metadata,quality_metadata\n"
            "2026-01-01T00:00:00Z,site-a,asset-a,1.0,\"{\"\"ok\"\": true}\",\"{\"\"missingness_rate\"\": 0.2}\"\n"
            "2026-01-01T00:01:00Z,site-a,asset-a,2.0,[],\"{\"\"missingness_rate\"\": 0.2}\"\n"
            "2026-01-01T00:02:00Z,site-a,asset-a,3.0,\"{\"\"ok\"\": true}\",not-json\n"
        ),
        encoding="utf-8",
    )

    payloads, ingest_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)
    assert len(payloads) == 1
    assert len(ingest_errors) == 2
    assert all(error["code"] == "invalid_payload" for error in ingest_errors)


def test_csv_structural_failure_still_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_structural.csv"
    csv_path.write_text("timestamp,site_id,temp\n2026-01-01T00:00:00Z,site-a,1.0\n", encoding="utf-8")
    with pytest.raises(SIIValidationError, match="CSV missing required columns"):
        load_frames_from_csv(str(csv_path))


def test_cross_path_json_csv_consistency_partial_success(tmp_path: Path) -> None:
    json_path = tmp_path / "mixed.json"
    json_path.write_text(json.dumps([_valid_payload(), 7]), encoding="utf-8")
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text(
        (
            "timestamp,site_id,asset_id,temp,metadata\n"
            "2026-01-01T00:00:00Z,site-a,asset-a,1.0,\"{\"\"ok\"\": true}\"\n"
            "2026-01-01T00:01:00Z,site-a,asset-a,2.0,{bad-json\n"
        ),
        encoding="utf-8",
    )

    json_payloads, json_errors = load_frames_from_json(str(json_path), return_ingest_errors=True)
    csv_payloads, csv_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)

    assert len(json_payloads) == len(csv_payloads) == 1
    assert len(json_errors) == len(csv_errors) == 1
    assert set(json_errors[0].keys()) >= {"index", "code", "message"}
    assert set(csv_errors[0].keys()) >= {"index", "code", "message"}
    assert json_errors[0]["code"] == csv_errors[0]["code"] == "invalid_payload"


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

    def run_payloads(self, payloads: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{"frame_id": i} for i, _ in enumerate(payloads)]

    def write_output_file(self, output_path: Path, outputs: list[dict[str, object]]) -> None:
        output_path.write_text(json.dumps(outputs), encoding="utf-8")


def test_cli_summary_distinguishes_empty_partial_failed_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sii_cli, "configure_structured_logging", lambda _level: _DummyLogger())
    monkeypatch.setattr(sii_cli.SIIApplication, "from_config", lambda _config: _DummyApp())
    monkeypatch.setattr(sii_cli, "run_structural_pipeline", lambda _payloads, config=None: {"validation_results": {}})

    cases = [
        ("empty.json", "[]", {"input_empty": True, "all_failed": False, "partial_success": False, "frames_succeeded": 0, "frames_failed": 0}, 0),
        ("all_failed.json", json.dumps([1, 2]), {"input_empty": False, "all_failed": True, "partial_success": False, "frames_succeeded": 0, "frames_failed": 2}, 1),
        ("partial.json", json.dumps([_valid_payload(), 1]), {"input_empty": False, "all_failed": False, "partial_success": True, "frames_succeeded": 1, "frames_failed": 1}, 1),
        ("full.json", json.dumps([_valid_payload(), _valid_payload("2026-01-01T00:01:00Z")]), {"input_empty": False, "all_failed": False, "partial_success": False, "frames_succeeded": 2, "frames_failed": 0}, 0),
    ]

    for filename, payload, expected, expected_exit_code in cases:
        input_path = tmp_path / filename
        output_path = tmp_path / f"{filename}.out.json"
        input_path.write_text(payload, encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_path)])
        assert sii_cli.main() == expected_exit_code
        summary = json.loads(capsys.readouterr().out)
        for key, value in expected.items():
            assert summary[key] == value
        assert summary["results_emitted"] == summary["frames_succeeded"]
        assert set(summary.keys()) >= {
            "frames_succeeded",
            "frames_failed",
            "results_emitted",
            "input_empty",
            "all_failed",
            "partial_success",
            "ingest_errors",
        }
