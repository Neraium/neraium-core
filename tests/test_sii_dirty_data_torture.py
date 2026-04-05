from __future__ import annotations

import json
from pathlib import Path

import pytest

from neraium_core import sii_cli
from neraium_core.sii.errors import SIIValidationError
from neraium_core.sii.ingestion import load_frames_from_csv, load_frames_from_json


def _valid_payload(ts: str = "2026-01-01T00:00:00Z") -> dict[str, object]:
    return {
        "timestamp": ts,
        "site_id": "site-a",
        "asset_id": "asset-a",
        "sensor_values": {"temperature": 10.0},
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

    def run_payloads(self, payloads: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{"frame_id": i} for i, _ in enumerate(payloads)]

    def write_output_file(self, output_path: Path, outputs: list[dict[str, object]]) -> None:
        output_path.write_text(json.dumps(outputs), encoding="utf-8")


def _patch_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sii_cli, "configure_structured_logging", lambda _level: _DummyLogger())
    monkeypatch.setattr(sii_cli.SIIApplication, "from_config", lambda _config: _DummyApp())
    monkeypatch.setattr(sii_cli, "run_structural_pipeline", lambda _payloads, config=None: {"validation_results": {}})


def _run_cli(
    *,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    input_path: Path,
    output_path: Path,
) -> tuple[int, dict[str, object]]:
    _patch_cli(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["sii_cli", "--input", str(input_path), "--output", str(output_path)],
    )
    code = sii_cli.main()
    return code, json.loads(capsys.readouterr().out)


def test_json_dirty_mixed_records_lock_typed_error_contract(tmp_path: Path) -> None:
    json_path = tmp_path / "dirty.json"
    json_path.write_text(
        json.dumps(
            [
                _valid_payload(),
                7,
                {
                    "timestamp": "2026-01-01T00:01:00Z",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "metadata": {"ok": True},
                },
                {
                    "timestamp": "2026-01-01T00:02:00Z",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"temperature": 1.0},
                    "metadata": [],
                },
                {
                    "timestamp": "not-a-time",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"temperature": 2.0},
                    "quality_metadata": [],
                    "source_metadata": [],
                    "unexpected": "field",
                },
            ]
        ),
        encoding="utf-8",
    )

    payloads, ingest_errors = load_frames_from_json(str(json_path), return_ingest_errors=True)

    assert len(payloads) == 2
    assert [error["code"] for error in ingest_errors] == [
        "non_object_payload",
        "missing_required_field",
        "invalid_metadata_json",
    ]
    assert payloads[1]["timestamp"] == "0"
    assert "invalid_or_missing_timestamp_defaulted_to_epoch_zero" in payloads[1]["quality_metadata"]["validation_issues"]


def test_json_dirty_duplicate_source_record_id_is_not_deduplicated(tmp_path: Path) -> None:
    json_path = tmp_path / "dupe_ids.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    **_valid_payload("2026-01-01T00:00:00Z"),
                    "source_metadata": {"source_record_id": "dup-1"},
                },
                {
                    **_valid_payload("2026-01-01T00:01:00Z"),
                    "source_metadata": {"source_record_id": "dup-1"},
                },
            ]
        ),
        encoding="utf-8",
    )

    payloads, ingest_errors = load_frames_from_json(str(json_path), return_ingest_errors=True)

    assert ingest_errors == []
    assert [p["source_metadata"]["source_record_id"] for p in payloads] == ["dup-1", "dup-1"]


@pytest.mark.parametrize(
    "filename,csv_text,expected_payloads,expected_codes",
    [
        (
            "one_bad_row.csv",
            (
                "timestamp,site_id,asset_id,temp,metadata\n"
                "2026-01-01T00:00:00Z,site-a,asset-a,1.0,\"{\"\"ok\"\": true}\"\n"
                "2026-01-01T00:01:00Z,site-a,asset-a,2.0,{bad-json\n"
                "2026-01-01T00:02:00Z,site-a,asset-a,3.0,\"{\"\"ok\"\": true}\"\n"
            ),
            2,
            ["invalid_metadata_json"],
        ),
        (
            "multi_bad_rows.csv",
            (
                "timestamp,site_id,asset_id,temp,quality_metadata,source_metadata\n"
                "2026-01-01T00:00:00Z,site-a,asset-a,1.0,not-json,\n"
                "2026-01-01T00:01:00Z,site-a,asset-a,2.0,\"{\"\"missingness_rate\"\": 0.2}\",not-json\n"
                "2026-01-01T00:02:00Z,site-a,asset-a,3.0,\"{\"\"missingness_rate\"\": 0.2}\",\n"
            ),
            1,
            ["invalid_quality_metadata_json", "invalid_source_metadata_json"],
        ),
        (
            "all_invalid_rows.csv",
            (
                "timestamp,site_id,asset_id,temp,metadata\n"
                "2026-01-01T00:00:00Z,site-a,asset-a,1.0,{bad\n"
                "2026-01-01T00:01:00Z,site-a,asset-a,2.0,[]oops\n"
            ),
            0,
            ["invalid_metadata_json", "invalid_metadata_json"],
        ),
    ],
)
def test_csv_dirty_row_isolation_and_typed_codes(
    tmp_path: Path,
    filename: str,
    csv_text: str,
    expected_payloads: int,
    expected_codes: list[str],
) -> None:
    csv_path = tmp_path / filename
    csv_path.write_text(csv_text, encoding="utf-8")

    payloads, ingest_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)

    assert len(payloads) == expected_payloads
    assert [error["code"] for error in ingest_errors] == expected_codes
    assert all(set(error.keys()) >= {"index", "code", "message"} for error in ingest_errors)


def test_csv_dirty_blank_optional_fields_and_invalid_numeric_are_salvaged(tmp_path: Path) -> None:
    csv_path = tmp_path / "optional_blank.csv"
    csv_path.write_text(
        (
            "timestamp,site_id,asset_id,temp,quality_metadata,source_metadata,metadata\n"
            "2026-01-01T00:00:00Z,site-a,asset-a,not-a-number,,,\n"
            ",site-a,asset-a,2.0,,,\n"
        ),
        encoding="utf-8",
    )

    payloads, ingest_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)

    assert ingest_errors == []
    assert len(payloads) == 2
    assert payloads[0]["variables"]["temp"] is None
    assert payloads[0]["quality_metadata"]["validation_issues"]
    assert payloads[1]["timestamp"] == "0"


def test_csv_structural_missing_required_columns_fails_hard(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text("timestamp,site_id,temp\n2026-01-01T00:00:00Z,site-a,1.0\n", encoding="utf-8")

    with pytest.raises(SIIValidationError, match="CSV missing required columns"):
        load_frames_from_csv(str(csv_path))


def test_empty_inputs_lock_semantics_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "empty.json"
    csv_path = tmp_path / "header_only.csv"
    json_path.write_text("[]", encoding="utf-8")
    csv_path.write_text("timestamp,site_id,asset_id,temp\n", encoding="utf-8")

    json_payloads, json_errors = load_frames_from_json(str(json_path), return_ingest_errors=True)
    csv_payloads, csv_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)

    assert json_payloads == [] and json_errors == []
    assert csv_payloads == [] and csv_errors == []


def test_cross_path_same_bad_data_locks_compatible_semantics(tmp_path: Path) -> None:
    json_path = tmp_path / "cross.json"
    csv_path = tmp_path / "cross.csv"
    json_path.write_text(
        json.dumps(
            [
                _valid_payload(),
                {
                    "timestamp": "2026-01-01T00:01:00Z",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"temperature": 2.0},
                    "metadata": [],
                    "source_metadata": {"source_record_id": "row-2"},
                },
            ]
        ),
        encoding="utf-8",
    )
    csv_path.write_text(
        (
            "timestamp,site_id,asset_id,temp,metadata\n"
            "2026-01-01T00:00:00Z,site-a,asset-a,1.0,\"{\"\"ok\"\": true}\"\n"
            "2026-01-01T00:01:00Z,site-a,asset-a,2.0,[]\n"
        ),
        encoding="utf-8",
    )

    json_payloads, json_errors = load_frames_from_json(str(json_path), return_ingest_errors=True)
    csv_payloads, csv_errors = load_frames_from_csv(str(csv_path), return_ingest_errors=True)

    assert len(json_payloads) == len(csv_payloads) == 1
    assert len(json_errors) == len(csv_errors) == 1
    assert json_errors[0]["code"] == csv_errors[0]["code"] == "invalid_metadata_json"




def test_cli_error_truncation_fields_under_cap_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    input_path = tmp_path / "under_cap.json"
    output_path = tmp_path / "under_cap.out.json"
    input_path.write_text(json.dumps([_valid_payload(), 1, 2]), encoding="utf-8")

    code, summary = _run_cli(monkeypatch=monkeypatch, capsys=capsys, input_path=input_path, output_path=output_path)

    assert code == 1
    assert summary["frames_failed"] == 2
    assert summary["errors_truncated"] is False
    assert summary["total_error_count"] == 2
    assert summary["returned_error_count"] == 2
    assert len(summary["ingest_errors"]) == 2


def test_cli_error_truncation_fields_over_cap_are_explicit_and_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    input_path = tmp_path / "over_cap.json"
    output_path = tmp_path / "over_cap.out.json"
    input_path.write_text(json.dumps([1 for _ in range(25)]), encoding="utf-8")

    code, summary = _run_cli(monkeypatch=monkeypatch, capsys=capsys, input_path=input_path, output_path=output_path)

    assert code == 1
    assert summary["frames_failed"] == 25
    assert summary["errors_truncated"] is True
    assert summary["total_error_count"] == 25
    assert summary["returned_error_count"] == 21
    assert len(summary["ingest_errors"]) == 21
    assert summary["ingest_errors"][-1] == {"code": "truncated", "message": "additional errors omitted"}

def test_cli_summary_contract_shape_is_frozen_for_partial_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    input_path = tmp_path / "partial.json"
    output_path = tmp_path / "partial.out.json"
    input_path.write_text(json.dumps([_valid_payload(), {"timestamp": "2026-01-01T00:01:00Z"}]), encoding="utf-8")

    code, summary = _run_cli(monkeypatch=monkeypatch, capsys=capsys, input_path=input_path, output_path=output_path)

    assert code == 1
    assert summary["frames_succeeded"] == 1
    assert summary["frames_failed"] == 1
    assert summary["results_emitted"] == 1
    assert summary["input_empty"] is False
    assert summary["all_failed"] is False
    assert summary["partial_success"] is True
    assert isinstance(summary["ingest_errors"], list) and len(summary["ingest_errors"]) == 1
    assert set(summary["ingest_errors"][0].keys()) >= {"index", "code", "message"}
    assert summary["errors_truncated"] is False
    assert summary["total_error_count"] == 1
    assert summary["returned_error_count"] == 1


def test_cli_semantic_states_exit_codes_and_flags_are_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cases = [
        ("full_success.json", json.dumps([_valid_payload()]), 0, 1, 0, False, False, False, False),
        ("partial_success.json", json.dumps([_valid_payload(), 1]), 1, 1, 1, False, False, True, False),
        ("all_failed.json", json.dumps([1, 2]), 1, 0, 2, False, True, False, False),
        ("empty_input.json", "[]", 0, 0, 0, True, False, False, False),
    ]

    for filename, payload, expected_code, ok_count, fail_count, input_empty, all_failed, partial_success, structural in cases:
        input_path = tmp_path / filename
        output_path = tmp_path / f"{filename}.out.json"
        input_path.write_text(payload, encoding="utf-8")

        code, body = _run_cli(monkeypatch=monkeypatch, capsys=capsys, input_path=input_path, output_path=output_path)

        assert code == expected_code
        assert body["frames_succeeded"] == ok_count
        assert body["frames_failed"] == fail_count
        assert body["results_emitted"] == ok_count
        assert body["input_empty"] is input_empty
        assert body["all_failed"] is all_failed
        assert body["partial_success"] is partial_success
        assert bool(body["ingest_errors"]) is (fail_count > 0)
        assert body["errors_truncated"] is False
        assert body["total_error_count"] == fail_count
        assert body["returned_error_count"] == fail_count
        assert structural is False


def test_cli_structural_failure_is_distinct_contract_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    input_path = tmp_path / "structural.csv"
    output_path = tmp_path / "structural.out.json"
    input_path.write_text("timestamp,site_id,temp\n2026-01-01T00:00:00Z,site-a,1.0\n", encoding="utf-8")

    _patch_cli(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["sii_cli", "--input", str(input_path), "--output", str(output_path)],
    )

    code = sii_cli.main()
    body = json.loads(capsys.readouterr().out)

    assert code == 2
    assert "error" in body
    assert "CSV missing required columns" in body["error"]
    assert "frames_succeeded" not in body
    assert "ingest_errors" not in body
