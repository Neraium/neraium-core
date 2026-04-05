from __future__ import annotations

import json
from pathlib import Path

from neraium_core.sii import cli
from neraium_core import sii_cli as hardened_cli


def _write_payload(path: Path, *, delta: float) -> None:
    payload = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.0 + delta, "pressure": 100.0},
        },
        {
            "timestamp": "2026-01-01T00:01:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.2 + delta, "pressure": 99.7},
        },
        {
            "timestamp": "2026-01-01T00:02:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.4 + delta, "pressure": 99.5},
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_batch_mode_writes_aggregate_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    _write_payload(input_dir / "b.json", delta=0.3)
    _write_payload(input_dir / "a.json", delta=0.0)
    aggregate_path = tmp_path / "aggregate.json"

    exit_code = cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--batch",
            "--quiet",
            "--aggregate-output",
            str(aggregate_path),
        ]
    )

    assert exit_code == 0
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert [run["source_file"] for run in aggregate["runs"]] == ["a.json", "b.json"]
    assert aggregate["aggregate_summary"]["total_runs"] == 2
    assert aggregate["aggregate_summary"]["successful_runs"] == 2
    assert aggregate["aggregate_summary"]["failed_files"] == 0
    assert aggregate["aggregate_summary"]["skipped_files"] == 0


def test_batch_mode_isolates_bad_file_and_keeps_processing(tmp_path: Path) -> None:
    input_dir = tmp_path / "mixed"
    input_dir.mkdir()
    _write_payload(input_dir / "good-a.json", delta=0.0)
    (input_dir / "bad.json").write_text("{not-valid-json", encoding="utf-8")
    _write_payload(input_dir / "good-b.json", delta=0.1)
    aggregate_path = tmp_path / "aggregate_mixed.json"

    exit_code = cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--batch",
            "--quiet",
            "--aggregate-output",
            str(aggregate_path),
        ]
    )

    assert exit_code == 0
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    statuses = {run["source_file"]: run["status"] for run in aggregate["runs"]}
    assert statuses["bad.json"] == "failed"
    assert statuses["good-a.json"] == "success"
    assert statuses["good-b.json"] == "success"
    assert aggregate["aggregate_summary"]["total_discovered_files"] == 3
    assert aggregate["aggregate_summary"]["successful_runs"] == 2
    assert aggregate["aggregate_summary"]["failed_files"] == 1
    assert aggregate["aggregate_summary"]["skipped_files"] == 0


def test_watch_mode_processes_new_files_once(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    _write_payload(watch_dir / "first.json", delta=0.0)

    exit_code = cli.main(
        [
            "--watch",
            str(watch_dir),
            "--watch-iterations",
            "2",
            "--quiet",
        ]
    )

    assert exit_code == 0


def test_watch_mode_does_not_reprocess_old_files(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch-no-replay"
    watch_dir.mkdir()
    _write_payload(watch_dir / "first.json", delta=0.0)
    aggregate_path = tmp_path / "watch_aggregate.json"

    exit_code = cli.main(
        [
            "--watch",
            str(watch_dir),
            "--watch-iterations",
            "2",
            "--poll-interval",
            "0.1",
            "--quiet",
            "--aggregate-output",
            str(aggregate_path),
        ]
    )

    assert exit_code == 0
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert [run["status"] for run in aggregate["runs"]] == ["success", "skipped"]
    assert aggregate["runs"][1]["skip_reason"] == "already_processed"
    assert aggregate["aggregate_summary"]["total_discovered_files"] == 2
    assert aggregate["aggregate_summary"]["successful_runs"] == 1
    assert aggregate["aggregate_summary"]["skipped_files"] == 1


def test_partial_write_unstable_file_is_skipped(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "unstable"
    input_dir.mkdir()
    _write_payload(input_dir / "unstable.json", delta=0.0)
    aggregate_path = tmp_path / "unstable_aggregate.json"

    monkeypatch.setattr(cli, "_is_file_size_stable", lambda _path: False)
    exit_code = cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--batch",
            "--quiet",
            "--aggregate-output",
            str(aggregate_path),
        ]
    )

    assert exit_code == 0
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["runs"][0]["status"] == "skipped"
    assert aggregate["runs"][0]["skip_reason"] == "partial_write_unstable_size"
    assert aggregate["aggregate_summary"]["successful_runs"] == 0
    assert aggregate["aggregate_summary"]["failed_files"] == 0
    assert aggregate["aggregate_summary"]["skipped_files"] == 1


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


def test_hardened_cli_exit_code_all_invalid_records(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "all_invalid.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    monkeypatch.setattr(hardened_cli, "configure_structured_logging", lambda _level: _DummyLogger())
    monkeypatch.setattr(hardened_cli.SIIApplication, "from_config", lambda _config: _DummyApp())
    monkeypatch.setattr(hardened_cli, "run_structural_pipeline", lambda _payloads, config=None: {"validation_results": {}})
    monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_path)])

    code = hardened_cli.main()
    summary = json.loads(capsys.readouterr().out)

    assert code == 1
    assert summary["frames_succeeded"] == 0
    assert summary["frames_failed"] == 3
    assert summary["all_failed"] is True
    assert summary["ingest_errors"]


def test_hardened_cli_exit_code_structural_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "bad.json"
    output_path = tmp_path / "out.json"
    input_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(hardened_cli, "configure_structured_logging", lambda _level: _DummyLogger())
    monkeypatch.setattr(hardened_cli.SIIApplication, "from_config", lambda _config: _DummyApp())
    monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_path)])

    code = hardened_cli.main()
    err = json.loads(capsys.readouterr().out)

    assert code == 2
    assert "error" in err
