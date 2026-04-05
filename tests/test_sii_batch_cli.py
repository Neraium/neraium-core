from __future__ import annotations

import json
from pathlib import Path

from neraium_core.sii import cli


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
