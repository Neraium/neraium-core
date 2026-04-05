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


def test_watch_mode_processes_new_files_once(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    _write_payload(watch_dir / "first.json", delta=0.0)

    exit_code = cli.main(
        [
            "--watch",
            str(watch_dir),
            "--watch-iterations",
            "1",
            "--quiet",
        ]
    )

    assert exit_code == 0
