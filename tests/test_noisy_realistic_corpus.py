from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from tools.run_validation import execute_validation
from validation.corpus.registry import CorpusSnapshotRegistry, load_records_for_run


def _load_noisy_records() -> list[dict]:
    payload = json.loads(Path("validation/corpus/snapshots/data/noisy_realistic_ops_full.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    events = payload.get("events", [])
    assert isinstance(events, list)
    assert all(isinstance(item, dict) for item in events)
    return events


def test_noisy_realistic_ops_registry_points_to_full_snapshot_data() -> None:
    registry = CorpusSnapshotRegistry(Path("validation/corpus"))
    snapshot = registry.get_snapshot("noisy_realistic_ops")

    assert snapshot.corpus_id == "noisy_realistic_ops"
    assert snapshot.data_files[0]["path"] == "snapshots/data/noisy_realistic_ops_full.json"


def test_noisy_realistic_ops_full_dataset_has_expected_scale_and_assets() -> None:
    records = _load_noisy_records()

    assert 90 <= len(records) <= 120
    assert len({str(row.get("asset_id")) for row in records}) >= 4
    assert len({float(row.get("timestamp", 0.0)) for row in records}) >= 25


def test_noisy_realistic_ops_loads_by_corpus_id_via_validation_pipeline() -> None:
    registry = CorpusSnapshotRegistry(Path("validation/corpus"))
    rows, source, is_canonical = load_records_for_run(
        corpus_id="noisy_realistic_ops",
        input_path=None,
        input_format=None,
        registry=registry,
    )

    assert is_canonical is True
    assert source["corpus"]["corpus_id"] == "noisy_realistic_ops"
    assert len(rows) >= 90

    payload = execute_validation(
        Namespace(
            input=None,
            format=None,
            corpus_id="noisy_realistic_ops",
            corpus_root="validation/corpus",
            output="/tmp/noisy_realistic_ops_registry_test_report.json",
            core_output="/tmp/noisy_realistic_ops_registry_test_core.json",
            release_gate_output="/tmp/noisy_realistic_ops_registry_test_release.json",
            prior_core_report=None,
            history_root="/tmp/validation_history_tests",
            write_history=False,
            mark_baseline=False,
        )
    )

    assert payload["core_validation_report"]["corpus_source"]["corpus_id"] == "noisy_realistic_ops"
    assert payload["core_validation_report"]["corpus_summary"]["total_records"] >= 90


def test_noisy_realistic_ops_manifest_never_uses_tiny_legacy_snapshot() -> None:
    registry = CorpusSnapshotRegistry(Path("validation/corpus"))
    rows, source, _ = load_records_for_run(
        corpus_id="noisy_realistic_ops",
        input_path=None,
        input_format=None,
        registry=registry,
    )

    manifest_paths = [str(item.get("path") or "") for item in source["corpus"].get("manifest", [])]
    assert len(rows) >= 50
    assert manifest_paths
    assert all("tiny_legacy" not in path for path in manifest_paths)
    assert any("noisy_realistic_ops_full.json" in path for path in manifest_paths)


def test_registry_rejects_duplicate_noisy_corpus_ids(tmp_path: Path) -> None:
    root = tmp_path / "validation" / "corpus"
    snaps = root / "snapshots"
    data = snaps / "data"
    data.mkdir(parents=True)

    sample = {"events": [{"timestamp": 1, "asset_id": "a1", "domain": "water", "scenario_family": "x", "system_type": "pump"}]}
    (data / "full.json").write_text(json.dumps(sample), encoding="utf-8")
    (snaps / "a.json").write_text(json.dumps({"corpus_id": "noisy_realistic_ops", "data_files": [{"path": "snapshots/data/full.json", "format": "json"}], "quality_requirements": {"min_dataset_size": 1, "min_intervention_coverage": 0.0, "min_domain_diversity": 1}}), encoding="utf-8")
    (snaps / "b.json").write_text(json.dumps({"corpus_id": "noisy_realistic_ops", "data_files": [{"path": "snapshots/data/full.json", "format": "json"}], "quality_requirements": {"min_dataset_size": 1, "min_intervention_coverage": 0.0, "min_domain_diversity": 1}}), encoding="utf-8")
    (root / "registry.json").write_text(json.dumps({"snapshots": [{"corpus_id": "noisy_realistic_ops", "snapshot_file": "snapshots/a.json"}, {"corpus_id": "noisy_realistic_ops", "snapshot_file": "snapshots/b.json"}]}), encoding="utf-8")

    registry = CorpusSnapshotRegistry(root)
    try:
        registry.get_snapshot("noisy_realistic_ops")
        assert False, "expected duplicate corpus id failure"
    except ValueError as exc:
        assert "Duplicate corpus_id" in str(exc)


def test_registry_rejects_snapshot_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "validation" / "corpus"
    snaps = root / "snapshots"
    data = snaps / "data"
    data.mkdir(parents=True)

    sample = {"events": [{"timestamp": 1, "asset_id": "a1", "domain": "water", "scenario_family": "x", "system_type": "pump"}]}
    (data / "full.json").write_text(json.dumps(sample), encoding="utf-8")
    (snaps / "wrong.json").write_text(json.dumps({"corpus_id": "noisy_realistic_ops_tiny_legacy", "data_files": [{"path": "snapshots/data/full.json", "format": "json"}], "quality_requirements": {"min_dataset_size": 1, "min_intervention_coverage": 0.0, "min_domain_diversity": 1}}), encoding="utf-8")
    (root / "registry.json").write_text(json.dumps({"snapshots": [{"corpus_id": "noisy_realistic_ops", "snapshot_file": "snapshots/wrong.json"}]}), encoding="utf-8")

    registry = CorpusSnapshotRegistry(root)
    try:
        registry.get_snapshot("noisy_realistic_ops")
        assert False, "expected snapshot identity mismatch failure"
    except ValueError as exc:
        assert "Snapshot identity mismatch" in str(exc)
