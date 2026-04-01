from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _write_snapshot(corpus_root: Path, corpus_id: str, corpus_type: str, *, num_events: int = 4) -> None:
    data_dir = corpus_root / "snapshots" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / f"{corpus_id}.json"
    events = [
        {
            "timestamp": idx + 1,
            "asset_id": f"{corpus_id}-{idx}",
            "observation": {"x": float((idx % 7) + 1)},
            "outcome_label": "neutral" if idx % 3 == 0 else ("helpful" if idx % 3 == 1 else "harmful"),
            "domain": "water" if idx % 2 == 0 else "energy",
            "system_type": "pump" if idx % 2 == 0 else "compressor",
            "actual_intervention": "remove_top_driver_contribution" if idx % 5 == 0 else None,
        }
        for idx in range(num_events)
    ]
    data_file.write_text(json.dumps({"events": events}), encoding="utf-8")
    sha = hashlib.sha256(data_file.read_bytes()).hexdigest()
    snapshot = {
        "corpus_id": corpus_id,
        "description": "test",
        "created_at": "2026-03-31T00:00:00Z",
        "schema_version": "1.0",
        "corpus_type": corpus_type,
        "expected_difficulty": "low",
        "coverage_tags": ["unit_test"],
        "source_datasets": [{"name": "synthetic"}],
        "metadata_summary": {"domain_coverage": ["water", "energy"], "system_types": ["pump", "compressor"], "number_of_trajectories": 4, "intervention_coverage": 0.5},
        "ingestion_parameters": {},
        "quality_requirements": {"min_dataset_size": 1, "min_intervention_coverage": 0.0, "min_domain_diversity": 1},
        "data_files": [{"path": str(data_file), "format": "json", "sha256": sha}],
    }
    (corpus_root / "snapshots" / f"{corpus_id}.json").write_text(json.dumps(snapshot), encoding="utf-8")


def test_run_release_candidate_outputs_status(tmp_path: Path) -> None:
    corpus_root = tmp_path / "validation" / "corpus"
    _write_snapshot(corpus_root, "corpus_test", "baseline_clean")
    (corpus_root / "registry.json").write_text(
        json.dumps({"schema_version": "1.0", "snapshots": [{"corpus_id": "corpus_test", "snapshot_file": "snapshots/corpus_test.json", "corpus_type": "baseline_clean"}]}),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "tools/run_release_candidate.py",
        "--corpus-id",
        "corpus_test",
        "--corpus-root",
        str(corpus_root),
        "--history-root",
        str(tmp_path / "history"),
        "--output",
        str(tmp_path / "out.json"),
        "--core-output",
        str(tmp_path / "core.json"),
        "--release-gate-output",
        str(tmp_path / "gate.json"),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "RELEASE_" in res.stdout


def test_run_release_candidate_corpus_set_supports_types(tmp_path: Path) -> None:
    corpus_root = tmp_path / "validation" / "corpus"
    _write_snapshot(corpus_root, "c_base", "baseline_clean", num_events=60)
    _write_snapshot(corpus_root, "c_noise", "noisy_realistic", num_events=60)
    _write_snapshot(corpus_root, "c_adv", "adversarial", num_events=60)
    _write_snapshot(corpus_root, "c_transfer", "transfer_cross_domain", num_events=60)
    (corpus_root / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "snapshots": [
                    {"corpus_id": "c_base", "snapshot_file": "snapshots/c_base.json", "corpus_type": "baseline_clean"},
                    {"corpus_id": "c_noise", "snapshot_file": "snapshots/c_noise.json", "corpus_type": "noisy_realistic"},
                    {"corpus_id": "c_adv", "snapshot_file": "snapshots/c_adv.json", "corpus_type": "adversarial"},
                    {"corpus_id": "c_transfer", "snapshot_file": "snapshots/c_transfer.json", "corpus_type": "transfer_cross_domain"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "multi_report.json"
    cmd = [
        sys.executable,
        "tools/run_release_candidate.py",
        "--corpus-set",
        "baseline_clean,noisy_realistic,adversarial,transfer_cross_domain",
        "--corpus-root",
        str(corpus_root),
        "--history-root",
        str(tmp_path / "history"),
        "--output",
        str(tmp_path / "out.json"),
        "--multi-corpus-output",
        str(report_path),
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "RELEASE_" in res.stdout
    assert payload["corpus_results"]
    assert "class_summary" in payload


def test_resolve_corpus_ids_excludes_deprecated_and_small_corpora(tmp_path: Path) -> None:
    from tools.run_release_candidate import _resolve_corpus_ids
    from validation.corpus.registry import CorpusSnapshotRegistry

    corpus_root = tmp_path / "validation" / "corpus"
    _write_snapshot(corpus_root, "legacy_small", "baseline_clean")
    _write_snapshot(corpus_root, "baseline_clean_ops", "baseline_clean")
    big_data = corpus_root / "snapshots" / "data" / "baseline_clean_ops.json"
    big_data.write_text(json.dumps({"events": [{"timestamp": i, "asset_id": f"a-{i}", "observation": {"x": i}} for i in range(100)]}), encoding="utf-8")
    big_sha = hashlib.sha256(big_data.read_bytes()).hexdigest()
    big_snapshot_path = corpus_root / "snapshots" / "baseline_clean_ops.json"
    big_snapshot = json.loads(big_snapshot_path.read_text(encoding="utf-8"))
    big_snapshot["data_files"] = [{"path": str(big_data), "format": "json", "sha256": big_sha}]
    big_snapshot_path.write_text(json.dumps(big_snapshot), encoding="utf-8")

    (corpus_root / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "snapshots": [
                    {
                        "corpus_id": "legacy_small",
                        "snapshot_file": "snapshots/legacy_small.json",
                        "corpus_type": "baseline_clean",
                        "deprecated": True,
                        "exclude_from_release_gating": True,
                    },
                    {
                        "corpus_id": "baseline_clean_ops",
                        "snapshot_file": "snapshots/baseline_clean_ops.json",
                        "corpus_type": "baseline_clean",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = CorpusSnapshotRegistry(root=corpus_root)
    resolved = _resolve_corpus_ids(registry, corpus_id=None, corpus_set="baseline_clean")
    assert resolved == ["baseline_clean_ops"]
