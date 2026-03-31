from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _write_snapshot(corpus_root: Path, corpus_id: str, corpus_type: str) -> None:
    data_dir = corpus_root / "snapshots" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / f"{corpus_id}.json"
    data_file.write_text(
        json.dumps(
            {
                "events": [
                    {"timestamp": 1, "asset_id": f"{corpus_id}-A", "observation": {"x": 1.0}, "outcome_label": "neutral", "domain": "water", "system_type": "pump"},
                    {"timestamp": 2, "asset_id": f"{corpus_id}-B", "observation": {"x": 1.2}, "outcome_label": "helpful", "domain": "energy", "system_type": "compressor", "actual_intervention": "remove_top_driver_contribution"},
                    {"timestamp": 3, "asset_id": f"{corpus_id}-C", "observation": {"x": 1.3}, "outcome_label": "harmful", "domain": "energy", "system_type": "compressor", "actual_intervention": "restore_relationship_cluster_to_baseline"},
                    {"timestamp": 4, "asset_id": f"{corpus_id}-D", "observation": {"x": 0.9}, "outcome_label": "neutral", "domain": "water", "system_type": "pump"},
                ]
            }
        ),
        encoding="utf-8",
    )
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
    _write_snapshot(corpus_root, "c_base", "baseline_clean")
    _write_snapshot(corpus_root, "c_noise", "noisy_realistic")
    _write_snapshot(corpus_root, "c_adv", "adversarial")
    _write_snapshot(corpus_root, "c_transfer", "transfer_cross_domain")
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
