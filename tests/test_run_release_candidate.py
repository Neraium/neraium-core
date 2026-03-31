from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_release_candidate_outputs_status(tmp_path: Path) -> None:
    corpus_root = tmp_path / "validation" / "corpus"
    data_dir = corpus_root / "snapshots" / "data"
    data_dir.mkdir(parents=True)
    data_file = data_dir / "events.json"
    data_file.write_text(json.dumps({"events": [{"timestamp": 1, "asset_id": "A", "observation": {"x": 1.0}, "outcome_label": "neutral", "domain": "water", "system_type": "pump"}]}), encoding="utf-8")

    import hashlib

    sha = hashlib.sha256(data_file.read_bytes()).hexdigest()
    snapshot = {
        "corpus_id": "corpus_test",
        "description": "test",
        "created_at": "2026-03-31T00:00:00Z",
        "schema_version": "1.0",
        "source_datasets": [{"name": "synthetic"}],
        "metadata_summary": {"domain_coverage": ["water"], "system_types": ["pump"], "number_of_trajectories": 1, "intervention_coverage": 0.0},
        "ingestion_parameters": {},
        "quality_requirements": {"min_dataset_size": 1, "min_intervention_coverage": 0.0, "min_domain_diversity": 1},
        "data_files": [{"path": str(data_file), "format": "json", "sha256": sha}],
    }
    (corpus_root / "snapshots").mkdir(parents=True, exist_ok=True)
    (corpus_root / "snapshots" / "corpus_test.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (corpus_root / "registry.json").write_text(json.dumps({"schema_version": "1.0", "snapshots": [{"corpus_id": "corpus_test", "snapshot_file": "snapshots/corpus_test.json"}]}), encoding="utf-8")

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
