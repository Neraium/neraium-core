from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.run_validation import execute_validation
from validation.corpus.registry import CorpusSnapshotRegistry, compute_config_hash, load_records_for_run
from validation.history.tracker import load_history_index, resolve_baseline_report


def _write_corpus(tmp_path: Path, corpus_id: str = "corpus_test") -> Path:
    root = tmp_path / "validation" / "corpus"
    data_dir = root / "snapshots" / "data"
    data_dir.mkdir(parents=True)

    data_file = data_dir / f"{corpus_id}.json"
    payload = {
        "events": [
            {
                "timestamp": 1,
                "asset_id": "A",
                "observation": {"x": 0.1},
                "actual_intervention": "remove_top_driver_contribution",
                "outcome_label": "helpful",
                "domain": "water",
                "system_type": "pump",
            },
            {
                "timestamp": 2,
                "asset_id": "B",
                "observation": {"x": 0.2},
                "actual_intervention": "restore_relationship_cluster_to_baseline",
                "outcome_label": "harmful",
                "domain": "energy",
                "system_type": "compressor",
            },
            {
                "timestamp": 3,
                "asset_id": "C",
                "observation": {"x": 0.3},
                "actual_intervention": None,
                "outcome_label": "neutral",
                "domain": "water",
                "system_type": "pump",
            },
        ]
    }
    data_file.write_text(json.dumps(payload), encoding="utf-8")

    import hashlib

    sha = hashlib.sha256(data_file.read_bytes()).hexdigest()
    snapshot = {
        "corpus_id": corpus_id,
        "description": "test corpus",
        "created_at": "2026-03-31T00:00:00Z",
        "schema_version": "1.0",
        "source_datasets": [{"name": "synthetic"}],
        "metadata_summary": {
            "domain_coverage": ["water", "energy"],
            "system_types": ["pump", "compressor"],
            "number_of_trajectories": 3,
            "intervention_coverage": 0.66,
        },
        "ingestion_parameters": {"format": "json"},
        "quality_requirements": {
            "min_dataset_size": 3,
            "min_intervention_coverage": 0.0,
            "min_domain_diversity": 2,
        },
        "data_files": [{"path": str(data_file), "format": "json", "sha256": sha}],
    }
    snap_path = root / "snapshots" / f"{corpus_id}.json"
    snap_path.write_text(json.dumps(snapshot), encoding="utf-8")
    (root / "registry.json").write_text(
        json.dumps({"schema_version": "1.0", "snapshots": [{"corpus_id": corpus_id, "snapshot_file": f"snapshots/{corpus_id}.json"}]}),
        encoding="utf-8",
    )
    return root


def test_corpus_snapshot_loading_is_reproducible(tmp_path: Path) -> None:
    corpus_root = _write_corpus(tmp_path)
    registry = CorpusSnapshotRegistry(root=corpus_root)
    records_a, source_a, canonical_a = load_records_for_run(corpus_id="corpus_test", input_path=None, input_format=None, registry=registry)
    records_b, source_b, canonical_b = load_records_for_run(corpus_id="corpus_test", input_path=None, input_format=None, registry=registry)

    assert canonical_a is True and canonical_b is True
    assert records_a == records_b
    assert source_a["corpus"]["manifest"][0]["sha256"] == source_b["corpus"]["manifest"][0]["sha256"]


def test_validation_run_tags_and_history_tracking(tmp_path: Path) -> None:
    corpus_root = _write_corpus(tmp_path)
    history_root = tmp_path / "reports" / "validation" / "history"

    args = argparse.Namespace(
        input=None,
        format=None,
        corpus_id="corpus_test",
        corpus_root=str(corpus_root),
        output=str(tmp_path / "real_world_validation_report.json"),
        core_output=str(tmp_path / "core_validation_report.json"),
        release_gate_output=str(tmp_path / "release_gate_report.json"),
        prior_core_report=None,
        history_root=str(history_root),
        write_history=True,
        mark_baseline=True,
    )
    report = execute_validation(args)

    assert report["core_validation_report"]["run_metadata"]["corpus_id"] == "corpus_test"
    assert report["release_gate_report"]["run_metadata"]["run_id"]
    assert report["core_validation_report"]["run_metadata"]["config_hash"]

    idx = load_history_index(history_root)
    assert len(idx["runs"]) == 1
    assert idx["runs"][0]["is_baseline"] is True


def test_baseline_resolution_uses_latest_passing_run(tmp_path: Path) -> None:
    corpus_root = _write_corpus(tmp_path)
    history_root = tmp_path / "reports" / "validation" / "history"

    common = {
        "input": None,
        "format": None,
        "corpus_id": "corpus_test",
        "corpus_root": str(corpus_root),
        "prior_core_report": None,
        "history_root": str(history_root),
        "write_history": True,
        "mark_baseline": True,
    }
    execute_validation(argparse.Namespace(**common, output=str(tmp_path / "r1.json"), core_output=str(tmp_path / "c1.json"), release_gate_output=str(tmp_path / "g1.json")))
    common["mark_baseline"] = False
    execute_validation(argparse.Namespace(**common, output=str(tmp_path / "r2.json"), core_output=str(tmp_path / "c2.json"), release_gate_output=str(tmp_path / "g2.json")))

    baseline = resolve_baseline_report(history_root, "corpus_test")
    assert baseline is not None
    assert baseline["run_metadata"]["corpus_id"] == "corpus_test"


def test_config_hash_is_stable() -> None:
    payload = {"corpus_id": "corpus_v1", "format": "json"}
    assert compute_config_hash(payload) == compute_config_hash(dict(payload))
