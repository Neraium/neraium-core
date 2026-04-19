from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from tools.run_validation import execute_validation
from validation.corpus.registry import CorpusSnapshotRegistry, load_records_for_run


def _load_transfer_records(corpus_id: str) -> list[dict]:
    registry = CorpusSnapshotRegistry(Path("validation/corpus"))
    records, _, _ = load_records_for_run(
        corpus_id=corpus_id,
        input_path=None,
        input_format=None,
        registry=registry,
    )
    return records


def test_default_release_transfer_corpora_are_credible_and_sequential() -> None:
    registry = CorpusSnapshotRegistry(Path("validation/corpus"))
    corpus_ids = registry.list_release_gating_corpus_ids_by_type("transfer_cross_domain", minimum_records=50)

    assert set(corpus_ids) == {"transfer_cross_domain_small_scale", "transfer_cross_domain_large_scale"}

    for corpus_id in corpus_ids:
        records = _load_transfer_records(corpus_id)
        assert len(records) >= 50

        by_asset: dict[str, list[dict]] = defaultdict(list)
        for row in records:
            by_asset[str(row.get("asset_id"))].append(row)

        assert len(by_asset) >= 4
        for rows in by_asset.values():
            ts = [float(r.get("timestamp", 0.0)) for r in rows]
            assert ts == sorted(ts)
            assert len(set(ts)) >= 10


def test_run_release_candidate_transfer_corpora_complete_without_dataset_size_error(tmp_path: Path) -> None:
    payload = execute_validation(
        argparse.Namespace(
            input=None,
            format=None,
            corpus_id="transfer_cross_domain_small_scale",
            corpus_root="validation/corpus",
            output=str(tmp_path / "small_out.json"),
            core_output=str(tmp_path / "small_core.json"),
            release_gate_output=str(tmp_path / "small_gate.json"),
            prior_core_report=None,
            history_root=str(tmp_path / "history"),
            write_history=False,
            mark_baseline=False,
        )
    )
    assert payload["core_validation_report"]["corpus_summary"]["total_records"] >= 50

    payload = execute_validation(
        argparse.Namespace(
            input=None,
            format=None,
            corpus_id="transfer_cross_domain_large_scale",
            corpus_root="validation/corpus",
            output=str(tmp_path / "large_out.json"),
            core_output=str(tmp_path / "large_core.json"),
            release_gate_output=str(tmp_path / "large_gate.json"),
            prior_core_report=None,
            history_root=str(tmp_path / "history"),
            write_history=False,
            mark_baseline=False,
        )
    )
    assert payload["core_validation_report"]["corpus_summary"]["total_records"] >= 50
