from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform
from validation import RealWorldValidationPipeline


def _load_baseline_records() -> list[dict]:
    snapshot_path = Path("validation/corpus/snapshots/data/baseline_clean_ops.json")
    records = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert all(isinstance(item, dict) for item in records)
    return records


def test_baseline_clean_ops_snapshot_is_top_level_list_of_100_dicts() -> None:
    snapshot_path = Path("validation/corpus/snapshots/data/baseline_clean_ops.json")
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert isinstance(data, list)
    assert len(data) == 100
    assert all(isinstance(item, dict) for item in data)


def test_baseline_clean_ops_is_real_sequential_corpus() -> None:
    records = _load_baseline_records()

    assert 90 <= len(records) <= 120

    by_asset: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_asset[str(row.get("asset_id"))].append(row)

    assert len(by_asset) >= 4
    assert all(len(rows) >= 20 for rows in by_asset.values())

    scenario_by_asset = {}
    for asset, rows in by_asset.items():
        sorted_rows = sorted(rows, key=lambda r: float(r.get("timestamp", 0.0)))
        timestamps = [float(r.get("timestamp", 0.0)) for r in sorted_rows]
        assert len(set(timestamps)) >= 20
        assert timestamps == sorted(timestamps)
        scenario_by_asset[asset] = {str(r.get("scenario_family") or "unknown") for r in sorted_rows}

    # non-degenerate progression: asset-level mean of feature x is not identical
    mean_x = {}
    for asset, rows in by_asset.items():
        xs = [float((r.get("observation") or {}).get("x", 0.0) or 0.0) for r in rows]
        mean_x[asset] = round(sum(xs) / max(1, len(xs)), 6)
    assert len(set(mean_x.values())) >= 3

    # non-degenerate temporal behavior: novelty/support vary through time
    novelty_values = {round(float(r.get("novelty", 0.0) or 0.0), 4) for r in records}
    support_values = {int(r.get("support_count", 0) or 0) for r in records}
    assert len(novelty_values) > 10
    assert len(support_values) > 8

    # progression families are distinct and present
    scenario_counts = Counter(str(r.get("scenario_family") or "unknown") for r in records)
    assert len(scenario_counts) >= 4
    assert all(v >= 20 for v in scenario_counts.values())

    # each asset should mostly map to a single trajectory family in this baseline
    for families in scenario_by_asset.values():
        assert len(families) == 1


def test_baseline_clean_ops_is_loadable_by_validation_pipeline() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    pipeline = RealWorldValidationPipeline(decision_fn=platform.update)
    report = pipeline.run(_load_baseline_records())

    corpus_summary = report["core_validation_report"]["corpus_summary"]
    assert corpus_summary["total_records"] >= 90
    assert corpus_summary["asset_count"] >= 4
    assert report["core_validation_report"]["summary"]["decision_accuracy"] >= 0.0
