from __future__ import annotations

import json
from pathlib import Path

from validation.history.trends import build_multi_corpus_trend_summary


def test_multi_corpus_trend_summary_contains_cross_corpus_stability(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir(parents=True)
    index = {
        "runs": [
            {"timestamp": "2026-03-31T00:00:01Z", "corpus_id": "b", "run_id": "1", "decision_accuracy": 0.9, "harm_rate": 0.1, "release_passed": True, "corpus_type": "baseline_clean"},
            {"timestamp": "2026-03-31T00:00:02Z", "corpus_id": "t", "run_id": "2", "decision_accuracy": 0.7, "harm_rate": 0.2, "release_passed": False, "corpus_type": "transfer_cross_domain"},
        ]
    }
    (history_root / "index.json").write_text(json.dumps(index), encoding="utf-8")

    summary = build_multi_corpus_trend_summary(history_root)
    assert "per_corpus" in summary
    assert "cross_corpus_stability" in summary
    assert summary["cross_corpus_stability"]["latest_accuracy_stddev"] > 0
