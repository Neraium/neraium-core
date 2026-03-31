from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.history.tracker import load_history_index


def _series(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        out.append({"timestamp": row.get("timestamp"), "corpus_id": row.get("corpus_id"), "run_id": row.get("run_id"), "value": value})
    return out


def build_trend_summary(history_root: Path) -> dict[str, Any]:
    index = load_history_index(history_root)
    rows = list(index.get("runs", []))
    total = len(rows)
    release_pass = sum(1 for r in rows if r.get("release_passed"))

    regressions = 0
    by_corpus: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_corpus.setdefault(str(row.get("corpus_id")), []).append(row)

    for corpus_rows in by_corpus.values():
        ordered = sorted(corpus_rows, key=lambda r: r.get("timestamp", ""))
        prev = None
        for row in ordered:
            if prev and row.get("decision_accuracy") is not None and prev.get("decision_accuracy") is not None:
                if float(row["decision_accuracy"]) < float(prev["decision_accuracy"]):
                    regressions += 1
            prev = row

    return {
        "run_count": total,
        "accuracy_over_time": _series(rows, "decision_accuracy"),
        "harm_rate_over_time": _series(rows, "harm_rate"),
        "calibration_over_time": _series(rows, "calibration_quality"),
        "release_gate_pass_rate": round(release_pass / max(1, total), 6),
        "regression_frequency": round(regressions / max(1, total), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute validation trend summary from history index")
    parser.add_argument("--history-root", default="reports/validation/history")
    parser.add_argument("--output", default="reports/validation/history/trend_summary.json")
    args = parser.parse_args()

    history_root = Path(args.history_root)
    summary = build_trend_summary(history_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
