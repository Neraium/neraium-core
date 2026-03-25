from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from neraium_core.alignment import StructuralEngine
from neraium_core.diagnostics import summarize_results


def _rows_from_input(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError("Only CSV inputs are supported")
    required = {"entity_id", "time"}
    if not required.issubset(df.columns):
        raise ValueError("Input CSV must include entity_id and time columns")
    return df.to_dict(orient="records")


def _build_frame(row: dict[str, Any], feature_cols: list[str]) -> dict[str, Any]:
    return {
        "timestamp": float(row["time"]),
        "site_id": "eval-site",
        "asset_id": str(row["entity_id"]),
        "sensor_values": {c: float(row[c]) for c in feature_cols},
    }


def run_eval(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    feature_cols = [c for c in rows[0].keys() if c not in {"entity_id", "time", "scenario"}]
    engine = StructuralEngine(baseline_window=24, recent_window=12)
    results = []
    for row in rows:
        results.append(engine.process_frame(_build_frame(row, feature_cols)))
    summary = summarize_results(results)

    stability = [float((r.get("robustness", {}).get("stability", {}) or {}).get("overall_stability", 0.0)) for r in results]
    sensitivity_pop = [1.0 if (r.get("sensitivity", {}).get("top_drivers", [])) else 0.0 for r in results]
    summary.update(
        {
            "stability_mean": round(sum(stability) / len(stability), 4) if stability else 0.0,
            "sensitivity_summary_rate": round(sum(sensitivity_pop) / len(sensitivity_pop), 4) if sensitivity_pop else 0.0,
        }
    )
    return results, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate structural pipeline robustness layer.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--csv-out", type=Path, default=None)
    args = p.parse_args()

    rows = _rows_from_input(args.input)
    results, summary = run_eval(rows)

    print("metric,value")
    for k, v in summary.items():
        print(f"{k},{v}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "sample_result": results[-1]}, indent=2), encoding="utf-8")
        print(f"json_summary={args.json_out}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in summary.items():
                w.writerow([k, v])
        print(f"csv_summary={args.csv_out}")


if __name__ == "__main__":
    main()
