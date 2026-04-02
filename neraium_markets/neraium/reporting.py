"""Day 5 validation reporting and output persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from neraium.baselines import compare_to_baselines
from neraium.evaluation import evaluate_confidence_calibration


def build_validation_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact Day 5 validation summary dictionary."""
    useful_cols = [c for c in df.columns if c.startswith("action_useful_")]

    usefulness_summary = {
        c.replace("action_useful_", ""): float(df[c].mean()) for c in useful_cols
    }

    by_regime: dict[str, dict[str, float]] = {}
    if "regime_label" in df.columns and useful_cols:
        grouped = df.groupby("regime_label", observed=False)[useful_cols].mean(numeric_only=True)
        for regime, row in grouped.iterrows():
            by_regime[str(regime)] = {
                c.replace("action_useful_", ""): float(row[c]) for c in useful_cols
            }

    calibration = evaluate_confidence_calibration(df)
    baseline = compare_to_baselines(df)

    report: dict[str, Any] = {
        "total_signals": int(len(df)),
        "regime_counts": df.get("regime_label", pd.Series(dtype=str)).value_counts().to_dict(),
        "action_counts": df.get("action_posture", pd.Series(dtype=str)).value_counts().to_dict(),
        "average_confidence": float(df.get("confidence_score", pd.Series([0.0])).mean()),
        "usefulness_summary": usefulness_summary,
        "usefulness_by_regime": by_regime,
        "confidence_calibration": calibration.to_dict(orient="records"),
        "baseline_comparison": baseline.to_dict(orient="records"),
    }
    return report


def save_validation_outputs(
    signals_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path = "output",
) -> dict[str, Path]:
    """Persist Day 5 validation outputs (CSV/JSON) under ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_path = out_dir / "signals_with_forward_returns.csv"
    calib_path = out_dir / "confidence_calibration.csv"
    baseline_path = out_dir / "baseline_comparison.csv"
    summary_path = out_dir / "validation_summary.json"

    signals_df.to_csv(signals_path, index=False)
    calibration_df.to_csv(calib_path, index=False)
    baseline_df.to_csv(baseline_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    return {
        "signals": signals_path,
        "calibration": calib_path,
        "baseline": baseline_path,
        "summary": summary_path,
    }
