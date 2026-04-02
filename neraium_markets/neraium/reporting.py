"""Day 5 validation reporting and Day 6 reliability reporting."""

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


def build_day6_reliability_report(
    df: pd.DataFrame,
    persistence_summary: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    transition_quality: pd.DataFrame,
    filtered_comparison: pd.DataFrame,
) -> dict[str, Any]:
    """
    Aggregate Day 6 diagnostics: persistence, transitions, stability, false positives, filtering.
    """
    flag_cols = [
        "low_persistence_flag",
        "contradiction_flag",
        "unstable_signal_flag",
        "likely_false_positive_flag",
    ]
    false_positive_counts = {c: int(df[c].sum()) for c in flag_cols if c in df.columns}

    stability_summary: dict[str, float] = {}
    for c in ("regime_flip_flag", "action_flip_flag", "confidence_jump_flag"):
        if c in df.columns:
            stability_summary[f"mean_{c}"] = float(df[c].astype(float).mean())
    if "stability_score" in df.columns:
        stability_summary["mean_stability_score"] = float(df["stability_score"].astype(float).mean())

    top_problematic = pd.DataFrame()
    if not transition_quality.empty and "avg_usefulness_1d" in transition_quality.columns:
        tq = transition_quality.copy()
        if "transition_count" in tq.columns:
            tq = tq[tq["transition_count"] >= 2]
        if not tq.empty:
            ucols = [c for c in tq.columns if c.startswith("avg_usefulness_")]
            if ucols:
                tq["_u"] = tq[ucols].mean(axis=1)
                top_problematic = tq.nsmallest(min(10, len(tq)), "_u").drop(columns=["_u"], errors="ignore")

    pattern_rows: list[dict[str, Any]] = []
    if false_positive_counts:
        pattern_rows.append(
            {
                "pattern": "likely_false_positive_total",
                "count": false_positive_counts.get("likely_false_positive_flag", 0),
            }
        )
        if "action_quick_reversal_flag" in df.columns:
            pattern_rows.append(
                {
                    "pattern": "action_quick_reversal",
                    "count": int(df["action_quick_reversal_flag"].sum()),
                }
            )
        if "high_drift_poor_outcome_flag" in df.columns:
            pattern_rows.append(
                {
                    "pattern": "high_drift_poor_outcome",
                    "count": int(df["high_drift_poor_outcome_flag"].sum()),
                }
            )

    avg_run = float(df["regime_run_length"].mean()) if "regime_run_length" in df.columns else float("nan")

    unfiltered_avg = filtered_comparison[filtered_comparison["version"] == "unfiltered"]
    filtered_avg = filtered_comparison[filtered_comparison["version"] == "filtered"]
    u_mean = 0.0
    f_mean = 0.0
    if not unfiltered_avg.empty:
        u_mean = float(
            unfiltered_avg[["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]]
            .mean(axis=1)
            .iloc[0]
        )
    if not filtered_avg.empty:
        f_mean = float(
            filtered_avg[["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]]
            .mean(axis=1)
            .iloc[0]
        )

    filtering_improved = bool(f_mean > u_mean + 1e-9) if filtered_comparison.shape[0] >= 2 else False

    return {
        "average_regime_run_length": avg_run,
        "regime_persistence_by_label": persistence_summary.to_dict(orient="records"),
        "transition_matrix_rows": int(len(transition_matrix)),
        "signal_stability_summary": stability_summary,
        "false_positive_flag_counts": false_positive_counts,
        "filtered_vs_unfiltered": filtered_comparison.to_dict(orient="records"),
        "filtering_improved_mean_usefulness": filtering_improved,
        "top_problematic_regime_transitions": top_problematic.to_dict(orient="records"),
        "top_suppressible_false_positive_patterns": pattern_rows,
    }


def save_day6_outputs(
    persistence_summary: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    transition_quality: pd.DataFrame,
    filtered_comparison: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path = "output",
) -> dict[str, Path]:
    """Write Day 6 CSV/JSON artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "regime_persistence": out_dir / "regime_persistence.csv",
        "transition_matrix": out_dir / "transition_matrix.csv",
        "transition_quality": out_dir / "transition_quality.csv",
        "filtered_signal_comparison": out_dir / "filtered_signal_comparison.csv",
        "day6_summary": out_dir / "day6_reliability_summary.json",
    }

    persistence_summary.to_csv(paths["regime_persistence"], index=False)
    transition_matrix.to_csv(paths["transition_matrix"], index=False)
    transition_quality.to_csv(paths["transition_quality"], index=False)
    filtered_comparison.to_csv(paths["filtered_signal_comparison"], index=False)
    with paths["day6_summary"].open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    return paths


def save_day8_outputs(
    similarity_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    propagation_df: pd.DataFrame,
    asset_influence_df: pd.DataFrame,
    sector_influence_df: pd.DataFrame,
    market_state_df: pd.DataFrame,
    market_vs_asset_df: pd.DataFrame,
    output_dir: str | Path = "output",
) -> dict[str, Path]:
    """Persist Day 8 cross-asset / market-state CSV artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "asset_similarity_matrix": out_dir / "asset_similarity_matrix.csv",
        "asset_clusters": out_dir / "asset_clusters.csv",
        "cluster_summary": out_dir / "cluster_summary.csv",
        "regime_propagation": out_dir / "regime_propagation.csv",
        "asset_influence_scores": out_dir / "asset_influence_scores.csv",
        "sector_influence_scores": out_dir / "sector_influence_scores.csv",
        "market_state": out_dir / "market_state.csv",
        "market_vs_asset_comparison": out_dir / "market_vs_asset_comparison.csv",
    }
    similarity_df.to_csv(paths["asset_similarity_matrix"], index=True)
    clusters_df.to_csv(paths["asset_clusters"], index=False)
    cluster_summary_df.to_csv(paths["cluster_summary"], index=False)
    propagation_df.to_csv(paths["regime_propagation"], index=False)
    asset_influence_df.to_csv(paths["asset_influence_scores"], index=False)
    sector_influence_df.to_csv(paths["sector_influence_scores"], index=False)
    market_state_df.to_csv(paths["market_state"], index=False)
    market_vs_asset_df.to_csv(paths["market_vs_asset_comparison"], index=False)
    return paths
