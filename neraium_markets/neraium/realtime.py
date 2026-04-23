"""Day 11: single-cycle and polling realtime monitoring over the Days 1–10 pipeline."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from config import ASSETS, ASSET_TO_SECTOR, DATE_COLUMN, DAY8_CLUSTER_ASSETS
from neraium.alignment import align_close_series
from neraium.alignment_filters import (
    apply_timeframe_alignment_filter,
    compare_alignment_filtered_vs_unfiltered,
)
from neraium.baselines import compare_to_baselines
from neraium.clustering import (
    build_asset_panel,
    cluster_assets,
    compute_asset_similarity_matrix,
    summarize_clusters,
)
from neraium.data_loader import load_all_assets
from neraium.ingestion import ingest_market_data
from neraium.settings import Settings
from neraium.decisions import attach_outcomes, generate_decision_record
from neraium.diagnostics import compute_signal_stability
from neraium.evaluation import compute_forward_returns, evaluate_confidence_calibration, score_action_usefulness
from neraium.evidence import log_batch_events
from neraium.features import build_feature_table
from neraium.filtering import apply_signal_filters, compare_filtered_vs_unfiltered, identify_false_positive_patterns
from neraium.market_state import (
    aggregate_panel_per_timestamp,
    attach_spy_forward_returns,
    attach_asset_forward_returns as panel_attach_forwards,
    generate_market_action_posture,
    generate_market_explanation,
    score_panel_action_usefulness,
    synthesize_market_state,
)
from neraium.propagation import (
    build_regime_propagation_table,
    compute_asset_influence_scores,
    compute_sector_influence_scores,
)
from neraium.reporting import (
    build_day6_reliability_report,
    build_day7_alignment_report,
    build_day9_report,
    build_validation_report,
)
from neraium.scenarios import label_scenario_paths, summarize_scenario_paths
from neraium.signals import generate_signals
from neraium.structural import build_structural_snapshot
from neraium.trajectories import compute_market_state_runs, compute_market_state_trajectory
from neraium.timeframe_alignment import (
    apply_timeframe_confidence_adjustment,
    build_timeframe_alignment_table,
    compute_action_agreement,
    compute_regime_agreement,
)
from neraium.transitions import (
    build_transition_matrix,
    compute_regime_runs,
    summarize_regime_persistence,
    summarize_transition_quality,
)
from neraium.validation import validate_all
from neraium.warnings import (
    adjust_market_action_for_path,
    compare_path_aware_vs_static_market_usefulness,
    compute_early_warning_flags,
    compute_path_persistence_score,
    compute_reversal_risk_score,
    generate_market_warning_level,
    generate_path_aware_market_explanation,
    path_aware_usefulness_breakdowns,
    refine_early_warnings_with_reversal,
)


def _expand_timeframe_from_daily(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "daily":
        return df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    if timeframe == "1h":
        steps, freq = 7, "60min"
    elif timeframe == "15m":
        steps, freq = 26, "15min"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    base = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    price_cols = [c for c in base.columns if c != "timestamp"]
    rows: list[dict[str, object]] = []

    for i in range(len(base)):
        ts = pd.Timestamp(base.loc[i, "timestamp"])
        intra_idx = pd.date_range(ts + pd.Timedelta(hours=9, minutes=30), periods=steps, freq=freq)

        nxt = base.iloc[i + 1] if i + 1 < len(base) else base.iloc[i]
        cur = base.iloc[i]

        for j, bar_ts in enumerate(intra_idx):
            alpha = float(j + 1) / float(steps)
            row: dict[str, object] = {"timestamp": bar_ts}
            for col in price_cols:
                start = float(cur[col])
                end = float(nxt[col])
                row[col] = start + alpha * (end - start)
            rows.append(row)

    return pd.DataFrame(rows).sort_values("timestamp", ascending=True).reset_index(drop=True)


def _run_pipeline_for_timeframe(merged_prices: pd.DataFrame, timeframe: str) -> dict[str, object]:
    price_df = _expand_timeframe_from_daily(merged_prices, timeframe)
    features = build_feature_table(price_df)
    structural = build_structural_snapshot(features)
    signals = generate_signals(structural)
    signals["timeframe"] = timeframe

    evaluated = compute_forward_returns(signals, price_col="spy", horizons=[1, 5, 10])
    evaluated = score_action_usefulness(evaluated)
    calibration = evaluate_confidence_calibration(evaluated)
    baseline_comparison = compare_to_baselines(evaluated)
    summary = build_validation_report(evaluated)

    with_runs = compute_regime_runs(evaluated)
    persistence_summary = summarize_regime_persistence(with_runs)
    transition_matrix = build_transition_matrix(with_runs)
    stable = compute_signal_stability(with_runs)
    transition_quality = summarize_transition_quality(stable)
    flagged = identify_false_positive_patterns(stable)
    filtered = apply_signal_filters(flagged)
    filtered = score_action_usefulness(
        filtered,
        action_col="filtered_action_posture",
        out_prefix="filtered_useful",
    )
    filtered_comparison = compare_filtered_vs_unfiltered(filtered)
    day6_report = build_day6_reliability_report(
        filtered,
        persistence_summary,
        transition_matrix,
        transition_quality,
        filtered_comparison,
    )

    return {
        "signals": filtered,
        "summary": summary,
        "calibration": calibration,
        "baseline": baseline_comparison,
        "persistence": persistence_summary,
        "transition_matrix": transition_matrix,
        "transition_quality": transition_quality,
        "filtered_comparison": filtered_comparison,
        "day6_report": day6_report,
        "structural": structural,
        "evaluated": evaluated,
    }


def run_full_pipeline(
    project_root: Path | None = None,
    *,
    settings: Settings | None = None,
    preloaded_data: dict[str, pd.DataFrame] | None = None,
    emit_evidence_log: bool = True,
    data_dir: Path | None = None,
    active_assets: list[str] | None = None,
    evidence_log_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute Days 1–10 (load → validate → … → decision records).

    When ``settings`` is provided (Day 13), data is loaded via the ingestion layer
    (connector → normalize → source quality), unless ``preloaded_data`` is set (Day 14
    incremental path). Otherwise legacy CSV loading is used.

    Returns a dict with ``pipeline_success``, ``market_state``, ``decision_df``, alignment frames,
    and intermediate artifacts. On validation failure, ``pipeline_success`` is False.
    """
    root = project_root or Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ingestion_summary: dict[str, Any] | None = None
    if settings is not None and preloaded_data is not None:
        data = {k.lower(): v.copy() for k, v in preloaded_data.items()}
        assets_list = list(settings.active_assets) if settings.active_assets else list(ASSETS)
        if len(data) != len(assets_list):
            return {
                "pipeline_success": False,
                "validation_errors": [
                    f"preloaded_data: expected {len(assets_list)} assets, got {len(data)}"
                ],
                "ingestion_summary": None,
                "market_state": pd.DataFrame(),
                "decision_df": pd.DataFrame(),
                "merged": pd.DataFrame(),
                "daily": {},
                "hourly": {},
                "intraday": {},
                "alignment_df": pd.DataFrame(),
            }
    elif settings is not None:
        assets_list = list(settings.active_assets) if settings.active_assets else list(ASSETS)
        data, ingestion_summary = ingest_market_data(assets_list, "daily", settings)
        if len(data) != len(assets_list):
            return {
                "pipeline_success": False,
                "validation_errors": [
                    "ingestion: incomplete or rejected assets "
                    f"(expected {len(assets_list)}, got {len(data)}; "
                    f"rejected={ingestion_summary.get('assets_rejected', [])})"
                ],
                "ingestion_summary": ingestion_summary,
                "market_state": pd.DataFrame(),
                "decision_df": pd.DataFrame(),
                "merged": pd.DataFrame(),
                "daily": {},
                "hourly": {},
                "intraday": {},
                "alignment_df": pd.DataFrame(),
            }
    else:
        data = load_all_assets(data_dir=data_dir, assets=active_assets)

    errors = validate_all(data)
    if errors:
        return {
            "pipeline_success": False,
            "validation_errors": errors,
            "ingestion_summary": ingestion_summary,
            "market_state": pd.DataFrame(),
            "decision_df": pd.DataFrame(),
            "merged": pd.DataFrame(),
            "daily": {},
            "hourly": {},
            "intraday": {},
            "alignment_df": pd.DataFrame(),
        }

    merged = align_close_series(data)

    daily = _run_pipeline_for_timeframe(merged, "daily")
    hourly = _run_pipeline_for_timeframe(merged, "1h")
    intraday = _run_pipeline_for_timeframe(merged, "15m")

    structural_i = intraday["structural"]
    evaluated_i = intraday["evaluated"]
    assets = list(dict.fromkeys(DAY8_CLUSTER_ASSETS))

    asset_panel = build_asset_panel(structural_i, assets)
    asset_panel = panel_attach_forwards(asset_panel, merged, assets)
    asset_panel = score_panel_action_usefulness(asset_panel)

    similarity = compute_asset_similarity_matrix(asset_panel, assets, structural_i)
    clusters = cluster_assets(similarity)
    cluster_summary = summarize_clusters(clusters, asset_panel)

    propagation = build_regime_propagation_table(asset_panel, assets)
    asset_influence = compute_asset_influence_scores(propagation)
    sector_influence = compute_sector_influence_scores(propagation, ASSET_TO_SECTOR)

    panel_agg = aggregate_panel_per_timestamp(asset_panel)
    wide_market = structural_i.merge(panel_agg, on=DATE_COLUMN, how="left")
    market_state = synthesize_market_state(wide_market, cluster_summary, asset_influence)
    market_state = generate_market_action_posture(market_state)
    market_state = generate_market_explanation(market_state)
    market_state = attach_spy_forward_returns(market_state, evaluated_i)

    market_state = compute_market_state_trajectory(market_state)
    market_state = compute_market_state_runs(market_state)
    market_state = label_scenario_paths(market_state)
    market_state = compute_path_persistence_score(market_state)
    market_state = compute_reversal_risk_score(market_state)
    market_state = compute_early_warning_flags(market_state)
    market_state = refine_early_warnings_with_reversal(market_state)
    market_state = generate_market_warning_level(market_state)
    market_state = adjust_market_action_for_path(market_state)
    market_state = generate_path_aware_market_explanation(market_state)

    scenario_path_summary = summarize_scenario_paths(market_state)
    path_comparison = compare_path_aware_vs_static_market_usefulness(market_state)
    by_scen, by_wl = path_aware_usefulness_breakdowns(market_state)
    day9_report = build_day9_report(
        market_state,
        scenario_path_summary,
        path_comparison,
        by_scen,
        by_wl,
    )

    decision_df = generate_decision_record(market_state.copy())
    decision_df = attach_outcomes(decision_df)

    ev_path = evidence_log_path or (root / "output" / "evidence_log.jsonl")
    if emit_evidence_log:
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        log_batch_events(decision_df, path=ev_path)

    alignment_df = build_timeframe_alignment_table(
        daily_df=daily["signals"],
        hourly_df=hourly["signals"],
        intraday_df=intraday["signals"],
    )
    alignment_df = compute_regime_agreement(alignment_df)
    alignment_df = compute_action_agreement(alignment_df)
    alignment_df = apply_timeframe_confidence_adjustment(alignment_df)
    alignment_df = apply_timeframe_alignment_filter(alignment_df)

    comparison = compare_alignment_filtered_vs_unfiltered(alignment_df)
    for _, row in comparison.iterrows():
        prefix = f"comparison_{row['version']}"
        for c in comparison.columns:
            if c != "version":
                alignment_df[f"{prefix}_{c}"] = row[c]
    alignment_df["comparison_version"] = "embedded"

    alignment_report = build_day7_alignment_report(alignment_df)

    return {
        "pipeline_success": True,
        "validation_errors": [],
        "merged": merged,
        "daily": daily,
        "hourly": hourly,
        "intraday": intraday,
        "market_state": market_state,
        "decision_df": decision_df,
        "asset_panel": asset_panel,
        "similarity": similarity,
        "clusters": clusters,
        "cluster_summary": cluster_summary,
        "propagation": propagation,
        "asset_influence": asset_influence,
        "sector_influence": sector_influence,
        "scenario_path_summary": scenario_path_summary,
        "path_comparison": path_comparison,
        "by_scen": by_scen,
        "by_wl": by_wl,
        "day9_report": day9_report,
        "alignment_df": alignment_df,
        "alignment_report": alignment_report,
        "comparison": comparison,
        "project_root": root,
        "ingestion_summary": ingestion_summary,
    }


def _row_to_snapshot(row: pd.Series) -> dict[str, Any]:
    ts = row[DATE_COLUMN]
    if hasattr(ts, "isoformat"):
        ts_out = pd.Timestamp(ts).isoformat()
    else:
        ts_out = str(ts)
    return {
        "timestamp": ts_out,
        "market_regime_label": str(row.get("market_regime_label", "")),
        "market_confidence_score": float(row.get("market_confidence_score", float("nan"))),
        "market_warning_level": str(row.get("market_warning_level", "")),
        "path_adjusted_market_action": str(row.get("path_adjusted_market_action", "")),
        "market_explanation": str(row.get("market_explanation", "")),
        "scenario_path_label": str(row.get("scenario_path_label", "")),
    }


def run_realtime_cycle(
    project_root: Path | None = None,
    *,
    settings: Settings | None = None,
    emit_evidence_log: bool = False,
    data_dir: Path | None = None,
    active_assets: list[str] | None = None,
    evidence_log_path: Path | None = None,
) -> dict[str, Any]:
    """
    One monitoring cycle: full pipeline + latest-row snapshot dict.

    Returns keys including ``snapshot`` (structured dict), ``pipeline_success``, ``market_state``,
    and the full ``run_full_pipeline`` payload.
    """
    result = run_full_pipeline(
        project_root,
        settings=settings,
        emit_evidence_log=emit_evidence_log,
        data_dir=data_dir,
        active_assets=active_assets,
        evidence_log_path=evidence_log_path,
    )
    if not result["pipeline_success"]:
        return {**result, "snapshot": None, "cycle_log": "validation_failed"}

    ms = result["market_state"]
    if ms.empty:
        return {**result, "snapshot": None, "cycle_log": "empty_market_state"}

    last = ms.iloc[-1]
    snapshot = _row_to_snapshot(last)
    return {**result, "snapshot": snapshot, "cycle_log": "ok"}


def run_polling_loop(
    interval_seconds: int = 60,
    max_cycles: int = 1,
    project_root: Path | None = None,
    *,
    settings: Settings | None = None,
    emit_evidence_log: bool = False,
    data_dir: Path | None = None,
    active_assets: list[str] | None = None,
    evidence_log_path: Path | None = None,
    alert_hook: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Repeated ``run_realtime_cycle`` calls with optional alert hook
    ``alert_hook(prior_snapshot, current_result) -> None``.

    Sleeps ``interval_seconds`` between cycles (no sleep before the first cycle).
    """
    prior_snap: dict[str, Any] | None = None
    logs: list[dict[str, Any]] = []
    for i in range(max_cycles):
        if i > 0:
            time.sleep(interval_seconds)
        cyc = run_realtime_cycle(
            project_root,
            settings=settings,
            emit_evidence_log=emit_evidence_log,
            data_dir=data_dir,
            active_assets=active_assets,
            evidence_log_path=evidence_log_path,
        )
        snap = cyc.get("snapshot")
        logs.append({"cycle_index": i, "snapshot": snap, "pipeline_success": cyc.get("pipeline_success")})
        if alert_hook is not None and snap is not None:
            alert_hook(prior_snap, cyc)
        if isinstance(snap, dict):
            prior_snap = snap
    return logs


def write_realtime_snapshot_json(snapshot: dict[str, Any], output_path: str | Path) -> Path:
    """Write a single realtime snapshot JSON file (overwrite)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True, default=str)
    return p
