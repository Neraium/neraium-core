"""Day 12: argparse CLI for batch, realtime, health, and rebuild."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.alerts import (  # noqa: E402
    build_alert_record,
    build_operator_inbox,
    detect_alert_conditions,
    load_recent_alerts,
    log_alert_record,
    should_suppress_alert,
)
from neraium.decisions import (  # noqa: E402
    build_decision_memory,
    decision_quality_summary_table,
    evaluate_decision_quality,
)
from config import ASSETS  # noqa: E402
from neraium.connectors.registry import get_connector  # noqa: E402
from neraium.health import compute_system_health  # noqa: E402
from neraium.ingestion import ingest_market_data  # noqa: E402
from neraium.manifests import build_run_manifest, save_run_manifest  # noqa: E402
from neraium.market_state import compare_market_vs_asset_usefulness  # noqa: E402
from neraium.reporting import (  # noqa: E402
    save_day6_outputs,
    save_day7_outputs,
    save_day8_outputs,
    save_day9_outputs,
    save_validation_outputs,
)
from neraium.realtime import (  # noqa: E402
    run_full_pipeline,
    run_polling_loop,
    run_realtime_cycle,
    write_realtime_snapshot_json,
)
from neraium.review import (  # noqa: E402
    build_decision_review_table,
    identify_top_failures,
    identify_top_successes,
)
from neraium.runtime import (  # noqa: E402
    alert_log_path,
    atomic_write_json,
    ensure_output_layout,
    evidence_log_path,
    fallback_to_last_good_output,
    health_log_path,
    new_run_id,
    operator_inbox_path,
    pipeline_state_path,
    realtime_snapshot_path,
    run_resume_summary_path,
    run_status_path,
    safe_run_pipeline,
    save_run_status,
    build_run_status_payload,
)
from neraium.settings import Settings, load_settings, validate_settings, with_overrides  # noqa: E402

SUBCOMMANDS = frozenset(
    {
        "run-batch",
        "run-realtime",
        "run-day11-realtime",
        "health-check",
        "rebuild-outputs",
        "inspect-sources",
        "validate-source",
        "ingest-preview",
        "show-state",
        "clear-cache",
        "rebuild-from-snapshots",
        "run-incremental",
    }
)


def _parse_save_outputs(s: str | None) -> bool | None:
    if s is None:
        return None
    return s.strip().lower() in {"1", "true", "yes", "on"}


def _apply_common_flags(ns: argparse.Namespace, base: Settings) -> Settings:
    kw: dict[str, Any] = {}
    if getattr(ns, "output_dir", None):
        out = Path(ns.output_dir)
        root = base.project_root
        od = out if out.is_absolute() else (root / out).resolve()
        kw["output_dir"] = od
        kw["latest_dir"] = od / "latest"
        kw["runs_dir"] = od / "runs"
        kw["manifests_dir"] = od / "manifests"
        kw["alerts_dir"] = od / "alerts"
        kw["evidence_dir"] = od / "evidence"
        kw["reviews_dir"] = od / "reviews"
        kw["health_dir"] = od / "health"
        kw["state_dir"] = od / "state"
        kw["snapshot_dir"] = od / "snapshots"
        kw["cache_dir"] = od / "cache"
    if getattr(ns, "max_cycles", None) is not None:
        kw["max_cycles"] = int(ns.max_cycles)
    if getattr(ns, "interval_seconds", None) is not None:
        kw["polling_interval_seconds"] = int(ns.interval_seconds)
    so = _parse_save_outputs(getattr(ns, "save_outputs", None))
    if so is not None:
        kw["save_outputs"] = so
    st = getattr(ns, "source_type", None)
    if st:
        kw["source_type"] = st
    if not kw:
        return base
    return with_overrides(base, **kw)


def _latest_ts_ms(ms: pd.DataFrame) -> str | None:
    if ms is None or ms.empty or "timestamp" not in ms.columns:
        return None
    ts = ms["timestamp"].iloc[-1]
    if hasattr(ts, "isoformat"):
        return pd.Timestamp(ts).isoformat()
    return str(ts)


def _batch_inner(settings: Settings, save_outputs: bool, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Run full batch pipeline, optional artifact save; returns (pipeline_result, output_paths)."""
    from neraium.cache import cache_exists, make_cache_key, save_cache_artifact
    from neraium.incremental import build_resume_summary, detect_new_data, filter_to_incremental_window
    from neraium.snapshots import save_input_snapshot, save_output_snapshot
    from neraium.state_store import build_default_state, load_pipeline_state

    pipeline_path = pipeline_state_path(settings)
    prior = load_pipeline_state(pipeline_path)
    state = dict(prior) if prior else build_default_state(settings)
    prior_found = prior is not None

    assets = list(settings.active_assets) if settings.active_assets else None
    assets_list = list(ASSETS) if not assets else list(assets)
    data, ingestion_summary = ingest_market_data(assets_list, "daily", settings)

    new_info = detect_new_data(data, state)
    fingerprint = {
        a: str(pd.Timestamp(df["timestamp"].max()).isoformat())
        for a, df in data.items()
        if df is not None and not df.empty and "timestamp" in df.columns
    }
    merged_cache_key = make_cache_key("merged", settings.profile, "daily", fingerprint)
    cache_hit = settings.enable_cache and cache_exists(settings.cache_dir, merged_cache_key)

    if settings.enable_incremental and prior_found and not new_info["any_new_data"]:
        resume = build_resume_summary(
            prior_state_found=True,
            incremental_mode_used=True,
            new_data_summary=new_info,
            cache_hits=1 if cache_hit else 0,
            cache_misses=0 if cache_hit else 1,
            input_snapshot_written=False,
            output_snapshot_written=False,
            state_updated=False,
            skipped_full_pipeline=True,
            notes="No new data since last run",
        )
        atomic_write_json(run_resume_summary_path(settings), resume)
        return {
            "pipeline_success": True,
            "_day14_skip": True,
            "ingestion_summary": ingestion_summary,
            "new_data_detection": new_info,
            "resume_summary": resume,
            "run_id": run_id,
            "cache_key_merged": merged_cache_key,
        }, {}

    data_to_use = data
    incremental_used = bool(settings.enable_incremental and prior_found and new_info["any_new_data"])
    if incremental_used:
        data_to_use = filter_to_incremental_window(data, state, settings.incremental_lookback_buffer)

    result = run_full_pipeline(
        settings.project_root,
        settings=settings,
        preloaded_data=data_to_use,
        emit_evidence_log=True,
        data_dir=settings.data_dir,
        active_assets=assets,
        evidence_log_path=evidence_log_path(settings),
    )
    result["ingestion_summary"] = ingestion_summary
    result["_day14_timestamps"] = {
        a: str(pd.Timestamp(df["timestamp"].max()).isoformat())
        for a, df in data.items()
        if df is not None and not df.empty and "timestamp" in df.columns
    }
    result["_day14_incremental_used"] = incremental_used
    result["_day14_new_data"] = new_info
    result["_day14_cache_key"] = merged_cache_key
    result["_day14_run_id"] = run_id

    if not result["pipeline_success"]:
        raise RuntimeError("validation failed:\n" + "\n".join(result["validation_errors"]))

    daily = result["daily"]
    hourly = result["hourly"]
    intraday = result["intraday"]
    market_state = result["market_state"]
    decision_df = result["decision_df"]
    asset_panel = result["asset_panel"]
    similarity = result["similarity"]
    clusters = result["clusters"]
    cluster_summary = result["cluster_summary"]
    propagation = result["propagation"]
    asset_influence = result["asset_influence"]
    sector_influence = result["sector_influence"]
    scenario_path_summary = result["scenario_path_summary"]
    path_comparison = result["path_comparison"]
    by_scen = result["by_scen"]
    by_wl = result["by_wl"]
    day9_report = result["day9_report"]
    alignment_df = result["alignment_df"]
    alignment_report = result["alignment_report"]
    comparison = result["comparison"]

    decision_quality = evaluate_decision_quality(decision_df)
    decision_review = build_decision_review_table(decision_df)
    top_failures = identify_top_failures(decision_df, n=3)
    top_successes = identify_top_successes(decision_df, n=3)
    decision_memory = build_decision_memory(decision_df)
    decision_quality_df = decision_quality_summary_table(decision_quality)

    out_dir = settings.output_dir
    evidence_path = evidence_log_path(settings)
    market_vs_asset = compare_market_vs_asset_usefulness(asset_panel, market_state)

    summary = intraday["summary"]
    calibration = intraday["calibration"]
    baseline_comparison = intraday["baseline"]
    day6_report = intraday["day6_report"]
    filtered_comparison = intraday["filtered_comparison"]
    transition_matrix = intraday["transition_matrix"]

    print("Total signals:", summary["total_signals"])
    print("\nCounts by regime:")
    for k, v in summary["regime_counts"].items():
        print(f"  {k}: {v}")

    print("\nCounts by action posture:")
    for k, v in summary["action_counts"].items():
        print(f"  {k}: {v}")

    print("\nAverage usefulness by horizon (unfiltered):")
    for k, v in summary["usefulness_summary"].items():
        print(f"  {k}: {v:.4f}")

    monotonic = bool(calibration["monotonic_non_decreasing"].iloc[0]) if not calibration.empty else False
    print("\nHigher confidence improved usefulness (monotonic bin check):", monotonic)

    print("\nBaseline comparison summary (avg_usefulness):")
    pivot = baseline_comparison.pivot(index="model", columns="horizon", values="avg_usefulness")
    print(pivot.round(4).to_string())

    print("\n--- Day 6 reliability ---")
    arl = day6_report.get("average_regime_run_length", float("nan"))
    print(f"Average regime run length: {arl:.4f}")

    if not transition_matrix.empty:
        top_t = transition_matrix.sort_values("transition_count", ascending=False).head(5)
        print("\nMost common regime transitions (top 5):")
        for _, r in top_t.iterrows():
            print(
                f"  {r['from_regime']} -> {r['to_regime']}: "
                f"count={int(r['transition_count'])} "
                f"p={float(r['p_to_given_from']):.3f}"
            )

    lfp = day6_report.get("false_positive_flag_counts", {}).get("likely_false_positive_flag", 0)
    print(f"\nCount of likely false positives (flag): {lfp}")

    print("\nFiltered vs unfiltered usefulness:")
    print(filtered_comparison.round(4).to_string(index=False))

    improved_d6 = day6_report.get("filtering_improved_mean_usefulness", False)
    print(f"\nFiltering improved mean usefulness (1d/5d/10d avg): {improved_d6}")

    print("\n--- Day 7 multi-timeframe alignment ---")
    print("Total 15m aligned rows:", len(alignment_df))
    print("\nRegime alignment counts:")
    for k, v in alignment_report.get("regime_alignment_counts", {}).items():
        print(f"  {k}: {v}")

    print("\nAction alignment counts:")
    for k, v in alignment_report.get("action_alignment_counts", {}).items():
        print(f"  {k}: {v}")

    print(f"\nAverage adjusted confidence: {alignment_report['average_adjusted_confidence']:.4f}")
    print(f"Suppressed by alignment filter: {alignment_report['filter_suppression_count']}")

    print("\nAligned vs unaligned usefulness:")
    print(comparison.round(4).to_string(index=False))

    improved = False
    if len(comparison) >= 2:
        u = comparison.loc[comparison["version"] == "unaligned", ["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]].mean(axis=1)
        f = comparison.loc[comparison["version"] == "alignment_filtered", ["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]].mean(axis=1)
        improved = bool((f.iloc[0] if len(f) else 0.0) > (u.iloc[0] if len(u) else 0.0))
    print("\nMulti-timeframe alignment improved reliability:", improved)

    print("\n--- Day 8 market structure ---")
    print("\nAsset clusters (asset -> cluster_id):")
    print(clusters.to_string(index=False))

    print("\nTop influence assets:")
    if not asset_influence.empty:
        print(asset_influence.head(8).to_string(index=False))
    else:
        print("  (empty)")

    print("\nTop influence sectors:")
    if not sector_influence.empty:
        print(sector_influence.head(8).to_string(index=False))
    else:
        print("  (empty)")

    if "market_regime_label" in market_state.columns:
        print("\nMarket regime distribution:")
        print(market_state["market_regime_label"].value_counts().to_string())

    if "market_action_posture" in market_state.columns:
        print("\nMarket action posture distribution:")
        print(market_state["market_action_posture"].value_counts().to_string())

    print("\nMarket vs asset usefulness comparison:")
    print(market_vs_asset.round(4).to_string(index=False))

    print("\n--- Day 9 trajectory & path intelligence ---")
    if "trajectory_direction" in market_state.columns:
        print("\nTrajectory direction counts:")
        print(market_state["trajectory_direction"].value_counts().to_string())
    if "market_warning_level" in market_state.columns:
        print("\nMarket warning level:")
        print(market_state["market_warning_level"].value_counts().to_string())
    print("\nPath vs static usefulness (mean 1d/5d/10d):")
    print(path_comparison.round(4).to_string(index=False))
    print("\nPath-adjusted improved mean usefulness:", day9_report.get("path_adjusted_improved_mean_usefulness", False))

    print("\n--- Day 10 decision audit ---")
    print("Total decisions:", decision_quality["total_decisions"])
    print(
        "Avg usefulness (1d/5d/10d):",
        round(decision_quality["avg_usefulness_1d"], 4),
        round(decision_quality["avg_usefulness_5d"], 4),
        round(decision_quality["avg_usefulness_10d"], 4),
    )
    print("False positive rate (act/reduce & harmful 5d):", round(decision_quality["false_positive_rate"], 4))
    print("Abstention rate:", round(decision_quality["abstention_rate"], 4))
    print("\nTop 3 failures (high confidence, harmful 5d):")
    print(top_failures.to_string(index=False) if not top_failures.empty else "  (none)")
    print("\nTop 3 successes (high confidence, useful 5d):")
    print(top_successes.to_string(index=False) if not top_successes.empty else "  (none)")

    output_paths: dict[str, str] = {
        "evidence_log": str(evidence_path),
    }

    if save_outputs:
        base_paths = save_validation_outputs(
            signals_df=intraday["signals"],
            calibration_df=intraday["calibration"],
            baseline_df=intraday["baseline"],
            summary=intraday["summary"],
            output_dir=out_dir,
        )
        day6_paths = save_day6_outputs(
            persistence_summary=intraday["persistence"],
            transition_matrix=intraday["transition_matrix"],
            transition_quality=intraday["transition_quality"],
            filtered_comparison=intraday["filtered_comparison"],
            summary=intraday["day6_report"],
            output_dir=out_dir,
        )
        day7_paths = save_day7_outputs(
            alignment_df=alignment_df,
            comparison_df=comparison,
            summary=alignment_report,
            output_dir=out_dir,
        )
        day8_paths = save_day8_outputs(
            similarity_df=similarity,
            clusters_df=clusters,
            cluster_summary_df=cluster_summary,
            propagation_df=propagation,
            asset_influence_df=asset_influence,
            sector_influence_df=sector_influence,
            market_state_df=market_state,
            market_vs_asset_df=market_vs_asset,
            output_dir=out_dir,
        )
        day9_paths = save_day9_outputs(
            market_state_day9_df=market_state,
            scenario_summary_df=scenario_path_summary,
            path_comparison_df=path_comparison,
            summary=day9_report,
            output_dir=out_dir,
        )

        decision_df.to_csv(out_dir / "decision_records.csv", index=False)
        decision_review.to_csv(out_dir / "decision_review.csv", index=False)
        decision_quality_df.to_csv(out_dir / "decision_quality_summary.csv", index=False)
        top_failures.to_csv(out_dir / "top_failures.csv", index=False)
        top_successes.to_csv(out_dir / "top_successes.csv", index=False)
        decision_memory.to_csv(out_dir / "decision_memory.csv", index=False)
        day10_paths = {
            "decision_records": out_dir / "decision_records.csv",
            "decision_review": out_dir / "decision_review.csv",
            "decision_quality_summary": out_dir / "decision_quality_summary.csv",
            "top_failures": out_dir / "top_failures.csv",
            "top_successes": out_dir / "top_successes.csv",
            "decision_memory": out_dir / "decision_memory.csv",
        }

        print("\nSaved outputs:")
        for dct in (base_paths, day6_paths, day7_paths, day8_paths, day9_paths, day10_paths):
            for key, path in dct.items():
                sp = str(path)
                print(f"  {key}: {sp}")
                output_paths[key] = sp

    snap_in = False
    snap_out = False
    if settings.persist_input_snapshots:
        save_input_snapshot(data, settings.snapshot_dir, run_id, "daily")
        snap_in = True
    if settings.persist_output_snapshots:
        save_output_snapshot(
            {"market_state": result["market_state"], "day9_report": result["day9_report"]},
            settings.snapshot_dir,
            run_id,
        )
        snap_out = True
    if settings.enable_cache and "merged" in result:
        save_cache_artifact(result["merged"], settings.cache_dir, merged_cache_key)

    resume = build_resume_summary(
        prior_state_found=prior_found,
        incremental_mode_used=incremental_used,
        new_data_summary=new_info,
        cache_hits=1 if cache_hit else 0,
        cache_misses=0 if cache_hit else 1,
        input_snapshot_written=snap_in,
        output_snapshot_written=snap_out,
        state_updated=False,
        skipped_full_pipeline=False,
        notes=None,
    )
    result["resume_summary"] = resume
    atomic_write_json(run_resume_summary_path(settings), resume)

    return result, output_paths


def cmd_run_batch(ns: argparse.Namespace) -> int:
    base = load_settings(ns.profile)
    settings = _apply_common_flags(ns, base)
    if getattr(ns, "command", None) == "run-incremental":
        settings = with_overrides(settings, enable_incremental=True)
    validate_settings(settings)
    ensure_output_layout(settings)

    save_out = settings.save_outputs
    so = _parse_save_outputs(getattr(ns, "save_outputs", None))
    if so is not None:
        save_out = so

    run_id = new_run_id()

    def _work() -> tuple[dict[str, Any], dict[str, str]]:
        return _batch_inner(settings, save_outputs=save_out, run_id=run_id)

    status = safe_run_pipeline(_work, stage="batch", settings=settings)
    success = bool(status["success"])
    err = status["error_message"]
    inner = status.get("result")
    output_paths: dict[str, str] = {}
    pr: dict[str, Any] | None = None
    if isinstance(inner, tuple) and len(inner) == 2:
        pr, output_paths = inner
    elif inner is not None:
        pr = inner  # type: ignore[assignment]

    extra_meta: dict[str, Any] | None = None
    if isinstance(pr, dict) and pr.get("ingestion_summary"):
        ing = pr["ingestion_summary"]
        extra_meta = {
            "ingestion_summary": ing,
            "source_metadata": ing.get("source_metadata") if isinstance(ing, dict) else None,
            "source_type": settings.source_type,
        }
    if isinstance(pr, dict) and (pr.get("resume_summary") or pr.get("_day14_skip")):
        extra_meta = extra_meta or {}
        extra_meta["day14"] = {
            "resume_summary": pr.get("resume_summary"),
            "incremental_mode_used": pr.get("_day14_incremental_used"),
            "skipped_full_pipeline": pr.get("_day14_skip", False),
            "cache_key_merged": pr.get("_day14_cache_key") or pr.get("cache_key_merged"),
            "new_data_detection": pr.get("_day14_new_data") or pr.get("new_data_detection"),
        }

    manifest = build_run_manifest(
        settings,
        mode="batch",
        output_paths=output_paths,
        run_id=run_id,
        success=success,
        error_message=err if not success else None,
        extra_metadata=extra_meta,
    )
    mf_path = settings.manifests_dir / f"{run_id}.json"
    save_run_manifest(manifest, mf_path)

    if success and isinstance(pr, dict):
        from neraium.state_store import build_default_state, load_pipeline_state, save_pipeline_state

        st = load_pipeline_state(pipeline_state_path(settings)) or build_default_state(settings)
        st["last_successful_run_id"] = run_id
        if pr.get("_day14_timestamps"):
            st["latest_timestamp_by_asset"] = pr["_day14_timestamps"]
        st["latest_manifest_path"] = str(mf_path)
        st["source_type"] = settings.source_type
        if pr.get("cache_key_merged"):
            st["cache_key_meta"] = {"merged": pr.get("cache_key_merged")}
        save_pipeline_state(st, pipeline_state_path(settings))
        save_pipeline_state(st, settings.state_dir / "pipeline_state.json")

    health_st = "ok" if success else "degraded"
    latest_ts = _latest_ts_ms(pr["market_state"]) if isinstance(pr, dict) and pr.get("market_state") is not None else None

    day14_payload: dict[str, Any] | None = None
    if isinstance(pr, dict) and pr.get("resume_summary"):
        rs = pr["resume_summary"]
        day14_payload = {
            "incremental_mode_used": rs.get("incremental_mode_used"),
            "cache_hits": rs.get("cache_hits"),
            "cache_misses": rs.get("cache_misses"),
            "input_snapshot_written": rs.get("input_snapshot_written"),
            "output_snapshot_written": rs.get("output_snapshot_written"),
            "skipped_full_pipeline": rs.get("skipped_full_pipeline", False),
        }

    rsp = build_run_status_payload(
        run_id=run_id,
        mode="batch",
        success=success,
        settings=settings,
        manifest_path=str(mf_path),
        outputs_written=output_paths,
        error_message=err if not success else None,
        health_status=health_st,
        latest_timestamp_processed=latest_ts,
        alerts_fired=0,
        day14=day14_payload,
    )
    save_run_status(settings, rsp)

    if not success:
        print("Batch failed:", err, file=sys.stderr)
        fb = fallback_to_last_good_output(settings.output_dir)
        if fb.get("best_path"):
            print("Fallback reference:", fb["best_path"], file=sys.stderr)
        return 1
    if isinstance(pr, dict) and pr.get("_day14_skip"):
        print("Day 14: no new data — skipped full pipeline.")
    return 0


def _realtime_single_cycle(settings: Settings, *, day11_label: bool) -> int:
    ensure_output_layout(settings)
    snap_path = realtime_snapshot_path(settings)
    prior: dict | None = None
    if snap_path.is_file():
        try:
            prior = json.loads(snap_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = None

    assets = list(settings.active_assets) if settings.active_assets else None
    cyc = run_realtime_cycle(
        settings.project_root,
        settings=settings,
        emit_evidence_log=False,
        data_dir=settings.data_dir,
        active_assets=assets,
        evidence_log_path=evidence_log_path(settings),
    )
    snapshot = cyc.get("snapshot")
    ps = bool(cyc.get("pipeline_success"))

    run_id = new_run_id()
    output_paths: dict[str, str] = {
        "realtime_snapshot": str(snap_path),
    }

    if not ps or snapshot is None:
        print("Realtime cycle failed or empty snapshot.", file=sys.stderr)
        ms = cyc.get("market_state")
        if not isinstance(ms, pd.DataFrame):
            ms = pd.DataFrame()
        health = compute_system_health(ms, pipeline_success=False, recent_alert_count=0)
        log_health_snapshot(health, health_log_path(settings))
        manifest = build_run_manifest(
            settings,
            mode="realtime",
            output_paths=output_paths,
            run_id=run_id,
            success=False,
            error_message="pipeline_failed_or_empty_snapshot",
        )
        mf_path = settings.manifests_dir / f"{run_id}.json"
        save_run_manifest(manifest, mf_path)
        rsp = build_run_status_payload(
            run_id=run_id,
            mode="realtime",
            success=False,
            settings=settings,
            manifest_path=str(mf_path),
            outputs_written=output_paths,
            error_message="pipeline_failed_or_empty_snapshot",
            health_status=str(health.get("health_status", "degraded")),
            latest_timestamp_processed=None,
            alerts_fired=0,
        )
        save_run_status(settings, rsp)
        return 1

    a_path = alert_log_path(settings)
    info = detect_alert_conditions(snapshot, prior)
    recent = load_recent_alerts(a_path, limit=20)
    rec = build_alert_record(snapshot, info)
    fired = False
    if info.get("alert_triggered") and not should_suppress_alert(rec, recent):
        log_alert_record(rec, a_path)
        fired = True
        recent = load_recent_alerts(a_path, limit=20)

    health = compute_system_health(
        cyc["market_state"],
        pipeline_success=True,
        recent_alert_count=len(recent),
    )
    log_health_snapshot(health, health_log_path(settings))
    inbox = build_operator_inbox(cyc["market_state"], alert_log_path=a_path)
    inbox_path = operator_inbox_path(settings)
    inbox.to_csv(inbox_path, index=False)
    write_realtime_snapshot_json(snapshot, snap_path)

    output_paths["operator_inbox"] = str(inbox_path)
    output_paths["alert_log"] = str(a_path)
    output_paths["health_log"] = str(health_log_path(settings))

    manifest = build_run_manifest(
        settings,
        mode="realtime",
        output_paths=output_paths,
        run_id=run_id,
        success=True,
    )
    mf_path = settings.manifests_dir / f"{run_id}.json"
    save_run_manifest(manifest, mf_path)

    rsp = build_run_status_payload(
        run_id=run_id,
        mode="realtime",
        success=True,
        settings=settings,
        manifest_path=str(mf_path),
        outputs_written=output_paths,
        error_message=None,
        health_status=str(health.get("health_status", "ok")),
        latest_timestamp_processed=snapshot.get("timestamp"),
        alerts_fired=int(fired),
    )
    save_run_status(settings, rsp)

    title = "--- Day 11 realtime (advisory only; not execution) ---" if day11_label else "--- Realtime (advisory only) ---"
    print(title)
    print("Latest regime:", snapshot.get("market_regime_label"))
    print("Latest warning level:", snapshot.get("market_warning_level"))
    print("Path-adjusted action:", snapshot.get("path_adjusted_market_action"))
    print("Alert fired:", fired)
    if fired:
        print("Alert message:", rec.get("alert_message"))
    print("Health status:", health.get("health_status"))
    print("Outputs:", snap_path, inbox_path)
    return 0


def cmd_run_realtime(ns: argparse.Namespace) -> int:
    base = load_settings(ns.profile)
    settings = _apply_common_flags(ns, base)
    validate_settings(settings)
    ensure_output_layout(settings)

    interval = int(ns.interval_seconds) if getattr(ns, "interval_seconds", None) is not None else settings.polling_interval_seconds
    max_c = int(ns.max_cycles) if getattr(ns, "max_cycles", None) is not None else settings.max_cycles

    if max_c <= 1:
        return _realtime_single_cycle(settings, day11_label=False)

    assets = list(settings.active_assets) if settings.active_assets else None
    evp = evidence_log_path(settings)

    def _hook(prior: dict | None, cyc: dict[str, Any]) -> None:
        pass

    run_polling_loop(
        interval_seconds=interval,
        max_cycles=max_c,
        project_root=settings.project_root,
        settings=settings,
        emit_evidence_log=False,
        data_dir=settings.data_dir,
        active_assets=assets,
        evidence_log_path=evp,
        alert_hook=_hook,
    )
    run_id = new_run_id()
    snap = realtime_snapshot_path(settings)
    output_paths: dict[str, str] = {"realtime_snapshot": str(snap)} if snap.is_file() else {}
    manifest = build_run_manifest(
        settings,
        mode="realtime_poll",
        output_paths=output_paths,
        run_id=run_id,
        success=True,
    )
    mf_path = settings.manifests_dir / f"{run_id}.json"
    save_run_manifest(manifest, mf_path)
    rsp = build_run_status_payload(
        run_id=run_id,
        mode="realtime_poll",
        success=True,
        settings=settings,
        manifest_path=str(mf_path),
        outputs_written=output_paths,
        error_message=None,
        health_status="ok",
        latest_timestamp_processed=None,
        alerts_fired=0,
    )
    save_run_status(settings, rsp)
    print(f"Completed {max_c} polling cycles (interval={interval}s). Advisory only.")
    return 0


def cmd_run_day11_realtime(ns: argparse.Namespace) -> int:
    base = load_settings(ns.profile)
    settings = _apply_common_flags(ns, base)
    validate_settings(settings)
    max_c = getattr(ns, "max_cycles", None)
    if max_c is None:
        max_c = settings.max_cycles
    else:
        max_c = int(max_c)
    if max_c > 1:
        return cmd_run_realtime(ns)
    return _realtime_single_cycle(settings, day11_label=True)


def cmd_health_check(ns: argparse.Namespace) -> int:
    settings = _apply_common_flags(ns, load_settings(ns.profile))
    p = run_status_path(settings)
    if not p.is_file():
        print("No run_status.json at", p, file=sys.stderr)
        return 1
    print(p.read_text(encoding="utf-8"))
    return 0


def cmd_rebuild_outputs(ns: argparse.Namespace) -> int:
    ns2 = argparse.Namespace(**vars(ns))
    ns2.save_outputs = "true"
    return cmd_run_batch(ns2)


def cmd_inspect_sources(ns: argparse.Namespace) -> int:
    base = load_settings(ns.profile)
    settings = _apply_common_flags(ns, base)
    validate_settings(settings)
    conn = get_connector(settings.source_type, settings)
    payload = {
        "connector_name": conn.get_name(),
        "source_type": settings.source_type,
        "metadata": conn.get_source_metadata(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_validate_source(ns: argparse.Namespace) -> int:
    base = load_settings(ns.profile)
    settings = _apply_common_flags(ns, base)
    validate_settings(settings)
    assets = list(settings.active_assets) if settings.active_assets else list(ASSETS)
    _, summary = ingest_market_data(assets, "daily", settings)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_ingest_preview(ns: argparse.Namespace) -> int:
    base = load_settings(ns.profile)
    settings = _apply_common_flags(ns, base)
    validate_settings(settings)
    preview_assets = [a for a in ASSETS[:3]]
    data, summary = ingest_market_data(preview_assets, "daily", settings)
    print("ingested keys:", list(data.keys()))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_show_state(ns: argparse.Namespace) -> int:
    from neraium.state_store import load_pipeline_state

    settings = _apply_common_flags(ns, load_settings(ns.profile))
    validate_settings(settings)
    st = load_pipeline_state(pipeline_state_path(settings))
    print(json.dumps(st or {}, indent=2, default=str))
    return 0


def cmd_clear_cache(ns: argparse.Namespace) -> int:
    import shutil

    settings = _apply_common_flags(ns, load_settings(ns.profile))
    validate_settings(settings)
    cdir = settings.cache_dir
    if cdir.is_dir():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    print("Cache cleared:", cdir)
    return 0


def cmd_rebuild_from_snapshots(ns: argparse.Namespace) -> int:
    from neraium.snapshots import load_latest_snapshot

    settings = _apply_common_flags(ns, load_settings(ns.profile))
    validate_settings(settings)
    ms = load_latest_snapshot(str(settings.snapshot_dir), "market_state")
    if ms is None:
        print("No market_state snapshot found under", settings.snapshot_dir, file=sys.stderr)
        return 1
    print("Rows:", len(ms))
    print(ms.tail(5).to_string(index=False))
    print("(rebuild-from-snapshots: preview only; run-batch runs the full pipeline.)")
    return 0


def cmd_run_incremental(ns: argparse.Namespace) -> int:
    return cmd_run_batch(ns)


def _common_flag_parser() -> argparse.ArgumentParser:
    """Shared flags (must appear after the subcommand, e.g. ``run-batch --profile dev``)."""
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--profile", default="dev", choices=("dev", "test", "prod_local"), help="Configuration profile")
    c.add_argument("--output-dir", default=None, help="Override output base directory")
    c.add_argument("--max-cycles", type=int, default=None, help="Max realtime polling cycles")
    c.add_argument("--interval-seconds", type=int, default=None, help="Polling interval (realtime)")
    c.add_argument(
        "--save-outputs",
        default=None,
        metavar="BOOL",
        help="true/false: save batch CSV/JSON artifacts (batch / rebuild-outputs)",
    )
    c.add_argument(
        "--source-type",
        default=None,
        choices=("csv", "mock_api"),
        help="Override data source connector (Day 13)",
    )
    return c


def _build_parser() -> argparse.ArgumentParser:
    common = _common_flag_parser()
    p = argparse.ArgumentParser(
        prog="neraium",
        description="Neraium Markets CLI (read-only advisory; local deployment).",
    )
    sub = p.add_subparsers(dest="command", required=False)
    sub.add_parser("run-batch", parents=[common], help="Run full batch pipeline (Days 1–10 + evidence)")
    sub.add_parser("run-realtime", parents=[common], help="Realtime monitoring cycle(s); multi-cycle uses polling loop")
    sub.add_parser("run-day11-realtime", parents=[common], help="Single-cycle Day 11 realtime (same pipeline, explicit label)")
    sub.add_parser("health-check", parents=[common], help="Print latest run_status.json")
    sub.add_parser("rebuild-outputs", parents=[common], help="Run batch with artifact saving enabled")
    sub.add_parser("inspect-sources", parents=[common], help="Print active connector name and metadata (Day 13)")
    sub.add_parser("validate-source", parents=[common], help="Run ingestion quality summary for configured assets (Day 13)")
    sub.add_parser("ingest-preview", parents=[common], help="Ingest first three assets and print summary (Day 13)")
    sub.add_parser("show-state", parents=[common], help="Print pipeline state JSON (Day 14)")
    sub.add_parser("clear-cache", parents=[common], help="Remove cached intermediate artifacts (Day 14)")
    sub.add_parser(
        "rebuild-from-snapshots",
        parents=[common],
        help="Load latest market_state snapshot for inspection (Day 14)",
    )
    sub.add_parser("run-incremental", parents=[common], help="Run batch with incremental mode forced on (Day 14)")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m neraium.cli [run-batch] [flags]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["run-batch"] + argv
    ns = parser.parse_args(argv)
    cmd = ns.command or "run-batch"
    if cmd == "run-batch":
        return cmd_run_batch(ns)
    if cmd == "run-realtime":
        return cmd_run_realtime(ns)
    if cmd == "run-day11-realtime":
        return cmd_run_day11_realtime(ns)
    if cmd == "health-check":
        return cmd_health_check(ns)
    if cmd == "rebuild-outputs":
        return cmd_rebuild_outputs(ns)
    if cmd == "inspect-sources":
        return cmd_inspect_sources(ns)
    if cmd == "validate-source":
        return cmd_validate_source(ns)
    if cmd == "ingest-preview":
        return cmd_ingest_preview(ns)
    if cmd == "show-state":
        return cmd_show_state(ns)
    if cmd == "clear-cache":
        return cmd_clear_cache(ns)
    if cmd == "rebuild-from-snapshots":
        return cmd_rebuild_from_snapshots(ns)
    if cmd == "run-incremental":
        return cmd_run_incremental(ns)
    return 1


def legacy_main(argv: list[str] | None = None) -> int:
    """``main.py`` compatibility: ``--mode``, ``--realtime``, ``--save-output``, ``--interval``, ``--max-cycles``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = argparse.ArgumentParser(description="Neraium Markets (legacy main.py flags)")
    legacy.add_argument("--mode", choices=("batch", "realtime"), default="batch")
    legacy.add_argument("--realtime", action="store_true")
    legacy.add_argument("--interval", type=int, default=60)
    legacy.add_argument("--max-cycles", type=int, default=1)
    legacy.add_argument("--save-output", action="store_true")
    legacy.add_argument("--profile", default="dev", choices=("dev", "test", "prod_local"))
    legacy.add_argument("--output-dir", default=None)
    ns = legacy.parse_args(argv)

    if ns.realtime:
        ns.mode = "realtime"

    ns2 = argparse.Namespace(
        profile=ns.profile,
        output_dir=ns.output_dir,
        max_cycles=ns.max_cycles,
        interval_seconds=ns.interval,
        save_outputs="true" if ns.save_output else None,
        command="run-realtime" if ns.mode == "realtime" else "run-batch",
    )

    if ns.mode == "realtime":
        return cmd_run_realtime(ns2)
    return cmd_run_batch(ns2)


def dispatch_main() -> int:
    """Route ``python main.py`` argv: subcommands vs legacy flags."""
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        return main(argv)
    return legacy_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
