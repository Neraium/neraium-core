"""
CMAPSS FD replay: run NASA FD00x test matrices through Neraium StructuralEngine.

Output tiers (see --output-mode):
  - slim: product-facing columns only
  - diagnostics: wide flattening (legacy-style) for validation and research
  - both: write slim and diagnostics CSVs (default)

Examples:
  python run_fd_test.py --input /path/test_FD004.txt --output /tmp/fd004_results.csv
  python run_fd_test.py --input /path/test_FD004.txt --output-mode slim --slim-output /tmp/fd004_slim.csv
  python run_fd_test.py --input /path/test_FD004.txt --output-mode both --debug-jsonl /tmp/fd004_debug.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from neraium_core.fd004_transition import (
    detect_structural_transition,
    plot_transition_histograms,
    plot_unit_transition,
    save_transition_artifacts,
    summarize_fd004_transitions,
)
from run_engine import StructuralEngine


CMAPSS_COLUMNS = ["unit", "cycle"] + ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]


@dataclass(frozen=True)
class ResolvedReplayOutputs:
    """Paths for slim and diagnostics CSVs (None if that tier is not requested)."""

    slim: Path | None
    diagnostics: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a CMAPSS FD00x dataset through Neraium StructuralEngine and write CSV results."
    )
    parser.add_argument("--input", type=Path, default=Path("test_FD001.txt"), help="Path to CMAPSS dataset text file.")
    parser.add_argument("--output", type=Path, default=Path("fd_results.csv"), help="Base path for replay CSV outputs (stem used for default slim/diagnostics names).")
    parser.add_argument(
        "--output-mode",
        choices=["slim", "diagnostics", "both"],
        default="both",
        help="Which replay CSV tiers to write (default: both).",
    )
    parser.add_argument("--slim-output", type=Path, default=None, help="Explicit path for slim product CSV.")
    parser.add_argument("--diagnostics-output", type=Path, default=None, help="Explicit path for full diagnostics CSV.")
    parser.add_argument("--debug-jsonl", type=Path, default=None, help="Optional JSONL path: one object per row with full engine result plus unit/cycle metadata.")
    parser.add_argument("--max-units", type=int, default=None, help="Optional cap on number of units to replay.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional cap on cycles per unit.")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Only write the replay CSV; skip transition summary and plots.",
    )
    parser.add_argument(
        "--summary-only",
        type=Path,
        default=None,
        metavar="CSV",
        help="Skip replay; load this existing replay CSV and only run transition summary + optional plots.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Path for per-unit transition summary CSV (default: <output stem>_transition_summary.csv).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path for JSON aggregate metrics (default: next to summary CSV).",
    )
    parser.add_argument("--min-cycle", type=int, default=30, help="Ignore cycles before this (warmup guard) for transition detection.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Normalized pressure threshold for sustained crossing.")
    parser.add_argument("--window", type=int, default=5, help="Rolling mean window for pressure smoothing.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, save transition histograms and single-unit plot PNGs here.",
    )
    parser.add_argument("--plot-unit", type=int, default=1, help="Unit id for the single-unit transition plot.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose replay logs (first JSON result snippet). Set NERAIUM_DEBUG_ENGINE=1 for engine internals.",
    )
    parser.add_argument(
        "--calibration-profile",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "early_warning"],
        help="StructuralEngine calibration preset (default: balanced).",
    )
    parser.add_argument(
        "--fleet-summary-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional path for fleet priority summary JSON (per-asset ranks from slim replay rows).",
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace, base_output: Path) -> ResolvedReplayOutputs:
    """Derive slim/diagnostics paths from --output stem and explicit overrides."""
    parent = base_output.parent
    stem = base_output.stem
    default_slim = parent / f"{stem}_slim.csv"
    default_diag = parent / f"{stem}_diagnostics.csv"

    slim_explicit = args.slim_output.expanduser().resolve() if args.slim_output else None
    diag_explicit = args.diagnostics_output.expanduser().resolve() if args.diagnostics_output else None

    mode = args.output_mode
    slim_path: Path | None = None
    diag_path: Path | None = None

    if mode in ("slim", "both"):
        slim_path = slim_explicit or default_slim
    if mode in ("diagnostics", "both"):
        diag_path = diag_explicit or default_diag

    return ResolvedReplayOutputs(slim=slim_path, diagnostics=diag_path)


def load_dataset(path: Path) -> pd.DataFrame:
    input_path = path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    df = pd.read_csv(input_path, sep=r"\s+", header=None)
    expected_columns = len(CMAPSS_COLUMNS)
    if df.shape[1] != expected_columns:
        raise ValueError(f"Unexpected CMAPSS column count: got {df.shape[1]}, expected {expected_columns}")

    df.columns = CMAPSS_COLUMNS
    return df


def _safe_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _safe_list_item(items: Any, index: int) -> Any:
    if not isinstance(items, list):
        return None
    if index < 0 or index >= len(items):
        return None
    return items[index]


def flatten_slim_result(unit_id: int, cycle: int, result: dict[str, Any]) -> dict[str, Any]:
    """Product-facing columns for FD replay CSV (no research-only flattening)."""
    experimental = _safe_get(result, "experimental_analytics") or {}
    early_warning = _safe_get(result, "early_warning") or {}

    return {
        "unit": unit_id,
        "cycle": cycle,
        "state": _safe_get(result, "state"),
        "structural_drift_score": _safe_get(result, "structural_drift_score"),
        "latest_instability": _safe_get(result, "latest_instability"),
        "transition_pressure": _safe_get(result, "transition_pressure"),
        "transition_state": _safe_get(result, "transition_state"),
        "transition_outputs_actionable": _safe_get(result, "transition_outputs_actionable"),
        "engine_stabilization_progress": _safe_get(result, "engine_stabilization_progress"),
        "readiness_reason": _safe_get(result, "readiness", "reason"),
        "confidence_score": _safe_get(result, "confidence_score"),
        "interpreted_state": _safe_get(result, "interpreted_state"),
        "regime_name": _safe_get(result, "regime_name"),
        "early_warning_state": _safe_get(early_warning, "early_warning_state"),
        "early_warning_pre_instability_score": _safe_get(early_warning, "pre_instability_score"),
        "horizon_analysis_risk_horizon": _safe_get(experimental, "horizon_analysis", "risk_horizon"),
        "counterfactual_baseline_projected_horizon": _safe_get(
            experimental, "counterfactual_simulation", "baseline_future", "projected_horizon"
        ),
        "predicted_impact": _safe_get(result, "predicted_impact"),
        "structural_driver": _safe_get(result, "structural_driver"),
        "decision_severity": _safe_get(result, "decision_layer", "severity"),
        "decision_trajectory": _safe_get(result, "decision_layer", "trajectory"),
        "decision_time_horizon": _safe_get(result, "decision_layer", "time_horizon"),
        "decision_confidence": _safe_get(result, "decision_layer", "confidence"),
        "decision_actionability": _safe_get(result, "decision_layer", "actionability"),
        "recommended_next_action": _safe_get(result, "decision_layer", "recommended_next_action"),
        "recommended_urgency": _safe_get(result, "decision_layer", "recommended_urgency"),
        "inspection_priority": _safe_get(result, "decision_layer", "inspection_priority"),
        "trust_score": _safe_get(result, "trust_layer", "trust_score"),
        "safe_to_act": _safe_get(result, "trust_layer", "safe_to_act"),
        "operational_status": _safe_get(result, "operational_recommendation", "status"),
        "operational_recommendation_confidence": _safe_get(result, "operational_recommendation", "recommendation_confidence"),
    }


def flatten_diagnostics_result(
    unit_id: int,
    cycle: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Wide flattening of engine payload for diagnostics and validation (legacy-style)."""
    experimental = _safe_get(result, "experimental_analytics") or {}
    geometry = _safe_get(result, "geometry") or {}
    state_space_statistics = _safe_get(result, "state_space_statistics") or {}
    state_graph = _safe_get(result, "state_graph") or {}
    counterfactuals = _safe_get(experimental, "counterfactual_simulation", "counterfactuals")
    pressure_relief = _safe_list_item(counterfactuals, 0)
    continued_degradation = _safe_list_item(counterfactuals, 2)
    early_warning = _safe_get(result, "early_warning") or {}
    signal_degradation = _safe_get(result, "signal_degradation") or {}

    return {
        "unit": unit_id,
        "cycle": cycle,
        "state": _safe_get(result, "state"),
        "structural_drift_score": _safe_get(result, "structural_drift_score"),
        "latest_instability": _safe_get(result, "latest_instability"),
        "transition_pressure": _safe_get(result, "transition_pressure"),
        "transition_state": _safe_get(result, "transition_state"),
        "transition_outputs_actionable": _safe_get(result, "transition_outputs_actionable"),
        "engine_stabilization_progress": _safe_get(result, "engine_stabilization_progress"),
        "readiness_reason": _safe_get(result, "readiness", "reason"),
        "confidence_score": _safe_get(result, "confidence_score"),
        "interpreted_state": _safe_get(result, "interpreted_state"),
        "regime_name": _safe_get(result, "regime_name"),
        "predicted_impact": _safe_get(result, "predicted_impact"),
        "structural_driver": _safe_get(result, "structural_driver"),
        "trajectory_analysis_dominant_path": _safe_get(experimental, "trajectory_analysis", "dominant_path"),
        "trajectory_analysis_path_confidence": _safe_get(experimental, "trajectory_analysis", "path_confidence"),
        "branching_analysis_is_branching": _safe_get(experimental, "branching_analysis", "is_branching"),
        "branching_analysis_commitment": _safe_get(experimental, "branching_analysis", "commitment"),
        "branching_analysis_decision_tension": _safe_get(experimental, "branching_analysis", "decision_tension"),
        "branching_analysis_branch_count_estimate": _safe_get(experimental, "branching_analysis", "branch_count_estimate"),
        "constraint_analysis_lock_in_score": _safe_get(experimental, "constraint_analysis", "lock_in_score"),
        "constraint_analysis_point_of_no_return_risk": _safe_get(experimental, "constraint_analysis", "point_of_no_return_risk"),
        "constraint_analysis_recovery_margin": _safe_get(experimental, "constraint_analysis", "recovery_margin"),
        "early_warning_state": _safe_get(early_warning, "early_warning_state"),
        "early_warning_pre_instability_score": _safe_get(early_warning, "pre_instability_score"),
        "early_warning_stability_erosion_score": _safe_get(early_warning, "stability_erosion_score"),
        "early_warning_coherence_breakdown_score": _safe_get(early_warning, "coherence_breakdown_score"),
        "early_warning_structural_strain_score": _safe_get(early_warning, "structural_strain_score"),
        "early_warning_pre_commitment_score": _safe_get(early_warning, "pre_commitment_score"),
        "signal_degradation_signal_instability_score": _safe_get(signal_degradation, "signal_instability_score"),
        "signal_degradation_shape_change_score": _safe_get(signal_degradation, "shape_change_score"),
        "signal_degradation_spectral_shift_score": _safe_get(signal_degradation, "spectral_shift_score"),
        "signal_degradation_volatility_erosion_score": _safe_get(signal_degradation, "volatility_erosion_score"),
        "signal_degradation_coherence_loss_score": _safe_get(signal_degradation, "coherence_loss_score"),
        "signal_degradation_state": _safe_get(signal_degradation, "signal_degradation_state"),
        "geometry_path_length": _safe_get(geometry, "path_length"),
        "geometry_local_velocity_norm": _safe_get(geometry, "local_velocity_norm"),
        "geometry_local_acceleration_norm": _safe_get(geometry, "local_acceleration_norm"),
        "geometry_curvature": _safe_get(geometry, "curvature"),
        "geometry_directional_consistency": _safe_get(geometry, "directional_consistency"),
        "geometry_angular_change": _safe_get(geometry, "angular_change"),
        "geometry_path_smoothness": _safe_get(geometry, "path_smoothness"),
        "state_space_statistics_local_volume": _safe_get(state_space_statistics, "local_volume"),
        "state_space_statistics_local_density": _safe_get(state_space_statistics, "local_density"),
        "state_space_statistics_covariance_trace": _safe_get(state_space_statistics, "covariance_trace"),
        "state_space_statistics_principal_direction_strength": _safe_get(state_space_statistics, "principal_direction_strength"),
        "state_space_statistics_anisotropy": _safe_get(state_space_statistics, "anisotropy"),
        "state_space_statistics_state_contraction_score": _safe_get(state_space_statistics, "state_contraction_score"),
        "state_space_statistics_state_expansion_score": _safe_get(state_space_statistics, "state_expansion_score"),
        "state_graph_node_count": _safe_get(state_graph, "node_count"),
        "state_graph_edge_count": _safe_get(state_graph, "edge_count"),
        "state_graph_branching_factor": _safe_get(state_graph, "branching_factor"),
        "state_graph_transition_entropy": _safe_get(state_graph, "transition_entropy"),
        "state_graph_revisit_rate": _safe_get(state_graph, "revisit_rate"),
        "state_graph_path_commitment_score": _safe_get(state_graph, "path_commitment_score"),
        "state_graph_graph_divergence_score": _safe_get(state_graph, "graph_divergence_score"),
        "temporal_consistency_score": _safe_get(result, "temporal_consistency_score"),
        "ordering_stability_score": _safe_get(result, "ordering_stability_score"),
        "timestamp_gap_irregularity": _safe_get(result, "timestamp_gap_irregularity"),
        "alignment_confidence": _safe_get(result, "alignment_confidence"),
        "effective_sampling_density": _safe_get(result, "effective_sampling_density"),
        "hierarchy_analysis_origin_scope": _safe_get(experimental, "hierarchy_analysis", "origin_scope"),
        "hierarchy_analysis_propagation_risk": _safe_get(experimental, "hierarchy_analysis", "propagation_risk"),
        "horizon_analysis_risk_horizon": _safe_get(experimental, "horizon_analysis", "risk_horizon"),
        "counterfactual_baseline_projected_path": _safe_get(
            experimental, "counterfactual_simulation", "baseline_future", "projected_path"
        ),
        "counterfactual_baseline_projected_lock_in_risk": _safe_get(
            experimental, "counterfactual_simulation", "baseline_future", "projected_lock_in_risk"
        ),
        "counterfactual_baseline_projected_horizon": _safe_get(
            experimental, "counterfactual_simulation", "baseline_future", "projected_horizon"
        ),
        "counterfactual_pressure_relief_projected_path": _safe_get(pressure_relief or {}, "projected_path"),
        "counterfactual_continued_degradation_projected_path": _safe_get(
            continued_degradation or {}, "projected_path"
        ),
        "decision_layer_json": json.dumps(_safe_get(result, "decision_layer") or {}, default=str),
        "attribution_json": json.dumps(_safe_get(result, "attribution") or {}, default=str),
        "operational_recommendation_json": json.dumps(_safe_get(result, "operational_recommendation") or {}, default=str),
        "trust_layer_json": json.dumps(_safe_get(result, "trust_layer") or {}, default=str),
    }


def write_debug_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")


def replay_unit(
    unit_df: pd.DataFrame,
    *,
    max_cycles: int | None = None,
    verbose: bool = False,
    debug_records: list[dict[str, Any]] | None = None,
    collect_diagnostics: bool = True,
    collect_slim: bool = True,
    calibration_profile: str = "balanced",
    calibration: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = StructuralEngine(calibration_profile=calibration_profile, calibration=calibration)
    diag_rows: list[dict[str, Any]] = []
    slim_rows: list[dict[str, Any]] = []
    printed_debug_result = False

    ordered = unit_df.sort_values("cycle")
    if max_cycles is not None and max_cycles > 0:
        ordered = ordered.head(int(max_cycles))
    unit_id = int(ordered["unit"].iloc[0])

    for _, row in ordered.iterrows():
        cycle = int(row["cycle"])
        sensor_values = {sensor: row[sensor] for sensor in SENSOR_COLUMNS}

        frame = {
            "timestamp": str(cycle),
            "site_id": "cmapss",
            "asset_id": f"unit_{unit_id}",
            "sensor_values": sensor_values,
        }
        result = engine.process_frame(frame)
        if verbose and not printed_debug_result:
            print(json.dumps(result, indent=2)[:1000])
            printed_debug_result = True

        if collect_diagnostics:
            diag_rows.append(flatten_diagnostics_result(unit_id=unit_id, cycle=cycle, result=result))
        if collect_slim:
            slim_rows.append(flatten_slim_result(unit_id=unit_id, cycle=cycle, result=result))

        if debug_records is not None:
            debug_records.append(
                {
                    "unit": unit_id,
                    "cycle": cycle,
                    "result": result,
                }
            )

    if verbose:
        print(
            f"unit={unit_id} replay_rows={len(ordered)} diag_rows={len(diag_rows)} slim_rows={len(slim_rows)} "
            f"last_cycle={ordered['cycle'].iloc[-1] if len(ordered) else 'n/a'}"
        )

    return diag_rows, slim_rows


def run_transition_postprocess(
    replay_df: pd.DataFrame,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Per-unit transition detection, CSV/JSON summary, optional plots."""
    summary_df, json_summary = summarize_fd004_transitions(
        replay_df,
        min_cycle=int(args.min_cycle),
        window=int(args.window),
        threshold=float(args.threshold),
        pressure_col="transition_pressure",
        verbose=bool(args.verbose),
    )
    csv_out = args.summary_output or (output_path.parent / f"{output_path.stem}_transition_summary.csv")
    json_out = args.summary_json or csv_out.with_suffix(".json")
    save_transition_artifacts(summary_df, json_summary, csv_path=csv_out, json_path=json_out)
    print("Transition summary:")
    print(f"  units_evaluated={json_summary.get('units_evaluated')}  with_detection={json_summary.get('units_with_detection')}  without_detection={json_summary.get('units_without_detection')}")
    print(f"  mean_remaining_life_normalized={json_summary.get('mean_remaining_life_normalized')}")
    print(f"  mean_lead_time_cycles={json_summary.get('mean_lead_time_cycles')}")
    print(f"  wrote {csv_out}")
    print(f"  wrote {json_out}")

    if args.plot_dir is not None:
        plot_dir = args.plot_dir.expanduser().resolve()
        plot_transition_histograms(summary_df, output_dir=plot_dir, show=False)
        pu = int(args.plot_unit)
        g = replay_df[replay_df["unit"] == pu]
        if len(g):
            det = detect_structural_transition(
                g,
                min_cycle=int(args.min_cycle),
                window=int(args.window),
                threshold=float(args.threshold),
                verbose=bool(args.verbose),
            )
            plot_unit_transition(
                g,
                det,
                unit_id=pu,
                output_path=plot_dir / f"fd004_unit_{pu}_transition.png",
                min_cycle=int(args.min_cycle),
                show=False,
            )
        else:
            print(f"  plot skipped: unit {pu} not in replay data")


def transition_dataframe_for_postprocess(
    args: argparse.Namespace,
    all_diag: list[dict[str, Any]],
    all_slim: list[dict[str, Any]],
) -> pd.DataFrame:
    """Prefer diagnostics rows for transition analysis; fall back to slim."""
    if args.output_mode in ("both", "diagnostics") and all_diag:
        return pd.DataFrame(all_diag)
    return pd.DataFrame(all_slim)


def main() -> None:
    args = parse_args()
    base_output = args.output.expanduser().resolve()

    if args.summary_only is not None:
        csv_in = args.summary_only.expanduser().resolve()
        if not csv_in.is_file():
            raise FileNotFoundError(f"--summary-only file not found: {csv_in}")
        replay_df = pd.read_csv(csv_in)
        run_transition_postprocess(replay_df, args, output_path=base_output)
        return

    input_path = args.input.expanduser().resolve()
    dataset = load_dataset(input_path)

    resolved = resolve_output_paths(args, base_output)
    debug_records: list[dict[str, Any]] | None = [] if args.debug_jsonl else None

    all_diag: list[dict[str, Any]] = []
    all_slim: list[dict[str, Any]] = []
    unit_ids = sorted(dataset["unit"].unique())
    if args.max_units is not None and args.max_units > 0:
        unit_ids = unit_ids[: int(args.max_units)]

    collect_diagnostics = args.output_mode in ("both", "diagnostics")
    collect_slim = args.output_mode in ("both", "slim")

    for unit_id in unit_ids:
        unit_df = dataset[dataset["unit"] == unit_id]
        drows, srows = replay_unit(
            unit_df,
            max_cycles=args.max_cycles,
            verbose=bool(args.verbose),
            debug_records=debug_records,
            collect_diagnostics=collect_diagnostics,
            collect_slim=collect_slim,
            calibration_profile=str(args.calibration_profile),
        )
        all_diag.extend(drows)
        all_slim.extend(srows)

    if resolved.slim:
        resolved.slim.parent.mkdir(parents=True, exist_ok=True)
    if resolved.diagnostics:
        resolved.diagnostics.parent.mkdir(parents=True, exist_ok=True)

    if args.output_mode in ("both", "slim") and resolved.slim is not None:
        pd.DataFrame(all_slim).to_csv(resolved.slim, index=False)
    if args.output_mode in ("both", "diagnostics") and resolved.diagnostics is not None:
        pd.DataFrame(all_diag).to_csv(resolved.diagnostics, index=False)

    if debug_records is not None and args.debug_jsonl:
        dbg_path = args.debug_jsonl.expanduser().resolve()
        write_debug_jsonl(dbg_path, debug_records)
        print(f"Debug JSONL: {dbg_path}  rows={len(debug_records)}")

    if args.fleet_summary_json is not None and all_slim:
        fs_path = args.fleet_summary_json.expanduser().resolve()
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fleet_payload = build_fleet_summary(pd.DataFrame(all_slim))
        fs_path.write_text(json.dumps(fleet_payload, indent=2), encoding="utf-8")
        print(f"Fleet summary JSON: {fs_path}")

    print("CMAPSS replay complete")
    print(f"Input path: {input_path}")
    print(f"Base output: {base_output}")
    if resolved.slim and args.output_mode in ("both", "slim"):
        print(f"Slim CSV: {resolved.slim}")
    if resolved.diagnostics and args.output_mode in ("both", "diagnostics"):
        print(f"Diagnostics CSV: {resolved.diagnostics}")
    print(f"Units processed: {len(unit_ids)}")
    print(f"Rows per tier: slim={len(all_slim)} diagnostics={len(all_diag)}")

    if not args.replay_only:
        tdf = transition_dataframe_for_postprocess(args, all_diag, all_slim)
        run_transition_postprocess(tdf, args, output_path=base_output)


if __name__ == "__main__":
    main()
