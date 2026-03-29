import argparse
import json
import time
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
from neraium_core.integrations.aux_time_client import AUX_TIMEClient, unavailable_payload
from run_engine import StructuralEngine


CMAPSS_COLUMNS = ["unit", "cycle"] + ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a CMAPSS FD00x dataset through Neraium StructuralEngine and write CSV results."
    )
    parser.add_argument("--input", type=Path, default=Path("test_FD001.txt"), help="Path to CMAPSS dataset text file.")
    parser.add_argument("--output", type=Path, default=Path("fd_results.csv"), help="Path to output CSV file.")
    parser.add_argument("--use-aux_time", action="store_true", help="Use AUX-TIME aligned time as an optional timestamp source.")
    parser.add_argument("--aux_time-cache-ms", type=int, default=500, help="Optional AUX-TIME cache duration in milliseconds.")
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
        help="Verbose replay logs (first JSON row, AUX_TIME debug). Set NERAIUM_DEBUG_ENGINE=1 for engine internals.",
    )
    return parser.parse_args()


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


def flatten_result(
    unit_id: int,
    cycle: int,
    result: dict[str, Any],
    aux_time: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experimental = _safe_get(result, "experimental_analytics") or {}
    geometry = _safe_get(result, "geometry") or {}
    state_space_statistics = _safe_get(result, "state_space_statistics") or {}
    state_graph = _safe_get(result, "state_graph") or {}
    counterfactuals = _safe_get(experimental, "counterfactual_simulation", "counterfactuals")
    pressure_relief = _safe_list_item(counterfactuals, 0)
    continued_degradation = _safe_list_item(counterfactuals, 2)
    early_warning = _safe_get(result, "early_warning") or {}
    signal_degradation = _safe_get(result, "signal_degradation") or {}
    aux_time_payload = aux_time or unavailable_payload("disabled")

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
        "aux_time_time": aux_time_payload.get("aux_time_time"),
        "aux_time_drift_ms": aux_time_payload.get("drift_ms"),
        "aux_time_wobble_ms": aux_time_payload.get("wobble_ms"),
        "aux_time_live_ms": aux_time_payload.get("live_ms"),
        "aux_time_fractal_factor": aux_time_payload.get("fractal_factor"),
        "aux_time_available": aux_time_payload.get("available", False),
        "aux_time_reason": aux_time_payload.get("reason"),
    }


def replay_unit(
    unit_df: pd.DataFrame,
    *,
    use_aux_time: bool = False,
    aux_time_cache_ms: int = 500,
    max_cycles: int | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    engine = StructuralEngine()
    rows: list[dict[str, Any]] = []
    printed_debug_result = False
    aux_time_client = AUX_TIMEClient(cache_ms=0) if use_aux_time else None
    printed_aux_time_debug = 0
    cache_ttl_ms = aux_time_cache_ms
    last_aux_time_payload: dict[str, Any] | None = None
    last_aux_time_fetch_time = 0.0
    aux_time_api_calls = 0

    ordered = unit_df.sort_values("cycle")
    if max_cycles is not None and max_cycles > 0:
        ordered = ordered.head(int(max_cycles))
    unit_id = int(ordered["unit"].iloc[0])

    def get_cached_aux_time_time() -> dict[str, Any]:
        nonlocal last_aux_time_payload, last_aux_time_fetch_time, aux_time_api_calls
        current_time = time.time() * 1000  # ms
        should_refresh = last_aux_time_payload is None or (current_time - last_aux_time_fetch_time) > cache_ttl_ms
        if verbose:
            print(f"AUX_TIME fetch reused: {not should_refresh}")

        if should_refresh:
            aux_time_api_calls += 1
            try:
                last_aux_time_payload = aux_time_client.get_time() if aux_time_client is not None else unavailable_payload("disabled")
                last_aux_time_fetch_time = current_time
            except Exception:
                # fallback: reuse last known value
                if last_aux_time_payload is None:
                    last_aux_time_payload = unavailable_payload("error")
                last_aux_time_fetch_time = current_time

        if last_aux_time_payload is None:
            return unavailable_payload("unavailable")
        return last_aux_time_payload

    for _, row in ordered.iterrows():
        cycle = int(row["cycle"])
        sensor_values = {sensor: row[sensor] for sensor in SENSOR_COLUMNS}

        timestamp = str(cycle)
        aux_time_payload = unavailable_payload("disabled")
        if aux_time_client is not None:
            aux_time_payload = get_cached_aux_time_time()
            if aux_time_payload.get("available") and aux_time_payload.get("aux_time_time") is not None:
                timestamp = str(aux_time_payload["aux_time_time"])
            else:
                aux_time_payload = {**aux_time_payload, "reason": aux_time_payload.get("reason") or "unavailable"}

            if verbose and printed_aux_time_debug < 3:
                print(
                    "DEBUG AUX_TIME:",
                    f"available={aux_time_payload.get('available', False)}",
                    f"aux_time_time={aux_time_payload.get('aux_time_time')}",
                    f"drift_ms={aux_time_payload.get('drift_ms')}",
                )
                printed_aux_time_debug += 1

        frame = {
            "timestamp": timestamp,
            "site_id": "cmapss",
            "asset_id": f"unit_{unit_id}",
            "sensor_values": sensor_values,
        }
        result = engine.process_frame(frame)
        if verbose and not printed_debug_result:
            print(json.dumps(result, indent=2)[:1000])
            printed_debug_result = True
        rows.append(flatten_result(unit_id=unit_id, cycle=cycle, result=result, aux_time=aux_time_payload))
    if verbose:
        print(f"unit={unit_id} rows={len(rows)} last_cycle={ordered['cycle'].iloc[-1] if len(ordered) else 'n/a'}")

    if aux_time_client is not None and verbose:
        print(f"AUX_TIME API calls for unit {unit_id}: {aux_time_api_calls} / rows={len(ordered)}")

    return rows


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


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()

    if args.summary_only is not None:
        csv_in = args.summary_only.expanduser().resolve()
        if not csv_in.is_file():
            raise FileNotFoundError(f"--summary-only file not found: {csv_in}")
        replay_df = pd.read_csv(csv_in)
        run_transition_postprocess(replay_df, args, output_path=csv_in)
        return

    input_path = args.input.expanduser().resolve()
    dataset = load_dataset(input_path)

    all_rows: list[dict[str, Any]] = []
    unit_ids = sorted(dataset["unit"].unique())
    if args.max_units is not None and args.max_units > 0:
        unit_ids = unit_ids[: int(args.max_units)]

    for unit_id in unit_ids:
        unit_df = dataset[dataset["unit"] == unit_id]
        all_rows.extend(
            replay_unit(
                unit_df,
                use_aux_time=args.use_aux_time,
                aux_time_cache_ms=args.aux_time_cache_ms,
                max_cycles=args.max_cycles,
                verbose=bool(args.verbose),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False)

    print("CMAPSS replay complete")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Units processed: {len(unit_ids)}")
    print(f"Rows written: {len(all_rows)}")

    if not args.replay_only:
        run_transition_postprocess(pd.DataFrame(all_rows), args, output_path=output_path)


if __name__ == "__main__":
    main()
