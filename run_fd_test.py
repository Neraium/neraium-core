import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from neraium_core.integrations.gal2_client import GAL2Client, unavailable_payload
from run_engine import StructuralEngine


CMAPSS_COLUMNS = ["unit", "cycle"] + ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a CMAPSS FD00x dataset through Neraium StructuralEngine and write CSV results."
    )
    parser.add_argument("--input", type=Path, default=Path("test_FD001.txt"), help="Path to CMAPSS dataset text file.")
    parser.add_argument("--output", type=Path, default=Path("fd_results.csv"), help="Path to output CSV file.")
    parser.add_argument("--use-gal2", action="store_true", help="Use GAL-2 aligned time as an optional timestamp source.")
    parser.add_argument("--gal2-cache-ms", type=int, default=0, help="Optional GAL-2 cache duration in milliseconds.")
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
    gal2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experimental = _safe_get(result, "experimental_analytics") or {}
    counterfactuals = _safe_get(experimental, "counterfactual_simulation", "counterfactuals")
    pressure_relief = _safe_list_item(counterfactuals, 0)
    continued_degradation = _safe_list_item(counterfactuals, 2)
    gal2_payload = gal2 or unavailable_payload("disabled")

    return {
        "unit": unit_id,
        "cycle": cycle,
        "state": _safe_get(result, "state"),
        "structural_drift_score": _safe_get(result, "structural_drift_score"),
        "latest_instability": _safe_get(result, "latest_instability"),
        "transition_pressure": _safe_get(result, "transition_pressure"),
        "transition_state": _safe_get(result, "transition_state"),
        "confidence_score": _safe_get(result, "confidence_score"),
        "interpreted_state": _safe_get(result, "interpreted_state"),
        "regime_name": _safe_get(result, "regime_name"),
        "trajectory_analysis_dominant_path": _safe_get(experimental, "trajectory_analysis", "dominant_path"),
        "trajectory_analysis_path_confidence": _safe_get(experimental, "trajectory_analysis", "path_confidence"),
        "branching_analysis_is_branching": _safe_get(experimental, "branching_analysis", "is_branching"),
        "branching_analysis_commitment": _safe_get(experimental, "branching_analysis", "commitment"),
        "branching_analysis_decision_tension": _safe_get(experimental, "branching_analysis", "decision_tension"),
        "constraint_analysis_lock_in_score": _safe_get(experimental, "constraint_analysis", "lock_in_score"),
        "constraint_analysis_point_of_no_return_risk": _safe_get(experimental, "constraint_analysis", "point_of_no_return_risk"),
        "constraint_analysis_recovery_margin": _safe_get(experimental, "constraint_analysis", "recovery_margin"),
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
        "gal2_time": gal2_payload.get("gal2_time"),
        "gal2_drift_ms": gal2_payload.get("drift_ms"),
        "gal2_wobble_ms": gal2_payload.get("wobble_ms"),
        "gal2_live_ms": gal2_payload.get("live_ms"),
        "gal2_fractal_factor": gal2_payload.get("fractal_factor"),
        "gal2_available": gal2_payload.get("available", False),
        "gal2_reason": gal2_payload.get("reason"),
    }


def replay_unit(
    unit_df: pd.DataFrame,
    *,
    use_gal2: bool = False,
    gal2_cache_ms: int = 0,
) -> list[dict[str, Any]]:
    engine = StructuralEngine()
    rows: list[dict[str, Any]] = []
    printed_debug_result = False
    gal2_client = GAL2Client(cache_ms=gal2_cache_ms) if use_gal2 else None
    printed_gal2_debug = 0

    ordered = unit_df.sort_values("cycle")
    unit_id = int(ordered["unit"].iloc[0])

    for _, row in ordered.iterrows():
        cycle = int(row["cycle"])
        sensor_values = {sensor: row[sensor] for sensor in SENSOR_COLUMNS}

        timestamp = str(cycle)
        gal2_payload = unavailable_payload("disabled")
        if gal2_client is not None:
            gal2_payload = gal2_client.get_time()
            if gal2_payload.get("available") and gal2_payload.get("gal2_time") is not None:
                timestamp = str(gal2_payload["gal2_time"])
            else:
                gal2_payload = {**gal2_payload, "reason": gal2_payload.get("reason") or "unavailable"}

            if printed_gal2_debug < 3:
                print(
                    "DEBUG GAL2:",
                    f"available={gal2_payload.get('available', False)}",
                    f"gal2_time={gal2_payload.get('gal2_time')}",
                    f"drift_ms={gal2_payload.get('drift_ms')}",
                )
                printed_gal2_debug += 1

        frame = {
            "timestamp": timestamp,
            "site_id": "cmapss",
            "asset_id": f"unit_{unit_id}",
            "sensor_values": sensor_values,
        }
        result = engine.process_frame(frame)
        if not printed_debug_result:
            print(json.dumps(result, indent=2)[:1000])
            printed_debug_result = True
        rows.append(flatten_result(unit_id=unit_id, cycle=cycle, result=result, gal2=gal2_payload))

    return rows


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    dataset = load_dataset(input_path)

    all_rows: list[dict[str, Any]] = []
    unit_ids = sorted(dataset["unit"].unique())

    for unit_id in unit_ids:
        unit_df = dataset[dataset["unit"] == unit_id]
        all_rows.extend(replay_unit(unit_df, use_gal2=args.use_gal2, gal2_cache_ms=args.gal2_cache_ms))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False)

    print("CMAPSS replay complete")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Units processed: {len(unit_ids)}")
    print(f"Rows written: {len(all_rows)}")


if __name__ == "__main__":
    main()
