import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from run_engine import StructuralEngine


CMAPSS_COLUMNS = ["unit", "cycle"] + ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a CMAPSS FD00x dataset through Neraium StructuralEngine and write CSV results."
    )
    parser.add_argument("--input", type=Path, default=Path("test_FD001.txt"), help="Path to CMAPSS dataset text file.")
    parser.add_argument("--output", type=Path, default=Path("fd_results.csv"), help="Path to output CSV file.")
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


def flatten_result(unit_id: int, cycle: int, result: dict[str, Any]) -> dict[str, Any]:
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
        "trajectory_analysis_dominant_path": _safe_get(result, "trajectory_analysis", "dominant_path"),
        "branching_analysis_is_branching": _safe_get(result, "branching_analysis", "is_branching"),
        "branching_analysis_commitment": _safe_get(result, "branching_analysis", "commitment"),
        "constraint_analysis_lock_in_score": _safe_get(result, "constraint_analysis", "lock_in_score"),
        "constraint_analysis_point_of_no_return_risk": _safe_get(result, "constraint_analysis", "point_of_no_return_risk"),
        "hierarchy_analysis_origin_scope": _safe_get(result, "hierarchy_analysis", "origin_scope"),
        "hierarchy_analysis_propagation_risk": _safe_get(result, "hierarchy_analysis", "propagation_risk"),
        "horizon_analysis_risk_horizon": _safe_get(result, "horizon_analysis", "risk_horizon"),
    }


def replay_unit(unit_df: pd.DataFrame) -> list[dict[str, Any]]:
    engine = StructuralEngine()
    rows: list[dict[str, Any]] = []

    ordered = unit_df.sort_values("cycle")
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
        rows.append(flatten_result(unit_id=unit_id, cycle=cycle, result=result))

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
        all_rows.extend(replay_unit(unit_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False)

    print("CMAPSS replay complete")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Units processed: {len(unit_ids)}")
    print(f"Rows written: {len(all_rows)}")


if __name__ == "__main__":
    main()
