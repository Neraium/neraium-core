import argparse
import os

import pandas as pd

from neraium_core.engine import StructuralEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay CMAPSS FD004 through StructuralEngine")
    parser.add_argument(
        "--input",
        default=os.getenv("FD004_INPUT"),
        required=os.getenv("FD004_INPUT") is None,
        help="Path to CMAPSS train_FD004.txt (or set FD004_INPUT)",
    )
    parser.add_argument("--output", default="fd004_results.csv", help="Output CSV path")
    return parser.parse_args()


def run_fd004(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, sep=r"\s+", header=None)
    columns = ["unit", "cycle"] + ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = columns

    results = []
    for unit_id in df["unit"].unique():
        engine = StructuralEngine()
        unit_data = df[df["unit"] == unit_id]

        for _, row in unit_data.iterrows():
            frame = {
                "timestamp": str(row["cycle"]),
                "site_id": "nasa",
                "asset_id": f"engine_{unit_id}",
                "sensor_values": {f"s{i}": row[f"s{i}"] for i in range(1, 22)},
            }
            result = engine.process_frame(frame)
            results.append(
                {
                    "unit": unit_id,
                    "cycle": row["cycle"],
                    "drift": result["structural_drift_score"],
                    "mahal": result["mahalanobis_score"],
                    "cov": result["covariance_drift_score"],
                    "velocity": result["drift_velocity"],
                    "state": result["state"],
                }
            )

    pd.DataFrame(results).to_csv(output_path, index=False)
    print("FD004 replay complete")


if __name__ == "__main__":
    args = parse_args()
    run_fd004(args.input, args.output)
