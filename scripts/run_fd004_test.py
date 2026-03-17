from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay CMAPSS FD004 through StructuralEngine")
    parser.add_argument("--input", default=os.getenv("FD004_PATH"), help="Path to train_FD004.txt")
    parser.add_argument("--output", default="fd004_results.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input:
        raise SystemExit("Provide --input or set FD004_PATH")

    import pandas as pd

    from neraium_core.engine import StructuralEngine

    df = pd.read_csv(args.input, sep=r"\s+", header=None)
    columns = ["unit", "cycle"] + ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    df.columns = columns

    results = []
    sensor_names = [f"s{i}" for i in range(1, 22)]

    for unit_id in df["unit"].unique():
        engine = StructuralEngine()
        unit_data = df[df["unit"] == unit_id]

        for _, row in unit_data.iterrows():
            frame = {
                "timestamp": str(row["cycle"]),
                "site_id": "nasa",
                "asset_id": f"engine_{unit_id}",
                "sensor_values": {sensor: row[sensor] for sensor in sensor_names},
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

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"FD004 replay complete -> {args.output}")


if __name__ == "__main__":
    main()
