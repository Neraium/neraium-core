import argparse
from pathlib import Path

import pandas as pd

from run_engine import StructuralEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a CMAPSS FD00x dataset through the structural engine."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to FD00x dataset file (e.g., train_FD004.txt, test_FD001.txt, "
            "test_FD004.txt)"
        ),
    )
    return parser.parse_args()


def resolve_input_path(input_arg: str | None) -> Path:
    base_dir = Path(__file__).resolve().parent

    if input_arg:
        file_path = Path(input_arg).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
    else:
        candidates = [
            base_dir / "train_FD004.txt",
            base_dir / "test_FD004.txt",
            base_dir / "test_FD001.txt",
            Path.cwd() / "train_FD004.txt",
            Path.cwd() / "test_FD004.txt",
            Path.cwd() / "test_FD001.txt",
        ]
        file_path = next((path for path in candidates if path.exists()), candidates[0])

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return file_path


args = parse_args()
file_path = resolve_input_path(args.input)

df = pd.read_csv(file_path, sep=r"\s+", header=None)

columns = (
    ["unit", "cycle"]
    + ["op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

df.columns = columns

results = []

for unit_id in df["unit"].unique():

    engine = StructuralEngine()

    unit_data = df[df["unit"] == unit_id]

    for _, row in unit_data.iterrows():

        sensors = {}

        for s in [f"s{i}" for i in range(1, 22)]:
            sensors[s] = row[s]

        frame = {
            "timestamp": str(row["cycle"]),
            "site_id": "nasa",
            "asset_id": f"engine_{unit_id}",
            "sensor_values": sensors
        }

        result = engine.process_frame(frame)

        results.append({
            "unit": unit_id,
            "cycle": row["cycle"],
            "drift": result["structural_drift_score"],
            "mahal": result["mahalanobis_score"],
            "cov": result["covariance_drift_score"],
            "velocity": result["drift_velocity"],
            "state": result["state"]
        })

results_df = pd.DataFrame(results)

results_df.to_csv("fd004_results.csv", index=False)

print("FD004 replay complete")
print(f"Loaded dataset: {file_path}")
