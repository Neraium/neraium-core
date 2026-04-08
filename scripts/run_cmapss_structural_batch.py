from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neraium_core.alignment import StructuralEngine

CMAPSS_COLUMNS = ["unit", "cycle"] + [f"os{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
DEFAULT_DATASETS = ("FD001", "FD002", "FD003", "FD004")


def run_dataset(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    *,
    baseline_window: int,
    recent_window: int,
) -> Path:
    file_path = data_dir / f"test_{dataset}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"dataset file not found: {file_path}")

    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = CMAPSS_COLUMNS

    engine = StructuralEngine(
        baseline_window=baseline_window,
        recent_window=recent_window,
    )

    results: list[dict] = []
    global_t = 0.0

    for _, row in df.iterrows():
        frame = {
            "timestamp": global_t,
            "site_id": "nasa_cmapss",
            "asset_id": f"engine_{int(row['unit'])}",
            "sensor_values": {
                f"os{i}": float(row[f"os{i}"]) for i in range(1, 4)
            }
            | {
                f"s{i}": float(row[f"s{i}"]) for i in range(1, 22)
            },
        }

        out = engine.process_frame(frame)
        out["unit"] = int(row["unit"])
        out["cycle"] = int(row["cycle"])
        results.append(out)
        global_t += 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_results.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StructuralEngine over CMAPSS test datasets.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing test_FD00x.txt files.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for output CSVs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to process (default: FD001 FD002 FD003 FD004).",
    )
    parser.add_argument("--baseline-window", type=int, default=50)
    parser.add_argument("--recent-window", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        output_path = run_dataset(
            dataset,
            args.data_dir,
            args.output_dir,
            baseline_window=args.baseline_window,
            recent_window=args.recent_window,
        )
        row_count = sum(1 for _ in output_path.open("r", encoding="utf-8")) - 1
        print(f"finished {dataset} rows: {row_count} -> {output_path}")


if __name__ == "__main__":
    main()
