from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neraium_core.alignment import StructuralEngine

DATA_DIR = Path("/content/data")  # change to your folder
DATASET = "FD004"  # FD001, FD002, FD003, FD004
TETRA_VERTICES = {
    "STRUCTURAL": (1.0, 1.0, 1.0),
    "RELATIONAL": (1.0, -1.0, -1.0),
    "TRANSITION": (-1.0, 1.0, -1.0),
    "TEMPORAL": (-1.0, -1.0, 1.0),
}
TETRA_EDGES = [
    ("STRUCTURAL", "RELATIONAL"),
    ("STRUCTURAL", "TRANSITION"),
    ("STRUCTURAL", "TEMPORAL"),
    ("RELATIONAL", "TRANSITION"),
    ("RELATIONAL", "TEMPORAL"),
    ("TRANSITION", "TEMPORAL"),
]


def _plot_tetra_trajectory(trajectory: list[list[float]], unit: int) -> None:
    import matplotlib.pyplot as plt

    if not trajectory:
        print(f"no tetrahedral trajectory points found for unit {unit}")
        return

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    zs = [p[2] for p in trajectory]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    progression = [i / max(len(xs) - 1, 1) for i in range(len(xs))]
    colors = plt.cm.viridis(progression)

    for i in range(1, len(xs)):
        recency = (i - 1) / max(len(xs) - 2, 1)
        ax.plot(
            [xs[i - 1], xs[i]],
            [ys[i - 1], ys[i]],
            [zs[i - 1], zs[i]],
            color=colors[i],
            linewidth=1.0 + 2.0 * recency,
            alpha=0.25 + 0.7 * recency,
        )

    ax.scatter(xs, ys, zs, c=progression, cmap="viridis", s=12, label="trajectory")
    ax.scatter(xs[0], ys[0], zs[0], color="tab:green", s=70, marker="o", label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="tab:red", s=70, marker="^", label="end")

    vx = [coord[0] for coord in TETRA_VERTICES.values()]
    vy = [coord[1] for coord in TETRA_VERTICES.values()]
    vz = [coord[2] for coord in TETRA_VERTICES.values()]
    ax.scatter(vx, vy, vz, color="black", s=30, alpha=0.8, label="tetra vertices")

    for a, b in TETRA_EDGES:
        xa, ya, za = TETRA_VERTICES[a]
        xb, yb, zb = TETRA_VERTICES[b]
        ax.plot([xa, xb], [ya, yb], [za, zb], color="gray", alpha=0.25, linewidth=1.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{DATASET} tetrahedral trajectory (unit {unit})")
    ax.legend(loc="best")

    output_path = Path(f"fd00x_tetra_unit_{unit}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close(fig)
    print("saved", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FD00x structural engine")
    parser.add_argument("--plot-tetra", action="store_true", help="Plot tetrahedral trajectory for selected unit")
    parser.add_argument("--unit", type=int, default=1, help="Unit id to collect tetrahedral trajectory")
    args = parser.parse_args()

    file_path = DATA_DIR / f"test_{DATASET}.txt"

    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = ["unit", "cycle"] + [f"os{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

    engine = StructuralEngine(
        baseline_window=50,
        recent_window=12,
    )
    results = []
    tetra_trajectory: list[list[float]] = []
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

        if args.plot_tetra and int(row["unit"]) == args.unit:
            tetra_state = out.get("tetrahedral_state") if isinstance(out, dict) else None
            position = tetra_state.get("position") if isinstance(tetra_state, dict) else None
            if isinstance(position, (list, tuple)) and len(position) == 3:
                tetra_trajectory.append([float(position[0]), float(position[1]), float(position[2])])

        global_t += 1.0

    out_df = pd.DataFrame(results)
    out_df.to_csv(f"{DATASET}_results.csv", index=False)

    print("saved", f"{DATASET}_results.csv")
    print(out_df.head())
    print(out_df.columns.tolist())

    if args.plot_tetra:
        _plot_tetra_trajectory(tetra_trajectory, args.unit)


if __name__ == "__main__":
    main()
