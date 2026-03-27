#!/usr/bin/env python3
"""
NASA CMAPSS FD004 test-set evaluation: load data, run RUL predictions, plot results.

Usage (from repository root):
    python scripts/run_fd004_test.py

Optional:
    python scripts/run_fd004_test.py --test-path data/fd004/test_FD004.txt --rul-path data/fd004/RUL_FD004.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent

FD004_COLUMNS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)


def load_fd004(test_path: Path) -> pd.DataFrame:
    """Load test_FD004.txt (space-delimited). Drop empty columns."""
    df = pd.read_csv(test_path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    # trim to expected width if extra empty cols were dropped
    n = min(len(FD004_COLUMNS), df.shape[1])
    df = df.iloc[:, :n].copy()
    df.columns = list(FD004_COLUMNS[:n])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_rul_file(rul_path: Path) -> dict[int, int]:
    """RUL_FD004.txt: one RUL value per unit (line order = unit 1, 2, ...)."""
    lines = [ln.strip() for ln in rul_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: dict[int, int] = {}
    for idx, line in enumerate(lines, start=1):
        out[idx] = int(float(line.split()[0]))
    return out


def compute_true_rul(df: pd.DataFrame, rul_at_end: dict[int, int]) -> pd.DataFrame:
    """
    True RUL at each cycle: at the last observed cycle for a unit, RUL equals RUL_FD004;
    each cycle earlier adds 1 per cycle before the end.
    """
    df = df.copy()
    true_rul = np.zeros(len(df), dtype=float)
    for unit_id, g in df.groupby("unit"):
        idx = g.index.values
        cycles = g["cycle"].values
        c_max = float(np.max(cycles))
        r_end = float(rul_at_end.get(int(unit_id), 0))
        # RUL(c) = RUL(c_max) + (c_max - c)  with RUL(c_max) = r_end
        true_rul[idx] = r_end + (c_max - cycles)
    df["true_rul"] = true_rul
    return df


def preprocess_unit(df_unit: pd.DataFrame) -> np.ndarray:
    """Feature matrix for one unit (stub: operating settings + sensors)."""
    feature_cols = [c for c in df_unit.columns if c not in ("unit", "cycle", "true_rul")]
    return df_unit[feature_cols].to_numpy(dtype=np.float64)


def get_model():
    """
    Placeholder RUL model. Replace with a trained model that exposes .predict(X).

    predict(X) expects X shape (n_timesteps, n_features) for one engine and returns
    shape (n_timesteps,) predicted RUL at each cycle.
    """
    return PlaceholderRULModel()


class PlaceholderRULModel:
    """Simple monotonic RUL proxy for end-to-end pipeline testing."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.array([], dtype=float)
        n = X.shape[0]
        # crude health scalar from sensor magnitude (not a real model)
        w = float(np.mean(np.abs(X)))
        end_rul = float(np.clip(80.0 - 0.02 * w, 0.0, 125.0))
        start_rul = end_rul + (n - 1)
        return np.linspace(start_rul, end_rul, n)


def run_inference(
    df: pd.DataFrame,
    model,
) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    """
    Returns:
        pred_final: unit_id -> predicted RUL at last cycle
        pred_trajectories: unit_id -> full predicted RUL per row (same order as df rows for that unit)
    """
    pred_final: dict[int, float] = {}
    pred_trajectories: dict[int, np.ndarray] = {}

    for unit_id, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        X = preprocess_unit(g)
        traj = np.asarray(model.predict(X), dtype=float)
        pred_trajectories[int(unit_id)] = traj
        pred_final[int(unit_id)] = float(traj[-1]) if len(traj) else float("nan")

    return pred_final, pred_trajectories


def plot_results(
    *,
    units: list[int],
    true_final: np.ndarray,
    pred_final: np.ndarray,
    errors: np.ndarray,
    unit1_cycles: np.ndarray,
    unit1_pred: np.ndarray,
    unit1_true: np.ndarray | None,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) True vs Predicted RUL (final per unit)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    u = np.arange(len(units))
    ax1.plot(u, true_final, label="True RUL (final)", marker="o", ms=3)
    ax1.plot(u, pred_final, label="Predicted RUL (final)", marker="x", ms=3)
    ax1.set_xlabel("Unit index (sorted)")
    ax1.set_ylabel("RUL")
    ax1.set_title("FD004: True vs Predicted RUL (last cycle per unit)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "fd004_true_vs_pred.png", dpi=150)
    plt.close(fig1)

    # 2) Error curve (pred - true)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(u, errors, color="C2", marker="o", ms=3)
    ax2.axhline(0.0, color="gray", lw=0.8)
    ax2.set_xlabel("Unit index (sorted)")
    ax2.set_ylabel("Predicted - True")
    ax2.set_title("FD004: RUL error (final)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fd004_error.png", dpi=150)
    plt.close(fig2)

    # 3) Degradation: unit 1 — cycle vs predicted RUL (and true if provided)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(unit1_cycles, unit1_pred, label="Predicted RUL", color="C1")
    if unit1_true is not None and len(unit1_true) == len(unit1_cycles):
        ax3.plot(unit1_cycles, unit1_true, label="True RUL", color="C0", alpha=0.8)
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("RUL")
    ax3.set_title("FD004: Degradation (unit 1)")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "fd004_degradation_unit1.png", dpi=150)
    plt.close(fig3)

    print(f"Saved plots to {out_dir.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FD004 test evaluation and plots")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "data" / "fd004" / "test_FD004.txt",
        help="Path to test_FD004.txt",
    )
    parser.add_argument(
        "--rul-path",
        type=Path,
        default=REPO_ROOT / "data" / "fd004" / "RUL_FD004.txt",
        help="Path to RUL_FD004.txt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "fd004",
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    if not args.test_path.is_file():
        print(f"Missing test file: {args.test_path}", file=sys.stderr)
        print("Place NASA FD004 test_FD004.txt under data/fd004/ or pass --test-path.", file=sys.stderr)
        return 1
    if not args.rul_path.is_file():
        print(f"Missing RUL file: {args.rul_path}", file=sys.stderr)
        return 1

    df = load_fd004(args.test_path)
    rul_at_end = load_rul_file(args.rul_path)
    df = compute_true_rul(df, rul_at_end)

    model = get_model()
    pred_final_map, pred_traj = run_inference(df, model)

    # Align units with true final RUL (value at last cycle)
    units_sorted = sorted(pred_final_map.keys())
    true_final_list: list[float] = []
    for u in units_sorted:
        sub = df[df["unit"] == u]
        c_max = sub["cycle"].max()
        tf = float(sub.loc[sub["cycle"] == c_max, "true_rul"].iloc[0])
        true_final_list.append(tf)
    true_final = np.array(true_final_list)
    pred_final = np.array([pred_final_map[u] for u in units_sorted])
    errors = pred_final - true_final

    # Unit 1 trajectory
    u1 = 1
    g1 = df[df["unit"] == u1].sort_values("cycle")
    unit1_cycles = g1["cycle"].to_numpy()
    unit1_true = g1["true_rul"].to_numpy()
    unit1_pred = pred_traj.get(u1, np.array([]))

    try:
        plot_results(
            units=units_sorted,
            true_final=true_final,
            pred_final=pred_final,
            errors=errors,
            unit1_cycles=unit1_cycles,
            unit1_pred=unit1_pred,
            unit1_true=unit1_true,
            out_dir=args.out_dir,
        )
    except ImportError as e:
        print(f"matplotlib not available ({e}); skipping plots.", file=sys.stderr)
        return 0

    rmse = float(np.sqrt(np.mean(errors**2)))
    print(f"RMSE (final RUL): {rmse:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
