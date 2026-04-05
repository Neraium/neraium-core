from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, eigvals

DATA_PATH = Path(__file__).resolve().parent / "train_FD004.txt"

SIGNAL_COLUMNS = ["s_2", "s_3", "s_4", "s_7"]
WINDOW = 20
DEGRADATION_FRACTION = 0.8

EMA_ALPHA = 0.2
CUMULATIVE_N = 30

THRESHOLD_STD = 2.0
PERSISTENCE = 1
RUN_GRID_SEARCH = True


def load_fd004(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1)

    expected_cols = 26
    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}")

    cols = ["unit", "time"] + [f"op_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]
    df.columns = cols
    return df


def fit_var1(X: np.ndarray) -> np.ndarray:
    X_t = X[:-1]
    X_next = X[1:]
    A, _, _, _ = lstsq(X_t, X_next, rcond=None)
    return A


def run_multi_unit_experiment() -> pd.DataFrame:
    df = load_fd004(DATA_PATH)
    print("Loaded shape:", df.shape)
    print("Units:", df["unit"].nunique())

    units = sorted(df["unit"].unique())[:25]
    results = []

    print(f"Evaluating first {len(units)} units:", units)
    print("Signals:", SIGNAL_COLUMNS)

    for unit in units:
        data = df[df["unit"] == unit].copy()
        if data.empty:
            continue

        signals = data[SIGNAL_COLUMNS].reset_index(drop=True)

        # Normalize for numerical stability
        signals = (signals - signals.mean()) / (signals.std() + 1e-6)

        baseline_scores = []
        operator_drift_scores = []
        spectral_radius_scores = []

        for i in range(len(signals)):
            if i < WINDOW + 1:
                baseline_scores.append(0.0)
                operator_drift_scores.append(0.0)
                spectral_radius_scores.append(0.0)
                continue

            # Baseline z-score
            window_data = signals.iloc[i - WINDOW:i]
            mean = window_data.mean()
            std = window_data.std() + 1e-6
            z = ((signals.iloc[i] - mean) / std).abs().mean()
            baseline_scores.append(float(z))

            # Operator-based metrics
            X_prev = signals.iloc[i - WINDOW - 1:i - 1].values
            X_curr = signals.iloc[i - WINDOW:i].values

            try:
                A_prev = fit_var1(X_prev)
                A_curr = fit_var1(X_curr)

                drift = np.linalg.norm(A_curr - A_prev)
                spectral_radius = np.max(np.abs(eigvals(A_curr)))

            except Exception:
                drift = 0.0
                spectral_radius = 0.0

            operator_drift_scores.append(float(drift))
            spectral_radius_scores.append(float(spectral_radius))

        # --- Smoothed drift signals ---
        T = len(operator_drift_scores)
        drift_arr = np.array(operator_drift_scores)

        # Exponential moving average of drift (alpha=0.2)
        drift_ema = np.zeros(T)
        drift_ema[0] = drift_arr[0]
        for t in range(1, T):
            drift_ema[t] = EMA_ALPHA * drift_arr[t] + (1 - EMA_ALPHA) * drift_ema[t - 1]

        # Cumulative drift over last N=30 steps
        cumulative_drift = np.zeros(T)
        for t in range(T):
            start = max(0, t - CUMULATIVE_N + 1)
            cumulative_drift[t] = drift_arr[start:t + 1].sum()

        # Rolling linear slope of drift over window N=30 (drift_trend)
        drift_trend = np.zeros(T)
        for t in range(T):
            start = max(0, t - CUMULATIVE_N + 1)
            segment = drift_arr[start:t + 1]
            if len(segment) >= 2:
                x = np.arange(len(segment))
                slope, _ = np.polyfit(x, segment, 1)
                drift_trend[t] = slope

        degradation_start = int(len(signals) * DEGRADATION_FRACTION)

        baseline_peak_idx = int(np.argmax(baseline_scores))
        drift_peak_idx = int(np.argmax(operator_drift_scores))

        # --- Early warning: drift_ema threshold based on early baseline region ---
        early_end = max(1, int(T * 0.20))
        early_ema = drift_ema[:early_end]
        ema_threshold = early_ema.mean() + THRESHOLD_STD * early_ema.std()

        ema_warning_idx = None
        consecutive = 0
        for t in range(T):
            if drift_ema[t] > ema_threshold:
                consecutive += 1
                if consecutive >= PERSISTENCE:
                    ema_warning_idx = t
                    break
            else:
                consecutive = 0

        lead_vs_degradation = (
            degradation_start - ema_warning_idx if ema_warning_idx is not None else None
        )
        lead_vs_baseline = (
            baseline_peak_idx - ema_warning_idx if ema_warning_idx is not None else None
        )

        results.append(
            {
                "unit": int(unit),
                "timesteps": len(signals),
                "degradation_start": degradation_start,
                "baseline_peak": baseline_peak_idx,
                "drift_peak": drift_peak_idx,
                "ema_warning": ema_warning_idx,
                "lead_vs_degradation": lead_vs_degradation,
                "lead_vs_baseline": lead_vs_baseline,
            }
        )

    return pd.DataFrame(results)


def print_and_plot_results(results_df: pd.DataFrame) -> None:
    print("\n=== PER-UNIT RESULTS ===")
    print(results_df.to_string(index=False))

    valid_deg = results_df["lead_vs_degradation"].dropna()
    valid_base = results_df["lead_vs_baseline"].dropna()

    print("\n=== AGGREGATE SUMMARY ===")
    print("mean lead_vs_degradation:", valid_deg.mean() if not valid_deg.empty else None)
    print("median lead_vs_degradation:", valid_deg.median() if not valid_deg.empty else None)
    print("mean lead_vs_baseline:", valid_base.mean() if not valid_base.empty else None)
    print(
        "units where ema_warning < degradation_start:",
        int((results_df["lead_vs_degradation"] > 0).sum()),
    )
    print(
        "units where ema_warning < baseline_peak:",
        int((results_df["lead_vs_baseline"] > 0).sum()),
    )

    plt.figure(figsize=(10, 6))
    plt.hist(valid_deg.values, bins=20, edgecolor="black", alpha=0.8)
    plt.title("EMA Warning Lead vs Degradation")
    plt.xlabel("lead_vs_degradation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def run_experiment_with_params(threshold_std: float, persistence: int) -> dict:
    global THRESHOLD_STD, PERSISTENCE

    prev_threshold = THRESHOLD_STD
    prev_persistence = PERSISTENCE

    THRESHOLD_STD = threshold_std
    PERSISTENCE = persistence

    try:
        results_df = run_multi_unit_experiment()
    finally:
        THRESHOLD_STD = prev_threshold
        PERSISTENCE = prev_persistence

    false_positive_rate = (results_df["ema_warning"] >= results_df["degradation_start"]).mean()
    coverage = results_df["ema_warning"].notna().mean()
    mean_lead = results_df["lead_vs_degradation"].dropna().mean()

    return {
        "threshold": threshold_std,
        "persistence": persistence,
        "false_positive_rate": false_positive_rate,
        "coverage": coverage,
        "mean_lead": mean_lead,
    }


def grid_search() -> pd.DataFrame:
    thresholds = [1.5, 2.0, 2.5, 3.0]
    persistences = [3, 5, 8]
    records = []

    for threshold in thresholds:
        for persistence in persistences:
            row = run_experiment_with_params(threshold, persistence)
            records.append(row)
            print(
                f"threshold={threshold}, persistence={persistence} "
                f"-> FP={row['false_positive_rate']:.2f}, "
                f"coverage={row['coverage']:.2f}, lead={row['mean_lead']:.1f}"
            )

    return pd.DataFrame(records)


def score_row(row: pd.Series) -> float:
    return (
        row["mean_lead"] * 1.0
        + row["coverage"] * 50
        - row["false_positive_rate"] * 200
    )


def main():
    results_df = run_multi_unit_experiment()
    print_and_plot_results(results_df)


def run_grid_search_workflow():
    results_df = grid_search()
    results_df["score"] = results_df.apply(score_row, axis=1)
    ranked = results_df.sort_values("score", ascending=False).reset_index(drop=True)

    print("\n=== BEST CONFIG ===")
    print(ranked.iloc[0].to_string())

    print("\n=== RANKED CONFIGS ===")
    print(ranked.to_string(index=False))

    ranked.to_csv("fd004_grid_search_results.csv", index=False)


if __name__ == "__main__":
    if RUN_GRID_SEARCH:
        run_grid_search_workflow()
    else:
        main()
