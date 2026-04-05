from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, eigvals

DATA_PATH = Path(r"C:\Users\Owner\Desktop\CMAPSSData\train_FD004.txt")

UNIT_ID = 1
SIGNAL_COLUMNS = ["s_2", "s_3", "s_4", "s_7"]
WINDOW = 20
DEGRADATION_FRACTION = 0.8

EMA_ALPHA = 0.2
CUMULATIVE_N = 30


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


def main():
    df = load_fd004(DATA_PATH)
    print("Loaded shape:", df.shape)
    print("Units:", df["unit"].nunique())

    data = df[df["unit"] == UNIT_ID].copy()
    if data.empty:
        raise ValueError(f"No rows found for unit {UNIT_ID}")

    signals = data[SIGNAL_COLUMNS].reset_index(drop=True)

    # Normalize for numerical stability
    signals = (signals - signals.mean()) / (signals.std() + 1e-6)

    print(f"Using unit {UNIT_ID} with {len(signals)} timesteps")
    print("Signals:", SIGNAL_COLUMNS)

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

    print("\n=== SUMMARY ===")
    print("Degradation zone starts at timestep:", degradation_start)
    print("Max baseline score:", max(baseline_scores))
    print("Max operator drift:", max(operator_drift_scores))
    print("Max spectral radius:", max(spectral_radius_scores))
    print("Max drift EMA:", drift_ema.max())
    print("Max cumulative drift:", cumulative_drift.max())
    print("Max drift trend (slope):", drift_trend.max())

    # crude early-warning comparison
    baseline_peak_idx = int(np.argmax(baseline_scores))
    drift_peak_idx = int(np.argmax(operator_drift_scores))
    spectral_peak_idx = int(np.argmax(spectral_radius_scores))

    print("\n=== PEAK TIMESTEPS ===")
    print("Baseline peak:", baseline_peak_idx)
    print("Operator drift peak:", drift_peak_idx)
    print("Spectral radius peak:", spectral_peak_idx)

    # --- Early warning: drift_ema threshold based on early baseline region ---
    early_end = max(1, int(T * 0.20))
    early_ema = drift_ema[:early_end]
    ema_threshold = early_ema.mean() + 2 * early_ema.std()

    ema_warning_idx = None
    for t in range(T):
        if drift_ema[t] > ema_threshold:
            ema_warning_idx = t
            break

    print("\n=== EARLY WARNING (drift_ema > mean + 2*std of first 20%) ===")
    print(f"EMA threshold: {ema_threshold:.4f}")
    if ema_warning_idx is not None:
        print(f"drift_ema first exceeds threshold at timestep: {ema_warning_idx}")
        print(f"Baseline peak at timestep:                     {baseline_peak_idx}")
        lead = baseline_peak_idx - ema_warning_idx
        print(f"Lead over baseline peak: {lead} timesteps ({'earlier' if lead > 0 else 'later or simultaneous'})")
    else:
        print("drift_ema never exceeded threshold")
        print(f"Baseline peak at timestep: {baseline_peak_idx}")

    # --- Plot ---
    plt.figure(figsize=(14, 8))
    plt.plot(baseline_scores, label="Baseline (mean abs z-score)", alpha=0.7)
    plt.plot(operator_drift_scores, label="Raw drift ||A_t - A_{t-1}||", alpha=0.6)
    plt.plot(drift_ema, label=f"Drift EMA (α={EMA_ALPHA})", linewidth=2)
    plt.plot(cumulative_drift / (cumulative_drift.max() + 1e-9),
             label=f"Cumulative drift (N={CUMULATIVE_N}, normalized)", linestyle="--")
    plt.plot(drift_trend / (np.abs(drift_trend).max() + 1e-9),
             label=f"Drift trend/slope (N={CUMULATIVE_N}, normalized)", linestyle=":")
    plt.axvline(degradation_start, linestyle="--", color="red", label="Degradation start (proxy)")
    if ema_warning_idx is not None:
        plt.axvline(ema_warning_idx, linestyle="--", color="orange", label=f"EMA warning (t={ema_warning_idx})")
    plt.title(f"FD004 Unit {UNIT_ID}: sustained structural drift detection")
    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
