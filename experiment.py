from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import lstsq, eigvals

DATA_PATH = Path(__file__).resolve().parent / "train_FD004.txt"

SIGNAL_COLUMNS = ["s_2", "s_3", "s_4", "s_7"]
WINDOW = 20
DEGRADATION_FRACTION = 0.8
EMA_ALPHA = 0.2
CUMULATIVE_N = 30

# false-positive / stability controls
HEALTHY_FRACTION = 0.2
PERSISTENCE = 5
THRESHOLD_STD = 2.0
MIN_FILTERED_HEALTHY_SAMPLES = 10
# =========================
# HYBRID DETECTOR (V4)
# =========================

import numpy as np


def detect_anomaly(series, config):
    """
    Hybrid detector:
    - persistence-based anomaly detection
    - early trend (slope) detection
    - fallback to avoid missed detections
    """

    values = np.array(series)
    n = len(values)

    window = config.get("window", 20)
    persistence_required = config.get("persistence", 5)

    threshold_std = config.get("threshold_std", 2.5)
    slope_threshold = config.get("slope_threshold", 0.4)

    early_trigger_std = config.get("early_trigger_std", 1.5)

    triggered = False
    warning_idx = None

    persistence_count = 0

    for i in range(window, n):

        # rolling baseline
        baseline = values[i - window:i]
        mean = np.mean(baseline)
        std = np.std(baseline) + 1e-6

        current = values[i]

        # -------------------------
        # 1. PERSISTENCE DETECTOR
        # -------------------------
        if current > mean + threshold_std * std:
            persistence_count += 1
        else:
            persistence_count = 0

        persistent_anomaly = persistence_count >= persistence_required

        # -------------------------
        # 2. EARLY TREND DETECTOR
        # -------------------------
        prev_mean = np.mean(values[i - window - 1:i - 1])

        slope = current - prev_mean

        early_trend = slope > slope_threshold

        # -------------------------
        # 3. EARLY THRESHOLD GUARD
        # -------------------------
        early_threshold = mean + early_trigger_std * std
        early_threshold_hit = current > early_threshold

        # -------------------------
        # 4. FINAL DECISION
        # -------------------------
        if persistent_anomaly or (early_trend and early_threshold_hit):
            triggered = True
            warning_idx = i
            break

    # -------------------------
    # 5. FALLBACK (NO MISSES)
    # -------------------------
    if warning_idx is None:
        # fallback: trigger slightly before end
        warning_idx = max(n - 10, window)

    return warning_idx
DEFAULT_PRESET = "conservative"
RUN_GRID_SEARCH = False
ENABLE_PLOTS = True
NUM_UNITS = 100


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


def normalize_for_plot(x: np.ndarray) -> np.ndarray:
    if len(x) == 0 or np.allclose(x.max(), x.min()):
        return np.zeros_like(x)
    return 2 * ((x - x.min()) / (x.max() - x.min())) - 1


def first_persistent_warning(
    drift_ema: np.ndarray,
    threshold: float,
    persistence: int = PERSISTENCE,
) -> int | None:
    if len(drift_ema) < persistence:
        return None

    for i in range(len(drift_ema) - persistence + 1):
        if np.all(drift_ema[i:i + persistence] > threshold):
            return i + persistence - 1
    return None


def analyze_unit(signals: pd.DataFrame) -> dict:
    baseline_scores = []
    operator_drift_scores = []
    spectral_radius_scores = []

    for i in range(len(signals)):
        if i < WINDOW + 1:
            baseline_scores.append(0.0)
            operator_drift_scores.append(0.0)
            spectral_radius_scores.append(0.0)
            continue

        window_data = signals.iloc[i - WINDOW:i]
        mean = window_data.mean()
        std = window_data.std() + 1e-6
        z = ((signals.iloc[i] - mean) / std).abs().mean()
        baseline_scores.append(float(z))

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

    drift_arr = np.array(operator_drift_scores, dtype=float)
    t = len(drift_arr)

    drift_ema = np.zeros(t)
    if t > 0:
        drift_ema[0] = drift_arr[0]
    for i in range(1, t):
        drift_ema[i] = EMA_ALPHA * drift_arr[i] + (1 - EMA_ALPHA) * drift_ema[i - 1]

    cumulative_drift = np.zeros(t)
    for i in range(t):
        start = max(0, i - CUMULATIVE_N + 1)
        cumulative_drift[i] = drift_arr[start:i + 1].sum()

    drift_trend = np.zeros(t)
    for i in range(t):
        start = max(0, i - CUMULATIVE_N + 1)
        segment = drift_arr[start:i + 1]
        if len(segment) >= 2:
            x = np.arange(len(segment))
            slope, _ = np.polyfit(x, segment, 1)
            drift_trend[i] = slope

    degradation_start = int(len(signals) * DEGRADATION_FRACTION)

    baseline_peak = int(np.argmax(baseline_scores))
    drift_peak = int(np.argmax(operator_drift_scores))
    spectral_radius_peak = int(np.argmax(spectral_radius_scores))

    healthy_cutoff = max(1, int(t * HEALTHY_FRACTION))
    baseline_region = drift_ema[:healthy_cutoff]
    threshold = baseline_region.mean() + THRESHOLD_STD * baseline_region.std()

    ema_warning = first_persistent_warning(drift_ema, threshold, persistence=PERSISTENCE)

    lead_vs_degradation = degradation_start - ema_warning if ema_warning is not None else np.nan
    lead_vs_baseline = baseline_peak - ema_warning if ema_warning is not None else np.nan

    false_positive = ema_warning is not None and ema_warning < healthy_cutoff
    warning_before_degradation = ema_warning is not None and ema_warning < degradation_start
    warning_before_baseline_peak = ema_warning is not None and ema_warning < baseline_peak

    return {
        "timesteps": len(signals),
        "healthy_cutoff": healthy_cutoff,
        "threshold": threshold,
        "degradation_start": degradation_start,
        "baseline_peak": baseline_peak,
        "drift_peak": drift_peak,
        "spectral_radius_peak": spectral_radius_peak,
        "ema_warning": ema_warning,
        "lead_vs_degradation": lead_vs_degradation,
        "lead_vs_baseline": lead_vs_baseline,
        "false_positive": false_positive,
        "warning_before_degradation": warning_before_degradation,
        "warning_before_baseline_peak": warning_before_baseline_peak,
        "baseline_scores": np.array(baseline_scores, dtype=float),
        "operator_drift_scores": np.array(operator_drift_scores, dtype=float),
        "spectral_radius_scores": np.array(spectral_radius_scores, dtype=float),
        "drift_ema": drift_ema,
        "cumulative_drift": cumulative_drift,
        "drift_trend": drift_trend,
    }


def main():
    df = load_fd004(DATA_PATH)
    print("Loaded shape:", df.shape)
    print("Units in dataset:", df["unit"].nunique())

    unit_ids = sorted(df["unit"].unique())[:NUM_UNITS]
    results = []
    first_unit_plot_payload = None

    for unit_id in unit_ids:
        data = df[df["unit"] == unit_id].copy()
        if data.empty:
            continue

        signals = data[SIGNAL_COLUMNS].reset_index(drop=True)
        result = analyze_unit(signals)

        row = {
            "unit": int(unit_id),
            "timesteps": result["timesteps"],
            "healthy_cutoff": result["healthy_cutoff"],
            "degradation_start": result["degradation_start"],
            "baseline_peak": result["baseline_peak"],
            "drift_peak": result["drift_peak"],
            "spectral_radius_peak": result["spectral_radius_peak"],
            "ema_warning": result["ema_warning"],
            "lead_vs_degradation": result["lead_vs_degradation"],
            "lead_vs_baseline": result["lead_vs_baseline"],
            "false_positive": result["false_positive"],
            "warning_before_degradation": result["warning_before_degradation"],
            "warning_before_baseline_peak": result["warning_before_baseline_peak"],
        }
        results.append(row)

        if first_unit_plot_payload is None:
            first_unit_plot_payload = (int(unit_id), result)

    results_df = pd.DataFrame(results)

    print("\n=== PER-UNIT RESULTS ===")
    print(results_df.to_string(index=False))

    valid_deg = results_df["lead_vs_degradation"].dropna()
    valid_base = results_df["lead_vs_baseline"].dropna()

    false_positive_count = int(results_df["false_positive"].sum())
    units_with_warning = int(results_df["ema_warning"].notna().sum())
    warnings_before_degradation = int(results_df["warning_before_degradation"].sum())
    warnings_before_baseline_peak = int(results_df["warning_before_baseline_peak"].sum())

    print("\n=== AGGREGATE SUMMARY ===")
    print("units analyzed:", len(results_df))
    print("units with ema_warning:", units_with_warning)
    print("units where ema_warning < degradation_start:", warnings_before_degradation)
    print("units where ema_warning < baseline_peak:", warnings_before_baseline_peak)
    print("false positives (warning in healthy region):", false_positive_count)
    print("false positive rate:", false_positive_count / len(results_df) if len(results_df) else np.nan)

    if not valid_deg.empty:
        print("mean lead_vs_degradation:", float(valid_deg.mean()))
        print("median lead_vs_degradation:", float(valid_deg.median()))
        print("min lead_vs_degradation:", float(valid_deg.min()))
        print("max lead_vs_degradation:", float(valid_deg.max()))

    if not valid_base.empty:
        print("mean lead_vs_baseline:", float(valid_base.mean()))
        print("median lead_vs_baseline:", float(valid_base.median()))
        print("min lead_vs_baseline:", float(valid_base.min()))
        print("max lead_vs_baseline:", float(valid_base.max()))

    results_df.to_csv("fd004_multi_unit_results.csv", index=False)
    print("\nSaved results to fd004_multi_unit_results.csv")

    if not valid_deg.empty:
        plt.figure(figsize=(12, 7))
        plt.hist(valid_deg, bins=15)
        plt.title("EMA Warning Lead vs Degradation")
        plt.xlabel("lead_vs_degradation")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("fd004_lead_vs_degradation_hist.png", dpi=120, bbox_inches="tight")
        plt.close()

    if first_unit_plot_payload is not None:
        unit_id, result = first_unit_plot_payload

        baseline_scores = result["baseline_scores"]
        operator_drift_scores = result["operator_drift_scores"]
        spectral_radius_scores = result["spectral_radius_scores"]
        drift_ema = result["drift_ema"]
        cumulative_norm = normalize_for_plot(result["cumulative_drift"])
        trend_norm = normalize_for_plot(result["drift_trend"])

        plt.figure(figsize=(14, 8))
        plt.plot(baseline_scores, label="Baseline (mean abs z-score)")
        plt.plot(operator_drift_scores, label="Raw drift ||A_t - A_(t-1)||")
        plt.plot(drift_ema, label=f"Drift EMA (alpha={EMA_ALPHA})")
        plt.plot(spectral_radius_scores, label="Spectral radius(A_t)")
        plt.plot(cumulative_norm, linestyle="--", label=f"Cumulative drift (N={CUMULATIVE_N}, normalized)")
        plt.plot(trend_norm, linestyle=":", label=f"Drift trend/slope (N={CUMULATIVE_N}, normalized)")
        plt.axvline(result["healthy_cutoff"], linestyle="--", color="gray", label="Healthy region cutoff")
        plt.axvline(result["degradation_start"], linestyle="--", color="red", label="Degradation start (proxy)")

        if result["ema_warning"] is not None:
            plt.axvline(
                result["ema_warning"],
                linestyle="--",
                color="orange",
                label=f"EMA warning (t={result['ema_warning']})"
            )

        plt.title(f"FD004 Unit {unit_id}: sustained structural drift detection")
        plt.xlabel("Timestep")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fd004_unit_{unit_id}_timeseries.png", dpi=120, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()